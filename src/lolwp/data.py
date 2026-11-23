"""Load matches from SQLite into arrays: draft, player-history features, auxiliary targets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import db

# Per-player inputs. HISTORY_FEATURES and STAT_FEATURES are computed from games that *started
# before* the game being predicted; PREGAME_FEATURES describe the game itself but are fixed
# before it starts (account level, role assignment, summoner spells, keystone rune).
HISTORY_FEATURES = (
    "champion_games",     # log1p(games on this champion) / 3
    "champion_winrate",   # shrunk towards 50%: (wins - games/2) / (games + 4)
    "total_games",        # log1p(games seen) / 4
    "total_winrate",      # (wins - games/2) / (games + 8)
    "role_share",         # share of their games in today's role (0 if no games or role unknown)
    "streak",             # current win (+) / loss (-) streak, clipped to +-5, / 5
    "hours_since_last",   # log1p(hours since previous game, capped at a week) / 5
    "recent_winrate",     # exponential average of results (weight 0.1 per game), centred on 0
    "gold_margin",        # mean scaled team gold margin of their games, shrunk: sum / (games + 5)
    "kill_margin",        # same for kills
    "tower_margin",       # same for towers
    "champion_gold_margin",  # gold margin on this champion: sum / (games + 3)
    "role_winrate",       # (wins - games/2) / (games + 6) in today's role
    "game_length",        # mean standardised game length: sum / (games + 5)
    "games_24h",          # log1p(games in the last 24 hours) / 3
    "games_3h",           # log1p(games in the last 3 hours) / 2  (current session)
    "duo",                # log1p(most earlier games shared with one of today's teammates) / 2
    "champion_pool",      # log1p(distinct champions played) / 3
    "champion_recency",   # exp(-days since last game on this champion / 3); 0 if never
)

# Performance in earlier games, from matches.pstats. Each value compares the player with the
# opponent in the same role (with the enemy team average if roles are unknown), so that roles
# and champions with naturally high numbers do not look skilled. Stored as sum / (games + 5).
PERFORMANCE = (
    "gold", "damage", "cs", "level", "vision", "kda", "death_share", "kill_participation",
    "objective_damage", "time_dead", "lane_lead", "early_lane_lead", "max_cs_lead",
    "solo_kills", "plates", "afk", "cs_at_10", "first_blood", "control_wards",
)
RECENT = ("gold", "damage", "cs", "vision", "kda", "death_share", "kill_participation",
          "lane_lead")
RECENT_WINDOW = 10
IN_ROLE = ("gold", "kda", "lane_lead")
ADJUSTED = ("gold", "damage", "kda", "death_share")
STAT_FEATURES = tuple(f"perf_{name}" for name in PERFORMANCE) + (
    "perf_games",            # log1p(earlier games with statistics) / 4
    "champion_perf_gold",    # comparisons on today's champion: sum / (games + 3)
    "champion_perf_kda",
    "primary_role_rate",     # share of earlier games in which they got their primary role
    "autofill_rate",         # ... in which they were autofilled
    "champion_perf_damage",
    "champion_perf_lane_lead",
) + tuple(f"recent_{name}" for name in RECENT) + (   # last RECENT_WINDOW games: sum / (games + 3)
    "recent_games",          # games in that window / RECENT_WINDOW
) + tuple(f"role_perf_{name}" for name in IN_ROLE) + (   # in today's role: sum / (games + 3)
    "role_perf_games",       # log1p(earlier games with statistics in today's role) / 4
) + tuple(f"adjusted_{name}" for name in ADJUSTED) + (
    # ^ the comparison minus its average for games with the same result, so that a player
    #   who was carried and one who carried no longer look alike: sum / (games + 5)
    "matchup_games",         # log1p(earlier games against today's lane opponent's champion) / 2
    "matchup_winrate",       # (wins - games/2) / (games + 3) in those games
    "matchup_gold",          # gold comparison in those games: sum / (games + 3)
)
SPELLS = (4, 14, 12, 11, 7, 3, 6, 21, 1)   # Flash Ignite Teleport Smite Heal Exhaust Ghost Barrier Cleanse
KEYSTONES = (8005, 8008, 8021, 8010, 8112, 8128, 9923, 8214, 8229, 8230, 8437, 8439, 8465,
             8351, 8360, 8369, 8124)
PREGAME_FEATURES = (
    "pregame_known",         # 1 if the inputs below are available for this player
    "account_level",         # log1p(summoner level) / 7
    "role_primary", "role_secondary", "role_autofill",
) + tuple(f"spell_{i}" for i in SPELLS) + tuple(f"keystone_{i}" for i in KEYSTONES)

PLAYER_FEATURES = HISTORY_FEATURES + STAT_FEATURES + PREGAME_FEATURES

# Training-only targets (never inputs): three that flip sign with the side, one that does not.
AUX_TARGETS = ("gold_diff", "kill_diff", "tower_diff", "duration")
N_AUX_ANTI = 3


@dataclass
class DraftData:
    champs: np.ndarray       # [N, 10] int64 vocabulary indices; columns 0-4 blue, 5-9 red
    roles: np.ndarray        # [N, 10] int64; 1..5 = TOP..UTILITY, 0 = unknown
    y: np.ndarray            # [N] float32, 1 = blue side won
    weight: np.ndarray       # [N] float32 sample weight (older patches count less)
    patch: np.ndarray        # [N] str
    time: np.ndarray         # [N] int64 game creation, ms; rows are sorted by this
    feats: np.ndarray        # [N, 10, F] float32 player-history features (PLAYER_FEATURES)
    known: np.ndarray        # [N, 10] bool: the player's id was available
    aux: np.ndarray          # [N, 4] float32 auxiliary targets (AUX_TARGETS), roughly unit scale
    aux_mask: np.ndarray     # [N, 4] bool
    champion_ids: list[int]  # vocabulary: index i+1 <-> champion_ids[i]; index 0 = unknown
    patches: list[str]       # patches used, newest first
    clean_after: int | None  # ms; after this, every platform has had its first ladder snapshot
    clean: np.ndarray | None = None  # [N] bool: played after the first snapshot of its platform,
                                     # i.e. free of the ladder-snapshot selection bias

    def __len__(self) -> int:
        return len(self.y)


def select_patches(counts: list[tuple[str, int]], min_matches: int, max_patches: int) -> list[str]:
    """Newest patch first; add older patches only while there is too little data."""
    chosen, total = [], 0
    for patch, n in counts:  # newest first
        if chosen and (total >= min_matches or len(chosen) >= max_patches):
            break
        chosen.append(patch)
        total += n
    return chosen


def load_champion_names(path: str | Path, fetch: bool = True) -> dict[int, dict]:
    """{championId: {"id": ..., "name": ...}} from data/champions.json (or Data Dragon)."""
    path = Path(path)
    if not path.is_file():
        if not fetch:
            return {}
        from .riot import ddragon_champions, ddragon_versions
        version = ddragon_versions()[0]
        champs = ddragon_champions(version)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"version": version, "champions": {str(k): v for k, v in sorted(champs.items())}},
            indent=1))
        return champs
    return {int(k): v for k, v in json.loads(path.read_text())["champions"].items()}


class PlayerHistory:
    """Running per-player record. For every game call `features` for all ten players first,
    then `update` for all ten and `add_team` for both teams."""

    RECENT = 25  # timestamps kept per player for the games_24h / games_3h counts

    def __init__(self):
        # puuid -> [games, wins, last_ms, streak, role games x5 (4..8), role wins x5 (9..13),
        #           recent winrate, games with stats, gold, kills, towers, length sum (14..19),
        #           recent start times]
        self.total: dict[str, list] = {}
        # (puuid, champion id) -> [games, wins, games with stats, gold margin sum, last_ms]
        self.champ: dict[tuple[str, int], list] = {}
        self.pool: dict[str, int] = {}          # puuid -> distinct champions
        self.ids: dict[str, int] = {}           # puuid -> small int, for the pair keys
        self.pairs: dict[int, int] = {}         # (id_a << 32 | id_b), a < b -> games as teammates

    def _pair(self, a: str, b: str) -> int | None:
        ia, ib = self.ids.get(a), self.ids.get(b)
        if ia is None or ib is None:
            return None
        return (ia << 32 | ib) if ia < ib else (ib << 32 | ia)

    def features(self, puuid: str, champion: int, role: int, now_ms: int,
                 teammates: tuple[str, ...] | list[str] = ()) -> list[float]:
        """role: 0..4, or -1 if unknown. teammates: the puuids on the same team today."""
        t = self.total.get(puuid)
        if t is None:
            return [0.0] * len(HISTORY_FEATURES)
        n, w = t[0], t[1]
        cn, cw, cm, cgold, clast = self.champ.get((puuid, champion), (0, 0, 0, 0.0, 0))
        hours = min(max((now_ms - t[2]) / 3.6e6, 0.0), 168.0)
        shared = 0
        for other in teammates:
            if other != puuid:
                key = self._pair(puuid, other)
                if key is not None:
                    shared = max(shared, self.pairs.get(key, 0))
        nm = t[15]
        return [
            math.log1p(cn) / 3,
            (cw - cn / 2) / (cn + 4),
            math.log1p(n) / 4,
            (w - n / 2) / (n + 8),
            t[4 + role] / n if role >= 0 else 0.0,
            max(-5, min(5, t[3])) / 5,
            math.log1p(hours) / 5,
            t[14],
            t[16] / (nm + 5),
            t[17] / (nm + 5),
            t[18] / (nm + 5),
            cgold / (cm + 3),
            (t[9 + role] - t[4 + role] / 2) / (t[4 + role] + 6) if role >= 0 else 0.0,
            t[19] / (n + 5),
            math.log1p(sum(1 for u in t[20] if now_ms - u < 86.4e6)) / 3,
            math.log1p(sum(1 for u in t[20] if now_ms - u < 10.8e6)) / 2,
            math.log1p(shared) / 2,
            math.log1p(self.pool.get(puuid, 0)) / 3,
            math.exp(-max(now_ms - clast, 0) / 2.592e8) if cn else 0.0,
        ]

    def update(self, puuid: str, champion: int, role: int, won: bool, now_ms: int,
               margins: list[float] | None = None, length: float = 0.0) -> None:
        """margins: this player's team's scaled (gold, kill, tower) margins, if known;
        length: standardised game length."""
        t = self.total.get(puuid)
        if t is None:
            t = self.total[puuid] = [0, 0, now_ms, 0] + [0] * 10 + [0.0, 0, 0.0, 0.0, 0.0, 0.0, []]
        t[0] += 1
        t[1] += won
        t[2] = now_ms
        t[3] = (max(t[3], 0) + 1) if won else (min(t[3], 0) - 1)
        if role >= 0:
            t[4 + role] += 1
            t[9 + role] += won
        t[14] = 0.9 * t[14] + 0.1 * (0.5 if won else -0.5)
        t[19] += length
        recent = t[20]
        recent.append(now_ms)
        if len(recent) > self.RECENT:
            del recent[0]
        c = self.champ.get((puuid, champion))
        if c is None:
            c = self.champ[(puuid, champion)] = [0, 0, 0, 0.0, now_ms]
            self.pool[puuid] = self.pool.get(puuid, 0) + 1
        c[0] += 1
        c[1] += won
        c[4] = now_ms
        if margins is not None:
            t[15] += 1
            t[16] += margins[0]
            t[17] += margins[1]
            t[18] += margins[2]
            c[2] += 1
            c[3] += margins[0]

    def add_team(self, players: list[str]) -> None:
        """Count one more game together for every pair of these teammates."""
        ids = [self.ids.setdefault(p, len(self.ids)) for p in players]
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                key = (a << 32 | b) if a < b else (b << 32 | a)
                self.pairs[key] = self.pairs.get(key, 0) + 1


def _prior_sums(keys: np.ndarray, values: np.ndarray, valid: np.ndarray, window: int = 0):
    """For every row: the sum of `values` over *earlier* valid rows with the same key (only
    the last `window` of them if window > 0), and how many rows that sum covers.
    Rows must already be in time order. values [E, P], valid [E]."""
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    ok = valid[order]
    first = np.r_[True, k[1:] != k[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(k)), 0))
    before = np.cumsum(ok) - ok          # valid rows before this one, over all keys
    low = before[start]                  # ... before the first row of this key
    if window:
        low = np.maximum(low, before - window)
    sums = np.empty(values.shape, dtype=np.float32)
    counts = np.empty(len(k), dtype=np.int64)
    counts[order] = before - low
    for col in range(values.shape[1]):   # one column at a time keeps the float64 copies small
        running = np.r_[0.0, np.cumsum(values[order, col][ok].astype(np.float64))]
        sums[order, col] = running[before] - running[low]
    return sums, counts


def match_performance(ps: np.ndarray, ok: np.ndarray, duration: np.ndarray) -> np.ndarray:
    """[M, 10, len(PERFORMANCE)] comparison of every player with their opponent.
    ps: [M, 10, len(PSTAT_FIELDS)] raw statistics; ok: [M] roles known; duration: [M] seconds."""
    from .riot import PSTAT as I
    m = len(ps)

    def opponent(x):
        swapped = np.concatenate([x[:, 5:], x[:, :5]], axis=1)
        mean = np.repeat(swapped.reshape(m, 2, 5).mean(axis=2), 5, axis=1)
        return np.where(ok[:, None], swapped, mean)

    def team(x):
        return np.repeat(x.reshape(m, 2, 5).sum(axis=2), 5, axis=1)

    def rel(name, k, scale):
        x = ps[:, :, I[name]]
        xo = opponent(x)
        return (x - xo) / (x + xo + k) / scale

    def diff(name, scale, clip=3.0):
        x = ps[:, :, I[name]]
        return np.clip((x - opponent(x)) / scale, -clip, clip)

    kills, deaths, assists = (ps[:, :, I[n]] for n in ("kills", "deaths", "assists"))
    kda = np.log((kills + assists + 1) / (deaths + 1))
    kp = (kills + assists) / (team(kills) + 1)
    dead = ps[:, :, I["time_dead"]] / np.maximum(duration, 1)[:, None]
    out = np.stack([
        rel("gold", 1, 0.1), rel("damage", 1, 0.25), rel("cs", 20, 0.15),
        diff("champ_level", 2), rel("vision", 5, 0.2), kda - opponent(kda),
        (deaths / (team(deaths) + 1) - 0.2) / 0.1, (kp - opponent(kp)) / 0.2,
        rel("objective_damage", 500, 0.4), np.clip((dead - opponent(dead)) / 0.04, -3, 3),
        2 * ps[:, :, I["lane_lead"]] - 1, 2 * ps[:, :, I["early_lane_lead"]] - 1,
        diff("max_cs_lead", 20), diff("solo_kills", 2), diff("plates", 3),
        ps[:, :, I["afk"]], rel("cs_at_10", 10, 0.15), ps[:, :, I["first_blood"]] - 0.2,
        rel("control_wards", 2, 0.4),
    ], axis=-1)
    return out.astype(np.float32)


def stat_features(ps: np.ndarray, has: np.ndarray, ok: np.ndarray, duration: np.ndarray,
                  pid: np.ndarray, champ: np.ndarray, blue_win: np.ndarray) -> np.ndarray:
    """[M, 10, len(STAT_FEATURES) + len(PREGAME_FEATURES)] for matches in time order.

    ps: [M, 10, len(PSTAT_FIELDS)] raw per-player statistics (zeros where missing);
    has: [M] the match has statistics; ok: [M] roles known; pid: [M, 10] player index,
    -1 = unknown; champ: [M, 10] champion ids; blue_win: [M].
    History columns use earlier matches only; the pre-game columns read nothing but the
    fields of this match that are fixed before it starts."""
    from .riot import PSTAT as I
    m = len(ps)
    col = PERFORMANCE.index
    valid = (has[:, None] & (pid >= 0)).ravel()
    keys = pid.ravel().astype(np.int64)
    assignment = ps[:, :, I["role_assignment"]]
    perf = match_performance(ps, ok, duration).reshape(m * 10, len(PERFORMANCE))
    won = np.concatenate([np.repeat(blue_win[:, None], 5, axis=1),
                          np.repeat(~blue_win[:, None], 5, axis=1)], axis=1).ravel()
    slot_role = np.where(ok[:, None], np.tile(np.arange(5), 2)[None, :], -1).ravel()
    enemy = np.where(ok[:, None], np.concatenate([champ[:, 5:], champ[:, :5]], axis=1), -1).ravel()

    def shrunk(key, columns, k, window=0, rows=valid):
        sums, n = _prior_sums(key, columns, rows, window)
        return sums / (n[:, None] + k), n

    overall, n = shrunk(keys, perf, 5)
    assigned = np.stack([assignment.ravel() == 1, assignment.ravel() == 3], axis=1)
    rates, _ = shrunk(keys, assigned.astype(np.float32), 2, rows=valid & (assignment.ravel() > 0))
    on_champ, _ = shrunk(keys * 4096 + champ.ravel(),
                         perf[:, [col("gold"), col("kda"), col("damage"), col("lane_lead")]], 3)
    recent, n_recent = shrunk(keys, perf[:, [col(c) for c in RECENT]], 3, RECENT_WINDOW)
    in_role, n_role = shrunk(keys * 8 + slot_role, perf[:, [col(c) for c in IN_ROLE]], 3,
                             rows=valid & (slot_role >= 0))
    # what a winner / a loser showed on average in all *earlier* matches, removed from each
    # game before averaging (a running mean, so that no later game can influence it)
    adjusted = perf[:, [col(c) for c in ADJUSTED]].copy()
    for result in (False, True):
        rows = valid & (won == result)
        per_match = (adjusted * rows[:, None]).reshape(m, 10, -1).sum(axis=1, dtype=np.float64)
        count = rows.reshape(m, 10).sum(axis=1)
        before = np.cumsum(per_match, axis=0) - per_match
        n_before = np.cumsum(count) - count
        mean = (before / np.maximum(n_before, 1)[:, None]).astype(np.float32)
        adjusted -= np.repeat(mean, 10, axis=0) * (won == result)[:, None]
    adjusted, _ = shrunk(keys, adjusted, 5)
    versus = np.stack([won - 0.5, perf[:, col("gold")]], axis=1).astype(np.float32)
    matchup, n_matchup = shrunk(keys * 4096 + enemy, versus, 3, rows=valid & (enemy >= 0))

    history = np.concatenate([
        overall, (np.log1p(n) / 4)[:, None], on_champ[:, :2], rates, on_champ[:, 2:],
        recent, (n_recent / RECENT_WINDOW)[:, None],
        in_role, (np.log1p(n_role) / 4)[:, None],
        adjusted,
        (np.log1p(n_matchup) / 2)[:, None], matchup,
    ], axis=1)
    history[keys < 0] = 0

    spells = ps[:, :, [I["spell1"], I["spell2"]]]
    pregame = np.concatenate([
        np.ones((m, 10, 1), dtype=np.float32),
        (np.log1p(ps[:, :, I["summoner_level"]]) / 7)[..., None],
        np.stack([assignment == 1, assignment == 2, assignment == 3], axis=-1),
        np.stack([(spells == s).any(axis=-1) for s in SPELLS], axis=-1),
        np.stack([ps[:, :, I["keystone"]] == k for k in KEYSTONES], axis=-1),
    ], axis=-1).astype(np.float32) * has[:, None, None]
    return np.concatenate([history.reshape(m, 10, -1), pregame], axis=-1).astype(np.float32)


def aux_targets(stats_json: str | None, duration: int) -> tuple[list[float], list[bool]]:
    """Blue-minus-red margins scaled to roughly unit variance, plus standardised game length."""
    dur = (duration - 1750) / 420
    if not stats_json:
        return [0.0, 0.0, 0.0, dur], [False, False, False, True]
    s = json.loads(stats_json)
    (gb, gr), (kb, kr), (tb, tr) = s["gold"], s["kills"], s["towers"]
    return [
        (gb - gr) / max(gb + gr, 1) / 0.08,
        (kb - kr) / (kb + kr + 1) / 0.25,
        (tb - tr) / 11 / 0.5,
        dur,
    ], [True, True, True, True]


def load_draft_data(
    db_path: str | Path,
    min_matches: int = 150_000,
    max_patches: int = 3,
    patches: list[str] | None = None,
    patch_decay: float = 0.7,
    extra_champion_ids: tuple[int, ...] | list[int] = (),
) -> DraftData:
    if not Path(db_path).is_file():
        raise SystemExit(f"{db_path} not found - run lolwp-collect first")
    conn = db.connect(db_path)
    try:
        counts = db.patch_counts(conn)
        if not counts:
            raise SystemExit(f"{db_path} contains no matches - run lolwp-collect first")
        if patches:
            available = {p for p, _ in counts}
            missing = [p for p in patches if p not in available]
            if missing:
                raise SystemExit(f"no matches for patch(es) {missing}; available: {sorted(available)}")
            chosen = sorted(set(patches), key=db.patch_key, reverse=True)
        else:
            chosen = select_patches(counts, min_matches, max_patches)
        clean_after = db.clean_cutoff_ms(conn)
        seeds = db.initial_seeds_ms(conn)
        # Every stored match feeds the player histories, also those of patches not trained on.
        rows = conn.execute(
            "SELECT blue, red, roles_ok, blue_win, patch, game_creation, duration, "
            "blue_puuids, red_puuids, stats, pstats, platform FROM matches "
            "ORDER BY game_creation, match_id"
        ).fetchall()
    finally:
        conn.close()

    use = set(chosen)
    n = sum(1 for r in rows if r[4] in use)
    n_feat = len(HISTORY_FEATURES)
    from .riot import PSTAT_FIELDS
    m = len(rows)
    selected_rows = np.zeros(m, dtype=bool)
    pid = np.full((m, 10), -1, dtype=np.int64)
    champ_all = np.zeros((m, 10), dtype=np.int64)
    ps = np.zeros((m, 10, len(PSTAT_FIELDS)), dtype=np.float32)
    has_ps = np.zeros(m, dtype=bool)
    player_index: dict[str, int] = {}
    raw = np.empty((n, 10), dtype=np.int64)
    roles_ok = np.empty(n, dtype=bool)
    y = np.empty(n, dtype=np.float32)
    patch = np.empty(n, dtype=object)
    t = np.empty(n, dtype=np.int64)
    feats = np.zeros((n, 10, n_feat), dtype=np.float32)
    known = np.zeros((n, 10), dtype=bool)
    clean = np.zeros(n, dtype=bool)
    aux = np.zeros((n, len(AUX_TARGETS)), dtype=np.float32)
    aux_mask = np.zeros((n, len(AUX_TARGETS)), dtype=bool)

    history = PlayerHistory()
    i = 0
    for j, (blue, red, ok, win, p, created, duration, blue_ids, red_ids, stats,
            pstats, platform) in enumerate(rows):
        champions = [int(c) for c in blue.split(",")] + [int(c) for c in red.split(",")]
        players = (blue_ids.split(",") + red_ids.split(",")) if blue_ids and red_ids else None
        if players is not None and len(players) != 10:
            players = None
        selected = p in use
        selected_rows[j] = selected
        champ_all[j] = champions
        if players is not None:
            pid[j] = [player_index.setdefault(x, len(player_index)) for x in players]
            if pstats:
                try:
                    ps[j] = json.loads(pstats)
                    has_ps[j] = True
                except ValueError:   # other shape than this version expects: treat as missing
                    pass
        if selected:
            raw[i] = champions
            roles_ok[i], y[i], patch[i], t[i] = bool(ok), win, p, created
            clean[i] = platform in seeds and created > seeds[platform]
            aux[i], aux_mask[i] = aux_targets(stats, duration)
        if players is not None:
            if selected:
                for slot, (puuid, champion) in enumerate(zip(players, champions)):
                    team = players[:5] if slot < 5 else players[5:]
                    feats[i, slot] = history.features(
                        puuid, champion, slot % 5 if ok else -1, created, team)
                    known[i, slot] = True
            values, mask = aux_targets(stats, duration)
            for slot, (puuid, champion) in enumerate(zip(players, champions)):  # after reading!
                won = bool(win) if slot < 5 else not win
                sign = 1.0 if slot < 5 else -1.0
                history.update(puuid, champion, slot % 5 if ok else -1, won, created,
                               [sign * v for v in values[:3]] if mask[0] else None, values[3])
            history.add_team(players[:5])
            history.add_team(players[5:])
        i += selected

    extra = stat_features(
        ps, has_ps, np.array([bool(r[2]) for r in rows], dtype=bool),
        np.array([r[6] for r in rows], dtype=np.float32), pid, champ_all,
        np.array([bool(r[3]) for r in rows], dtype=bool))
    del ps
    feats = np.concatenate([feats, extra[selected_rows]], axis=-1)

    champion_ids = sorted(set(np.unique(raw).tolist()) | {int(c) for c in extra_champion_ids})
    lookup = {c: i + 1 for i, c in enumerate(champion_ids)}
    champs = np.vectorize(lookup.__getitem__, otypes=[np.int64])(raw) if n else raw

    role_row = np.tile(np.arange(1, 6, dtype=np.int64), 2)
    roles = np.where(roles_ok[:, None], role_row[None, :], 0)

    age = {p: i for i, p in enumerate(chosen)}  # 0 = newest patch
    weight = np.array([patch_decay ** age[p] for p in patch], dtype=np.float32)

    return DraftData(champs, roles, y, weight, patch.astype(str), t, feats, known, aux, aux_mask,
                     champion_ids, chosen, clean_after, clean)


def chronological_split(n: int, val_frac: float, test_frac: float):
    """Oldest games train, newest games test: mirrors predicting future games."""
    if not (0 < val_frac < 1 and 0 <= test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError("need 0 < val_frac, 0 <= test_frac, val_frac + test_frac < 1")
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_val - n_test
    idx = np.arange(n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
