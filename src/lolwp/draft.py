"""Pick and ban recommendations on top of the win model.

    lolwp-draft pick --side blue --role mid --ally top=Aatrox jg=Vi --enemy Ahri Jinx --bans Zed
    lolwp-draft ban  --side red --ally mid=Syndra

How it decides
    The win model only scores complete 5v5 drafts, so an unfinished draft is valued by
    Monte Carlo: every open slot is filled by sampling champions the way Diamond+ players
    actually pick them in that role (pick shares from the training games), and the model's
    win probability is averaged over those completions ("expected win rate against the field").
    The same sampled completions are reused for every candidate, so candidates are compared
    on identical games (common random numbers).

    pick  For each candidate in your role: expected win rate with that champion locked in.
          For the best few it also searches the enemy's open roles for the reply that hurts
          most (a one-step lookahead in the spirit of Chen et al. 2018, "The Art of Drafting",
          who run tree search over a win predictor).
    ban   Solo queue bans are blind and come before all picks. For each champion:
          gain = P(enemy picks it) x (win rate without it - win rate when the enemy has it).
          "net" additionally subtracts what your own team loses by not being able to pick it.

    --me  With your Riot ID (--me "Name#TAG" --platform euw1) the pick ranking is personal:
          your recent ranked games are fetched (lolwp.live) and the model is told who is on
          each candidate - games and results on it, how you did against your lane opponents,
          form, role. `draft %` shows the same pick for an anonymous player.

Roles: allies are given as role=Champion. Enemy roles may be omitted; they are inferred from
how each champion is usually played.

Bans and pick shares: champions that are banned a lot are picked more often than their raw
pick share says in the games where they get through. Once the bans are known (--bans with at
least BANS_KNOWN champions) open slots are sampled from pick shares given availability.
"""

from __future__ import annotations

import argparse
import itertools
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .predict import Predictor

ROLE_NAMES = ("top", "jungle", "mid", "bot", "support")
BANS_KNOWN = 6          # with this many bans entered, the ban phase is taken to be over
PERSONAL_MIN_GAMES = 2  # own champions outside the usual picks of a role are still candidates
ROLE_ALIASES = {
    "top": 0, "jungle": 1, "jg": 1, "jgl": 1, "jung": 1, "mid": 2, "middle": 2,
    "bot": 3, "bottom": 3, "adc": 3, "support": 4, "sup": 4, "supp": 4, "utility": 4,
}


def parse_role(name: str) -> int:
    try:
        return ROLE_ALIASES[name.strip().lower()]
    except KeyError:
        raise ValueError(f"unknown role {name!r} (use top, jungle, mid, bot, support)") from None


def role_games_from_db(db_path: str | Path, predictor: Predictor) -> np.ndarray:
    """[n_champions + 1, 5] games per vocabulary index and role, for the model's patches.

    Fallback for checkpoints trained before role counts were stored in them."""
    if not Path(db_path).is_file():
        raise SystemExit(
            "this model file has no role statistics and the match database was not found - "
            "rerun lolwp-train (or pass --db)")
    counts = np.zeros((len(predictor.index) + 1, 5), dtype=np.int64)
    patches = predictor.ckpt["patches"]
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            f"SELECT blue, red FROM matches WHERE roles_ok=1 AND patch IN "
            f"({','.join('?' * len(patches))})", patches)
        for blue, red in rows:
            for team in (blue, red):
                for role, cid in enumerate(team.split(",")):
                    idx = predictor.index.get(int(cid))
                    if idx:
                        counts[idx, role] += 1
    finally:
        conn.close()
    return counts


@dataclass
class PickOption:
    champion: str
    win: float                 # expected P(my team wins) with this pick
    vs_field: float            # win minus the pick-rate weighted average candidate
    spread: float              # std of `win` across ensemble members (model disagreement)
    role_games: int            # training games of this champion in this role
    counter: str | None = None        # enemy reply that hurts most, e.g. "Fizz (mid)"
    win_if_countered: float | None = None
    champion_id: int = 0              # Riot champion id
    draft_win: float | None = None    # the same pick by an anonymous player (set with --me)
    my_games: int | None = None       # the player's recent ranked games / wins on it (--me)
    my_wins: int | None = None


@dataclass
class BanOption:
    champion: str
    enemy_pick_chance: float   # P(some open enemy slot picks it)
    win_if_enemy_has: float    # expected P(my team wins) when the enemy has it
    gain: float                # expected win probability gained by denying it to the enemy
    net: float                 # gain minus what my own team loses by not having it
    spread: float              # std of `gain` across ensemble members
    champion_id: int = 0       # Riot champion id
    ban_rate: float = 0.0      # share of training games in which it was banned


class DraftAdvisor:
    def __init__(self, predictor: Predictor, role_games: np.ndarray | None = None,
                 db_path: str | Path = "data/matches.sqlite", samples: int | None = None,
                 min_role_share: float = 0.005, seed: int = 0):
        """samples: sampled completions per option; default 256 on a GPU, 96 on a CPU.
        The enemy-reply search, by far the largest part of a pick request, uses two ensemble
        members and a quarter of the samples to stay responsive."""
        self.p = predictor
        samples = samples or (256 if predictor.device.type != "cpu" else 96)
        self.search_members = 2
        if role_games is None:
            stored = predictor.ckpt.get("champion_role_games")
            if stored is not None:
                role_games = np.vstack([np.zeros((1, 5), dtype=np.int64), np.asarray(stored)])
            else:
                role_games = role_games_from_db(db_path, predictor)
        self.role_games = np.asarray(role_games, dtype=np.float64)   # [V + 1, 5], row 0 unused
        if self.role_games.shape != (len(predictor.index) + 1, 5):
            raise ValueError("role_games does not match the model's champion vocabulary")
        totals = self.role_games.sum(axis=0)
        if (totals <= 0).any():
            raise SystemExit("no role information in the training data - cannot advise drafts")
        self.role_share = self.role_games / totals                   # P(champion | role)
        # ban rates, and pick shares in games where the champion was not banned
        n_vocab = len(self.role_games)
        ban_games = predictor.ckpt.get("champion_ban_games")
        ban_total = predictor.ckpt.get("ban_games") or 0
        self.ban_rate = np.zeros(n_vocab)
        if ban_games is not None and ban_total > 0:
            self.ban_rate[1:] = np.asarray(ban_games, dtype=np.float64) / ban_total
        open_games = self.role_games / np.clip(1 - self.ban_rate, 0.1, 1)[:, None]
        self.open_share = open_games / open_games.sum(axis=0)        # P(champion | role, not banned)
        self.samples = samples
        self.min_role_share = min_role_share
        self.seed = seed
        self.id_of = {idx: cid for cid, idx in predictor.index.items()}

    # ---- helpers ---------------------------------------------------------------------

    def name(self, idx: int) -> str:
        return self.p.display(self.id_of[idx])

    def index(self, champion) -> int:
        return self.p.index[self.p.champion_id(champion)]

    def viable(self, role: int) -> list[int]:
        """Champions that make up at least `min_role_share` of all picks in this role."""
        return np.flatnonzero(self.role_share[:, role] >= self.min_role_share).tolist()

    def infer_roles(self, champions: list[int], fixed: dict[int, int]) -> dict[int, int]:
        """Most likely role assignment {role: champion} for champions with unknown roles."""
        free = [r for r in range(5) if r not in fixed]
        if len(champions) > len(free):
            raise ValueError("more than five champions on one team")
        prob = self.role_games / np.maximum(self.role_games.sum(axis=1, keepdims=True), 1)
        best, best_score = (), -np.inf
        for roles in itertools.permutations(free, len(champions)):
            score = sum(np.log(prob[c, r] + 1e-4) for c, r in zip(champions, roles))
            if score > best_score:
                best, best_score = roles, score
        return {**fixed, **dict(zip(best, champions))}

    def parse_team(self, tokens: list[str], infer: bool) -> tuple[dict[int, int], set[int]]:
        """['mid=Ahri', 'Jinx'] -> ({role: vocabulary index}, roles that were inferred)."""
        fixed: dict[int, int] = {}
        loose: list[int] = []
        for tok in tokens or []:
            if "=" in tok:
                role_name, champ = tok.split("=", 1)
                role = parse_role(role_name)
                if role in fixed:
                    raise ValueError(f"two champions given for {ROLE_NAMES[role]}")
                fixed[role] = self.index(champ)
            elif infer:
                loose.append(self.index(tok))
            else:
                raise ValueError(f"give allies as role=Champion, e.g. mid={tok}")
        team = self.infer_roles(loose, fixed)
        return team, set(team) - set(fixed)

    def _state(self, ally: dict[int, int], enemy: dict[int, int], banned: set[int]):
        picked = list(ally.values()) + list(enemy.values())
        if len(set(picked)) != len(picked):
            raise ValueError("a champion appears twice among the picks")
        clash = banned & set(picked)
        if clash:
            raise ValueError(f"banned champion already picked: {self.name(next(iter(clash)))}")
        fixed = np.zeros(10, dtype=np.int64)  # columns 0-4 my team by role, 5-9 enemy by role
        for role, c in ally.items():
            fixed[role] = c
        for role, c in enemy.items():
            fixed[5 + role] = c
        return fixed

    def share(self, banned: set[int] | frozenset[int]) -> np.ndarray:
        """Pick shares to fill open slots with: given availability once the bans are known."""
        return self.open_share if len(banned) >= BANS_KNOWN else self.role_share

    def _sample(self, fixed: np.ndarray, banned: set[int], skip: tuple[int, ...] = (),
                n: int | None = None) -> np.ndarray:
        """[n, 10] completions: open slots sampled from role pick shares, no duplicates.
        Columns in `skip` stay 0 (they are filled in by the caller)."""
        n = n or self.samples
        share = self.share(banned)
        rng = np.random.default_rng(self.seed)
        out = np.tile(fixed, (n, 1))
        taken = np.zeros((n, self.role_share.shape[0]), dtype=bool)
        taken[:, [c for c in fixed if c] + sorted(banned)] = True
        rows = np.arange(n)
        for col in range(10):
            if fixed[col] or col in skip:
                continue
            w = np.where(taken, 0.0, share[:, col % 5][None, :])
            cdf = w.cumsum(axis=1)
            u = rng.random(n) * cdf[:, -1]
            pick = (cdf <= u[:, None]).sum(axis=1)
            out[:, col] = pick
            taken[rows, pick] = True
        return out

    @torch.no_grad()
    def _win(self, drafts: np.ndarray, side: str, my=None, batch: int = 16384,
             members: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """P(my team wins) for [M, 10] drafts (my team first): (ensemble [M], members [K, M]).

        my: optional (role, feats [M, F]): player-history inputs of the player in that role of
        my team; everybody else stays anonymous."""
        mine_first = torch.from_numpy(drafts)
        champs = mine_first if side == "blue" else torch.cat(
            [mine_first[:, 5:], mine_first[:, :5]], dim=1)
        roles = torch.tensor([1, 2, 3, 4, 5] * 2).expand(len(champs), 10)
        scale, bias = self.p.scale, self.p.bias
        if my is None:
            z = self.p.member_logits(champs, roles, members=members)
        else:
            if self.p.player_calibration is None:
                raise ValueError("this model was trained without player features")
            cal = self.p.ckpt.get("calibration_solo") or self.p.player_calibration
            scale, bias = cal["scale"], cal["bias"]
            col = my[0] if side == "blue" else 5 + my[0]
            mine = torch.as_tensor(my[1], dtype=torch.float32)
            parts = []
            for i in range(0, len(champs), batch):   # the [B, 10, F] inputs are built per batch
                f = mine[i:i + batch].to(self.p.device)
                feats = torch.zeros(len(f), 10, f.shape[1], device=self.p.device)
                feats[:, col] = f
                known = torch.zeros(len(f), 10, dtype=torch.bool, device=self.p.device)
                known[:, col] = True
                parts.append(self.p.member_logits(champs[i:i + batch], roles[i:i + batch],
                                                  feats, known, members=members))
            z = torch.cat(parts, dim=1)
        z = scale * z.double().cpu().numpy() + bias
        ens, members = 1 / (1 + np.exp(-z.mean(axis=0))), 1 / (1 + np.exp(-z))
        return (ens, members) if side == "blue" else (1 - ens, 1 - members)

    def _values(self, base: np.ndarray, overrides: list[dict[int, int]], side: str, my=None,
                members: int | None = None):
        """Expected win rate when `overrides` ({column: champion}) are forced into `base`.

        Samples in which a forced champion was also sampled elsewhere are ignored, which is
        the right conditioning: once it is locked in, nobody else can have it.
        my: optional (role, feats [S, F]), the player inputs that go with each override.
        Returns (mean [S], member means [K, S])."""
        n = len(base)
        drafts = np.tile(base, (len(overrides), 1, 1))
        valid = np.ones((len(overrides), n), dtype=bool)
        for s, forced in enumerate(overrides):
            others = [c for c in range(10) if c not in forced]
            for col, champ in forced.items():
                valid[s] &= ~(base[:, others] == champ).any(axis=1)
                drafts[s, :, col] = champ
        if my is not None:
            my = (my[0], np.repeat(np.asarray(my[1], dtype=np.float32), n, axis=0))
        ens, each = self._win(drafts.reshape(-1, 10), side, my, members=members)
        weight = valid / np.maximum(valid.sum(axis=1, keepdims=True), 1)
        mean = (ens.reshape(len(overrides), n) * weight).sum(axis=1)
        member_mean = (each.reshape(len(each), len(overrides), n) * weight).sum(axis=2)
        mean[valid.sum(axis=1) == 0] = np.nan
        return mean, member_mean

    def _player_feats(self, player, champions: list[int], role: int, fixed: np.ndarray):
        """[len(champions), F] model inputs for `player` on these champions (vocabulary
        indices) in `role`, against the enemy in the same role if that pick is known."""
        if player.result_means is None:
            player.result_means = self.p.ckpt.get("result_means")
        return player.features([self.id_of[c] for c in champions], role,
                               self.id_of.get(int(fixed[5 + role]), 0))

    # ---- picks -----------------------------------------------------------------------

    def evaluate(self, side: str, ally: dict[int, int], enemy: dict[int, int],
                 banned: set[int] = frozenset(), player=None, role: int | None = None
                 ) -> tuple[float, float]:
        """Expected P(my team wins) of the draft as it stands, open slots filled the way the
        field picks: (win rate, std across ensemble members). With `player` (lolwp.live.Player)
        and their `role` already filled, that player's history is taken into account."""
        fixed = self._state(ally, enemy, set(banned))
        base = self._sample(fixed, set(banned))
        my = None
        if player is not None and role is not None and fixed[role] and self.p.player_calibration:
            feats = self._player_feats(player, [int(fixed[role])], role, fixed)
            my = (role, np.repeat(feats, len(base), axis=0))
        ens, members = self._win(base, side, my)
        return float(ens.mean()), float(members.mean(axis=1).std())

    def recommend_picks(self, side: str, role: int, ally: dict[int, int],
                        enemy: dict[int, int], banned: set[int] = frozenset(),
                        pool: list[int] | None = None, top: int = 10,
                        counter_check: int = 8, player=None) -> list[PickOption]:
        """player: optional lolwp.live.Player; the ranking then uses their own record."""
        if side not in ("blue", "red"):
            raise ValueError("side must be blue or red")
        if role in ally:
            raise ValueError(f"your role ({ROLE_NAMES[role]}) is already filled on your team")
        if player is not None and self.p.player_calibration is None:
            raise ValueError("this model was trained without player features")
        fixed = self._state(ally, enemy, set(banned))
        unavailable = set(banned) | set(fixed.tolist())
        if pool is not None:
            candidates = list(dict.fromkeys(pool))
        else:
            candidates = self.viable(role)
            if player is not None:   # what they actually play, even if unusual in this role
                own = [self.p.index[c] for c, rec in player.records.items()
                       if rec.games >= PERSONAL_MIN_GAMES and c in self.p.index]
                candidates += [c for c in own if c not in candidates]
        candidates = [c for c in candidates if c not in unavailable]
        if not candidates:
            raise ValueError("no candidate champions left for this role")
        base = self._sample(fixed, set(banned), skip=(role,))
        overrides = [{role: c} for c in candidates]
        draft_mean, member_mean = self._values(base, overrides, side)
        mean, feats = draft_mean, None
        if player is not None:
            feats = self._player_feats(player, candidates, role, fixed)
            mean, member_mean = self._values(base, overrides, side, (role, feats))
        share = self.share(banned)[candidates, role] + 1e-9
        field = float(np.nansum(mean * share) / share.sum())
        order = [i for i in np.argsort(-mean) if not np.isnan(mean[i])][:top]
        options = []
        for i in order:
            c = candidates[i]
            opt = PickOption(self.name(c), float(mean[i]), float(mean[i] - field),
                             float(member_mean[:, i].std()), int(self.role_games[c, role]),
                             champion_id=self.id_of[c])
            if player is not None:
                rec = player.records.get(self.id_of[c])
                opt.draft_win = float(draft_mean[i])
                opt.my_games, opt.my_wins = (rec.games, rec.wins) if rec else (0, 0)
            options.append(opt)

        open_enemy = [r for r in range(5) if not fixed[5 + r]]
        if counter_check and open_enemy:
            small = base[: max(24, len(base) // 4)]
            for opt, i in list(zip(options, order))[:counter_check]:
                c = candidates[i]
                replies = [(r, e) for r in open_enemy for e in self.viable(r)
                           if e != c and e not in unavailable]
                if not replies:
                    continue
                my = None if feats is None else (role, np.repeat(feats[i:i + 1], len(replies), axis=0))
                vals, _ = self._values(small, [{role: c, 5 + r: e} for r, e in replies], side, my,
                                       members=self.search_members)
                # the search may use fewer models and samples: report the drop it found
                # relative to the same estimate without a reply, on top of the full value
                ref, _ = self._values(small, [{role: c}], side,
                                      None if my is None else (role, my[1][:1]),
                                      members=self.search_members)
                vals = vals - ref[0] + opt.win
                worst = int(np.nanargmin(vals))
                r, e = replies[worst]
                opt.counter = f"{self.name(e)} ({ROLE_NAMES[r]})"
                opt.win_if_countered = float(vals[worst])
        return options

    # ---- bans ------------------------------------------------------------------------

    def recommend_bans(self, side: str, ally: dict[int, int] | None = None,
                       enemy: dict[int, int] | None = None, banned: set[int] = frozenset(),
                       top: int = 10, sort: str = "gain", with_net: bool = True
                       ) -> list[BanOption]:
        """with_net=False skips what your own team loses (`net` then equals `gain`): about
        half the work, for when only `gain` is shown."""
        if side not in ("blue", "red"):
            raise ValueError("side must be blue or red")
        if sort not in ("gain", "net"):
            raise ValueError("sort must be 'gain' or 'net'")
        ally, enemy = ally or {}, enemy or {}
        fixed = self._state(ally, enemy, set(banned))
        unavailable = set(banned) | set(fixed.tolist())
        base = self._sample(fixed, set(banned))
        base_ens, base_members = self._win(base, side)
        open_enemy = [r for r in range(5) if not fixed[5 + r]]
        open_ally = [r for r in range(5) if not fixed[r]]
        if not open_enemy:
            raise ValueError("the enemy team is already complete - nothing left to ban")

        # (champion, column, pick share) for every plausible champion-in-open-slot
        share = self.share(banned)
        cases = [(c, col, share[c, col % 5])
                 for col in [5 + r for r in open_enemy] + (open_ally if with_net else [])
                 for c in self.viable(col % 5) if c not in unavailable]
        vals, member_vals = self._values(base, [{col: c} for c, col, _ in cases], side)

        results = []
        for c in sorted({c for c, col, _ in cases if col >= 5}):
            absent = ~(base == c).any(axis=1)
            if not absent.any():
                continue
            without = base_ens[absent].mean()
            without_members = base_members[:, absent].mean(axis=1)
            gain, lost, chance, weighted = 0.0, 0.0, 0.0, 0.0
            gain_members = np.zeros(len(base_members))
            for i, (ci, col, share) in enumerate(cases):
                if ci != c or np.isnan(vals[i]):
                    continue
                if col >= 5:   # enemy would have it
                    gain += share * (without - vals[i])
                    gain_members += share * (without_members - member_vals[:, i])
                    chance += share
                    weighted += share * vals[i]
                else:          # my own team would have had it
                    lost += share * (vals[i] - without)
            if chance <= 0:
                continue
            results.append(BanOption(self.name(c), float(min(1.0, chance)),
                                     float(weighted / chance), float(gain),
                                     float(gain - lost), float(gain_members.std()),
                                     champion_id=self.id_of[c], ban_rate=float(self.ban_rate[c])))
        results.sort(key=lambda b: -(b.gain if sort == "gain" else b.net))
        return results[:top]


# ---- command line ------------------------------------------------------------------------

def _describe(advisor: DraftAdvisor, team: dict[int, int], inferred: set[int]) -> str:
    if not team:
        return "none yet"
    return ", ".join(f"{advisor.name(team[r])} ({ROLE_NAMES[r]}{'?' if r in inferred else ''})"
                     for r in sorted(team))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["pick", "ban"])
    ap.add_argument("--side", required=True, choices=["blue", "red"], help="your team's side")
    ap.add_argument("--role", help="pick: the role you are picking for")
    ap.add_argument("--ally", nargs="*", default=[], metavar="ROLE=CHAMP",
                    help="teammates' picks (for bans: also intended picks, including your own)")
    ap.add_argument("--enemy", nargs="*", default=[], metavar="[ROLE=]CHAMP",
                    help="enemy picks; roles are inferred when omitted")
    ap.add_argument("--bans", nargs="*", default=[], metavar="CHAMP",
                    help="champions already banned by either team")
    ap.add_argument("--pool", nargs="*", metavar="CHAMP",
                    help="pick: only consider these champions (e.g. the ones you play)")
    ap.add_argument("--me", metavar="NAME#TAG",
                    help="pick: your Riot ID; ranks the picks with your own recent record")
    ap.add_argument("--platform", default="euw1", help="--me: your server, e.g. euw1 na1 kr")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--sort", choices=["gain", "net"], default="gain", help="ban: ranking")
    ap.add_argument("--samples", type=int, default=None,
                    help="sampled completions per option (default 256 on a GPU, 96 on a CPU)")
    ap.add_argument("--no-counter", action="store_true", help="pick: skip the enemy-reply search")
    ap.add_argument("--model", default="models/draft_model.pt")
    ap.add_argument("--db", default="data/matches.sqlite",
                    help="only read if the model file lacks role statistics")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    predictor = Predictor.load(args.model, device=args.device)
    advisor = DraftAdvisor(predictor, db_path=args.db, samples=args.samples)
    try:
        ally, _ = advisor.parse_team(args.ally, infer=False)
        enemy, inferred = advisor.parse_team(args.enemy, infer=True)
        banned = {advisor.index(c) for c in args.bans}
        print(f"{args.side} side | allies: {_describe(advisor, ally, set())} | "
              f"enemies: {_describe(advisor, enemy, inferred)}"
              + (" | '?' = role inferred" if inferred else ""))
        if args.action == "pick":
            if not args.role:
                raise ValueError("pick needs --role")
            role = parse_role(args.role)
            pool = [advisor.index(c) for c in args.pool] if args.pool is not None else None
            player = None
            if args.me:
                from .live import PlayerLookup
                from .riot import RiotAPIError, RiotAuthError
                try:
                    player = PlayerLookup.open(db_path=args.db).player(
                        args.me, args.platform,
                        progress=lambda k, n: print(f"\rfetching your games {k + 1}/{n}",
                                                    end="", flush=True))
                except (RiotAuthError, RiotAPIError) as e:
                    raise SystemExit(f"error: {e}")
                print(f"\r{player.name}: {player.games} ranked games in the last month, "
                      f"{player.wins} won" + " " * 10)
            options = advisor.recommend_picks(
                args.side, role, ally, enemy, banned, pool, args.top,
                counter_check=0 if args.no_counter else 8, player=player)
            print(f"\nBest picks for {ROLE_NAMES[role]}:")
            mine = f"{'draft %':>9}{'you':>9}" if player else ""
            print(f"{'champion':<16}{'win %':>7}{mine}{'vs field':>10}{'+-model':>9}{'games':>8}"
                  "   worst enemy reply")
            for o in options:
                reply = (f"{o.counter} -> {100 * o.win_if_countered:.1f}%"
                         if o.counter else "")
                mine = (f"{100 * o.draft_win:>9.1f}{f'{o.my_wins}-{o.my_games - o.my_wins}':>9}"
                        if player else "")
                print(f"{o.champion:<16}{100 * o.win:>7.1f}{mine}{100 * o.vs_field:>+10.1f}"
                      f"{100 * o.spread:>9.1f}{o.role_games:>8}   {reply}")
            if player:
                print("\ndraft %: the same pick by an anonymous player. you: your wins-losses "
                      "on the champion in the last month.")
            print("\nwin %: expected win rate with this pick, other open slots filled the way "
                  "Diamond+ players pick.\n+-model: disagreement between ensemble members; "
                  "gaps smaller than this are not meaningful.\nworst enemy reply: the minimum "
                  "over many noisy estimates, so it overstates the danger somewhat.")
        else:
            options = advisor.recommend_bans(args.side, ally, enemy, banned, args.top, args.sort)
            print(f"\nBest bans (sorted by {args.sort}):")
            print(f"{'champion':<16}{'enemy picks':>12}{'win % if so':>13}{'gain':>8}"
                  f"{'net':>8}{'+-model':>9}{'banned':>8}")
            for o in options:
                print(f"{o.champion:<16}{100 * o.enemy_pick_chance:>11.1f}%"
                      f"{100 * o.win_if_enemy_has:>13.1f}{100 * o.gain:>+8.2f}"
                      f"{100 * o.net:>+8.2f}{100 * o.spread:>9.2f}{100 * o.ban_rate:>7.0f}%")
            print("\ngain: win-rate points won by denying it to the enemy "
                  "(chance they pick it x damage when they do).\nnet: gain minus what your own "
                  "team loses by not being able to pick it; near zero for champions that are "
                  "simply strong for whoever gets them.\nbanned: how often it is banned anyway. "
                  "enemy picks: how often it reaches an enemy when you do not ban it.")
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    print(f"[model trained on patch {', '.join(predictor.ckpt['patches'])}]")


if __name__ == "__main__":
    main()
