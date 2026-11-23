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

Roles: allies are given as role=Champion. Enemy roles may be omitted; they are inferred from
how each champion is usually played.
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


@dataclass
class BanOption:
    champion: str
    enemy_pick_chance: float   # P(some open enemy slot picks it)
    win_if_enemy_has: float    # expected P(my team wins) when the enemy has it
    gain: float                # expected win probability gained by denying it to the enemy
    net: float                 # gain minus what my own team loses by not having it
    spread: float              # std of `gain` across ensemble members


class DraftAdvisor:
    def __init__(self, predictor: Predictor, role_games: np.ndarray | None = None,
                 db_path: str | Path = "data/matches.sqlite", samples: int = 256,
                 min_role_share: float = 0.005, seed: int = 0):
        self.p = predictor
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

    def _sample(self, fixed: np.ndarray, banned: set[int], skip: tuple[int, ...] = (),
                n: int | None = None) -> np.ndarray:
        """[n, 10] completions: open slots sampled from role pick shares, no duplicates.
        Columns in `skip` stay 0 (they are filled in by the caller)."""
        n = n or self.samples
        rng = np.random.default_rng(self.seed)
        out = np.tile(fixed, (n, 1))
        taken = np.zeros((n, self.role_share.shape[0]), dtype=bool)
        taken[:, [c for c in fixed if c] + sorted(banned)] = True
        rows = np.arange(n)
        for col in range(10):
            if fixed[col] or col in skip:
                continue
            w = np.where(taken, 0.0, self.role_share[:, col % 5][None, :])
            cdf = w.cumsum(axis=1)
            u = rng.random(n) * cdf[:, -1]
            pick = (cdf <= u[:, None]).sum(axis=1)
            out[:, col] = pick
            taken[rows, pick] = True
        return out

    @torch.no_grad()
    def _win(self, drafts: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
        """P(my team wins) for [M, 10] drafts (my team first): (ensemble [M], members [K, M])."""
        mine_first = torch.from_numpy(drafts)
        champs = mine_first if side == "blue" else torch.cat(
            [mine_first[:, 5:], mine_first[:, :5]], dim=1)
        roles = torch.tensor([1, 2, 3, 4, 5] * 2).expand(len(champs), 10)
        z = self.p.member_logits(champs, roles).double().cpu().numpy()
        z = self.p.scale * z + self.p.bias
        ens, members = 1 / (1 + np.exp(-z.mean(axis=0))), 1 / (1 + np.exp(-z))
        return (ens, members) if side == "blue" else (1 - ens, 1 - members)

    def _values(self, base: np.ndarray, overrides: list[dict[int, int]], side: str):
        """Expected win rate when `overrides` ({column: champion}) are forced into `base`.

        Samples in which a forced champion was also sampled elsewhere are ignored, which is
        the right conditioning: once it is locked in, nobody else can have it.
        Returns (mean [S], member means [K, S])."""
        n = len(base)
        drafts = np.tile(base, (len(overrides), 1, 1))
        valid = np.ones((len(overrides), n), dtype=bool)
        for s, forced in enumerate(overrides):
            others = [c for c in range(10) if c not in forced]
            for col, champ in forced.items():
                valid[s] &= ~(base[:, others] == champ).any(axis=1)
                drafts[s, :, col] = champ
        ens, members = self._win(drafts.reshape(-1, 10), side)
        weight = valid / np.maximum(valid.sum(axis=1, keepdims=True), 1)
        mean = (ens.reshape(len(overrides), n) * weight).sum(axis=1)
        member_mean = (members.reshape(len(members), len(overrides), n) * weight).sum(axis=2)
        mean[valid.sum(axis=1) == 0] = np.nan
        return mean, member_mean

    # ---- picks -----------------------------------------------------------------------

    def recommend_picks(self, side: str, role: int, ally: dict[int, int],
                        enemy: dict[int, int], banned: set[int] = frozenset(),
                        pool: list[int] | None = None, top: int = 10,
                        counter_check: int = 8) -> list[PickOption]:
        if side not in ("blue", "red"):
            raise ValueError("side must be blue or red")
        if role in ally:
            raise ValueError(f"your role ({ROLE_NAMES[role]}) is already filled on your team")
        fixed = self._state(ally, enemy, set(banned))
        unavailable = set(banned) | set(fixed.tolist())
        candidates = [c for c in (pool if pool is not None else self.viable(role))
                      if c not in unavailable]
        if not candidates:
            raise ValueError("no candidate champions left for this role")
        base = self._sample(fixed, set(banned), skip=(role,))
        mean, member_mean = self._values(base, [{role: c} for c in candidates], side)
        share = self.role_share[candidates, role] + 1e-9
        field = float(np.nansum(mean * share) / share.sum())
        order = [i for i in np.argsort(-mean) if not np.isnan(mean[i])][:top]
        options = [
            PickOption(self.name(candidates[i]), float(mean[i]), float(mean[i] - field),
                       float(member_mean[:, i].std()), int(self.role_games[candidates[i], role]))
            for i in order
        ]

        open_enemy = [r for r in range(5) if not fixed[5 + r]]
        if counter_check and open_enemy:
            small = base[: max(32, len(base) // 3)]
            for opt, i in list(zip(options, order))[:counter_check]:
                c = candidates[i]
                replies = [(r, e) for r in open_enemy for e in self.viable(r)
                           if e != c and e not in unavailable]
                if not replies:
                    continue
                vals, _ = self._values(small, [{role: c, 5 + r: e} for r, e in replies], side)
                worst = int(np.nanargmin(vals))
                r, e = replies[worst]
                opt.counter = f"{self.name(e)} ({ROLE_NAMES[r]})"
                opt.win_if_countered = float(vals[worst])
        return options

    # ---- bans ------------------------------------------------------------------------

    def recommend_bans(self, side: str, ally: dict[int, int] | None = None,
                       enemy: dict[int, int] | None = None, banned: set[int] = frozenset(),
                       top: int = 10, sort: str = "gain") -> list[BanOption]:
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
        cases = [(c, col, self.role_share[c, col % 5])
                 for col in [5 + r for r in open_enemy] + open_ally
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
                                     float(gain - lost), float(gain_members.std())))
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
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--sort", choices=["gain", "net"], default="gain", help="ban: ranking")
    ap.add_argument("--samples", type=int, default=256, help="sampled completions per option")
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
            options = advisor.recommend_picks(
                args.side, role, ally, enemy, banned, pool, args.top,
                counter_check=0 if args.no_counter else 8)
            print(f"\nBest picks for {ROLE_NAMES[role]}:")
            print(f"{'champion':<16}{'win %':>7}{'vs field':>10}{'+-model':>9}{'games':>8}"
                  "   worst enemy reply")
            for o in options:
                reply = (f"{o.counter} -> {100 * o.win_if_countered:.1f}%"
                         if o.counter else "")
                print(f"{o.champion:<16}{100 * o.win:>7.1f}{100 * o.vs_field:>+10.1f}"
                      f"{100 * o.spread:>9.1f}{o.role_games:>8}   {reply}")
            print("\nwin %: expected win rate with this pick, other open slots filled the way "
                  "Diamond+ players pick.\n+-model: disagreement between ensemble members; "
                  "gaps smaller than this are not meaningful.\nworst enemy reply: the minimum "
                  "over many noisy estimates, so it overstates the danger somewhat.")
        else:
            options = advisor.recommend_bans(args.side, ally, enemy, banned, args.top, args.sort)
            print(f"\nBest bans (sorted by {args.sort}):")
            print(f"{'champion':<16}{'enemy picks':>12}{'win % if so':>13}{'gain':>8}"
                  f"{'net':>8}{'+-model':>9}")
            for o in options:
                print(f"{o.champion:<16}{100 * o.enemy_pick_chance:>11.1f}%"
                      f"{100 * o.win_if_enemy_has:>13.1f}{100 * o.gain:>+8.2f}"
                      f"{100 * o.net:>+8.2f}{100 * o.spread:>9.2f}")
            print("\ngain: win-rate points won by denying it to the enemy "
                  "(chance they pick it x damage when they do).\nnet: gain minus what your own "
                  "team loses by not being able to pick it; near zero for champions that are "
                  "simply strong for whoever gets them.\nPick chances come from games played "
                  "after bans, so champions that are banned a lot are underrated here.")
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    print(f"[model trained on patch {', '.join(predictor.ckpt['patches'])}]")


if __name__ == "__main__":
    main()
