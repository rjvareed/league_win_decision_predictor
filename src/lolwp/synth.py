"""Synthetic matches with a known ground truth, for tests and pipeline dry runs.

    python -m lolwp.synth --db data/synthetic.sqlite --champions data/synthetic_champions.json
    lolwp-train --db data/synthetic.sqlite --champions data/synthetic_champions.json \
                --out models/synthetic.pt

This is NOT League data: champions are "Champ001".. and outcomes are sampled from a made-up
model with champion strengths, ally synergies, enemy counters, a blue-side bonus and - unless
players=False - player skill and a bonus for playing one of your three main champions. It only
shows that the pipeline runs end to end and can recover structure that exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import db

COLUMNS = ("match_id, platform, game_version, patch, game_creation, duration, blue, red, "
           "roles_ok, blue_win, seed_tier, blue_puuids, red_puuids, bans, stats, pstats")


def make_synthetic(db_path, champions_path=None, n: int = 60_000, n_champions: int = 100,
                   seed: int = 0, effect: float = 0.25, blue_bonus: float = 0.08,
                   players: bool = True, n_players: int = 1500, skill: float = 0.3,
                   main_bonus: float = 0.3) -> np.ndarray:
    """Write n fake matches; returns the true P(blue wins) per match in time order."""
    rng = np.random.default_rng(seed)
    v = n_champions
    strength = rng.normal(0, effect, v)
    synergy = np.triu(rng.normal(0, effect, (v, v)), 1)
    synergy = synergy + synergy.T
    counter = np.triu(rng.normal(0, effect, (v, v)), 1)
    counter = counter - counter.T  # counter[a, b] = -counter[b, a]
    by_role = [np.arange(v)[np.arange(v) % 5 == r] for r in range(5)]

    # players: fixed role (id % 5), a skill level and three main champions of that role
    prng = np.random.default_rng(seed + 1000)
    player_skill = prng.normal(0, skill, n_players)
    mains = np.stack([prng.choice(by_role[p % 5], 3, replace=False) for p in range(n_players)])
    players_by_role = [np.arange(n_players)[np.arange(n_players) % 5 == r] for r in range(5)]

    conn = db.connect(db_path)
    rows, truth = [], np.empty(n)
    for i in range(n):
        if players:
            who = np.concatenate([prng.choice(pool, 2, replace=False) for pool in players_by_role])
            picks, on_main = np.empty(10, dtype=np.int64), np.zeros(10, dtype=bool)
            for slot, p in enumerate(who):  # slots: top blue, top red, jungle blue, ...
                if prng.random() < 0.7:
                    c, on_main[slot] = prng.choice(mains[p]), True
                else:
                    c = prng.choice(by_role[p % 5])
                    on_main[slot] = c in mains[p]
                if slot % 2 and c == picks[slot - 1]:  # lane opponent already has it
                    c = prng.choice(by_role[p % 5][by_role[p % 5] != c])
                    on_main[slot] = c in mains[p]
                picks[slot] = c
            player_logit = player_skill[who] + main_bonus * on_main
        else:
            picks = np.concatenate([rng.choice(pool, 2, replace=False) for pool in by_role])
            who, player_logit = None, np.zeros(10)
        blue, red = picks[0::2].copy(), picks[1::2].copy()  # one champion per role each
        logit = blue_bonus + strength[blue].sum() - strength[red].sum()
        logit += (synergy[np.ix_(blue, blue)].sum() - synergy[np.ix_(red, red)].sum()) / 2
        logit += counter[np.ix_(blue, red)].sum()
        logit += player_logit[0::2].sum() - player_logit[1::2].sum()
        p = 1 / (1 + np.exp(-logit))
        truth[i] = p
        blue_ids = [f"player{x}" for x in who[0::2]] if players else None
        red_ids = [f"player{x}" for x in who[1::2]] if players else None
        roles_ok = rng.random() > 0.05
        skills = (np.concatenate([player_skill[who[0::2]], player_skill[who[1::2]]])
                  if players else np.zeros(10))
        if not roles_ok:
            order_b, order_r = rng.permutation(5), rng.permutation(5)
            blue, red = blue[order_b], red[order_r]
            skills = np.concatenate([skills[:5][order_b], skills[5:][order_r]])
            if players:
                blue_ids = [blue_ids[j] for j in order_b]
                red_ids = [red_ids[j] for j in order_r]
        win = int(rng.random() < p)
        # end-of-game margins: follow the winner, larger for lopsided drafts
        margin = (1 if win else -1) * abs(prng.normal(0.6, 0.4)) + 0.3 * np.tanh(logit)
        gold = [int(55_000 * (1 + 0.08 * margin)), int(55_000 * (1 - 0.08 * margin))]
        kills = [max(0, int(25 + 10 * margin)), max(0, int(25 - 10 * margin))]
        towers = [int(np.clip(6 + 4 * margin, 0, 11)), int(np.clip(6 - 4 * margin, 0, 11))]
        stats = json.dumps({"gold": gold, "kills": kills, "towers": towers, "dragons": [2, 2],
                            "barons": [0, 0], "inhibitors": [0, 0], "surrender": 0})
        pstats = None
        if players:   # per-player statistics that follow the result and the player's skill
            from .riot import PSTAT, PSTAT_FIELDS
            table = np.zeros((10, len(PSTAT_FIELDS)), dtype=np.int64)
            lead = np.where(np.arange(10) < 5, margin, -margin) + 2 * skills + prng.normal(0, 0.5, 10)
            table[:, PSTAT["gold"]] = 11_000 * (1 + 0.08 * lead)
            table[:, PSTAT["damage"]] = 18_000 * (1 + 0.1 * lead)
            table[:, PSTAT["kills"]] = np.clip(5 + 2 * lead, 0, None)
            table[:, PSTAT["deaths"]] = np.clip(5 - 2 * lead, 0, None)
            table[:, PSTAT["assists"]] = 6
            table[:, PSTAT["cs"]] = 180
            table[:, PSTAT["summoner_level"]] = prng.integers(30, 900, 10)
            table[:, PSTAT["spell1"]] = 4
            table[:, PSTAT["spell2"]] = prng.choice([14, 12, 11, 7], 10)
            table[:, PSTAT["keystone"]] = prng.choice([8005, 8010, 8112, 8437], 10)
            table[:, PSTAT["role_assignment"]] = prng.choice([1, 2, 3], 10, p=[0.7, 0.25, 0.05])
            table[:, PSTAT["lane_lead"]] = lead > 0
            pstats = json.dumps(table.tolist(), separators=(",", ":"))
        patch = "99.1" if i < 0.3 * n else "99.2"
        rows.append((f"SYN_{i}", "syn", f"{patch}.1.1", patch, 1_700_000_000_000 + i * 60_000,
                     int(1750 - 300 * abs(margin) + prng.normal(0, 200)),
                     ",".join(str(c + 1) for c in blue), ",".join(str(c + 1) for c in red),
                     int(roles_ok), win, "DIAMOND",
                     ",".join(blue_ids) if players else None,
                     ",".join(red_ids) if players else None,
                     ",".join(["-1"] * 10), stats, pstats))
    conn.executemany(
        f"INSERT OR REPLACE INTO matches({COLUMNS}) VALUES({','.join('?' * 16)})", rows)
    conn.close()
    if champions_path:
        Path(champions_path).parent.mkdir(parents=True, exist_ok=True)
        Path(champions_path).write_text(json.dumps({
            "version": "synthetic",
            "champions": {str(c + 1): {"id": f"Champ{c + 1:03d}", "name": f"Champ {c + 1:03d}"}
                          for c in range(v)},
        }))
    return truth


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/synthetic.sqlite")
    ap.add_argument("--champions", default="data/synthetic_champions.json")
    ap.add_argument("-n", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-players", action="store_true")
    args = ap.parse_args(argv)
    make_synthetic(args.db, args.champions, n=args.n, seed=args.seed,
                   players=not args.no_players)
    print(f"wrote {args.n} synthetic matches to {args.db}")


if __name__ == "__main__":
    main()
