"""Crawl Diamond+ ranked solo queue matches from the Riot API into SQLite.

    lolwp-collect                       # kr + euw1 + na1 + vn2, two newest patches
    lolwp-collect --backfill-share 1    # only re-fetch old rows that lack player ids / statistics

Strategy
  1. Seed: list every Challenger/Grandmaster/Master/Diamond solo-queue player per platform
     (league-v4, platform-routed hosts), with their ladder position.
  2. Crawl top-down: players are visited from the top of the ladder downwards, and revisited
     every --recrawl-hours. Visiting a player lists their recent ranked solo games and fetches
     each unseen match (match-v5, region-routed hosts). Top-down order matters for the player
     features of the model: high-elo players mostly meet each other, so their match histories
     become dense instead of each game having one known player and nine strangers.
  3. Backfill: a share of the request budget re-fetches matches stored by older versions of
     this crawler to add what they did not keep: player ids, bans, team totals and the
     per-player statistics (newest matches first).

Each API host has its own rate-limit budget, so one thread per host runs in parallel.
The crawl is resumable: stop with Ctrl-C and rerun at any time (e.g. after renewing a dev key).

Selection bias, and why first_seen is tracked: the ladder list is a snapshot. A player who is
Diamond+ at the snapshot has, on average, been winning just before it, so games *older* than
the snapshot that were found through that player are slightly biased towards their team.
Games played after the snapshot are not. Players who join the list in a later snapshot are
therefore only crawled forward from the moment they were first listed, and lolwp-train
warns when its test games are older than the first snapshot.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import threading
import time
from collections import defaultdict
from pathlib import Path

from . import db
from .riot import (
    APEX_TIERS,
    DIVISION_PRIORITY,
    DIVISIONS,
    PLATFORM_TO_REGION,
    TIER_PRIORITY,
    RiotAuthError,
    RiotClient,
    ddragon_champions,
    ddragon_versions,
    load_api_key,
    parse_match,
    patch_of,
)

RESEED_AFTER_S = 3 * 24 * 3600
PAGE = 100  # match ids per request (API maximum)

INSERT_MATCH = (
    "INSERT OR IGNORE INTO matches(match_id, platform, game_version, patch, game_creation, "
    "duration, blue, red, roles_ok, blue_win, seed_tier, blue_puuids, red_puuids, bans, stats, "
    "pstats) VALUES(:match_id, :platform, :game_version, :patch, :game_creation, :duration, "
    ":blue, :red, :roles_ok, :blue_win, :seed_tier, :blue_puuids, :red_puuids, :bans, :stats, "
    ":pstats)"
)


class Crawler:
    def __init__(self, args, client: RiotClient, min_patch: tuple[int, ...], target_patch: str):
        self.args = args
        self.client = client
        self.min_patch = min_patch
        self.target_patch = target_patch
        self.stop = threading.Event()
        self.seeded = {p: threading.Event() for p in args.platforms}
        self.errors: list[str] = []
        self.since = int(time.time()) - args.days * 86400
        self.fetched = 0     # new matches fetched this run (approximate, unsynchronised)
        self.backfilled = 0  # old matches re-fetched this run

    # ---- seeding ---------------------------------------------------------------------

    def _add_players(self, conn, platform: str, tier: str, priority: int,
                     entries: list[dict]) -> None:
        now = int(time.time())
        rows = []
        for e in entries:
            puuid = e.get("puuid")
            if not puuid and e.get("summonerId"):
                puuid = self.client.puuid_from_summoner_id(platform, e["summonerId"])
            if puuid:
                rows.append((puuid, platform, tier, priority, e.get("leaguePoints"),
                             e.get("wins"), e.get("losses"), now, now))
        conn.executemany(
            "INSERT INTO players(puuid, platform, tier, priority, lp, wins, losses, first_seen, "
            "seeded_at) VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(puuid) DO UPDATE SET "
            "tier=excluded.tier, platform=excluded.platform, priority=excluded.priority, "
            "lp=excluded.lp, wins=excluded.wins, losses=excluded.losses, "
            "seeded_at=excluded.seeded_at",   # first_seen is deliberately kept
            rows,
        )
        conn.executemany(
            "INSERT OR IGNORE INTO ladder_history(puuid, at, priority, lp, wins, losses) "
            "VALUES(?,?,?,?,?,?)", [(r[0], now, r[3], r[4], r[5], r[6]) for r in rows])

    def seed(self, platform: str) -> None:
        conn = db.connect(self.args.db)
        try:
            last = db.get_meta(conn, f"seeded:{platform}")
            unranked = conn.execute(
                "SELECT 1 FROM players WHERE platform=? AND priority IS NULL LIMIT 1",
                (platform,)).fetchone()  # database from before ladder positions were stored
            if last and not unranked and time.time() - float(last) < RESEED_AFTER_S:
                return
            for tier in APEX_TIERS:
                if self.stop.is_set():
                    return
                self._add_players(conn, platform, tier.upper(), TIER_PRIORITY[tier.upper()],
                                  self.client.apex_league(platform, tier))
            for division in DIVISIONS:
                page = 1
                while not self.stop.is_set():
                    entries = self.client.diamond_page(platform, division, page)
                    if not entries:
                        break
                    self._add_players(conn, platform, "DIAMOND", DIVISION_PRIORITY[division],
                                      entries)
                    page += 1
            if not self.stop.is_set():
                now = str(time.time())
                db.set_meta(conn, f"seeded:{platform}", now)
                if db.get_meta(conn, f"initial_seed:{platform}") is None:
                    db.set_meta(conn, f"initial_seed:{platform}", now)
        except RiotAuthError as e:
            self.errors.append(str(e))
            self.stop.set()
        except Exception as e:  # keep the other threads alive; this platform keeps what it has
            self.errors.append(f"seeding {platform} failed: {e!r}")
        finally:
            self.seeded[platform].set()
            conn.close()

    # ---- crawling --------------------------------------------------------------------

    def _next_players(self, conn, platforms: list[str]) -> list[tuple]:
        """Players never visited, or not visited for --recrawl-hours; top of the ladder first.

        Revisits of high players outrank first visits of lower ones, so the crawl settles at
        the ladder depth the request budget can keep fresh."""
        if self.args.order == "tier" and not all(self.seeded[p].is_set() for p in platforms):
            return []  # ladder positions are being refreshed; the caller backfills meanwhile
        marks = ",".join("?" * len(platforms))
        cutoff = int(time.time()) - int(self.args.recrawl_hours * 3600)
        order = ("priority IS NULL, priority, lp DESC" if self.args.order == "tier"
                 else "crawled_at IS NOT NULL, RANDOM()")
        return conn.execute(
            f"SELECT puuid, platform, crawled_at, tier, first_seen FROM players "
            f"WHERE platform IN ({marks}) AND (crawled_at IS NULL OR crawled_at < ?) "
            f"ORDER BY {order} LIMIT 20",
            (*platforms, cutoff),
        ).fetchall()

    def _history_start(self, conn, platform: str, crawled_at, first_seen) -> int:
        if crawled_at is not None:
            return max(self.since, crawled_at - 3600)
        initial = db.get_meta(conn, f"initial_seed:{platform}")
        if initial is not None and first_seen and first_seen > float(initial) + 3600:
            return max(self.since, int(first_seen))  # joined the list later: forward only
        return self.since

    def _crawl_player(self, conn, region, puuid, platform, crawled_at, tier, first_seen) -> int:
        """Returns the number of API requests spent."""
        start_time = self._history_start(conn, platform, crawled_at, first_seen)
        now = int(time.time())
        spent, offset, done = 0, 0, False
        while not done:
            ids = self.client.match_ids(region, puuid, start_time, start=offset)
            spent += 1
            for match_id in ids:
                if self.stop.is_set():
                    return spent  # player stays unmarked and is retried next run
                known = conn.execute(
                    "SELECT 1 FROM matches WHERE match_id=? UNION ALL "
                    "SELECT 1 FROM skipped WHERE match_id=?", (match_id, match_id)
                ).fetchone()
                if known:
                    continue
                payload = self.client.match(region, match_id)
                spent += 1
                if payload is None:
                    conn.execute("INSERT OR IGNORE INTO skipped VALUES(?,?)", (match_id, "404"))
                    continue
                row, reason = parse_match(payload)
                self.fetched += 1
                if row is None:
                    conn.execute("INSERT OR IGNORE INTO skipped VALUES(?,?)", (match_id, reason))
                    # ids are newest-first: a game from too old a patch means the rest is older
                    try:
                        version = payload["info"]["gameVersion"]
                        if db.patch_key(patch_of(version)) < self.min_patch:
                            done = True
                            break
                    except (KeyError, TypeError, ValueError):
                        pass
                    continue
                row["seed_tier"] = tier
                conn.execute(INSERT_MATCH, row)
                if db.patch_key(row["patch"]) < self.min_patch:
                    done = True
                    break
            if len(ids) < PAGE:
                done = True
            offset += PAGE
        conn.execute("UPDATE players SET crawled_at=? WHERE puuid=?", (now, puuid))
        return spent

    def _backfill(self, conn, region: str, platforms: list[str], limit: int) -> int:
        """Re-fetch up to `limit` matches that lack player ids / statistics, newest first."""
        marks = ",".join("?" * len(platforms))
        ids = [r[0] for r in conn.execute(
            f"SELECT match_id FROM matches WHERE pstats IS NULL AND platform IN ({marks}) "
            f"ORDER BY game_creation DESC LIMIT ?", (*platforms, limit))]
        for match_id in ids:
            if self.stop.is_set():
                break
            payload = self.client.match(region, match_id)
            row, _ = parse_match(payload) if payload else (None, "404")
            if row is None:
                conn.execute(
                    "UPDATE matches SET blue_puuids=COALESCE(blue_puuids, ''), "
                    "red_puuids=COALESCE(red_puuids, ''), pstats='' WHERE match_id=?",
                    (match_id,))
            else:
                conn.execute(
                    "UPDATE matches SET blue=:blue, red=:red, roles_ok=:roles_ok, "
                    "blue_puuids=:blue_puuids, red_puuids=:red_puuids, bans=:bans, stats=:stats, "
                    "pstats=:pstats "
                    "WHERE match_id=:match_id", row)
            self.backfilled += 1
        return len(ids)

    def crawl(self, region: str, platforms: list[str]) -> None:
        conn = db.connect(self.args.db)
        share = min(max(self.args.backfill_share, 0.0), 1.0)
        try:
            while not self.stop.is_set():
                players = self._next_players(conn, platforms) if share < 1 else []
                if not players:
                    # idle (ladder refresh running, or everyone is up to date): backfill
                    if self._backfill(conn, region, platforms, 20):
                        continue
                    if share >= 1 or all(self.seeded[p].is_set() for p in platforms):
                        if share >= 1 or self.args.once:
                            return
                        time.sleep(30)  # wait for players to become due for a revisit
                    else:
                        time.sleep(2)
                    continue
                for puuid, platform, crawled_at, tier, first_seen in players:
                    if self.stop.is_set():
                        return
                    spent = self._crawl_player(conn, region, puuid, platform, crawled_at, tier,
                                               first_seen)
                    if share > 0:
                        self._backfill(conn, region, platforms,
                                       math.ceil(spent * share / (1 - share)))
        except RiotAuthError as e:
            self.errors.append(str(e))
            self.stop.set()
        except Exception as e:
            self.errors.append(f"crawler {region} failed: {e!r}")
        finally:
            conn.close()


def save_champions(path: Path, version: str) -> None:
    champs = ddragon_champions(version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"version": version, "champions": {str(k): v for k, v in sorted(champs.items())}},
        indent=1,
    ))


def status_line(conn, crawler: Crawler, t0: float) -> str:
    counts = db.patch_counts(conn)
    total = sum(n for _, n in counts)
    top = ", ".join(f"{p}: {n}" for p, n in counts[:3]) or "none yet"
    done, players = conn.execute("SELECT COUNT(crawled_at), COUNT(*) FROM players").fetchone()
    todo = conn.execute("SELECT COUNT(*) FROM matches WHERE pstats IS NULL").fetchone()[0]
    minutes = max(1e-9, (time.time() - t0) / 60)
    return (f"[{time.strftime('%H:%M:%S')}] matches {total} ({top}) | "
            f"players visited {done}/{players} | new {crawler.fetched / minutes:.0f}/min | "
            f"backfill {crawler.backfilled / minutes:.0f}/min, {todo} left")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platforms", nargs="+", default=["kr", "euw1", "na1", "vn2"],
                    choices=sorted(PLATFORM_TO_REGION), metavar="PLATFORM",
                    help="platforms to crawl (default: kr euw1 na1 vn2, one per API region, "
                         "so all four rate budgets are used in parallel)")
    ap.add_argument("--db", default="data/matches.sqlite")
    ap.add_argument("--champions", default="data/champions.json")
    ap.add_argument("--patches", type=int, default=2,
                    help="how many of the newest patches to collect (default 2); matches of "
                         "older patches already stored are kept and still used for training")
    ap.add_argument("--days", type=int, default=35,
                    help="never look further back than this many days (default 35)")
    ap.add_argument("--target-matches", type=int, default=0,
                    help="stop once this many matches of the newest patch are stored "
                         "(default 0 = keep running until stopped)")
    ap.add_argument("--recrawl-hours", type=float, default=24,
                    help="revisit a player for new games after this many hours")
    ap.add_argument("--order", choices=["tier", "random"], default="tier",
                    help="visit players from the top of the ladder down (default), or randomly")
    ap.add_argument("--backfill-share", type=float, default=0.2,
                    help="share of requests spent re-fetching old rows without player ids "
                         "(default 0.2; 1 = backfill only, 0 = never)")
    ap.add_argument("--once", action="store_true",
                    help="exit when no player is due for a visit instead of waiting")
    args = ap.parse_args(argv)
    args.platforms = list(dict.fromkeys(args.platforms))

    try:
        client = RiotClient(load_api_key())
    except RiotAuthError as e:
        raise SystemExit(f"error: {e}")

    versions = ddragon_versions()
    patches = list(dict.fromkeys(patch_of(v) for v in versions if v[0].isdigit()))
    target_patch = patches[0]
    min_patch = db.patch_key(patches[min(args.patches, len(patches)) - 1])
    save_champions(Path(args.champions), versions[0])
    print(f"newest patch {target_patch}; collecting patches >= "
          f"{'.'.join(map(str, min_patch))} from {', '.join(args.platforms)}")

    conn = db.connect(args.db)  # also migrates an older database before the threads start
    crawler = Crawler(args, client, min_patch, target_patch)
    by_region = defaultdict(list)
    for p in args.platforms:
        by_region[PLATFORM_TO_REGION[p]].append(p)
    threads = [threading.Thread(target=crawler.seed, args=(p,), daemon=True)
               for p in args.platforms]
    threads += [threading.Thread(target=crawler.crawl, args=(r, ps), daemon=True)
                for r, ps in by_region.items()]
    random.seed()
    for t in threads:
        t.start()

    t0 = time.time()
    last_print = 0.0
    try:
        while any(t.is_alive() for t in threads) and not crawler.stop.is_set():
            time.sleep(1)
            if time.time() - last_print >= 60:
                last_print = time.time()
                print(status_line(conn, crawler, t0), flush=True)
                if args.target_matches:
                    # the live patch can be newer than Data Dragon's newest: count the newest seen
                    counts = [(p, n) for p, n in db.patch_counts(conn)
                              if db.patch_key(p) >= db.patch_key(target_patch)]
                    if counts and counts[0][1] >= args.target_matches:
                        print(f"reached {args.target_matches} matches on patch {counts[0][0]}")
                        break
    except KeyboardInterrupt:
        print("\nstopping (finishing in-flight requests)...")
    crawler.stop.set()
    for t in threads:
        t.join(timeout=90)
    print(status_line(conn, crawler, t0))
    for e in crawler.errors:
        print("ERROR:", e)
    if crawler.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
