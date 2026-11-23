import argparse
import json
import sqlite3
import time

import pytest

from lolwp import ROLES, db
from lolwp.collect import Crawler
from lolwp.riot import RateLimiter, parse_match, patch_of


def fake_match(match_id="KR_1", version="16.22.700.1", blue_win=True, duration=1800,
               queue=420, positions=None, early_surrender=False, champs=None):
    champs = champs or list(range(1, 11))
    parts = []
    for i in range(10):
        team = 100 if i < 5 else 200
        pos = (positions or list(ROLES) * 2)[i]
        parts.append({"teamId": team, "championId": champs[i], "teamPosition": pos,
                      "gameEndedInEarlySurrender": early_surrender, "puuid": f"{match_id}-p{i}",
                      "goldEarned": 10_000 + 100 * i, "kills": i})
    # shuffle participant order within the payload to prove we sort by role
    parts = parts[4::-1] + parts[:4:-1]
    return {
        "metadata": {"matchId": match_id},
        "info": {
            "queueId": queue, "gameVersion": version, "gameCreation": 1_760_000_000_000,
            "gameDuration": duration, "platformId": "KR", "participants": parts,
            "teams": [
                {"teamId": 100, "win": blue_win,
                 "bans": [{"championId": c, "pickTurn": j} for j, c in enumerate([51, 52, 53, 54, -1])],
                 "objectives": {"tower": {"kills": 9}, "dragon": {"kills": 3}, "baron": {"kills": 1},
                                "inhibitor": {"kills": 2}}},
                {"teamId": 200, "win": not blue_win, "bans": [{"championId": 61, "pickTurn": 6}],
                 "objectives": {"tower": {"kills": 2}}},
            ],
        },
    }


def test_patch_of():
    assert patch_of("16.22.700.1234") == "16.22"
    assert patch_of("16.3.1") == "16.3"
    with pytest.raises(ValueError):
        patch_of("garbage")


def test_patch_ordering_is_numeric():
    assert db.patch_key("16.10") > db.patch_key("16.9")


def test_parse_match_orders_by_role():
    row, reason = parse_match(fake_match())
    assert reason == ""
    assert row["blue"] == "1,2,3,4,5" and row["red"] == "6,7,8,9,10"
    assert row["roles_ok"] == 1 and row["blue_win"] == 1
    assert row["patch"] == "16.22" and row["platform"] == "kr"


def test_parse_match_red_win():
    row, _ = parse_match(fake_match(blue_win=False))
    assert row["blue_win"] == 0


def test_parse_match_bad_roles_kept_without_roles():
    positions = ["TOP", "TOP", "MIDDLE", "BOTTOM", "UTILITY"] + list(ROLES)
    row, _ = parse_match(fake_match(positions=positions))
    assert row["roles_ok"] == 0
    assert sorted(row["blue"].split(","), key=int) == ["1", "2", "3", "4", "5"]
    assert sorted(row["red"].split(","), key=int) == ["6", "7", "8", "9", "10"]


@pytest.mark.parametrize("kwargs,reason", [
    ({"duration": 200}, "remake"),
    ({"early_surrender": True}, "remake"),
    ({"queue": 440}, "queue"),
    ({"champs": [1, 1, 3, 4, 5, 6, 7, 8, 9, 10]}, "champions"),
])
def test_parse_match_rejects(kwargs, reason):
    row, why = parse_match(fake_match(**kwargs))
    assert row is None and why == reason


def test_parse_match_malformed():
    row, why = parse_match({"metadata": {"matchId": "X"}, "info": {"queueId": 420}})
    assert row is None and why.startswith("malformed")


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        assert s > 0
        self.t += s


def test_rate_limiter_never_exceeds_any_window():
    clock = FakeClock()
    limiter = RateLimiter(((20, 1.0), (100, 120.0)), clock=clock.now, sleep=clock.sleep)
    stamps = []
    for _ in range(350):
        limiter.acquire()
        stamps.append(clock.t)
    for limit, window in limiter.limits:
        for i, s in enumerate(stamps):
            in_window = sum(1 for x in stamps[i:] if x < s + window)
            assert in_window <= limit
    # and it is not needlessly slow: 350 requests need 3 full 120s windows, not more than ~4
    assert stamps[-1] < 4 * 124


def test_rate_limiter_header_and_block():
    clock = FakeClock()
    limiter = RateLimiter(clock=clock.now, sleep=clock.sleep)
    limiter.set_limits_from_header("500:10,30000:600")
    assert limiter.limits == ((500, 10.0), (30000, 600.0))
    limiter.set_limits_from_header("nonsense")
    assert limiter.limits == ((500, 10.0), (30000, 600.0))
    limiter.block_for(7)
    limiter.acquire()
    assert clock.t >= 7


class FakeClient:
    """Two Diamond+ players sharing one match; histories are newest-first."""

    def __init__(self):
        self.match_calls = []
        self.list_calls = []
        self.matches = {
            "KR_3": fake_match("KR_3", "16.22.1.1"),
            "KR_2": fake_match("KR_2", "16.21.1.1", blue_win=False),
            "KR_1": fake_match("KR_1", "16.20.1.1"),      # older than wanted
            "KR_0": fake_match("KR_0", "16.20.1.1"),      # must never be fetched
            "KR_R": fake_match("KR_R", "16.22.1.1", duration=100),  # remake
        }
        self.histories = {"p1": ["KR_3", "KR_R", "KR_2", "KR_1", "KR_0"], "p2": ["KR_3"]}

    def apex_league(self, platform, tier):
        entry = {"puuid": "p1", "leaguePoints": 900, "wins": 120, "losses": 80}
        return [entry] if tier == "challenger" else []

    def diamond_page(self, platform, division, page):
        entry = {"puuid": "p2", "leaguePoints": 40, "wins": 60, "losses": 55}
        return [entry] if (division, page) == ("I", 1) else []

    def match_ids(self, region, puuid, start_time, start=0, count=100):
        self.list_calls.append((puuid, start_time, start))
        return self.histories[puuid][start:start + count]

    def match(self, region, match_id):
        self.match_calls.append(match_id)
        return self.matches[match_id]


def crawl_args(tmp_path, **kw):
    base = dict(db=str(tmp_path / "m.sqlite"), platforms=["kr"], days=30, recrawl_hours=24,
                order="tier", backfill_share=0.2, once=True)
    return argparse.Namespace(**{**base, **kw})


def test_crawler_end_to_end(tmp_path):
    args = crawl_args(tmp_path)
    client = FakeClient()
    crawler = Crawler(args, client, min_patch=(16, 21), target_patch="16.22")
    crawler.seed("kr")
    crawler.crawl("asia", ["kr"])
    assert crawler.errors == []

    conn = db.connect(args.db)
    stored = dict(conn.execute("SELECT match_id, patch FROM matches").fetchall())
    # KR_1 is stored (already paid for) but ends the walk; KR_0 is never requested
    assert stored == {"KR_3": "16.22", "KR_2": "16.21", "KR_1": "16.20"}
    assert "KR_0" not in client.match_calls
    assert client.match_calls.count("KR_3") == 1  # shared match fetched once
    assert conn.execute("SELECT reason FROM skipped WHERE match_id='KR_R'").fetchone() == ("remake",)
    ladder = conn.execute("SELECT puuid, tier, priority, lp, wins, losses FROM players "
                          "ORDER BY puuid").fetchall()
    assert ladder == [("p1", "CHALLENGER", 0, 900, 120, 80), ("p2", "DIAMOND", 3, 40, 60, 55)]
    assert client.list_calls[0][0] == "p1"  # top of the ladder is visited first
    blue_ids, red_ids, bans, stats = conn.execute(
        "SELECT blue_puuids, red_puuids, bans, stats FROM matches WHERE match_id='KR_3'").fetchone()
    assert blue_ids == ",".join(f"KR_3-p{i}" for i in range(5))  # same role order as champions
    assert red_ids == ",".join(f"KR_3-p{i}" for i in range(5, 10))
    assert bans == "51,52,53,54,-1,61,-1,-1,-1,-1"
    assert json.loads(stats) == {
        "gold": [51_000, 53_500], "kills": [10, 35], "towers": [9, 2], "dragons": [3, 0],
        "barons": [1, 0], "inhibitors": [2, 0], "surrender": 0}
    assert db.clean_cutoff_ms(conn) == int(float(db.get_meta(conn, "initial_seed:kr")) * 1000)
    assert conn.execute("SELECT COUNT(*) FROM players WHERE crawled_at IS NULL").fetchone() == (0,)
    assert db.patch_counts(conn)[0] == ("16.22", 1)
    assert float(db.get_meta(conn, "seeded:kr")) <= time.time()


def test_crawler_pages_through_long_histories(tmp_path):
    client = FakeClient()
    client.histories = {"p1": [f"KR_L{i}" for i in range(150)], "p2": []}
    client.matches = {m: fake_match(m, "16.22.1.1") for m in client.histories["p1"]}
    crawler = Crawler(crawl_args(tmp_path, backfill_share=0), client, (16, 21), "16.22")
    crawler.seed("kr")
    crawler.crawl("asia", ["kr"])
    conn = db.connect(crawl_args(tmp_path).db)
    assert conn.execute("SELECT COUNT(*) FROM matches").fetchone() == (150,)
    assert [c[2] for c in client.list_calls if c[0] == "p1"] == [0, 100]


def test_late_joiners_are_only_crawled_forward(tmp_path):
    """A player first listed after the initial ladder snapshot must not contribute games
    from before they were listed (those are biased towards their own wins)."""
    args = crawl_args(tmp_path, backfill_share=0)
    conn = db.connect(args.db)
    now = int(time.time())
    db.set_meta(conn, "seeded:kr", str(now))
    db.set_meta(conn, "initial_seed:kr", str(now - 5 * 86400))
    conn.execute("INSERT INTO players(puuid, platform, tier, priority, lp, first_seen) VALUES"
                 "('p1','kr','MASTER',2,10,?), ('p2','kr','MASTER',2,5,?)",
                 (now - 5 * 86400, now - 86400))
    client = FakeClient()
    client.histories = {"p1": [], "p2": []}
    crawler = Crawler(args, client, (16, 21), "16.22")
    crawler.seeded["kr"].set()
    crawler.crawl("asia", ["kr"])
    starts = {puuid: start_time for puuid, start_time, _ in client.list_calls}
    assert starts["p1"] == crawler.since          # initial player: full window
    assert starts["p2"] == now - 86400            # late joiner: from first_seen


def old_schema_db(path):
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE players (puuid TEXT PRIMARY KEY, platform TEXT NOT NULL, tier TEXT NOT NULL,
                              crawled_at INTEGER);
        CREATE TABLE matches (match_id TEXT PRIMARY KEY, platform TEXT NOT NULL,
            game_version TEXT NOT NULL, patch TEXT NOT NULL, game_creation INTEGER NOT NULL,
            duration INTEGER NOT NULL, blue TEXT NOT NULL, red TEXT NOT NULL,
            roles_ok INTEGER NOT NULL, blue_win INTEGER NOT NULL, seed_tier TEXT);
        CREATE TABLE skipped (match_id TEXT PRIMARY KEY, reason TEXT);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO meta VALUES ('seeded:kr', '1794000000.5');
        INSERT INTO players VALUES ('p1', 'kr', 'DIAMOND', 1794000100);
        INSERT INTO matches VALUES ('KR_3', 'kr', '16.22.1.1', '16.22', 1793000000000, 1800,
                                    '1,2,3,4,5', '6,7,8,9,10', 1, 1, 'DIAMOND');
    """)
    conn.commit()
    conn.close()


def test_old_database_is_migrated_and_backfilled(tmp_path):
    args = crawl_args(tmp_path, backfill_share=1)
    old_schema_db(args.db)
    conn = db.connect(args.db)
    assert conn.execute("SELECT first_seen, priority FROM players").fetchone() == (1794000000, None)
    assert db.get_meta(conn, "initial_seed:kr") == "1794000000.5"
    assert conn.execute("SELECT blue_puuids FROM matches").fetchone() == (None,)
    db.connect(args.db).close()  # migrating twice is harmless

    client = FakeClient()
    crawler = Crawler(args, client, (16, 21), "16.22")
    crawler.seeded["kr"].set()
    crawler.crawl("asia", ["kr"])   # backfill only: no player is visited
    assert client.list_calls == [] and client.match_calls == ["KR_3"]
    row = conn.execute("SELECT blue, blue_puuids, bans, blue_win FROM matches").fetchone()
    assert row == ("1,2,3,4,5", ",".join(f"KR_3-p{i}" for i in range(5)),
                   "51,52,53,54,-1,61,-1,-1,-1,-1", 1)
    assert crawler.backfilled == 1
