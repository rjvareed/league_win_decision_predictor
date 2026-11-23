"""SQLite storage for crawled players and matches.

One connection per thread (WAL mode), so the crawler threads can write concurrently.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    puuid      TEXT PRIMARY KEY,
    platform   TEXT NOT NULL,
    tier       TEXT NOT NULL,
    crawled_at INTEGER,             -- unix seconds of the last match-list crawl, NULL = never
    priority   INTEGER,             -- 0 Challenger, 1 Grandmaster, 2 Master, 3..6 Diamond I..IV
    lp         INTEGER,             -- league points at the last ladder snapshot
    wins       INTEGER,             -- season wins / losses at the last ladder snapshot
    losses     INTEGER,
    first_seen INTEGER,             -- unix seconds when the player first appeared on the ladder list
    seeded_at  INTEGER              -- unix seconds of the last ladder snapshot
);
CREATE INDEX IF NOT EXISTS idx_players_todo ON players (platform, crawled_at);

CREATE TABLE IF NOT EXISTS matches (
    match_id      TEXT PRIMARY KEY,
    platform      TEXT NOT NULL,
    game_version  TEXT NOT NULL,    -- e.g. 16.22.725.1234
    patch         TEXT NOT NULL,    -- e.g. 16.22
    game_creation INTEGER NOT NULL, -- unix milliseconds
    duration      INTEGER NOT NULL, -- seconds
    blue          TEXT NOT NULL,    -- 5 champion ids, comma separated
    red           TEXT NOT NULL,
    roles_ok      INTEGER NOT NULL, -- 1: both teams are ordered TOP,JUNGLE,MIDDLE,BOTTOM,UTILITY
    blue_win      INTEGER NOT NULL,
    seed_tier     TEXT,             -- tier of the Diamond+ player whose history surfaced the match
    blue_puuids   TEXT,             -- 5 player ids in the same order as `blue`; NULL = not fetched
    red_puuids    TEXT,             --   yet (old rows, see lolwp-collect backfill), '' = unavailable
    bans          TEXT,             -- 10 champion ids (blue's 5, then red's 5), -1 = no ban
    stats         TEXT,             -- JSON, end-of-game team totals as [blue, red] pairs
    pstats        TEXT              -- JSON, 10 lists of per-player numbers (riot.PSTAT_FIELDS);
);                                  --   NULL = not fetched yet, '' = unavailable
CREATE INDEX IF NOT EXISTS idx_matches_patch ON matches (patch);

-- match ids that were fetched and rejected (remake, wrong queue, malformed), so we never refetch
CREATE TABLE IF NOT EXISTS skipped (
    match_id TEXT PRIMARY KEY,
    reason   TEXT
);

-- every ladder snapshot of every player, so that rank can later be used as it was *before* a game
CREATE TABLE IF NOT EXISTS ladder_history (
    puuid    TEXT NOT NULL,
    at       INTEGER NOT NULL,      -- unix seconds
    priority INTEGER,
    lp       INTEGER,
    wins     INTEGER,
    losses   INTEGER,
    PRIMARY KEY (puuid, at)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# columns added after the first release: (table, column, type)
ADDED_COLUMNS = [
    ("players", "priority", "INTEGER"), ("players", "lp", "INTEGER"),
    ("players", "wins", "INTEGER"), ("players", "losses", "INTEGER"),
    ("players", "first_seen", "INTEGER"), ("players", "seeded_at", "INTEGER"),
    ("matches", "blue_puuids", "TEXT"), ("matches", "red_puuids", "TEXT"),
    ("matches", "bans", "TEXT"), ("matches", "stats", "TEXT"),
    ("matches", "pstats", "TEXT"),
]


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring a database created by an older version up to the current schema."""
    added = False
    for table, column, kind in ADDED_COLUMNS:
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
            added = True
    if added:
        # Players of an old database were all listed by its (single) ladder snapshot.
        conn.execute(
            "UPDATE players SET first_seen = COALESCE((SELECT CAST(value AS INTEGER) FROM meta "
            "WHERE key = 'seeded:' || players.platform), 0) WHERE first_seen IS NULL")
        conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) "
            "SELECT 'initial_seed:' || substr(key, 8), value FROM meta WHERE key LIKE 'seeded:%'")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_players_priority ON players (platform, priority)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_time ON matches (game_creation)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_todo ON matches (platform, game_creation) "
                 "WHERE pstats IS NULL")


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=60, isolation_level=None)  # autocommit
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=60000")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def patch_key(patch: str) -> tuple[int, ...]:
    """'16.22' -> (16, 22), for correct numeric ordering of patches."""
    return tuple(int(p) for p in patch.split("."))


def patch_counts(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    """[(patch, n_matches)] sorted newest patch first."""
    rows = conn.execute("SELECT patch, COUNT(*) FROM matches GROUP BY patch").fetchall()
    return sorted(rows, key=lambda r: patch_key(r[0]), reverse=True)


def clean_cutoff_ms(conn: sqlite3.Connection) -> int | None:
    """Games created after this moment were played after the first ladder snapshot of every
    platform, i.e. the player list that surfaced them cannot depend on their outcome."""
    rows = conn.execute("SELECT value FROM meta WHERE key LIKE 'initial_seed:%'").fetchall()
    return int(max(float(v) for (v,) in rows) * 1000) if rows else None


def initial_seeds_ms(conn: sqlite3.Connection) -> dict[str, int]:
    """{platform: unix ms of its first ladder snapshot}."""
    rows = conn.execute("SELECT key, value FROM meta WHERE key LIKE 'initial_seed:%'").fetchall()
    return {k.split(":", 1)[1]: int(float(v) * 1000) for k, v in rows}
