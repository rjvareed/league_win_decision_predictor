"""A single player's recent games, fetched live, turned into the model's player inputs.

The win model was trained with each player's record *inside the crawl window* (about a month
of ranked games). To advise one player in champion select we need the same numbers for them:
this module looks the player up by Riot ID, collects their recent ranked solo games (from the
local match database where it already has them, otherwise from the Riot API, cached in
data/live_cache.sqlite) and computes PLAYER_FEATURES for "this player on champion X in role R,
right now" for any number of candidate champions at once.

    lookup = PlayerLookup.open()                      # needs RIOT_API_KEY in .env
    me = lookup.player("Some Name#TAG", "euw1")       # ~1 request per game not seen before
    feats = me.features([champion ids], role=2)       # [n, len(PLAYER_FEATURES)]
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import PLAYER_FEATURES, featurize
from .riot import PLATFORM_TO_REGION, RiotClient, load_api_key, parse_match

# account-v1 only exists on three regional hosts; any of them resolves any Riot ID
ACCOUNT_REGION = {"americas": "americas", "europe": "europe", "asia": "asia", "sea": "asia"}
ROW_COLUMNS = ("blue", "red", "roles_ok", "blue_win", "patch", "game_creation", "duration",
               "blue_puuids", "red_puuids", "stats", "pstats")
HISTORY_DAYS = 30     # about the span of player history the model saw in training
HISTORY_GAMES = 40
LIST_TTL_S = 300      # how long a fetched match list is reused

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (match_id TEXT PRIMARY KEY, row TEXT);   -- row: JSON, NULL = unusable
CREATE TABLE IF NOT EXISTS lists (puuid TEXT PRIMARY KEY, fetched_at REAL, ids TEXT);
CREATE TABLE IF NOT EXISTS accounts (riot_id TEXT PRIMARY KEY, puuid TEXT, name TEXT);
"""


@dataclass
class ChampionRecord:
    games: int = 0
    wins: int = 0


@dataclass
class Player:
    puuid: str
    name: str
    rows: list[tuple]                       # match rows in time order (featurize layout)
    records: dict[int, ChampionRecord] = field(default_factory=dict)   # champion id -> record
    role_games: list[int] = field(default_factory=lambda: [0] * 5)
    games: int = 0
    wins: int = 0
    result_means: list | None = None        # from the model file, see data.stat_features

    def __post_init__(self):
        for r in self.rows:
            players = r[7].split(",") + r[8].split(",")
            if self.puuid not in players:
                continue
            slot = players.index(self.puuid)
            champion = int((r[0].split(",") + r[1].split(","))[slot])
            won = bool(r[3]) == (slot < 5)
            rec = self.records.setdefault(champion, ChampionRecord())
            rec.games += 1
            rec.wins += won
            self.games += 1
            self.wins += won
            if r[2]:
                self.role_games[slot % 5] += 1

    def features(self, champions: list[int], role: int, lane_opponent: int = 0,
                 now_ms: int | None = None) -> np.ndarray:
        """[len(champions), len(PLAYER_FEATURES)]: the model's inputs for this player on each
        of these champions (Riot champion ids) in `role` (0..4 = top..support), as if a game
        started now. lane_opponent: champion id of the enemy in the same role, if known."""
        if not champions:
            return np.zeros((0, len(PLAYER_FEATURES)), dtype=np.float32)
        now_ms = int(time.time() * 1000) if now_ms is None else now_ms
        last = self.rows[-1][5] if self.rows else 0
        now_ms = max(now_ms, last + 1)
        mine = ",".join(self.puuid if r == role else "" for r in range(5))
        hypothetical = []
        for c in champions:
            blue = ",".join(str(int(c)) if r == role else "0" for r in range(5))
            red = ",".join(str(int(lane_opponent)) if r == role else "0" for r in range(5))
            hypothetical.append((blue, red, 1, 0, "", now_ms, 1800, mine, ",,,,", None, None))
        rows = list(self.rows) + hypothetical
        flag = np.r_[np.zeros(len(self.rows), dtype=bool), np.ones(len(champions), dtype=bool)]
        return featurize(rows, selected=flag, frozen=flag,
                         result_means=self.result_means)["feats"][:, role]


class PlayerLookup:
    def __init__(self, client: RiotClient, cache_path: str | Path = "data/live_cache.sqlite",
                 db_path: str | Path | None = "data/matches.sqlite"):
        self.client = client
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self.cache = sqlite3.connect(cache_path, isolation_level=None, check_same_thread=False)
        self.cache.executescript(CACHE_SCHEMA)
        self.db_path = Path(db_path) if db_path and Path(db_path).is_file() else None

    @classmethod
    def open(cls, env_file: str | Path = ".env", **kwargs) -> "PlayerLookup":
        """Raises riot.RiotAuthError if no API key is configured."""
        return cls(RiotClient(load_api_key(env_file), max_retries=3), **kwargs)

    # ---- lookups -----------------------------------------------------------------------

    def resolve(self, riot_id: str, platform: str) -> tuple[str, str]:
        """'Name#TAG' -> (puuid, display name)."""
        if "#" not in riot_id:
            raise ValueError("give the Riot ID as Name#TAG")
        if platform not in PLATFORM_TO_REGION:
            raise ValueError(f"unknown platform {platform!r} (e.g. {', '.join(PLATFORM_TO_REGION)})")
        name, tag = (part.strip() for part in riot_id.rsplit("#", 1))
        key = f"{name}#{tag}".lower()
        row = self.cache.execute("SELECT puuid, name FROM accounts WHERE riot_id=?", (key,)).fetchone()
        if row:
            return row[0], row[1]
        region = ACCOUNT_REGION[PLATFORM_TO_REGION[platform]]
        from urllib.parse import quote
        data = self.client.get(
            region, f"/riot/account/v1/accounts/by-riot-id/{quote(name)}/{quote(tag)}")
        if not data or not data.get("puuid"):
            raise ValueError(f"no account found for {riot_id!r}")
        shown = f"{data.get('gameName', name)}#{data.get('tagLine', tag)}"
        self.cache.execute("INSERT OR REPLACE INTO accounts VALUES(?,?,?)", (key, data["puuid"], shown))
        return data["puuid"], shown

    def _stored(self, match_ids: list[str]) -> dict[str, tuple | None]:
        """Rows already in the live cache or the crawl database."""
        found: dict[str, tuple | None] = {}
        marks = ",".join("?" * len(match_ids))
        for mid, row in self.cache.execute(
                f"SELECT match_id, row FROM matches WHERE match_id IN ({marks})", match_ids):
            found[mid] = tuple(json.loads(row)) if row else None
        missing = [m for m in match_ids if m not in found]
        if missing and self.db_path is not None:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            try:
                marks = ",".join("?" * len(missing))
                for mid, *row in conn.execute(
                        f"SELECT match_id, {', '.join(ROW_COLUMNS)} FROM matches "
                        f"WHERE match_id IN ({marks})", missing):
                    if row[7] and row[10]:      # only rows that have player ids and statistics
                        found[mid] = tuple(row)
            except sqlite3.Error:
                pass
            finally:
                conn.close()
        return found

    def history(self, puuid: str, platform: str, games: int = HISTORY_GAMES,
                days: int = HISTORY_DAYS, progress=None) -> list[tuple]:
        """The player's ranked solo games of the last `days` days (at most `games`, newest
        ones), as match rows in time order. One API request per game not stored yet."""
        region = PLATFORM_TO_REGION[platform]
        now = time.time()
        row = self.cache.execute("SELECT fetched_at, ids FROM lists WHERE puuid=?", (puuid,)).fetchone()
        if row and now - row[0] < LIST_TTL_S:
            ids = json.loads(row[1])
        else:
            ids = self.client.match_ids(region, puuid, int(now - days * 86400), count=100)
            self.cache.execute("INSERT OR REPLACE INTO lists VALUES(?,?,?)", (puuid, now, json.dumps(ids)))
        ids = ids[:games]
        found = self._stored(ids) if ids else {}
        todo = [m for m in ids if m not in found]
        for k, mid in enumerate(todo):
            if progress:
                progress(k, len(todo))
            payload = self.client.match(region, mid)
            parsed, _ = parse_match(payload) if payload else (None, "missing")
            usable = parsed is not None and parsed["blue_puuids"] and parsed["pstats"]
            found[mid] = tuple(parsed[c] for c in ROW_COLUMNS) if usable else None
            self.cache.execute("INSERT OR REPLACE INTO matches VALUES(?,?)",
                               (mid, json.dumps(found[mid]) if usable else None))
        rows = [r for r in (found.get(m) for m in ids) if r is not None]
        return sorted(rows, key=lambda r: r[5])

    def player(self, riot_id: str, platform: str, games: int = HISTORY_GAMES,
               days: int = HISTORY_DAYS, progress=None) -> Player:
        puuid, name = self.resolve(riot_id, platform)
        return Player(puuid, name, self.history(puuid, platform, games, days, progress))
