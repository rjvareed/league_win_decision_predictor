"""Minimal Riot API client: per-host rate limiting, retries, and match parsing.

Rate limits are enforced by Riot *per routing value* (each platform host such as
kr.api.riotgames.com and each regional host such as asia.api.riotgames.com has its
own budget), so we keep one limiter per host and crawl hosts in parallel.
"""

from __future__ import annotations

import bisect
import json
import os
import threading
import time
from pathlib import Path

import requests

from . import ROLES

# platform -> regional route used by match-v5
PLATFORM_TO_REGION = {
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe", "me1": "europe",
    "kr": "asia", "jp1": "asia",
    "oc1": "sea", "sg2": "sea", "tw2": "sea", "vn2": "sea",
}

# crawl order: lower = higher on the ladder
TIER_PRIORITY = {"CHALLENGER": 0, "GRANDMASTER": 1, "MASTER": 2}
DIVISION_PRIORITY = {"I": 3, "II": 4, "III": 5, "IV": 6}

RANKED_SOLO_QUEUE_ID = 420
RANKED_SOLO = "RANKED_SOLO_5x5"
APEX_TIERS = ("challenger", "grandmaster", "master")
DIVISIONS = ("I", "II", "III", "IV")
MIN_DURATION_S = 300  # anything shorter is a remake / aborted game

# Limits of a development / personal key. Replaced by the X-App-Rate-Limit header
# of the first response, so a production key is picked up automatically.
DEFAULT_LIMITS = ((20, 1.0), (100, 120.0))


class RiotAuthError(RuntimeError):
    """The API key is missing, invalid or expired (HTTP 401/403)."""


class RiotAPIError(RuntimeError):
    pass


def load_api_key(env_file: str | Path = ".env") -> str:
    key = os.environ.get("RIOT_API_KEY", "").strip()
    if not key:
        p = Path(env_file)
        if p.is_file():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line.startswith("RIOT_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("'\"")
    if not key or "xxxx" in key:
        raise RiotAuthError(
            "No Riot API key found. Get a free key at https://developer.riotgames.com and put it "
            "in .env as RIOT_API_KEY=RGAPI-... (or export RIOT_API_KEY)."
        )
    return key


class RateLimiter:
    """Sliding-window limiter for several (max_requests, window_seconds) limits at once.

    A sliding window that allows at most N requests in any interval of the (slightly
    padded) window length is stricter than Riot's fixed windows, so it cannot overrun them.
    """

    def __init__(self, limits=DEFAULT_LIMITS, clock=time.monotonic, sleep=time.sleep):
        self.limits = tuple(limits)
        self._clock = clock
        self._sleep = sleep
        self._stamps: list[float] = []
        self._blocked_until = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def _padded(window: float) -> float:
        return window * 1.03 + 0.25

    def set_limits_from_header(self, header: str | None) -> None:
        """Parse e.g. '20:1,100:120'."""
        if not header:
            return
        try:
            limits = tuple(
                (int(n), float(w)) for n, w in (part.split(":") for part in header.split(","))
            )
        except ValueError:
            return
        if limits and all(n > 0 and w > 0 for n, w in limits):
            self.limits = limits

    def block_for(self, seconds: float) -> None:
        with self._lock:
            self._blocked_until = max(self._blocked_until, self._clock() + seconds)

    def acquire(self) -> None:
        with self._lock:
            while True:
                now = self._clock()
                wait = self._blocked_until - now
                horizon = max(self._padded(w) for _, w in self.limits)
                cut = bisect.bisect_right(self._stamps, now - horizon)
                if cut:
                    del self._stamps[:cut]
                for limit, window in self.limits:
                    w = self._padded(window)
                    in_window = len(self._stamps) - bisect.bisect_right(self._stamps, now - w)
                    if in_window >= limit:
                        # the request that must leave the window before we may send again
                        blocker = self._stamps[len(self._stamps) - limit]
                        wait = max(wait, blocker + w - now)
                if wait <= 0:
                    self._stamps.append(now)
                    return
                self._sleep(wait)


class RiotClient:
    def __init__(self, api_key: str, max_retries: int = 6, timeout: float = 20.0):
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self._limiters: dict[str, RateLimiter] = {}
        self._limiters_lock = threading.Lock()
        self._local = threading.local()

    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"X-Riot-Token": self.api_key, "User-Agent": "lolwp/0.1"})
            self._local.session = s
        return s

    def limiter(self, host: str) -> RateLimiter:
        with self._limiters_lock:
            if host not in self._limiters:
                self._limiters[host] = RateLimiter()
            return self._limiters[host]

    def get(self, host: str, path: str, params: dict | None = None):
        """GET https://{host}.api.riotgames.com{path}. Returns parsed JSON, or None on 404."""
        url = f"https://{host}.api.riotgames.com{path}"
        limiter = self.limiter(host)
        last_error = "?"
        for attempt in range(self.max_retries):
            limiter.acquire()
            try:
                r = self._session().get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                last_error = repr(e)
                time.sleep(min(60, 2 ** attempt))
                continue
            limiter.set_limits_from_header(r.headers.get("X-App-Rate-Limit"))
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code in (401, 403):
                raise RiotAuthError(
                    f"HTTP {r.status_code} from Riot API: the key is invalid or expired "
                    "(development keys expire every 24h - regenerate it at "
                    "https://developer.riotgames.com and rerun; the crawl resumes)."
                )
            if r.status_code == 429:
                try:
                    retry_after = float(r.headers.get("Retry-After", 10))
                except ValueError:
                    retry_after = 10.0
                limiter.block_for(retry_after + 1.0)
                last_error = "429 rate limited"
                continue
            if r.status_code >= 500:
                last_error = f"HTTP {r.status_code}"
                time.sleep(min(60, 2 ** attempt))
                continue
            raise RiotAPIError(f"HTTP {r.status_code} for {url}: {r.text[:200]}")
        raise RiotAPIError(f"giving up on {url} after {self.max_retries} attempts ({last_error})")

    # ---- league-v4 (platform routed) -------------------------------------------------

    def apex_league(self, platform: str, tier: str) -> list[dict]:
        """Challenger / Grandmaster / Master ladder entries."""
        data = self.get(platform, f"/lol/league/v4/{tier}leagues/by-queue/{RANKED_SOLO}")
        return (data or {}).get("entries", [])

    def diamond_page(self, platform: str, division: str, page: int) -> list[dict]:
        data = self.get(
            platform, f"/lol/league/v4/entries/{RANKED_SOLO}/DIAMOND/{division}", {"page": page}
        )
        return data or []

    def puuid_from_summoner_id(self, platform: str, summoner_id: str) -> str | None:
        """Fallback for league entries that carry no puuid (older API responses)."""
        data = self.get(platform, f"/lol/summoner/v4/summoners/{summoner_id}")
        return (data or {}).get("puuid")

    # ---- match-v5 (region routed) ----------------------------------------------------

    def match_ids(self, region: str, puuid: str, start_time: int, start: int = 0,
                  count: int = 100) -> list[str]:
        """Ranked solo match ids, newest first; `start` pages further into the past."""
        data = self.get(
            region,
            f"/lol/match/v5/matches/by-puuid/{puuid}/ids",
            {"queue": RANKED_SOLO_QUEUE_ID, "type": "ranked", "startTime": start_time,
             "start": start, "count": count},
        )
        return data or []

    def match(self, region: str, match_id: str) -> dict | None:
        return self.get(region, f"/lol/match/v5/matches/{match_id}")


def patch_of(game_version: str) -> str:
    """'16.22.725.1234' -> '16.22'."""
    parts = game_version.split(".")
    if len(parts) < 2 or not (parts[0].isdigit() and parts[1].isdigit()):
        raise ValueError(f"unexpected gameVersion {game_version!r}")
    return f"{int(parts[0])}.{int(parts[1])}"


# Per-player columns of matches.pstats (one list per player, same order as blue + red).
# Indices 9-15 are known before the game starts; everything else is an outcome and may only
# reach the model through *earlier* games of the same player.
PSTAT_FIELDS = (
    "kills", "deaths", "assists", "gold", "damage", "damage_taken", "cs", "vision",
    "champ_level",
    "summoner_level", "spell1", "spell2", "keystone", "primary_style", "sub_style",
    "role_assignment",      # 1 primary, 2 secondary, 3 autofill / other, 0 unknown
    "objective_damage", "cc_time", "time_dead", "solo_kills", "cs_at_10",
    "max_cs_lead", "max_level_lead", "lane_lead", "early_lane_lead", "plates",
    "afk", "first_blood", "control_wards", "healing_shielding", "mitigated",
)
PSTAT = {name: i for i, name in enumerate(PSTAT_FIELDS)}


def _player_stats(p: dict) -> list:
    ch = p.get("challenges") or {}

    def num(source: dict, key: str) -> int:
        try:
            return int(round(float(source.get(key) or 0)))
        except (TypeError, ValueError):
            return 0

    styles = (p.get("perks") or {}).get("styles") or []
    primary = styles[0] if styles else {}
    selections = primary.get("selections") or [{}]
    assignment = str(p.get("selectedRolePreferences") or "").split(".")
    kind = assignment[1] if len(assignment) > 1 else ""
    return [
        num(p, "kills"), num(p, "deaths"), num(p, "assists"), num(p, "goldEarned"),
        num(p, "totalDamageDealtToChampions"), num(p, "totalDamageTaken"),
        num(p, "totalMinionsKilled") + num(p, "neutralMinionsKilled"), num(p, "visionScore"),
        num(p, "champLevel"),
        num(p, "summonerLevel"), num(p, "summoner1Id"), num(p, "summoner2Id"),
        num(selections[0], "perk"), num(primary, "style"),
        num(styles[1], "style") if len(styles) > 1 else 0,
        {"PRIMARY": 1, "SECONDARY": 2}.get(kind, 3 if kind else 0),
        num(p, "damageDealtToObjectives"), num(p, "timeCCingOthers"),
        num(p, "totalTimeSpentDead"), num(ch, "soloKills"),
        num(ch, "laneMinionsFirst10Minutes") + num(ch, "jungleCsBefore10Minutes"),
        num(ch, "maxCsAdvantageOnLaneOpponent"), num(ch, "maxLevelLeadLaneOpponent"),
        num(ch, "laningPhaseGoldExpAdvantage"), num(ch, "earlyLaningPhaseGoldExpAdvantage"),
        num(ch, "turretPlatesTaken"), int(bool(p.get("wasAfk"))),
        int(bool(p.get("firstBloodKill") or p.get("firstBloodAssist"))),
        num(ch, "controlWardsPlaced"),
        num(p, "totalHealsOnTeammates") + num(p, "totalDamageShieldedOnTeammates"),
        num(p, "damageSelfMitigated"),
    ]


def _extras(info: dict, ordered: dict) -> dict:
    """Optional columns: player ids, bans, end-of-game team totals and per-player statistics.

    The totals are outcomes, so they may only ever be used as training targets, never as
    model inputs. Anything missing is stored as '' / NULL rather than failing the match."""
    puuids = {tid: [str(p.get("puuid") or "") for p in members] for tid, members in ordered.items()}
    have_ids = all(all(ids) for ids in puuids.values())
    team_info = {t.get("teamId"): t for t in info.get("teams", [])}

    bans = []
    for tid in (100, 200):
        picked = [int(b.get("championId", -1)) for b in team_info.get(tid, {}).get("bans", [])]
        bans += (picked + [-1] * 5)[:5]

    def objective(tid: int, name: str) -> int:
        return int(team_info.get(tid, {}).get("objectives", {}).get(name, {}).get("kills", 0))

    def total(tid: int, key: str) -> int:
        return int(sum(p.get(key, 0) or 0 for p in ordered[tid]))

    stats = {
        "gold": [total(100, "goldEarned"), total(200, "goldEarned")],
        "kills": [total(100, "kills"), total(200, "kills")],
        "towers": [objective(100, "tower"), objective(200, "tower")],
        "dragons": [objective(100, "dragon"), objective(200, "dragon")],
        "barons": [objective(100, "baron"), objective(200, "baron")],
        "inhibitors": [objective(100, "inhibitor"), objective(200, "inhibitor")],
        "surrender": int(any(p.get("gameEndedInSurrender") for p in ordered[100] + ordered[200])),
    }
    return {
        "blue_puuids": ",".join(puuids[100]) if have_ids else "",
        "red_puuids": ",".join(puuids[200]) if have_ids else "",
        "bans": ",".join(map(str, bans)),
        "stats": json.dumps(stats, separators=(",", ":")) if stats["gold"][0] > 0 else None,
        "pstats": json.dumps([_player_stats(p) for p in ordered[100] + ordered[200]],
                             separators=(",", ":")) if stats["gold"][0] > 0 else "",
    }


def parse_match(match: dict) -> tuple[dict | None, str]:
    """Turn a match-v5 payload into a row for the matches table.

    Returns (row, "") for a usable game, or (None, reason) if it must be skipped.
    Only pre-game information (champions, roles, side) and the result are kept.
    """
    try:
        info = match["info"]
        match_id = match["metadata"]["matchId"]
        if info.get("queueId") != RANKED_SOLO_QUEUE_ID:
            return None, "queue"
        parts = info["participants"]
        if len(parts) != 10:
            return None, "participants"
        if info.get("gameDuration", 0) < MIN_DURATION_S or any(
            p.get("gameEndedInEarlySurrender") for p in parts
        ):
            return None, "remake"
        teams = {100: [], 200: []}
        for p in parts:
            teams[p["teamId"]].append(p)
        if len(teams[100]) != 5 or len(teams[200]) != 5:
            return None, "teams"
        if len({p["championId"] for p in parts}) != 10 or any(p["championId"] <= 0 for p in parts):
            return None, "champions"
        wins = {t["teamId"]: bool(t["win"]) for t in info["teams"]}
        if wins[100] == wins[200]:
            return None, "result"

        roles_ok = True
        ordered = {}
        for tid, members in teams.items():
            by_role = {p.get("teamPosition"): p for p in members}
            if len(by_role) == 5 and all(r in by_role for r in ROLES):
                ordered[tid] = [by_role[r] for r in ROLES]
            else:
                roles_ok = False
        if not roles_ok:
            ordered = dict(teams)

        row = {
            "match_id": match_id,
            "platform": str(info.get("platformId", match_id.split("_")[0])).lower(),
            "game_version": info["gameVersion"],
            "patch": patch_of(info["gameVersion"]),
            "game_creation": int(info["gameCreation"]),
            "duration": int(info["gameDuration"]),
            "blue": ",".join(str(p["championId"]) for p in ordered[100]),
            "red": ",".join(str(p["championId"]) for p in ordered[200]),
            "roles_ok": int(roles_ok),
            "blue_win": int(wins[100]),
            **_extras(info, ordered),
        }
        return row, ""
    except (KeyError, TypeError, ValueError, IndexError) as e:
        return None, f"malformed:{type(e).__name__}"


# ---- Data Dragon (static data, no key needed) ---------------------------------------

DDRAGON = "https://ddragon.leagueoflegends.com"


def ddragon_versions() -> list[str]:
    r = requests.get(f"{DDRAGON}/api/versions.json", timeout=20)
    r.raise_for_status()
    return r.json()


def ddragon_champions(version: str) -> dict[int, dict]:
    """{championId: {"id": "MonkeyKing", "name": "Wukong"}}"""
    r = requests.get(f"{DDRAGON}/cdn/{version}/data/en_US/champion.json", timeout=20)
    r.raise_for_status()
    return {
        int(c["key"]): {"id": c["id"], "name": c["name"]} for c in r.json()["data"].values()
    }
