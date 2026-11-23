"""Champion-select assistant in the browser.

    lolwp-app                      # then open http://127.0.0.1:8765

Enter the draft as it happens (bans, your team, the enemy picks you can see) and the page
shows, for the model trained by lolwp-train:

  * the win chance of the draft as it stands,
  * the best picks for your role, optionally restricted to the champions you play,
  * the best bans,
  * and, with your Riot ID, picks ranked with your own recent record on every champion
    (needs RIOT_API_KEY in .env; see lolwp.live).

Everything runs on this machine; only champion icons (Data Dragon) and, if you load your
history, Riot API requests leave it. The server listens on localhost only unless --host is
given.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from .draft import ROLE_NAMES, DraftAdvisor
from .predict import LOW_DATA_GAMES, Predictor

STATIC = Path(__file__).parent / "static"
PLAYER_TTL_S = 300
MAX_BODY = 64 * 1024


class BadRequest(ValueError):
    pass


class Assistant:
    """The model plus everything the page asks for. One instance serves all requests."""

    def __init__(self, model: str = "models/draft_model.pt", db: str = "data/matches.sqlite",
                 champions: str = "data/champions.json", device: str = "cpu",
                 samples: int | None = None,
                 env_file: str = ".env", cache: str = "data/live_cache.sqlite"):
        self.predictor = Predictor.load(model, device=device)
        self.advisor = DraftAdvisor(self.predictor, db_path=db, samples=samples)
        self.lock = threading.Lock()          # one model evaluation at a time
        self.fetch_lock = threading.Lock()    # one player lookup at a time (shared cache file)
        self.db, self.env_file, self.cache = db, env_file, cache
        self._lookup = None
        self._players: dict[tuple[str, str], tuple[float, object]] = {}
        self.ddragon = None
        try:
            self.ddragon = json.loads(Path(champions).read_text()).get("version")
        except (OSError, ValueError):
            pass

    # ---- player history ------------------------------------------------------------------

    def lookup(self):
        if self._lookup is None:
            from .live import PlayerLookup
            self._lookup = PlayerLookup.open(self.env_file, cache_path=self.cache, db_path=self.db)
        return self._lookup

    def can_lookup(self) -> bool:
        from .riot import RiotAuthError, load_api_key
        try:
            load_api_key(self.env_file)
        except RiotAuthError:
            return False
        return self.predictor.player_calibration is not None

    def player(self, spec: dict | None, fetch: bool = False):
        """The lolwp.live.Player for {"riot_id", "platform"}; None if spec is empty. Unless
        `fetch`, only players loaded before (POST /api/player) are returned."""
        if not spec or not spec.get("riot_id"):
            return None
        key = (str(spec["riot_id"]).strip().lower(), str(spec.get("platform", "")).lower())
        hit = self._players.get(key)
        if hit and time.time() - hit[0] < PLAYER_TTL_S:
            return hit[1]
        if not fetch and not hit:
            raise BadRequest("load your match history first")
        from .riot import RiotAPIError, RiotAuthError
        try:
            with self.fetch_lock:
                player = self.lookup().player(str(spec["riot_id"]), key[1])
        except (RiotAuthError, RiotAPIError) as e:
            if hit:                 # keep advising with the older copy
                return hit[1]
            raise BadRequest(str(e)) from None
        except ValueError as e:
            raise BadRequest(str(e)) from None
        self._players[key] = (time.time(), player)
        return player

    def player_summary(self, player) -> dict:
        index = self.predictor.index
        return {
            "name": player.name, "games": player.games, "wins": player.wins,
            "roles": player.role_games,
            "champions": sorted(
                ({"id": c, "games": r.games, "wins": r.wins}
                 for c, r in player.records.items() if c in index),
                key=lambda x: -x["games"]),
        }

    # ---- page data -----------------------------------------------------------------------

    def meta(self) -> dict:
        p, a = self.predictor, self.advisor
        champions = []
        for cid, idx in p.index.items():
            games = a.role_games[idx]
            total = games.sum()
            if cid not in p.names:
                continue
            champions.append({
                "id": cid, "name": p.names[cid], "key": p.ckpt["champion_keys"].get(cid, ""),
                "roles": [round(float(g / total), 3) if total else 0.0 for g in games],
                "pick": [round(float(x), 4) for x in a.role_share[idx]],
                "ban": round(float(a.ban_rate[idx]), 3),
                "few": bool(p.games.get(cid, 0) < LOW_DATA_GAMES),
            })
        champions.sort(key=lambda c: c["name"])
        from .riot import PLATFORM_TO_REGION
        return {
            "patches": p.ckpt["patches"], "trained_at": p.ckpt.get("trained_at"),
            "matches": p.ckpt.get("n_matches", {}).get("train"),
            "ddragon": self.ddragon, "roles": list(ROLE_NAMES), "champions": champions,
            "player_lookup": self.can_lookup(), "platforms": sorted(PLATFORM_TO_REGION),
        }

    def _index(self, cid) -> int:
        try:
            return self.predictor.index[int(cid)]
        except (KeyError, TypeError, ValueError):
            raise BadRequest(f"unknown champion id {cid!r}") from None

    def advise(self, req: dict) -> dict:
        a = self.advisor
        side = req.get("side")
        if side not in ("blue", "red"):
            raise BadRequest("side must be blue or red")
        try:
            role = int(req.get("role"))
            assert 0 <= role < 5
        except (TypeError, ValueError, AssertionError):
            raise BadRequest("role must be 0..4") from None
        ally: dict[int, int] = {}
        for r, cid in (req.get("ally") or {}).items():
            if cid:
                ally[int(r)] = self._index(cid)
        fixed, loose = {}, []
        for e in req.get("enemy") or []:
            if not e or not e.get("id"):
                continue
            if e.get("role") is None:
                loose.append(self._index(e["id"]))
            else:
                fixed[int(e["role"])] = self._index(e["id"])
        banned = {self._index(c) for c in req.get("bans") or [] if c}
        pool = req.get("pool")
        pool = [self._index(c) for c in pool] if pool else None
        top = max(1, min(int(req.get("top") or 12), 40))
        want = req.get("want") or "all"
        if want not in ("all", "picks", "bans"):
            raise BadRequest("want must be all, picks or bans")
        try:
            player = self.player(req.get("player")) if a.p.player_calibration else None
            enemy = a.infer_roles(loose, fixed)
            with self.lock:
                out = {
                    "enemy": {str(r): a.id_of[c] for r, c in enemy.items()},
                    "inferred": sorted(set(enemy) - set(fixed)),
                    "personal": player is not None,
                }
                if want in ("all", "picks"):
                    win, spread = a.evaluate(side, ally, enemy, banned, player, role)
                    out["state"] = {"win": win, "spread": spread,
                                    "complete": len(ally) == 5 and len(enemy) == 5,
                                    # your history only counts once your champion is known
                                    "personal": player is not None and role in ally}
                    others = {r: c for r, c in ally.items() if r != role}
                    picks = a.recommend_picks(side, role, others, enemy, banned, pool, top,
                                              counter_check=6, player=player)
                    out["picks"] = [vars(o) for o in picks]
                    out["current"] = a.id_of[ally[role]] if role in ally else None
                if want in ("all", "bans"):
                    sort = "net" if req.get("sort") == "net" else "gain"
                    out["bans"] = [vars(o) for o in a.recommend_bans(
                        side, ally, enemy, banned, top, sort, with_net=sort == "net")
                    ] if len(enemy) < 5 else []
        except BadRequest:
            raise
        except ValueError as e:
            raise BadRequest(str(e)) from None
        return out


def make_handler(assistant: Assistant):
    class Handler(BaseHTTPRequestHandler):
        server_version = "lolwp"

        def log_message(self, fmt, *args):   # quiet: one line per request is noise here
            pass

        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload) -> None:
            self._send(code, json.dumps(payload).encode(), "application/json")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/meta":
                self._json(200, assistant.meta())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 < length <= MAX_BODY:
                    raise BadRequest("bad request size")
                # a page on another site must not be able to drive this server
                origin = self.headers.get("Origin")
                if origin and origin.split("//", 1)[-1] != self.headers.get("Host"):
                    raise BadRequest("cross-origin request refused")
                try:
                    req = json.loads(self.rfile.read(length))
                except ValueError:
                    raise BadRequest("body must be JSON") from None
                if not isinstance(req, dict):
                    raise BadRequest("body must be a JSON object")
                if path == "/api/advise":
                    self._json(200, assistant.advise(req))
                elif path == "/api/player":
                    if not assistant.can_lookup():
                        raise BadRequest("player lookup needs RIOT_API_KEY in .env and a model "
                                         "trained with player features")
                    player = assistant.player(req, fetch=True)
                    if player is None:
                        raise BadRequest("give a Riot ID as Name#TAG")
                    self._json(200, assistant.player_summary(player))
                else:
                    self._json(404, {"error": "not found"})
            except BadRequest as e:
                self._json(400, {"error": str(e)})
            except Exception as e:   # noqa: BLE001 - report instead of dropping the connection
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/draft_model.pt")
    ap.add_argument("--db", default="data/matches.sqlite",
                    help="optional: saves API requests for games the crawl already has")
    ap.add_argument("--champions", default="data/champions.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--samples", type=int, default=None,
                    help="sampled completions per option (default 256 on a GPU, 96 on a CPU)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    if not Path(args.model).is_file():
        raise SystemExit(f"{args.model} not found - run lolwp-train first")
    assistant = Assistant(args.model, args.db, args.champions, args.device, args.samples)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(assistant))
    print(f"champion select assistant: http://{args.host}:{args.port}  "
          f"(model: patch {', '.join(assistant.predictor.ckpt['patches'])}; Ctrl-C to stop)",
          flush=True)
    if not assistant.can_lookup():
        print("note: no RIOT_API_KEY in .env (or a draft-only model) - personal pick ranking is off")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
