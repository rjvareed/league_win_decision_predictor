import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from lolwp.app import Assistant, BadRequest, make_handler
from lolwp.live import Player

from test_live import rows_of


@pytest.fixture(scope="module")
def assistant(trained, tmp_path_factory):
    db_path, champs, out, _, _ = trained
    d = tmp_path_factory.mktemp("app")
    return Assistant(str(out), str(db_path), str(champs), samples=64,
                     env_file=str(d / "no.env"), cache=str(d / "cache.sqlite"))


def test_meta_lists_champions_with_roles_and_ban_rates(assistant, monkeypatch):
    monkeypatch.delenv("RIOT_API_KEY", raising=False)
    meta = assistant.meta()
    assert meta["patches"] == ["99.2", "99.1"] and meta["ddragon"] == "synthetic"
    assert len(meta["champions"]) == 60 and meta["player_lookup"] is False
    c = meta["champions"][0]
    assert c["id"] == 1 and c["name"] == "Champ 001" and c["roles"] == [1.0, 0, 0, 0, 0]
    bans = [c["ban"] for c in meta["champions"]]
    assert 0 <= min(bans) < 0.1 < max(bans) < 1   # the synthetic world bans strong champions more
    assert sum(bans) == pytest.approx(10, abs=0.5)


def test_advise(assistant):
    req = {"side": "red", "role": 2, "ally": {"0": 1, "2": 3},
           "enemy": [{"id": 8, "role": None}, {"id": 9, "role": 3}], "bans": [13, 18]}
    out = assistant.advise(req)
    assert out["enemy"] == {"2": 8, "3": 9} and out["inferred"] == [2]
    assert out["current"] == 3 and not out["personal"] and not out["state"]["complete"]
    assert 0 < out["state"]["win"] < 1
    names = [o["champion_id"] for o in out["picks"]]
    assert 3 in names and not {1, 8, 9, 13, 18} & set(names)          # own pick is still ranked
    assert all(o["champion_id"] % 5 == 3 for o in out["picks"])        # mid champions only
    assert out["bans"] and not {1, 3, 8, 9, 13, 18} & {o["champion_id"] for o in out["bans"]}
    assert all(0 < o["ban_rate"] < 1 for o in out["bans"])
    # the two halves the page asks for separately
    assert "bans" not in assistant.advise({**req, "want": "picks"})
    assert set(assistant.advise({**req, "want": "bans"})) == {"enemy", "inferred", "personal", "bans"}
    assert [o["champion_id"] for o in assistant.advise({**req, "pool": [23, 28], "want": "picks"})["picks"]
            ] in ([23, 28], [28, 23])
    full = {"side": "blue", "role": 0, "ally": {str(r): r + 1 for r in range(5)},
            "enemy": [{"id": r + 6, "role": r} for r in range(5)]}
    out = assistant.advise(full)
    assert out["state"]["complete"] and out["bans"] == []
    assert out["state"]["win"] == pytest.approx(
        assistant.predictor.predict([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]), abs=1e-5)
    for bad in ({"side": "green", "role": 2}, {"side": "blue", "role": 7},
                {"side": "blue", "role": 2, "ally": {"0": 999}},
                {"side": "blue", "role": 2, "ally": {"0": 1}, "enemy": [{"id": 1, "role": 0}]},
                {"side": "blue", "role": 2, "player": {"riot_id": "x#y", "platform": "kr"}}):
        with pytest.raises(BadRequest):
            assistant.advise(bad)


def test_advise_for_a_known_player(assistant, trained):
    me = Player("player12", "Twelve", rows_of(trained[0], "player12"))
    assistant._players[("twelve#syn", "kr")] = (float("inf"), me)
    spec = {"riot_id": "Twelve#SYN", "platform": "kr"}
    out = assistant.advise({"side": "blue", "role": 2, "player": spec, "want": "picks"})
    assert out["personal"] and all(o["my_games"] is not None for o in out["picks"])
    summary = assistant.player_summary(me)
    assert summary["games"] == me.games and summary["champions"][0]["games"] >= summary["champions"][-1]["games"]


def test_http_round_trip(assistant):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(assistant))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path, body, headers=None):
        req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        with urllib.request.urlopen(base + "/") as r:
            assert b"Champion select assistant" in r.read()
        with urllib.request.urlopen(base + "/api/meta") as r:
            assert len(json.loads(r.read())["champions"]) == 60
        status, out = post("/api/advise", {"side": "blue", "role": 1, "want": "picks"})
        assert status == 200 and len(out["picks"]) == 12
        assert post("/api/advise", {"side": "blue"})[0] == 400
        assert post("/api/advise", {"side": "blue", "role": 1},
                    {"Origin": "http://elsewhere.example"}) == (400, {"error": "cross-origin request refused"})
        assert post("/api/player", {"riot_id": "a#b", "platform": "kr"})[0] == 400   # no API key
        assert post("/api/nothing", {})[0] == 404
    finally:
        server.shutdown()
        server.server_close()
