import sqlite3

import numpy as np
import pytest

from lolwp.data import PLAYER_FEATURES, featurize
from lolwp.draft import DraftAdvisor
from lolwp.live import ROW_COLUMNS, Player, PlayerLookup
from lolwp.predict import Predictor
from lolwp.riot import parse_match

from test_riot import fake_match

F = dict(zip(PLAYER_FEATURES, range(len(PLAYER_FEATURES))))


def rows_of(db_path, puuid, limit=None):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        f"SELECT {', '.join(ROW_COLUMNS)} FROM matches WHERE blue_puuids LIKE ? OR red_puuids "
        "LIKE ? ORDER BY game_creation", (f"%{puuid},%", f"%{puuid},%")).fetchall()
    conn.close()
    rows = [r for r in rows if puuid in (r[7] + "," + r[8]).split(",")]
    return rows[:limit] if limit else rows


def test_player_record_and_features_match_the_training_pipeline(trained):
    rows = rows_of(trained[0], "player7", limit=40)
    assert len(rows) == 40
    history, game = rows[:-1], rows[-1]
    me = Player("player7", "Seven", history)
    assert me.games == 39 and sum(r.games for r in me.records.values()) == 39
    assert me.wins == sum(r.wins for r in me.records.values())
    assert sum(me.role_games) <= 39 and max(me.role_games) == me.role_games[7 % 5]

    # the game that really came next: its inputs, as the training pipeline computes them
    players = game[7].split(",") + game[8].split(",")
    champions = [int(c) for c in (game[0] + "," + game[1]).split(",")]
    slot = players.index("player7")
    flag = np.r_[np.zeros(39, dtype=bool), True]
    reference = featurize(rows, flag)
    assert bool(game[2]), "pick a game with known roles for this test"
    live = me.features([champions[slot], 59], slot % 5, champions[(slot + 5) % 10],
                       now_ms=game[5])
    me.result_means = reference["result_means"].tolist()
    assert live.shape == (2, len(PLAYER_FEATURES))
    ref = reference["feats"][0, slot]
    unknown_live = [i for n, i in F.items() if n == "duo" or n in (
        "pregame_known", "account_level", "role_primary", "role_secondary", "role_autofill")
        or n.startswith(("spell_", "keystone_", "adjusted_"))]
    same = np.setdiff1d(np.arange(len(ref)), unknown_live)
    assert live[0, same] == pytest.approx(ref[same], abs=1e-5)
    assert live[0, F["pregame_known"]] == 0          # runes and spells are not chosen yet
    # another champion changes the champion-specific inputs and nothing else
    changed = {n for n, i in F.items() if abs(live[0, i] - live[1, i]) > 1e-7}
    assert changed and all(n.startswith(("champion_", "matchup_")) for n in changed)
    # hypothetical games never enter the record
    assert me.games == 39 and len(me.rows) == 39
    assert me.features([], 0).shape == (0, len(PLAYER_FEATURES))
    assert Player("nobody", "x", []).features([1, 2], 0).shape == (2, len(PLAYER_FEATURES))


def test_personal_picks_prefer_the_players_own_champions(trained):
    db_path, _, out, _, _ = trained
    advisor = DraftAdvisor(Predictor.load(str(out)), samples=96, min_role_share=0.0001)
    me = Player("player12", "Twelve", rows_of(db_path, "player12"))       # role 12 % 5 = mid
    role = 12 % 5
    anonymous = advisor.recommend_picks("blue", role, {}, {}, top=100, counter_check=0)
    personal = advisor.recommend_picks("blue", role, {}, {}, top=100, counter_check=2, player=me)
    assert {o.champion for o in anonymous} == {o.champion for o in personal}
    assert all(o.draft_win is None and o.my_games is None for o in anonymous)
    by_name = {o.champion: o for o in anonymous}
    for o in personal:
        assert o.draft_win == pytest.approx(by_name[o.champion].win, abs=1e-6)
        rec = me.records.get(o.champion_id)
        assert (o.my_games, o.my_wins) == ((rec.games, rec.wins) if rec else (0, 0))
    # the synthetic player has three main champions with most of their games: the model was
    # trained to value experience, so they gain the most from the personal view
    gain = {o.champion: o.win - o.draft_win for o in personal}
    mains = sorted(personal, key=lambda o: -o.my_games)[:3]
    rest = [g for name, g in gain.items() if name not in {m.champion for m in mains}]
    assert min(gain[m.champion] for m in mains) > np.mean(rest)
    assert personal[0].counter and personal[2].counter is None
    # the draft as it stands, with and without knowing who plays it
    ally = {role: advisor.index(mains[0].champion)}
    plain, _ = advisor.evaluate("blue", ally, {})
    mine, _ = advisor.evaluate("blue", ally, {}, player=me, role=role)
    assert 0 < plain < 1 and 0 < mine < 1 and mine != pytest.approx(plain, abs=1e-6)


class FakeClient:
    def __init__(self, ids):
        self.ids, self.calls = ids, []

    def get(self, host, path, params=None):
        self.calls.append(("get", host, path))
        if path.lower().endswith("/some%20one/euw"):     # Riot IDs are not case sensitive
            return {"puuid": "KR_1-p2", "gameName": "Some One", "tagLine": "EUW"}
        return None

    def match_ids(self, region, puuid, start_time, start=0, count=100):
        self.calls.append(("ids", region, puuid, count))
        return self.ids[:count]

    def match(self, region, match_id):
        self.calls.append(("match", region, match_id))
        if match_id == "KR_404":
            return None
        if match_id == "KR_9":
            return fake_match(match_id, queue=450)        # not ranked solo: unusable
        payload = fake_match("KR_1")                      # same ten players in every game
        payload["metadata"]["matchId"] = match_id
        payload["info"]["gameCreation"] += int(match_id.split("_")[1]) * 3_600_000
        return payload


def test_lookup_fetches_each_game_once_and_reuses_the_crawl_database(tmp_path):
    from lolwp import db
    main = tmp_path / "matches.sqlite"
    conn = db.connect(main)
    row, _ = parse_match(FakeClient([]).match("asia", "KR_3"))
    conn.execute(f"INSERT INTO matches({', '.join(row)}) VALUES({', '.join('?' * len(row))})",
                 list(row.values()))
    conn.close()

    client = FakeClient(["KR_5", "KR_404", "KR_9", "KR_3", "KR_2"])
    lookup = PlayerLookup(client, tmp_path / "cache.sqlite", main)
    with pytest.raises(ValueError, match="Name#TAG"):
        lookup.resolve("no tag", "kr")
    with pytest.raises(ValueError, match="unknown platform"):
        lookup.resolve("a#b", "moon")
    with pytest.raises(ValueError, match="no account"):
        lookup.resolve("Ghost#000", "kr")
    seen = []
    me = lookup.player("some one#euw", "kr", progress=lambda k, n: seen.append((k, n)))
    assert (me.puuid, me.name) == ("KR_1-p2", "Some One#EUW")
    assert [r[5] for r in me.rows] == sorted(r[5] for r in me.rows) and me.games == 3
    fetched = [c[2] for c in client.calls if c[0] == "match"]
    assert sorted(fetched) == ["KR_2", "KR_404", "KR_5", "KR_9"]      # KR_3 came from the database
    assert seen == [(0, 4), (1, 4), (2, 4), (3, 4)]
    assert me.records[3].games == 3 and me.role_games == [0, 0, 3, 0, 0]

    # second lookup: account, match list and all games (also the unusable ones) are cached
    before = len(client.calls)
    again = PlayerLookup(client, tmp_path / "cache.sqlite", main).player("Some One#EUW", "kr")
    assert len(client.calls) == before and again.games == 3
    assert again.features([3, 4], 2).shape == (2, len(PLAYER_FEATURES))
