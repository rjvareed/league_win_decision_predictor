import json
import math

import numpy as np
import pytest
import torch

from lolwp import db
from lolwp.baselines import fit_logistic, one_hot_features, player_team_features
from lolwp.data import (HISTORY_FEATURES, PREGAME_FEATURES, RECENT_WINDOW, STAT_FEATURES, PLAYER_FEATURES, PlayerHistory, aux_targets,
                        load_draft_data, match_performance, stat_features)
from lolwp.riot import PSTAT, PSTAT_FIELDS
from lolwp.model import DraftModel, swap_sides
from lolwp.synth import COLUMNS, make_synthetic
from lolwp.train import DEFAULT_HP, resolve_hp
from lolwp.tune import main as tune_main, sample_config

F = len(PLAYER_FEATURES)


def insert(conn, i, blue_ids, red_ids, blue_win, t, patch="16.22", roles_ok=1, stats=None,
           pstats=None):
    conn.execute(
        f"INSERT INTO matches({COLUMNS}) VALUES({','.join('?' * 16)})",
        (f"M{i}", "kr", patch + ".1.1", patch, t, 1750, "1,2,3,4,5", "6,7,8,9,10", roles_ok,
         blue_win, "MASTER", blue_ids, red_ids, None, stats, pstats))


def player_table(**columns):
    """pstats JSON for one match: every keyword is a list of 10 values (or one for all)."""
    table = np.zeros((10, len(PSTAT_FIELDS)), dtype=np.int64)
    for name, values in columns.items():
        table[:, PSTAT[name]] = values
    return json.dumps(table.tolist())


def test_player_history_is_strictly_past(tmp_path):
    """Features of a game only reflect games that started before it, and never its own result."""
    path = tmp_path / "h.sqlite"
    conn = db.connect(path)
    a = ",".join(f"a{i}" for i in range(5))
    b = ",".join(f"b{i}" for i in range(5))
    hour = 3_600_000
    insert(conn, 0, a, b, 1, 10 * hour, patch="16.21")   # a0..a4 win on blue (older patch)
    insert(conn, 1, a, b, 1, 12 * hour)                  # win again
    insert(conn, 2, b, a, 1, 13 * hour)                  # sides swapped: b wins, a loses
    insert(conn, 3, None, None, 0, 14 * hour)            # no player ids stored
    insert(conn, 4, a, b, 0, 20 * hour, roles_ok=0)
    conn.close()
    data = load_draft_data(path, patches=["16.22"])
    assert len(data) == 4 and data.feats.shape == (4, 10, F)
    f = dict(zip(PLAYER_FEATURES, range(F)))

    # game 1 (row 0): one earlier game, taken from the patch that is not trained on
    row = data.feats[0]
    assert data.known[0].all()
    assert row[0, f["total_games"]] == pytest.approx(math.log1p(1) / 4)
    assert row[0, f["total_winrate"]] == pytest.approx(0.5 / 9)       # blue player: 1 win
    assert row[5, f["total_winrate"]] == pytest.approx(-0.5 / 9)      # red player: 1 loss
    assert row[0, f["champion_games"]] == pytest.approx(math.log1p(1) / 3)
    assert row[0, f["role_share"]] == 1.0 and row[0, f["streak"]] == pytest.approx(0.2)
    assert row[0, f["hours_since_last"]] == pytest.approx(math.log1p(2) / 5)

    # game 2 (row 1): a0 now plays red top on champion 6 for the first time, after 2 wins
    a0 = data.feats[1, 5]
    assert a0[f["total_winrate"]] == pytest.approx(1 / 10) and a0[f["champion_games"]] == 0
    assert a0[f["streak"]] == pytest.approx(0.4) and a0[f["role_share"]] == 1.0

    # game 3 (row 2): no ids -> nothing known, all zeros
    assert not data.known[2].any() and not data.feats[2].any()

    # game 4 (row 3): a0 has 2 wins then 1 loss; roles unknown -> role share 0
    a0 = data.feats[3, 0]
    assert a0[f["total_games"]] == pytest.approx(math.log1p(3) / 4)
    assert a0[f["total_winrate"]] == pytest.approx(0.5 / 11)
    assert a0[f["streak"]] == pytest.approx(-0.2) and a0[f["role_share"]] == 0
    assert a0[f["champion_games"]] == pytest.approx(math.log1p(2) / 3)   # champion 1 twice
    # newer features: results W, W, L -> exponential average; three games in the last 24 h,
    # the last one 7 h ago; three games with each teammate; two champions (1 and 6) played
    assert a0[f["recent_winrate"]] == pytest.approx(0.9 * (0.9 * 0.05 + 0.05) - 0.05)
    assert a0[f["games_24h"]] == pytest.approx(math.log1p(3) / 3) and a0[f["games_3h"]] == 0
    assert a0[f["duo"]] == pytest.approx(math.log1p(3) / 2)
    assert a0[f["champion_pool"]] == pytest.approx(math.log1p(2) / 3)
    assert a0[f["champion_recency"]] == pytest.approx(math.exp(-8 / 72))  # champion 1: 8 h ago
    assert a0[f["gold_margin"]] == 0                                      # no stats stored

    # changing the result of the last game must not change any feature
    conn = db.connect(path)
    conn.execute("UPDATE matches SET blue_win = 1 WHERE match_id = 'M4'")
    conn.close()
    assert np.array_equal(load_draft_data(path, patches=["16.22"]).feats, data.feats)


def test_player_history_unseen_player():
    h = PlayerHistory()
    assert h.features("x", 1, 0, 0) == [0.0] * len(HISTORY_FEATURES)
    h.update("x", 1, -1, False, 1000)
    h.update("x", 1, 2, False, 2000)
    feats = h.features("x", 2, 2, 2000 + 3_600_000)
    assert feats[4] == 0.5 and feats[5] == pytest.approx(-0.4) and feats[0] == 0


def test_player_history_margins_and_duo():
    f = dict(zip(PLAYER_FEATURES, range(F)))
    h = PlayerHistory()
    h.update("x", 1, 2, True, 0, [1.0, 0.5, -0.5], 1.0)
    h.update("x", 2, 2, False, 1000, None, -1.0)           # a game without stored statistics
    h.add_team(["x", "y", "z"])
    h.add_team(["x", "y"])
    v = h.features("x", 1, 2, 2000, ["x", "z", "y", "stranger"])
    assert v[f["gold_margin"]] == pytest.approx(1 / 6) and v[f["tower_margin"]] == pytest.approx(-0.5 / 6)
    assert v[f["champion_gold_margin"]] == pytest.approx(1 / 4)
    assert v[f["role_winrate"]] == pytest.approx(0.0) and v[f["game_length"]] == 0
    assert v[f["duo"]] == pytest.approx(math.log1p(2) / 2)   # twice with y, once with z
    assert h.features("x", 1, 2, 2000)[f["duo"]] == 0        # teammates not given
    assert v[f["games_3h"]] == pytest.approx(math.log1p(2) / 2)


def test_stat_features_use_earlier_games_only(tmp_path):
    path = tmp_path / "s.sqlite"
    conn = db.connect(path)
    a = ",".join(f"a{i}" for i in range(5))
    b = ",".join(f"b{i}" for i in range(5))
    hour = 3_600_000
    rich = player_table(gold=[12_000] * 5 + [8_000] * 5, kills=[4] * 5 + [1] * 5, deaths=1,
                        role_assignment=[1] * 5 + [3] * 5, lane_lead=[1] * 5 + [0] * 5)
    insert(conn, 0, a, b, 1, 1 * hour, pstats=rich)
    insert(conn, 1, a, b, 1, 2 * hour)                       # no statistics for this game
    last = player_table(gold=[1] * 5 + [50_000] * 5, summoner_level=[30] * 5 + [500] * 5,
                        spell1=4, spell2=[14] * 5 + [11] * 5, keystone=8010,
                        role_assignment=[2] * 5 + [1] * 5)
    insert(conn, 2, a, b, 0, 3 * hour, pstats=last)
    conn.close()
    data = load_draft_data(path, patches=["16.22"])
    f = dict(zip(PLAYER_FEATURES, range(F)))
    assert data.feats.shape == (3, 10, F)

    first = data.feats[0]
    assert first[:, f["perf_gold"]].max() == 0 and first[:, f["perf_games"]].max() == 0
    assert first[0, f["pregame_known"]] == 1 and first[0, f["role_primary"]] == 1
    assert first[5, f["role_autofill"]] == 1
    assert not data.feats[1][:, f["pregame_known"]].any()    # nothing stored for game 2

    row = data.feats[2]   # history = game 0 only; its own statistics must not leak in
    gold = (12_000 - 8_000) / 20_001 / 0.1
    assert row[0, f["perf_gold"]] == pytest.approx(gold / 6, rel=1e-4)
    assert row[5, f["perf_gold"]] == pytest.approx(-gold / 6, rel=1e-4)
    assert row[0, f["perf_games"]] == pytest.approx(math.log1p(1) / 4)
    assert row[0, f["perf_kda"]] == pytest.approx(math.log(2.5) / 6, rel=1e-4)
    assert row[0, f["perf_lane_lead"]] == pytest.approx(1 / 6)
    assert row[0, f["champion_perf_gold"]] == pytest.approx(gold / 4, rel=1e-4)
    assert row[0, f["primary_role_rate"]] == pytest.approx(1 / 3) and row[0, f["autofill_rate"]] == 0
    assert row[5, f["autofill_rate"]] == pytest.approx(1 / 3)
    # ... while what is fixed before the game does come from the game itself
    assert row[0, f["account_level"]] == pytest.approx(math.log1p(30) / 7)
    assert row[0, f["role_secondary"]] == 1 and row[5, f["role_primary"]] == 1
    assert row[0, f["spell_4"]] == 1 and row[0, f["spell_14"]] == 1 and row[0, f["spell_11"]] == 0
    assert row[5, f["spell_11"]] == 1 and row[0, f["keystone_8010"]] == 1
    assert row[0, f["keystone_8005"]] == 0

    # outcome statistics of the last game are invisible to every feature
    conn = db.connect(path)
    conn.execute("UPDATE matches SET pstats = ? WHERE match_id = 'M2'", (player_table(
        gold=[50_000] * 5 + [1] * 5, kills=9, summoner_level=[30] * 5 + [500] * 5,
        spell1=4, spell2=[14] * 5 + [11] * 5, keystone=8010,
        role_assignment=[2] * 5 + [1] * 5),))
    conn.close()
    assert np.array_equal(load_draft_data(path, patches=["16.22"]).feats, data.feats)


def test_match_performance_is_relative_to_the_opponent():
    table = np.zeros((1, 10, len(PSTAT_FIELDS)), dtype=np.float32)
    table[0, :, PSTAT["gold"]] = [10, 20, 30, 40, 50, 30, 20, 10, 40, 50]
    perf = match_performance(table, np.array([True]), np.array([1800.0]))
    gold = perf[0, :, 0]
    assert gold[0] == pytest.approx(-20 / 41 / 0.1) and gold[1] == 0
    assert np.allclose(gold[:5], -gold[5:])                 # lane opponents mirror each other
    unknown = match_performance(table, np.array([False]), np.array([1800.0]))[0, :, 0]
    assert unknown[0] == pytest.approx((10 - 30) / 41 / 0.1)   # against the enemy average (30)
    # unknown players never receive history
    out = stat_features(table, np.array([True]), np.array([True]), np.array([1800.0]),
                        np.full((1, 10), -1), np.ones((1, 10), dtype=np.int64),
                        np.array([True]))
    assert not out[0, :, :len(STAT_FEATURES)].any()
    assert out.shape[-1] == len(STAT_FEATURES) + len(PREGAME_FEATURES)


def test_recent_role_and_matchup_statistics():
    """One player, 13 games with statistics; checks the windowed, in-role, result-adjusted
    and matchup columns of the 14th against values worked out by hand."""
    n = 14
    table = np.zeros((n, 10, len(PSTAT_FIELDS)), dtype=np.float32)
    gold = np.arange(1, n + 1, dtype=np.float32) * 1000       # slot 0 earns 1k, 2k, ...
    table[:, 0, PSTAT["gold"]] = gold
    table[:, 5, PSTAT["gold"]] = 5000                          # lane opponent: always 5k
    pid = np.full((n, 10), -1)
    pid[:, 0] = 0
    pid[:, 5] = 1
    champ = np.tile(np.arange(1, 11), (n, 1))
    champ[:3, 5] = 99                                          # a different opponent early on
    ok = np.ones(n, dtype=bool)
    ok[3] = False                                              # one game without roles
    win = np.arange(n) % 2 == 0
    out = stat_features(table, np.ones(n, dtype=bool), ok, np.full(n, 1800.0), pid, champ, win)
    f = {name: i for i, name in enumerate(STAT_FEATURES)}
    comparison = (gold - 5000) / (gold + 5001) / 0.1
    comparison[3] = (gold[3] - 1000) / (gold[3] + 1001) / 0.1  # vs the enemy average that game
    last = out[-1, 0]
    assert last[f["perf_gold"]] == pytest.approx(comparison[:13].sum() / 18, rel=1e-4)
    assert last[f["recent_gold"]] == pytest.approx(comparison[3:13].sum() / 13, rel=1e-4)
    assert last[f["recent_games"]] == 1.0 and out[4, 0, f["recent_games"]] == 4 / RECENT_WINDOW
    in_role = np.delete(comparison[:13], 3)
    assert last[f["role_perf_gold"]] == pytest.approx(in_role.sum() / 15, rel=1e-4)
    assert last[f["role_perf_games"]] == pytest.approx(math.log1p(12) / 4)
    # matchup: games 4..12 were against champion 6 with roles known
    assert last[f["matchup_games"]] == pytest.approx(math.log1p(9) / 2)
    wins = win[4:13].sum()
    assert last[f["matchup_winrate"]] == pytest.approx((wins - 4.5) / 12)
    assert last[f["matchup_gold"]] == pytest.approx(comparison[4:13].sum() / 12, rel=1e-4)
    # result-adjusted
    both = match_performance(table, ok, np.full(n, 1800.0))[:, [0, 5], 0]   # [n, 2 players]
    won = np.stack([win, ~win], axis=1)
    adjusted = comparison.copy()
    for g in range(n):   # minus the mean of earlier games with the same result (both players)
        earlier = both[:g][won[:g] == win[g]]
        adjusted[g] -= earlier.mean() if len(earlier) else 0.0
    assert last[f["adjusted_gold"]] == pytest.approx(adjusted[:13].sum() / 18, abs=1e-4)
    # the first game has no history at all
    assert not out[0, :, :len(STAT_FEATURES)].any()


def test_aux_targets():
    stats = json.dumps({"gold": [60_000, 50_000], "kills": [30, 19], "towers": [9, 3]})
    values, mask = aux_targets(stats, 1750 + 420)
    assert mask == [True] * 4
    assert values == pytest.approx([10 / 110 / 0.08, 11 / 50 / 0.25, 6 / 11 / 0.5, 1.0])
    values, mask = aux_targets(None, 1750)
    assert mask == [False, False, False, True] and values == [0, 0, 0, 0]


def small_model(**kw):
    torch.manual_seed(0)
    return DraftModel(30, d_model=16, n_layers=1, n_heads=2, d_ff=32, **kw).eval()


def drafts(batch=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    champs = torch.stack([torch.randperm(30, generator=g)[:10] + 1 for _ in range(batch)])
    roles = torch.tensor([[1, 2, 3, 4, 5] * 2] * batch)
    roles[0] = 0  # one draft without role information
    feats = torch.randn(batch, 10, F, generator=g)
    known = torch.rand(batch, 10, generator=g) < 0.7
    return champs, roles, feats, known


def test_lr_offset_reproduces_logistic_regression():
    """At initialisation the network must equal the fitted regression exactly."""
    rng = np.random.default_rng(0)
    champs = np.stack([rng.permutation(30)[:10] + 1 for _ in range(600)])
    roles = np.tile(np.array([1, 2, 3, 4, 5] * 2), (600, 1))
    roles[:100] = 0
    y = (rng.random(600) < 0.5).astype(np.float32)
    x = one_hot_features(champs, roles, 30)
    clf, _ = fit_logistic(x, y, np.ones(600), x, y, grid=(1.0,))
    model = small_model(lr_offset=True, n_player_feats=F, n_aux_anti=3, aux_duration=True)
    model.set_lr(clf.coef_[0], clf.intercept_[0])
    with torch.no_grad():
        logit = model(torch.tensor(champs), torch.tensor(roles))
    assert np.allclose(logit.numpy(), clf.decision_function(x), atol=1e-4)
    assert {"lr_weight", "lr_bias"} <= set(model.state_dict())
    assert not any(n.startswith("lr_") for n, _ in model.named_parameters())  # frozen


def test_antisymmetry_with_players_and_aux():
    model = small_model(n_player_feats=F, n_aux_anti=3, aux_duration=True)
    champs, roles, feats, known = drafts()
    with torch.no_grad():
        for p in model.head[-1].parameters():
            p.normal_(0, 0.5)
        a, aux_a = model(champs, roles, feats, known, return_aux=True)
        b, aux_b = model(swap_sides(champs), swap_sides(roles), swap_sides(feats),
                         swap_sides(known), return_aux=True)
    assert aux_a.shape == (6, 4)
    assert torch.allclose(a + b, 2 * model.side_bias.expand(6), atol=1e-5)   # side_effect is 0
    assert torch.allclose(aux_a[:, :3], -aux_b[:, :3], atol=1e-5)            # margins flip sign
    assert torch.allclose(aux_a[:, 3], aux_b[:, 3], atol=1e-5)               # game length does not
    assert not torch.allclose(a, -a.flip(0))


def test_missing_players_equal_draft_only():
    model = small_model(n_player_feats=F)
    champs, roles, feats, known = drafts()
    with torch.no_grad():
        model.no_player.normal_(0, 0.3)
        draft_only = model(champs, roles)
        nobody = model(champs, roles, feats, torch.zeros_like(known))
        somebody = model(champs, roles, feats, known)
    assert torch.allclose(draft_only, nobody, atol=1e-6)
    assert not torch.allclose(draft_only, somebody, atol=1e-3)


def test_first_version_checkpoints_still_fit():
    """Default arguments must build exactly the parameter set of the first release."""
    names = set(small_model().state_dict())
    assert not any(n.startswith(("player", "no_player", "lr_")) for n in names)
    assert small_model().head[-1].out_features == 1


def test_player_team_features():
    feats = np.zeros((1, 10, F), dtype=np.float32)
    feats[0, :, 3] = [0.1] * 5 + [0.3] * 5
    known = np.ones((1, 10), dtype=bool)
    known[0, 9] = False
    x = player_team_features(feats, known).toarray()
    assert x.shape == (1, F + 1)
    assert x[0, 3] == pytest.approx(0.5 - 1.2) and x[0, F] == 1


def test_resolve_hp_precedence(tmp_path):
    import argparse
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"dropout": 0.4, "lr": 5e-4}))
    args = argparse.Namespace(config=str(cfg), lr=2e-3, antisymmetric=False)
    hp = resolve_hp(args)
    assert hp["dropout"] == 0.4 and hp["lr"] == 2e-3 and hp["antisymmetric"] is False
    assert hp["layers"] == DEFAULT_HP["layers"]
    cfg.write_text(json.dumps({"typo": 1}))
    with pytest.raises(SystemExit):
        resolve_hp(args)


def test_tune_smoke(tmp_path):
    import random
    assert set(sample_config(random.Random(0))) == set(DEFAULT_HP)
    make_synthetic(tmp_path / "m.sqlite", tmp_path / "c.json", n=3000, n_champions=30,
                   n_players=300, seed=2)
    out = tmp_path / "best.json"
    tune_main(["--db", str(tmp_path / "m.sqlite"), "--champions", str(tmp_path / "c.json"),
               "--trials", "2", "--epochs", "1", "--device", "cpu", "--out", str(out)])
    best = json.loads(out.read_text())
    assert set(best) == set(DEFAULT_HP) and best["epochs"] == DEFAULT_HP["epochs"]
