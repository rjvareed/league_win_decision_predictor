import numpy as np
import pytest
import torch

from lolwp.baselines import one_hot_features
from lolwp.data import PLAYER_FEATURES, chronological_split, load_draft_data, select_patches
from lolwp.metrics import per_sample_log_loss, summarize
from lolwp.model import DraftModel
from lolwp.predict import Predictor
from lolwp.train import fit_calibration


def random_draft(batch, n_champions, seed=0):
    g = torch.Generator().manual_seed(seed)
    champs = torch.stack([torch.randperm(n_champions, generator=g)[:10] + 1 for _ in range(batch)])
    roles = torch.tensor([[1, 2, 3, 4, 5] * 2] * batch)
    return champs, roles


def swap_sides(x):
    return torch.cat([x[:, 5:], x[:, :5]], dim=1)


def test_antisymmetric_part_flips_with_sides():
    torch.manual_seed(0)
    model = DraftModel(30, d_model=16, n_layers=1, n_heads=2, d_ff=32).eval()
    champs, roles = random_draft(8, 30)
    with torch.no_grad():
        model.side_bias.fill_(0.3)
        model.side_effect.weight.normal_(0, 0.1)
        a = model(champs, roles)
        b = model(swap_sides(champs), swap_sides(roles))
        side = 0.3 + model.side_effect(champs).sum(dim=(1, 2))
    # logit(A,B) + logit(B,A) leaves only the side terms (same champions on the map)
    assert torch.allclose(a + b, 2 * side, atol=1e-5)
    assert not torch.allclose(a, b)


def test_permutation_invariant_within_team():
    torch.manual_seed(0)
    model = DraftModel(30, d_model=16, n_layers=1, n_heads=2, d_ff=32).eval()
    champs, roles = random_draft(4, 30)
    perm = [2, 0, 4, 1, 3, 9, 7, 5, 8, 6]
    with torch.no_grad():
        assert torch.allclose(model(champs, roles), model(champs[:, perm], roles[:, perm]), atol=1e-5)


def test_plain_mode_runs():
    model = DraftModel(30, d_model=16, n_layers=1, n_heads=2, d_ff=32, antisymmetric=False).eval()
    champs, roles = random_draft(4, 30)
    assert model(champs, roles).shape == (4,)


def test_select_patches():
    counts = [("16.22", 50), ("16.21", 100), ("16.20", 100), ("16.19", 100)]
    assert select_patches(counts, min_matches=40, max_patches=3) == ["16.22"]
    assert select_patches(counts, min_matches=120, max_patches=3) == ["16.22", "16.21"]
    assert select_patches(counts, min_matches=10**9, max_patches=3) == ["16.22", "16.21", "16.20"]


def test_chronological_split():
    tr, va, te = chronological_split(1000, 0.1, 0.1)
    assert (len(tr), len(va), len(te)) == (800, 100, 100)
    assert tr.max() < va.min() and va.max() < te.min()
    tr, va, te = chronological_split(1000, 0.1, 0.0)
    assert len(te) == 0 and len(tr) == 900


def test_one_hot_features():
    champs = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    with_roles = one_hot_features(champs, np.array([[1, 2, 3, 4, 5] * 2]), 10)
    without = one_hot_features(champs, np.zeros((1, 10), dtype=np.int64), 10)
    assert with_roles.shape == (1, 12 * 11)
    assert with_roles.sum() == 20 and without.sum() == 10
    assert with_roles[0, 1] == 1 and with_roles[0, 11 + 6] == 1  # blue champ 1, red champ 6


def test_calibration_recovers_scale():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 50_000)
    y = (rng.random(50_000) < 1 / (1 + np.exp(-(0.5 * z + 0.2)))).astype(np.float32)
    scale, bias = fit_calibration(z, y)
    assert abs(scale - 0.5) < 0.05 and abs(bias - 0.2) < 0.05


def test_metrics_perfectly_calibrated_coin():
    y = np.array([1, 0, 1, 0], dtype=np.float32)
    m = summarize(y, np.full(4, 0.5))
    assert m["log_loss"] == pytest.approx(np.log(2)) and m["brier"] == pytest.approx(0.25)
    assert m["ece"] == pytest.approx(0.0) and m["auc"] == 0.5


def test_data_loading(trained):
    db_path, *_ = trained
    data = load_draft_data(db_path, min_matches=10**9, max_patches=2)
    assert data.patches == ["99.2", "99.1"]
    assert len(data) == 24_000 and (np.diff(data.time) >= 0).all()
    assert data.champs.min() >= 1 and data.champs.max() <= len(data.champion_ids)
    assert sorted(round(float(w), 3) for w in np.unique(data.weight)) == [0.7, 1.0]
    assert ((data.roles == 0).all(axis=1) | (data.roles > 0).all(axis=1)).all()
    only_new = load_draft_data(db_path, min_matches=0)
    assert only_new.patches == ["99.2"] and len(only_new) == 16_800
    # player histories still come from *all* stored games, also the patch not trained on
    assert only_new.known.all() and (only_new.feats[0, :, 2] > 0).all()
    assert data.aux_mask.all() and abs(float(data.aux[:, 0].std()) - 1) < 0.5


def test_pipeline_learns_real_structure(trained):
    *_, truth, summary = trained
    m = summary["metrics"]
    prior = m["prior (blue winrate)"]["log_loss"]
    draft = m["transformer ensemble, draft"]["log_loss"]
    full = m["transformer ensemble, draft + players"]["log_loss"]
    n_test = summary["n_matches"]["test"]
    oracle = float(np.mean(-(truth[-n_test:] * np.log(truth[-n_test:])
                             + (1 - truth[-n_test:]) * np.log(1 - truth[-n_test:]))))
    # draft alone clearly beats ignoring it; player histories add a lot on top (the synthetic
    # world has strong player skill); nothing can beat the true probabilities
    assert oracle < full < draft - 0.01 < prior - 0.01
    assert m["transformer ensemble, draft + players"]["ece"] < 0.05
    assert summary["logloss_diffs"]["players_vs_draft"]["ci_high"] < 0
    # both patches were used, so every member went through the fine-tuning stage
    assert summary["patches"] == ["99.2", "99.1"] and len(summary["members"]) == 2
    assert summary["config"]["n_player_feats"] == len(PLAYER_FEATURES)
    # knowing a single player (what the champion-select assistant has) sits in between
    solo = m["transformer ensemble, draft + one known player"]["log_loss"]
    assert full < solo < draft


def test_checkpoint_carries_what_the_assistant_needs(trained):
    ckpt = torch.load(trained[2], map_location="cpu", weights_only=True)
    n = len(ckpt["champion_ids"])
    assert ckpt["calibration_solo"]["scale"] > 0 and ckpt["calibration_players"]["scale"] > 0
    assert ckpt["ban_games"] == ckpt["n_matches"]["train"]         # every synthetic game has bans
    bans = np.array(ckpt["champion_ban_games"])
    assert bans.shape == (n,) and bans.sum() == 10 * ckpt["ban_games"]
    means = np.array(ckpt["result_means"])                           # [loser, winner] x ADJUSTED
    assert means.shape[0] == 2 and (means[1, 0] > means[0, 0])       # winners out-earn losers
    assert ckpt["hyperparameters"]["refit"] and ckpt["hyperparameters"]["player_partial"] > 0


def test_refit_uses_validation_games_and_fixed_epochs(trained, monkeypatch):
    """The refit trains on training + validation rows for exactly the epochs found before."""
    import lolwp.train as T
    args = T.build_parser().parse_args([
        "--db", str(trained[0]), "--champions", str(trained[1]), "--max-patches", "2",
        "--device", "cpu", "--epochs", "3", "--finetune-epochs", "2"])
    hp = T.resolve_hp(args)
    bundle = T.prepare(args, log=lambda *_: None)
    calls = []
    real_fit = T.fit

    def spy(model, b, rows, val, hp, **kw):
        calls.append((len(rows), kw.get("stop_after")))
        return real_fit(model, b, rows[:2048], val, {**hp, "batch_size": 1024}, **kw)

    monkeypatch.setattr(T, "fit", spy)
    model = T.refit_member(bundle, hp, 0, {"pretrain_epoch": 2, "finetune_epoch": 1},
                           log=lambda *_: None)
    newest = int((bundle.data.patch[np.concatenate([bundle.tr, bundle.va])] == "99.2").sum())
    assert calls == [(len(bundle.tr) + len(bundle.va), 2), (newest, 1)]
    assert torch.isfinite(T.predict_logits(model, bundle, bundle.rows(bundle.va[:64]))).all()
    # one known player per game: the same player on every call, and never more than one
    solo = T.predict_logits(model, bundle, bundle.rows(bundle.va[:64]), solo=True)
    assert torch.equal(solo, T.predict_logits(model, bundle, bundle.rows(bundle.va[:64]), solo=True))
    assert not torch.allclose(solo, T.predict_logits(model, bundle, bundle.rows(bundle.va[:64])))


def test_predictor(trained):
    _, _, out, _, _ = trained
    p = Predictor.load(str(out))
    blue = ["Champ001", "champ 002", "CHAMP003", "4", "Champ005"]
    red = ["Champ006", "Champ007", "Champ008", "Champ009", "Champ010"]
    p_blue = p.predict(blue, red)
    assert 0 < p_blue < 1
    assert 0 < p.predict(blue, red, roles_known=False) < 1
    # player features: take the game whose blue players have the best record relative to
    # red's; giving the same histories to the other side must flip the advantage
    data = load_draft_data(trained[0], min_matches=10**9, max_patches=2)
    f = PLAYER_FEATURES.index("total_winrate")
    gap = data.feats[:, :5, f].sum(axis=1) - data.feats[:, 5:, f].sum(axis=1)
    good = data.feats[int(np.argmax(gap))]
    bad = np.concatenate([good[5:], good[:5]])
    assert p.predict(blue, red, player_feats=good) > p.predict(blue, red, player_feats=bad) + 0.1
    assert p.predict(blue, red, player_feats=good, players_known=[False] * 10) == pytest.approx(
        p.predict(blue, red, player_feats=np.zeros((10, len(PLAYER_FEATURES))), players_known=[False] * 10))
    # order within a team is irrelevant when roles are unknown
    assert p.predict(blue[::-1], red, roles_known=False) == pytest.approx(
        p.predict(blue, red, roles_known=False), abs=1e-5)
    with pytest.raises(ValueError, match="did you mean"):
        p.predict(["Chmp001"] + blue[1:], red)
    with pytest.raises(ValueError, match="only be picked once"):
        p.predict(blue, ["Champ001"] + red[1:])
    with pytest.raises(ValueError, match="exactly 5"):
        p.predict(blue[:4], red)
