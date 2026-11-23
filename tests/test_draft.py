import numpy as np
import pytest

from lolwp.draft import DraftAdvisor, parse_role, role_games_from_db
from lolwp.predict import Predictor

# synthetic champions: "Champ001".. with role = (number - 1) % 5, 60 champions
TOP, JG, MID, BOT, SUP = range(5)


@pytest.fixture(scope="module")
def advisor(trained):
    _, _, out, _, _ = trained
    return DraftAdvisor(Predictor.load(str(out)), samples=128, min_role_share=0.0001)


def champs_of_role(advisor, role):
    return [advisor.index(f"Champ{n:03d}") for n in range(1, 61) if (n - 1) % 5 == role]


def test_parse_role():
    assert parse_role("JG") == 1 and parse_role("adc") == 3 and parse_role("Support") == 4
    with pytest.raises(ValueError):
        parse_role("carry")


def test_role_statistics_match_database(trained, advisor):
    db_path = trained[0]
    stored = advisor.role_games
    for role in range(5):
        assert set(np.flatnonzero(stored[:, role]).tolist()) == set(champs_of_role(advisor, role))
    # the fallback for old checkpoints counts all games of the model's patches (train+val+test)
    from_db = role_games_from_db(db_path, advisor.p)
    assert (from_db >= stored).all() and from_db.sum() > stored.sum()
    assert ((from_db > 0) == (stored > 0)).all()


def test_infer_roles_and_parse_team(advisor):
    team, inferred = advisor.parse_team(["Champ003", "sup=Champ010", "Champ001"], infer=True)
    assert team == {MID: advisor.index("Champ003"), SUP: advisor.index("Champ010"),
                    TOP: advisor.index("Champ001")}
    assert inferred == {MID, TOP}
    with pytest.raises(ValueError, match="role=Champion"):
        advisor.parse_team(["Champ003"], infer=False)
    with pytest.raises(ValueError, match="two champions"):
        advisor.parse_team(["mid=Champ003", "mid=Champ008"], infer=True)


def test_sampling_respects_roles_bans_and_uniqueness(advisor):
    fixed = np.zeros(10, dtype=np.int64)
    fixed[MID] = advisor.index("Champ003")
    banned = {advisor.index("Champ001"), advisor.index("Champ006")}
    drafts = advisor._sample(fixed, banned)
    assert drafts.shape == (128, 10) and (drafts[:, MID] == fixed[MID]).all()
    assert all(len(set(row)) == 10 for row in drafts.tolist())
    assert not np.isin(drafts, list(banned)).any()
    for col in range(10):
        assert set(drafts[:, col].tolist()) <= set(champs_of_role(advisor, col % 5))


@pytest.mark.parametrize("side", ["blue", "red"])
def test_last_pick_value_equals_model_prediction(advisor, side):
    """With nine champions locked, no sampling is left: the value must be the model's output."""
    mine = ["Champ001", "Champ002", "Champ004", "Champ005"]            # top jg bot sup
    theirs = ["Champ006", "Champ007", "Champ008", "Champ009", "Champ010"]
    ally = {r: advisor.index(c) for r, c in zip([TOP, JG, BOT, SUP], mine)}
    enemy, _ = advisor.parse_team(theirs, infer=True)
    options = advisor.recommend_picks(side, MID, ally, enemy, top=100)
    assert [o.win for o in options] == sorted((o.win for o in options), reverse=True)
    assert all(o.counter is None for o in options)                     # enemy has no pick left
    for o in options[:3] + options[-3:]:
        full = mine[:2] + [o.champion] + mine[2:]
        p_blue = advisor.p.predict(full, theirs) if side == "blue" else advisor.p.predict(theirs, full)
        assert o.win == pytest.approx(p_blue if side == "blue" else 1 - p_blue, abs=1e-5)


def test_picks_follow_planted_strengths(advisor):
    # make_synthetic(seed=1) draws champion strengths first: reproduce them here
    strength = np.random.default_rng(1).normal(0, 0.25, 60)
    options = advisor.recommend_picks("blue", MID, {}, {}, top=100, counter_check=0)
    assert len(options) == 12
    true = [strength[int(o.champion.split()[1]) - 1] for o in options]
    assert np.corrcoef([o.win for o in options], true)[0, 1] > 0.35  # strength is only part of a champion's planted effect
    assert sum(o.vs_field > 0 for o in options) not in (0, len(options))


def test_pick_pool_bans_and_counter(advisor):
    banned = {advisor.index("Champ003")}
    enemy, _ = advisor.parse_team(["Champ008"], infer=True)            # enemy mid
    options = advisor.recommend_picks("red", MID, {}, enemy, banned, top=100, counter_check=2)
    names = {o.champion for o in options}
    assert "Champ 003" not in names and "Champ 008" not in names and len(names) == 10
    assert all(o.counter and o.win_if_countered <= o.win + 0.02 for o in options[:2])
    assert "(mid)" not in options[0].counter                           # enemy mid is taken
    assert options[2].counter is None
    pool = [advisor.index("Champ013"), advisor.index("Champ018")]
    assert {o.champion for o in advisor.recommend_picks("red", MID, {}, {}, pool=pool)} == {
        "Champ 013", "Champ 018"}
    with pytest.raises(ValueError, match="already filled"):
        advisor.recommend_picks("blue", MID, {MID: pool[0]}, {})
    with pytest.raises(ValueError, match="twice"):
        advisor.recommend_picks("blue", TOP, {MID: pool[0]}, {MID: pool[0]})


def test_bans_target_strong_enemy_champions(advisor):
    strength = np.random.default_rng(1).normal(0, 0.25, 60)
    bans = advisor.recommend_bans("blue", top=100)
    assert len(bans) == 60
    assert [b.gain for b in bans] == sorted((b.gain for b in bans), reverse=True)
    true = [strength[int(b.champion.split()[1]) - 1] for b in bans]
    # the enemy having a strong champion lowers my win rate; banning it gains the most
    assert np.corrcoef([b.win_if_enemy_has for b in bans], true)[0, 1] < -0.35
    assert np.mean(true[:10]) > np.mean(true[-10:])
    # with nothing known about either team, a champion helps whoever gets it: net ~ 0
    assert abs(np.mean([b.net for b in bans])) < np.mean([abs(b.gain) for b in bans])
    already = {advisor.index(bans[0].champion)}
    assert bans[0].champion not in {b.champion for b in advisor.recommend_bans("blue", banned=already, top=100)}
    by_net = advisor.recommend_bans("blue", top=100, sort="net")
    assert [b.net for b in by_net] == sorted((b.net for b in by_net), reverse=True)
