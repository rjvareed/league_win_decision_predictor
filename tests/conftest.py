import pytest

from lolwp.synth import make_synthetic
from lolwp.train import run as train_main


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic")
    db_path, champs, out = d / "m.sqlite", d / "champions.json", d / "model.pt"
    truth = make_synthetic(db_path, champs, n=24_000, n_champions=60, seed=1)
    summary = train_main([
        "--db", str(db_path), "--champions", str(champs), "--out", str(out),
        "--max-patches", "2", "--ensemble", "2", "--epochs", "10", "--finetune-epochs", "3",
        "--device", "cpu", "--batch-size", "512",
    ])
    return db_path, champs, out, truth, summary
