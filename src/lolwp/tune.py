"""Random search over hyperparameters, scored by validation log loss.

    lolwp-tune --trials 30                 # writes models/best_config.json
    lolwp-train --config models/best_config.json

Each trial trains one model (no ensemble) with a random draw from SEARCH_SPACE; trial 0 is
always the defaults, so the table shows whether searching helped at all. Only the validation
set is used, so the test set of lolwp-train stays untouched. Validation losses of single
models differ by noise of roughly +-0.0003 between seeds: do not read much into small gaps.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .train import DEFAULT_HP, add_data_args, prepare, train_member

SEARCH_SPACE = {
    "d_model": [32, 64, 128],
    "layers": [1, 2, 3],
    "dropout": [0.1, 0.2, 0.3, 0.4],
    "antisymmetric": [True, False],
    "lr_offset": [True, False],
    "lr": [3e-4, 1e-3, 2e-3],
    "weight_decay": [0.01, 0.05, 0.2],
    "batch_size": [512, 1024, 2048],
    "player_dropout": [0.1, 0.2, 0.35],
    "aux_weight": [0.0, 0.1, 0.3, 1.0],
    "finetune": [True, False],
    "finetune_lr_mult": [0.1, 0.3],
}


def sample_config(rng: random.Random) -> dict:
    hp = dict(DEFAULT_HP)
    for key, choices in SEARCH_SPACE.items():
        hp[key] = rng.choice(choices)
    hp["d_ff"] = 2 * hp["d_model"]
    return hp


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, help="cap the epochs per trial (faster, rougher search)")
    ap.add_argument("--out", default="models/best_config.json")
    args = ap.parse_args(argv)

    bundle = prepare(args)
    rng = random.Random(args.seed)
    results = []
    for trial in range(args.trials):
        hp = dict(DEFAULT_HP) if trial == 0 else sample_config(rng)
        if args.epochs:
            hp["epochs"] = args.epochs
        t0 = time.time()
        _, info = train_member(bundle, hp, seed=args.seed, log=lambda *_: None)
        results.append((info["val_loss"], trial, hp))
        changed = {k: v for k, v in hp.items() if DEFAULT_HP[k] != v and k != 'epochs'}
        print(f"trial {trial:3d}  val {info['val_loss']:.5f}  {time.time() - t0:4.0f}s  "
              f"{'defaults' if trial == 0 else changed}", flush=True)

    results.sort(key=lambda r: r[0])
    default_loss = next(loss for loss, trial, _ in results if trial == 0)
    best_loss, best_trial, best_hp = results[0]
    print(f"\nbest: trial {best_trial} with val {best_loss:.5f} "
          f"(defaults: {default_loss:.5f}, gain {default_loss - best_loss:+.5f})")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_hp["epochs"] = DEFAULT_HP["epochs"]  # a capped search still trains fully afterwards
    out.write_text(json.dumps(best_hp, indent=1))
    print(f"wrote {out}; train with: lolwp-train --config {out}")


if __name__ == "__main__":
    main()
