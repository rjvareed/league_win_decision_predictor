"""Train the win model and compare it against baselines on the newest games.

    lolwp-train                                   # defaults
    lolwp-train --config models/best_config.json  # hyperparameters found by lolwp-tune
    lolwp-train --patches 16.22 16.21             # force specific patch(es)
    lolwp-train --test-frac 0                     # final fit: no hold-out test set

Split is chronological (oldest games train, newest validate/test), early stopping and
calibration use the validation set only, and the test set is touched once for the report.

Training recipe per ensemble member
  1. Optional (--lr-offset): a logistic regression on champion / champion-in-role one-hots
     is fitted first and frozen inside the network as an offset; the network then starts
     exactly at that baseline and learns a correction on top.
  2. Pretrain on all selected patches (older ones down-weighted by --patch-decay).
  3. Fine-tune on the newest patch only at a lower learning rate; kept only if the validation
     loss improves (--no-finetune to disable).
  4. Refit: a fresh model is trained on training + validation games for the epoch counts
     found in 2 and 3, and that one is saved (--no-refit to disable).
Inputs are the draft and, when the database has player ids, each player's history before the
game (see lolwp.data.PLAYER_FEATURES). Player inputs are randomly hidden during training (for
everyone, or for a random part of the ten), so the same model also predicts from the draft
alone and from the draft plus a single known player. End-of-game margins (gold, kills, towers,
length) are extra training targets only (--aux-weight 0 to disable).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse

from .baselines import fit_logistic, one_hot_features, player_team_features
from .data import (
    N_AUX_ANTI,
    PLAYER_FEATURES,
    DraftData,
    chronological_split,
    load_champion_names,
    load_draft_data,
)
from .metrics import paired_bootstrap_logloss_diff, per_sample_log_loss, summarize
from .model import DraftModel

# Chosen by lolwp-tune on 513k real Diamond+ games (patches 16.19-16.22) with player
# histories and per-player statistics. On draft-only data (no player ids) a smaller model
# with --lr-offset did better. The masking shares, the longer schedule and the refit were
# compared afterwards on 511k games (16.21-16.23), see "Results" in the README.
DEFAULT_HP = {
    "d_model": 32, "layers": 2, "heads": 4, "d_ff": 64, "dropout": 0.1,
    "antisymmetric": True, "lr_offset": False,
    "lr": 0.0003, "weight_decay": 0.2, "batch_size": 512,
    "role_dropout": 0.15, "player_dropout": 0.2, "player_partial": 0.2, "aux_weight": 1.0,
    "epochs": 60, "patience": 7,
    "finetune": True, "finetune_lr_mult": 0.1, "finetune_epochs": 12,
    "refit": True,
}
MIN_FINETUNE_ROWS = 5000


@dataclass
class Bundle:
    """Everything a training run needs, prepared once (shared by lolwp-train and lolwp-tune)."""
    data: DraftData
    tr: np.ndarray
    va: np.ndarray
    te: np.ndarray
    device: torch.device
    champs: torch.Tensor
    roles: torch.Tensor
    y: torch.Tensor
    w: torch.Tensor
    feats: torch.Tensor
    known: torch.Tensor
    aux: torch.Tensor
    aux_mask: torch.Tensor
    prior: float
    has_players: bool
    lr_coef: np.ndarray
    lr_intercept: float
    lr_c: float
    x_draft: sparse.csr_matrix
    names: dict

    @property
    def n_champions(self) -> int:
        return len(self.data.champion_ids)

    def rows(self, idx: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(idx, device=self.device)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def prepare(args, log=print) -> Bundle:
    names = load_champion_names(args.champions)
    data = load_draft_data(args.db, args.min_matches, args.max_patches, args.patches,
                           args.patch_decay, extra_champion_ids=tuple(names))
    tr, va, te = chronological_split(len(data), args.val_frac, args.test_frac)
    if len(tr) < 1000 or len(va) < 200:
        raise SystemExit(f"only {len(data)} matches - collect more data first")
    per_patch = {p: int((data.patch == p).sum()) for p in data.patches}
    coverage = float(data.known[tr].mean())
    has_players = coverage > 0.02 and not args.no_players
    log(f"{len(data)} matches from patches {per_patch}; "
        f"train {len(tr)} / val {len(va)} / test {len(te)}; device {args.device}")
    log(f"player ids available for {100 * coverage:.0f}% of training picks"
        + ("" if has_players else " -> training draft-only"))
    if len(data) < 50_000:
        log("WARNING: fewer than 50k matches - the signal is weak, expect a noisy model")
    ev = te if len(te) else va
    if has_players and data.clean is not None and data.clean.any():
        early = 1 - float(data.clean[ev].mean())
        if early > 0:
            log(f"note: {100 * early:.0f}% of the evaluation games were played before the first "
                "ladder snapshot of their platform. Player-feature results on those can be "
                "optimistic (who is on the crawled ladder already hints at who won), so the "
                "report also scores the games played after the snapshot separately.")

    w_tr = data.weight[tr]
    prior = float(np.average(data.y[tr], weights=w_tr))
    n_champ = len(data.champion_ids)
    x_draft = one_hot_features(data.champs, data.roles, n_champ)
    # a share of role-less copies teaches the regression to cope with unknown roles too
    rng = np.random.default_rng(0)
    sub = tr[rng.random(len(tr)) < 0.15]
    x_noroles = one_hot_features(data.champs[sub], np.zeros_like(data.roles[sub]), n_champ)
    clf, best_c = fit_logistic(
        sparse.vstack([x_draft[tr], x_noroles]).tocsr(),
        np.concatenate([data.y[tr], data.y[sub]]),
        np.concatenate([w_tr, data.weight[sub]]), x_draft[va], data.y[va])

    device = torch.device(args.device)
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    return Bundle(
        data, tr, va, te, device, t(data.champs), t(data.roles), t(data.y), t(data.weight),
        t(data.feats), t(data.known), t(data.aux), t(data.aux_mask), prior, has_players,
        clf.coef_[0].astype(np.float32), float(clf.intercept_[0]), best_c, x_draft, names)


def model_config(bundle: Bundle, hp: dict) -> dict:
    aux = hp["aux_weight"] > 0
    return dict(
        n_champions=bundle.n_champions, d_model=hp["d_model"], n_layers=hp["layers"],
        n_heads=hp["heads"], d_ff=hp["d_ff"], dropout=hp["dropout"],
        antisymmetric=hp["antisymmetric"], lr_offset=hp["lr_offset"],
        n_player_feats=len(PLAYER_FEATURES) if bundle.has_players else 0,
        n_aux_anti=N_AUX_ANTI if aux else 0, aux_duration=aux)


def make_model(bundle: Bundle, hp: dict) -> DraftModel:
    model = DraftModel(**model_config(bundle, hp)).to(bundle.device)
    with torch.no_grad():
        if hp["lr_offset"]:
            model.set_lr(bundle.lr_coef, bundle.lr_intercept)
        else:
            model.side_bias.fill_(math.log(bundle.prior / (1 - bundle.prior)))
    return model


@torch.no_grad()
def predict_logits(model, bundle: Bundle, rows: torch.Tensor, players: bool = True,
                   roles: bool = True, batch: int = 8192, solo: bool = False) -> torch.Tensor:
    """solo: only one player per game is known (the same one on every call), the situation
    of someone using the model for their own champion select."""
    model.eval()
    out = []
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch]
        role_in = bundle.roles[r] if roles else torch.zeros_like(bundle.roles[r])
        if players and model.player is not None:
            known = bundle.known[r]
            if solo:
                known = known & (torch.arange(10, device=r.device)[None, :] == (r % 10)[:, None])
            out.append(model(bundle.champs[r], role_in, bundle.feats[r], known))
        else:
            out.append(model(bundle.champs[r], role_in))
    return torch.cat(out) if out else torch.zeros(0, device=bundle.device)


def fit(model, bundle: Bundle, rows: torch.Tensor, val: torch.Tensor, hp: dict, *, lr: float,
        epochs: int, patience: int, seed: int, weighted: bool = True,
        keep_current: bool = False, stop_after: int | None = None, log=print
        ) -> tuple[float, int]:
    """AdamW + warmup/cosine schedule; early-stop on validation log loss.

    keep_current: treat the model's present weights as the result to beat (fine-tuning).
    stop_after: no early stopping; follow the schedule of an `epochs`-long run for exactly
    this many epochs and keep the final weights (refitting with the validation games, when
    there is nothing left to select on).
    Returns (best validation log loss, best epoch; 0 = the starting weights were kept)."""
    device = bundle.device
    gen = torch.Generator(device=device).manual_seed(seed)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": hp["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    n, batch_size = len(rows), hp["batch_size"]
    total = epochs * math.ceil(n / batch_size)
    warmup = max(1, min(500, total // 20))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    def val_loss():
        return F.binary_cross_entropy_with_logits(
            predict_logits(model, bundle, val), bundle.y[val]).item()

    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    best_loss, best_epoch, best_state, bad = float("inf"), 0, None, 0
    if keep_current:
        best_loss, best_state = val_loss(), copy.deepcopy(model.state_dict())
    use_players = model.player is not None
    use_aux = hp["aux_weight"] > 0
    for epoch in range(1, (stop_after or epochs) + 1):
        model.train()
        perm = rows[torch.randperm(n, device=device, generator=gen)]
        running, seen = 0.0, 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            r = bundle.roles[idx]
            if hp["role_dropout"] > 0:  # cope with drafts whose roles are unknown
                drop = torch.rand(len(idx), device=device, generator=gen) < hp["role_dropout"]
                r = r.masked_fill(drop[:, None], 0)
            feats = known = None
            if use_players:  # cope with games where nothing is known about the players
                u = torch.rand(len(idx), device=device, generator=gen)
                drop = u < hp["player_dropout"]
                feats, known = bundle.feats[idx], bundle.known[idx] & ~drop[:, None]
                if hp["player_partial"] > 0:
                    # ... and with games where only some of the ten are known, down to a
                    # single one (champion select: you know yourself, nobody else)
                    partial = (u >= hp["player_dropout"]) & (
                        u < hp["player_dropout"] + hp["player_partial"])
                    n_keep = torch.randint(1, 10, (len(idx), 1), device=device, generator=gen)
                    rank = torch.rand(len(idx), 10, device=device, generator=gen).argsort(dim=1).argsort(dim=1)
                    known = known & ((rank < n_keep) | ~partial[:, None])
            logits, aux = model(bundle.champs[idx], r, feats, known, return_aux=True)
            wi = bundle.w[idx] if weighted else torch.ones(len(idx), device=device)
            bce = (F.binary_cross_entropy_with_logits(logits, bundle.y[idx], reduction="none")
                   * wi).sum() / wi.sum()
            loss = bce
            if use_aux:
                mask = bundle.aux_mask[idx]
                err = F.smooth_l1_loss(aux, bundle.aux[idx], reduction="none") * mask
                loss = loss + hp["aux_weight"] * err.sum() / mask.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += bce.item() * wi.sum().item()
            seen += wi.sum().item()
        if stop_after:
            log(f"  epoch {epoch:3d}  train {running / seen:.5f}")
            continue
        current = val_loss()
        log(f"  epoch {epoch:3d}  train {running / seen:.5f}  val {current:.5f}")
        if current < best_loss - 1e-6:
            best_loss, best_epoch, bad = current, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    if stop_after:
        return float("nan"), stop_after
    model.load_state_dict(best_state)
    return best_loss, best_epoch


def train_member(bundle: Bundle, hp: dict, seed: int, log=print):
    """Pretrain on all selected patches, then fine-tune on the newest one."""
    torch.manual_seed(seed)
    model = make_model(bundle, hp)
    tr, va = bundle.rows(bundle.tr), bundle.rows(bundle.va)
    loss, epoch = fit(model, bundle, tr, va, hp, lr=hp["lr"], epochs=hp["epochs"],
                      patience=hp["patience"], seed=seed, log=log)
    info = {"val_loss": loss, "pretrain_epoch": epoch, "finetune_epoch": 0}
    newest = bundle.tr[bundle.data.patch[bundle.tr] == bundle.data.patches[0]]
    if hp["finetune"] and len(bundle.data.patches) > 1 and len(newest) >= MIN_FINETUNE_ROWS:
        log(f"  fine-tuning on {len(newest)} games of patch {bundle.data.patches[0]}")
        loss, epoch = fit(model, bundle, bundle.rows(newest), va, hp,
                          lr=hp["lr"] * hp["finetune_lr_mult"], epochs=hp["finetune_epochs"],
                          patience=3, seed=seed + 1000, weighted=False, keep_current=True, log=log)
        info.update(val_loss=loss, finetune_epoch=epoch)
    return model, info


def refit_member(bundle: Bundle, hp: dict, seed: int, info: dict, log=print):
    """Train a fresh model on the training *and* validation games, for as many epochs as
    `train_member` found best with this seed (`info`).

    The validation games are the newest ones before the test period, and the ones whose
    player histories look most like those of future games (the crawl has had time to see
    each player often), so leaving them out of the final model costs accuracy. Nothing is
    selected on them here: the schedule and the stopping points come from the first run."""
    torch.manual_seed(seed)
    model = make_model(bundle, hp)
    both = np.concatenate([bundle.tr, bundle.va])
    va = bundle.rows(bundle.va)
    if info["pretrain_epoch"]:
        fit(model, bundle, bundle.rows(both), va, hp, lr=hp["lr"], epochs=hp["epochs"],
            patience=0, seed=seed, stop_after=info["pretrain_epoch"], log=log)
    if info["finetune_epoch"]:
        newest = both[bundle.data.patch[both] == bundle.data.patches[0]]
        fit(model, bundle, bundle.rows(newest), va, hp, lr=hp["lr"] * hp["finetune_lr_mult"],
            epochs=hp["finetune_epochs"], patience=0, seed=seed + 1000, weighted=False,
            stop_after=info["finetune_epoch"], log=log)
    return model


def fit_calibration(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt scaling p = sigmoid(scale * logit + bias), fitted on validation logits."""
    z = torch.tensor(logits, dtype=torch.float64)
    t = torch.tensor(y, dtype=torch.float64)
    params = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([params], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(params[0] * z + params[1], t)
        loss.backward()
        return loss

    opt.step(closure)
    scale, bias = params.detach().tolist()
    if not (math.isfinite(scale) and math.isfinite(bias)) or scale <= 0:
        return 1.0, 0.0
    return scale, bias


def format_table(rows: dict[str, dict[str, float]]) -> str:
    cols = ["log_loss", "brier", "auc", "accuracy", "ece"]
    width = max(len(k) for k in rows)
    lines = [f"{'model':<{width}}  " + "  ".join(f"{c:>8}" for c in cols)]
    for name, m in rows.items():
        lines.append(f"{name:<{width}}  " + "  ".join(f"{m[c]:8.4f}" for c in cols))
    return "\n".join(lines)


def add_data_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--db", default="data/matches.sqlite")
    ap.add_argument("--champions", default="data/champions.json")
    g = ap.add_argument_group("data")
    g.add_argument("--patches", nargs="+", help="use exactly these patches (e.g. 16.22 16.21)")
    g.add_argument("--min-matches", type=int, default=400_000,
                   help="add older patches until this many matches are available")
    g.add_argument("--max-patches", type=int, default=4)
    g.add_argument("--patch-decay", type=float, default=0.8,
                   help="pretraining sample weight multiplier per patch of age")
    g.add_argument("--val-frac", type=float, default=0.1)
    g.add_argument("--test-frac", type=float, default=0.1)
    g.add_argument("--no-players", action="store_true",
                   help="ignore player ids even if the database has them (draft-only model)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def add_hp_args(ap: argparse.ArgumentParser) -> None:
    """Hyperparameters default to None so that --config and DEFAULT_HP can fill them in."""
    flag = argparse.BooleanOptionalAction
    g = ap.add_argument_group("hyperparameters (defaults: DEFAULT_HP, or --config)")
    g.add_argument("--config", help="JSON file of hyperparameters, e.g. written by lolwp-tune")
    g.add_argument("--d-model", type=int)
    g.add_argument("--layers", type=int)
    g.add_argument("--heads", type=int)
    g.add_argument("--d-ff", type=int)
    g.add_argument("--dropout", type=float)
    g.add_argument("--antisymmetric", action=flag, default=None,
                   help="score s(A|B) - s(B|A) instead of a plain blue/red encoder")
    g.add_argument("--lr-offset", action=flag, default=None,
                   help="start from the frozen logistic regression and learn a correction")
    g.add_argument("--lr", type=float)
    g.add_argument("--weight-decay", type=float)
    g.add_argument("--batch-size", type=int)
    g.add_argument("--role-dropout", type=float)
    g.add_argument("--player-dropout", type=float,
                   help="share of training games shown without any player inputs")
    g.add_argument("--player-partial", type=float,
                   help="share of training games shown with only 1-9 random players known")
    g.add_argument("--aux-weight", type=float,
                   help="weight of the end-of-game margin targets (0 = off)")
    g.add_argument("--epochs", type=int)
    g.add_argument("--patience", type=int)
    g.add_argument("--finetune", action=flag, default=None,
                   help="fine-tune on the newest patch after pretraining on all")
    g.add_argument("--refit", action=flag, default=None,
                   help="retrain every member on training + validation games for the number "
                        "of epochs early stopping chose, and save that model")
    g.add_argument("--finetune-lr-mult", type=float)
    g.add_argument("--finetune-epochs", type=int)


def resolve_hp(args) -> dict:
    hp = dict(DEFAULT_HP)
    if getattr(args, "config", None):
        loaded = json.loads(Path(args.config).read_text())
        unknown = set(loaded) - set(hp)
        if unknown:
            raise SystemExit(f"unknown hyperparameters in {args.config}: {sorted(unknown)}")
        hp.update(loaded)
    for key in hp:
        value = getattr(args, key, None)
        if value is not None:
            hp[key] = value
    if hp["d_model"] % hp["heads"]:
        raise SystemExit("--d-model must be divisible by --heads")
    return hp


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    add_hp_args(ap)
    ap.add_argument("--out", default="models/draft_model.pt")
    ap.add_argument("--ensemble", type=int, default=5, help="number of seeds to average")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def run(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    hp = resolve_hp(args)
    b = prepare(args)
    data, y = b.data, b.data.y
    ev = b.te if len(b.te) else b.va          # games the final report is computed on
    ev_name = "test" if len(b.te) else "validation (no test split)"
    report: dict[str, dict[str, float]] = {}

    # ---- baselines ---------------------------------------------------------------------
    report["prior (blue winrate)"] = summarize(y[ev], np.full(len(ev), b.prior))
    p_lr = sigmoid(b.x_draft[ev] @ b.lr_coef + b.lr_intercept)
    report[f"logistic regression, draft (C={b.lr_c})"] = summarize(y[ev], p_lr)
    p_lr_players = None
    if b.has_players:
        x_players = sparse.hstack(
            [b.x_draft, player_team_features(data.feats, data.known)]).tocsr()
        clf, c = fit_logistic(x_players[b.tr], y[b.tr], data.weight[b.tr], x_players[b.va], y[b.va])
        p_lr_players = clf.predict_proba(x_players[ev])[:, 1]
        report[f"logistic regression, draft + players (C={c})"] = summarize(y[ev], p_lr_players)

    # ---- transformer ensemble ----------------------------------------------------------
    va_t, ev_t = b.rows(b.va), b.rows(ev)
    modes = {"draft": dict(players=False), "noroles": dict(players=False, roles=False)}
    if b.has_players:
        modes["players"] = dict(players=True)
        modes["solo"] = dict(players=True, solo=True)
    states, infos = [], []
    val_logits = {m: [] for m in modes}
    ev_logits = {m: [] for m in modes}
    t0 = time.time()
    for k in range(args.ensemble):
        print(f"model {k + 1}/{args.ensemble}")
        model, info = train_member(b, hp, args.seed + k)
        infos.append(info)
        final = model
        if hp["refit"]:
            print("  refitting on training + validation games")
            final = refit_member(b, hp, args.seed + k, info)
        states.append({n: t.cpu() for n, t in final.state_dict().items()})
        # Calibration always comes from the model that never saw the validation games; the
        # report uses the saved model, unless it would be scored on games it was trained on.
        scored = final if len(b.te) else model
        for m, kw in modes.items():
            val_logits[m].append(predict_logits(model, b, va_t, **kw).cpu().numpy())
            ev_logits[m].append(predict_logits(scored, b, ev_t, **kw).cpu().numpy())
    print(f"training took {time.time() - t0:.0f}s")

    def calibrated(mode: str, calibrate_on: str):
        scale, bias = fit_calibration(np.mean(val_logits[calibrate_on], axis=0), y[b.va])
        return sigmoid(scale * np.mean(ev_logits[mode], axis=0) + bias), (scale, bias)

    p_draft, cal_draft = calibrated("draft", "draft")
    report["transformer, draft (single, uncalibrated)"] = summarize(
        y[ev], sigmoid(ev_logits["draft"][0]))
    report["transformer ensemble, draft"] = summarize(y[ev], p_draft)
    report["  same, roles hidden"] = summarize(y[ev], calibrated("noroles", "draft")[0])
    p_best, cal_players, cal_solo = p_draft, None, None
    if b.has_players:
        p_solo, cal_solo = calibrated("solo", "solo")
        report["transformer ensemble, draft + one known player"] = summarize(y[ev], p_solo)
        p_best, cal_players = calibrated("players", "players")
        report["transformer ensemble, draft + players"] = summarize(y[ev], p_best)
        late = data.clean[ev] if data.clean is not None else np.zeros(len(ev), dtype=bool)
        if 500 <= late.sum() < len(ev):
            report[f"  same, {int(late.sum())} games after the ladder snapshot"] = summarize(
                y[ev][late], p_best[late])
            report["  draft only, on those games"] = summarize(y[ev][late], p_draft[late])

    print(f"\nResults on {ev_name} set ({len(ev)} newest games"
          f"{'' if len(b.te) else '; optimistic: also used for early stopping/calibration'}):")
    print(format_table(report))

    def compare(label, p_a, p_b):
        d = paired_bootstrap_logloss_diff(y[ev], p_a, p_b)
        print(f"log loss, {label}: {d[0]:+.5f} (95% CI {d[1]:+.5f} .. {d[2]:+.5f})")
        return {"mean": d[0], "ci_low": d[1], "ci_high": d[2]}

    print("\nPaired differences (negative = first model better):")
    diffs = {"draft_vs_logreg": compare("transformer draft - logistic regression draft",
                                        p_draft, p_lr)}
    if b.has_players:
        diffs["players_vs_draft"] = compare("transformer draft+players - transformer draft",
                                            p_best, p_draft)
        diffs["players_vs_logreg_players"] = compare(
            "transformer draft+players - logistic regression draft+players", p_best, p_lr_players)
    base = per_sample_log_loss(y[ev], np.full(len(ev), b.prior)).mean()
    best = per_sample_log_loss(y[ev], p_best).mean()
    print(f"log loss gained over the blue-winrate prior by the best model: {base - best:.5f} nats/game")

    n_champ = b.n_champions
    counts = np.bincount(data.champs[b.tr].ravel(), minlength=n_champ + 1)[1:]
    # how often each champion was played in each role: pick distributions for lolwp-draft
    role_games = np.zeros((n_champ + 1, 5), dtype=np.int64)
    has_role = data.roles[b.tr] > 0
    np.add.at(role_games, (data.champs[b.tr][has_role], data.roles[b.tr][has_role] - 1), 1)
    # how often each champion was banned (lolwp-draft corrects pick shares for availability)
    ban_games, n_ban_games = np.zeros(n_champ + 1, dtype=np.int64), 0
    if data.bans is not None:
        stored = data.bans[b.tr]
        stored = stored[stored[:, 0] >= 0]
        n_ban_games = len(stored)
        unique = np.sort(stored, axis=1)           # count a champion once per game even if
                                                   # both teams banned it
        first = np.ones_like(unique, dtype=bool)
        first[:, 1:] = unique[:, 1:] != unique[:, :-1]
        np.add.at(ban_games, unique[first], 1)
    names = b.names
    checkpoint = {
        "format": 2,
        "config": model_config(b, hp),
        "state_dicts": states,
        "calibration": {"scale": cal_draft[0], "bias": cal_draft[1]},
        "calibration_players": (
            {"scale": cal_players[0], "bias": cal_players[1]} if cal_players else None),
        "calibration_solo": (
            {"scale": cal_solo[0], "bias": cal_solo[1]} if cal_solo else None),
        "player_features": list(PLAYER_FEATURES) if b.has_players else [],
        "champion_ids": data.champion_ids,
        "champion_names": {int(c): names[c]["name"] for c in data.champion_ids if c in names},
        "champion_keys": {int(c): names[c]["id"] for c in data.champion_ids if c in names},
        "champion_train_games": counts.tolist(),
        "champion_role_games": role_games[1:].tolist(),
        "champion_ban_games": ban_games[1:].tolist(),
        "result_means": (data.result_means.tolist() if data.result_means is not None else None),
        "ban_games": n_ban_games,
        "patches": data.patches,
        "n_matches": {"train": len(b.tr), "val": len(b.va), "test": len(b.te)},
        "evaluated_on": ev_name,
        "metrics": report,
        "logloss_diffs": diffs,
        "members": infos,
        "hyperparameters": hp,
        "args": {k: v for k, v in vars(args).items()},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)
    summary = {k: v for k, v in checkpoint.items()
               if k not in ("state_dicts", "champion_names", "champion_keys",
                            "champion_ids", "champion_train_games", "champion_role_games",
                            "champion_ban_games")}
    out.with_suffix(".metrics.json").write_text(json.dumps(summary, indent=1))
    print(f"\nsaved {out} and {out.with_suffix('.metrics.json')}")
    return summary


def main(argv=None) -> None:
    run(argv)


if __name__ == "__main__":
    main()
