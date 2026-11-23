"""Predict win probabilities for a draft.

    lolwp-predict --blue Aatrox "Lee Sin" Ahri Jinx Thresh --red Garen Vi Syndra Caitlyn Lux

Champions are given in role order TOP JUNGLE MIDDLE BOTTOM SUPPORT. Pass --no-roles if the
order is arbitrary. Python API:

    from lolwp.predict import Predictor
    p_blue = Predictor.load("models/draft_model.pt").predict(blue, red)
"""

from __future__ import annotations

import argparse
import difflib
import re

import numpy as np
import torch

from .model import DraftModel

LOW_DATA_GAMES = 50


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


class Predictor:
    def __init__(self, checkpoint: dict, device: str = "cpu"):
        if checkpoint.get("format") not in (1, 2):
            raise ValueError("unsupported checkpoint format")
        self.ckpt = checkpoint
        self.device = torch.device(device)
        self.models = []
        for state in checkpoint["state_dicts"]:
            m = DraftModel(**checkpoint["config"])
            m.load_state_dict(state)
            self.models.append(m.to(self.device).eval())
        self.scale = checkpoint["calibration"]["scale"]
        self.bias = checkpoint["calibration"]["bias"]
        # separate calibration for predictions that use player-history features (format 2)
        self.player_calibration = checkpoint.get("calibration_players")
        self.index = {c: i + 1 for i, c in enumerate(checkpoint["champion_ids"])}
        self.games = dict(zip(checkpoint["champion_ids"], checkpoint["champion_train_games"]))
        self.names = {int(k): v for k, v in checkpoint["champion_names"].items()}
        self.aliases: dict[str, int] = {}
        for cid in checkpoint["champion_ids"]:
            self.aliases[str(cid)] = cid
        for table in (checkpoint["champion_keys"], checkpoint["champion_names"]):
            for cid, name in table.items():
                self.aliases[_norm(name)] = int(cid)

    @classmethod
    def load(cls, path: str = "models/draft_model.pt", device: str = "cpu") -> "Predictor":
        return cls(torch.load(path, map_location="cpu", weights_only=True), device)

    def champion_id(self, name: str | int) -> int:
        key = _norm(str(name))
        if key in self.aliases:
            return self.aliases[key]
        close = difflib.get_close_matches(key, list(self.aliases), n=3, cutoff=0.6)
        hint = ", ".join(self.display(self.aliases[c]) for c in close)
        raise ValueError(f"unknown champion {name!r}" + (f" - did you mean: {hint}?" if hint else ""))

    def display(self, cid: int) -> str:
        return self.names.get(cid, str(cid))

    def low_data_champions(self, ids: list[int]) -> list[str]:
        return [self.display(c) for c in ids if self.games.get(c, 0) < LOW_DATA_GAMES]

    @torch.no_grad()
    def member_logits(self, champs: torch.Tensor, roles: torch.Tensor, feats=None, known=None,
                      batch: int = 16384, members: int | None = None) -> torch.Tensor:
        """Raw blue-win logits of every ensemble member (or the first `members`):
        [n_models, n_drafts].
        champs/roles are [n_drafts, 10] vocabulary / role indices, blue first. feats/known
        are optional player-history inputs (lolwp.data.PLAYER_FEATURES); without them the
        prediction is draft-only."""
        champs, roles = champs.to(self.device), roles.to(self.device)
        if feats is not None:
            feats, known = feats.to(self.device), known.to(self.device)

        def run(m, i):
            part = slice(i, i + batch)
            if feats is None:
                return m(champs[part], roles[part])
            return m(champs[part], roles[part], feats[part], known[part])

        return torch.stack([
            torch.cat([run(m, i) for i in range(0, len(champs), batch)])
            for m in self.models[:members]])

    @torch.no_grad()
    def predict(self, blue: list, red: list, roles_known: bool = True,
                player_feats=None, players_known=None) -> float:
        """P(blue side wins). blue/red: 5 champion names or ids each, in role order
        TOP, JUNGLE, MIDDLE, BOTTOM, SUPPORT (any order if roles_known=False).

        player_feats: optional [10, F] history features per pick (blue first), with
        players_known [10] marking the players they are available for."""
        if len(blue) != 5 or len(red) != 5:
            raise ValueError("each team needs exactly 5 champions")
        ids = [self.champion_id(c) for c in list(blue) + list(red)]
        if len(set(ids)) != 10:
            dupes = sorted({self.display(c) for c in ids if ids.count(c) > 1})
            raise ValueError(f"a champion can only be picked once: {', '.join(dupes)}")
        champs = torch.tensor([[self.index[c] for c in ids]])
        role_row = [1, 2, 3, 4, 5] * 2 if roles_known else [0] * 10
        roles = torch.tensor([role_row])
        scale, bias = self.scale, self.bias
        feats = known = None
        if player_feats is not None:
            if self.player_calibration is None:
                raise ValueError("this model was trained without player features")
            feats = torch.as_tensor(player_feats, dtype=torch.float32).reshape(1, 10, -1)
            known = (torch.ones(1, 10, dtype=torch.bool) if players_known is None
                     else torch.as_tensor(players_known, dtype=torch.bool).reshape(1, 10))
            scale, bias = self.player_calibration["scale"], self.player_calibration["bias"]
        logit = float(self.member_logits(champs, roles, feats, known).mean())
        return float(1.0 / (1.0 + np.exp(-(scale * logit + bias))))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blue", nargs=5, required=True, metavar="CHAMP")
    ap.add_argument("--red", nargs=5, required=True, metavar="CHAMP")
    ap.add_argument("--no-roles", action="store_true",
                    help="champions are not listed in TOP JUNGLE MIDDLE BOTTOM SUPPORT order")
    ap.add_argument("--model", default="models/draft_model.pt")
    args = ap.parse_args(argv)

    predictor = Predictor.load(args.model)
    try:
        p_blue = predictor.predict(args.blue, args.red, roles_known=not args.no_roles)
        ids = [predictor.champion_id(c) for c in args.blue + args.red]
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    blue, red = (", ".join(predictor.display(c) for c in team) for team in (ids[:5], ids[5:]))
    print(f"Blue ({blue}): {100 * p_blue:.1f}%")
    print(f"Red  ({red}): {100 * (1 - p_blue):.1f}%")
    print(f"[model trained on patch {', '.join(predictor.ckpt['patches'])}]")
    low = predictor.low_data_champions(ids)
    if low:
        print(f"note: fewer than {LOW_DATA_GAMES} training games for {', '.join(low)}; "
              "their effect is poorly estimated")


if __name__ == "__main__":
    main()
