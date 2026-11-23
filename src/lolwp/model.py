"""Set-transformer draft model.

A draft is a set of 10 (champion, role, team[, player history]) tokens. A small pre-norm
transformer encoder lets every champion attend to every ally (synergy) and enemy (counter),
without the hand-built pairwise features older draft models needed.

Antisymmetric scoring (default)
    The encoder scores a team *from its own perspective*: s(own | opponent). The blue-win
    logit is

        logit = s(blue | red) - s(red | blue) + side_bias + sum_c side_effect[c]

    The first two terms flip sign exactly when the teams swap sides, which is what "how good
    is this draft into that draft" must do; every game therefore teaches the network about
    both teams with shared weights. The last two terms are the only side-dependent parts:
    the global blue-side advantage and a per-champion correction to it.

Optional parts (all off by default, so checkpoints of the first version still load)
    lr_offset        A frozen logistic regression on champion / champion-in-role one-hots is
                     added to the logit and the network's output layer starts at zero: the
                     network begins exactly at the baseline and only learns a correction.
    n_player_feats   Per-player history features are embedded and added to that player's
                     token. Unknown players get a learned "no information" embedding, which is
                     also what feats=None (draft-only prediction) uses for everyone.
    n_aux_anti / aux_duration
                     Extra outputs trained to predict end-of-game margins (gold, kills, towers:
                     antisymmetric like the logit) and game length (symmetric). They exist only
                     to give the encoder a richer training signal than one win/loss bit.
"""

from __future__ import annotations

import torch
from torch import nn


def swap_sides(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x[:, 5:], x[:, :5]], dim=1)


class DraftModel(nn.Module):
    def __init__(
        self,
        n_champions: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 128,
        dropout: float = 0.2,
        antisymmetric: bool = True,
        lr_offset: bool = False,
        n_player_feats: int = 0,
        n_aux_anti: int = 0,
        aux_duration: bool = False,
    ):
        super().__init__()
        self.antisymmetric = antisymmetric
        self.n_aux_anti = n_aux_anti
        self.aux_duration = aux_duration
        self.vocab = n_champions + 1
        self.champ = nn.Embedding(self.vocab, d_model)       # 0 = unknown champion
        self.role = nn.Embedding(6, d_model)                 # 0 = unknown role, 1..5 = TOP..UTILITY
        self.team = nn.Embedding(2, d_model)                 # 0 = own/blue, 1 = opponent/red
        self.emb_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, n_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        n_out = 1 + n_aux_anti + int(aux_duration)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_out)
        )
        self.side_bias = nn.Parameter(torch.zeros(1))
        self.side_effect = nn.Embedding(self.vocab, 1)
        nn.init.normal_(self.champ.weight, std=0.1)
        nn.init.normal_(self.role.weight, std=0.1)
        nn.init.normal_(self.team.weight, std=0.1)
        nn.init.zeros_(self.side_effect.weight)
        self.register_buffer("team_ids", torch.tensor([0] * 5 + [1] * 5), persistent=False)

        if n_player_feats:
            self.player = nn.Sequential(
                nn.Linear(n_player_feats, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            self.no_player = nn.Parameter(torch.zeros(d_model))
        else:
            self.player = None
        self.has_lr = lr_offset
        if lr_offset:
            self.register_buffer("lr_weight", torch.zeros(12 * self.vocab))
            self.register_buffer("lr_bias", torch.zeros(1))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def set_lr(self, coef, intercept: float) -> None:
        """Install logistic-regression weights laid out as baselines.one_hot_features."""
        self.lr_weight.copy_(torch.as_tensor(coef, dtype=torch.float32).reshape(-1))
        self.lr_bias.fill_(float(intercept))

    def lr_logit(self, champs: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
        v = self.vocab
        side = self.team_ids
        logit = self.lr_weight[side * v + champs].sum(dim=1) + self.lr_bias
        has_role = roles > 0
        idx = 2 * v + side * 5 * v + (roles - 1).clamp(min=0) * v + champs
        return logit + (self.lr_weight[idx] * has_role).sum(dim=1)

    def _score(self, champs, roles, feats, known) -> torch.Tensor:
        """Inputs [B, 10, ...], first five tokens are the team being scored. Returns [B, n_out]."""
        x = self.champ(champs) + self.role(roles) + self.team(self.team_ids)
        if self.player is not None:
            if feats is None:
                x = x + self.no_player
            else:
                x = x + torch.where(known.unsqueeze(-1), self.player(feats), self.no_player)
        h = self.encoder(self.emb_drop(x))
        pooled = torch.cat([h[:, :5].mean(dim=1), h[:, 5:].mean(dim=1)], dim=-1)
        return self.head(pooled)

    def forward(self, champs, roles, feats=None, known=None, return_aux: bool = False):
        """Logit of P(blue wins). champs/roles: [B, 10] with columns 0-4 blue, 5-9 red;
        feats [B, 10, F] / known [B, 10] are optional player-history inputs.
        With return_aux: (logit, aux [B, n_aux_anti + aux_duration])."""
        if self.antisymmetric:
            b = champs.shape[0]
            both = lambda x: None if x is None else torch.cat([x, swap_sides(x)])  # noqa: E731
            s = self._score(both(champs), both(roles), both(feats), both(known))
            own, opp = s[:b], s[b:]
            logit = (own[:, 0] - opp[:, 0] + self.side_bias
                     + self.side_effect(champs).sum(dim=(1, 2)))
            k = 1 + self.n_aux_anti
            aux = torch.cat([own[:, 1:k] - opp[:, 1:k], own[:, k:] + opp[:, k:]], dim=1)
        else:
            s = self._score(champs, roles, feats, known)
            logit, aux = s[:, 0] + self.side_bias, s[:, 1:]
        if self.has_lr:
            logit = logit + self.lr_logit(champs, roles)
        return (logit, aux) if return_aux else logit
