"""Reference models the transformer has to beat.

* prior: always predict the training blue-side winrate.
* logistic regression on one-hot "champion on blue" / "champion on red" features plus
  champion-in-role features. This is the classic draft model (Conley & Perry 2013; Semenov
  et al. 2016) and captures individual champion strength but no synergies or counters.
* the same regression with team totals of the player-history features added, which shows how
  much of the player signal a linear model already gets.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from .metrics import per_sample_log_loss


def one_hot_features(champs: np.ndarray, roles: np.ndarray, n_champions: int) -> sparse.csr_matrix:
    """Columns: [blue champ | red champ | blue champ x role | red champ x role]."""
    n = champs.shape[0]
    v = n_champions + 1
    side = np.repeat(np.array([0, 1]), 5)[None, :]            # 0 blue, 1 red
    rows = np.repeat(np.arange(n), 10)
    cols_champ = (side * v + champs).ravel()
    has_role = (roles > 0).ravel()
    cols_role = (2 * v + side * 5 * v + (roles - 1) * v + champs).ravel()
    r = np.concatenate([rows, rows[has_role]])
    c = np.concatenate([cols_champ, cols_role[has_role]])
    return sparse.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(n, 12 * v))


def player_team_features(feats: np.ndarray, known: np.ndarray) -> sparse.csr_matrix:
    """[N, F + 1]: blue-minus-red team totals of each player feature, and of known players."""
    f = feats * known[..., None]
    diff = f[:, :5].sum(axis=1) - f[:, 5:].sum(axis=1)
    count = known[:, :5].sum(axis=1) - known[:, 5:].sum(axis=1)
    return sparse.csr_matrix(np.hstack([diff, count[:, None]]).astype(np.float32))


def fit_logistic(x_train, y_train, w_train, x_val, y_val,
                 grid=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3)):
    """L2 logistic regression; regularisation strength chosen on the validation set."""
    best = None
    for c in grid:
        clf = LogisticRegression(C=c, max_iter=2000)
        clf.fit(x_train, y_train, sample_weight=w_train)
        loss = per_sample_log_loss(y_val, clf.predict_proba(x_val)[:, 1]).mean()
        if best is None or loss < best[0]:
            best = (loss, c, clf)
    return best[2], best[1]
