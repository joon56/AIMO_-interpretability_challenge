"""Repeated grouped K-fold for tiny labelled sets (leave-one-group-out is strongly biased on weak features)."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def grouped_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int = 5, reps: int = 20, C: float = 1.0):
    """Class-balanced logistic regression; returns (mean AUC, mean balanced acc, mean acc, OOF prob averaged)."""
    ug = np.unique(groups)
    aucs, baccs, accs, probs = [], [], [], np.zeros(len(y))
    for r in range(reps):
        perm = np.random.RandomState(r).permutation(ug)
        fold_of = {g: i % k for i, g in enumerate(perm)}
        fold = np.array([fold_of[g] for g in groups])
        prob = np.zeros(len(y))
        for f in range(k):
            te, trn = fold == f, fold != f
            if len(np.unique(y[trn])) < 2:
                prob[te] = y[trn].mean()
                continue
            lr = LogisticRegression(C=C, class_weight="balanced").fit(X[trn], y[trn])
            prob[te] = lr.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        aucs.append(roc_auc_score(y, prob))
        baccs.append(0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean()))
        accs.append((pred == y).mean())
        probs += prob / reps
    return float(np.mean(aucs)), float(np.mean(baccs)), float(np.mean(accs)), probs
