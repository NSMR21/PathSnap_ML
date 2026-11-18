"""Model zoo for the candidate classifier.

Each entry is the *untuned* starting configuration. The parameters are kept
in a plain dict so they can be logged next to the results, and so the
Bayesian search in the next stage can start from the same definitions.
"""
import copy
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import SEED


class RowCapped(BaseEstimator, ClassifierMixin):
    """Fit the wrapped model on at most `max_rows` random rows.

    Used for KNN: prediction cost grows with the number of stored rows, and
    1M stored rows would make every query a brute-force scan.
    """

    def __init__(self, estimator=None, max_rows=120_000, random_state=SEED):
        self.estimator = estimator
        self.max_rows = max_rows
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        if len(X) > self.max_rows:
            rng = np.random.default_rng(self.random_state)
            keep = rng.choice(len(X), self.max_rows, replace=False)
            X, y = X[keep], y[keep]
        self.model_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


class EpochMLP(BaseEstimator, ClassifierMixin):
    """MLP trained epoch by epoch so we get real train AND validation loss curves.

    sklearn's own early stopping only reports validation *accuracy*; here a
    random 10% of the training rows is held out, log loss is measured on it
    after every epoch and the best epoch's weights are kept.
    """

    def __init__(self, hidden=(64, 32), alpha=1e-4, lr=1e-3, batch_size=1024,
                 max_epochs=30, patience=4, val_share=0.1, random_state=SEED):
        self.hidden = hidden
        self.alpha = alpha
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_share = val_share
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y).astype(int)
        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(len(X))
        n_val = max(1, int(len(X) * self.val_share))
        va, tr = order[:n_val], order[n_val:]
        self.scaler_ = StandardScaler().fit(X[tr])
        x_tr, x_va = self.scaler_.transform(X[tr]), self.scaler_.transform(X[va])

        net = MLPClassifier(hidden_layer_sizes=self.hidden, alpha=self.alpha,
                            learning_rate_init=self.lr, batch_size=self.batch_size,
                            shuffle=True, random_state=self.random_state)
        best, best_net, wait = np.inf, None, 0
        self.history_ = []
        for epoch in range(1, self.max_epochs + 1):
            net.partial_fit(x_tr, y[tr], classes=np.array([0, 1]))
            val_loss = log_loss(y[va], net.predict_proba(x_va)[:, 1], labels=[0, 1])
            self.history_.append({"epoch": epoch, "train_loss": float(net.loss_), "val_loss": float(val_loss)})
            if val_loss < best - 1e-4:
                best, best_net, wait = val_loss, copy.deepcopy(net), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net_ = best_net
        self.best_epoch_ = int(np.argmin([h["val_loss"] for h in self.history_]) + 1)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.net_.predict_proba(self.scaler_.transform(np.asarray(X, dtype=np.float64)))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@dataclass
class ModelSpec:
    key: str
    name: str
    family: str
    params: dict = field(default_factory=dict)

    def build(self, **overrides):
        p = {**self.params, **overrides}
        return _BUILDERS[self.key](p)


_BUILDERS = {
    "logreg": lambda p: make_pipeline(StandardScaler(), LogisticRegression(
        C=p["C"], max_iter=p["max_iter"], random_state=SEED)),
    "gnb": lambda p: make_pipeline(
        QuantileTransformer(output_distribution="normal", n_quantiles=p["n_quantiles"], random_state=SEED),
        GaussianNB(var_smoothing=p["var_smoothing"])),
    "knn": lambda p: RowCapped(make_pipeline(StandardScaler(), KNeighborsClassifier(
        n_neighbors=p["n_neighbors"], weights=p["weights"], n_jobs=-1)), max_rows=p["max_rows"]),
    "dtree": lambda p: DecisionTreeClassifier(
        max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], random_state=SEED),
    "rf": lambda p: RandomForestClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
        max_features=p["max_features"], n_jobs=-1, random_state=SEED),
    "extratrees": lambda p: ExtraTreesClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
        max_features=p["max_features"], n_jobs=-1, random_state=SEED),
    "hgb": lambda p: HistGradientBoostingClassifier(
        learning_rate=p["learning_rate"], max_iter=p["max_iter"], max_leaf_nodes=p["max_leaf_nodes"],
        min_samples_leaf=p["min_samples_leaf"], l2_regularization=p["l2_regularization"],
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, random_state=SEED),
    "mlp": lambda p: EpochMLP(hidden=tuple(p["hidden"]), alpha=p["alpha"], lr=p["lr"],
                              batch_size=p["batch_size"], max_epochs=p["max_epochs"], patience=p["patience"]),
}

MODEL_ZOO = {
    "logreg": ModelSpec("logreg", "Logistic Regression", "Linear / probabilistic",
                        {"C": 1.0, "max_iter": 1000}),
    "gnb": ModelSpec("gnb", "Gaussian Naive Bayes (quantile-normalised)", "Linear / probabilistic",
                     {"n_quantiles": 500, "var_smoothing": 1e-9}),
    "knn": ModelSpec("knn", "Distance-weighted KNN", "Instance-based",
                     {"n_neighbors": 35, "weights": "distance", "max_rows": 120_000}),
    "dtree": ModelSpec("dtree", "Decision Tree", "Tree-based",
                       {"max_depth": 12, "min_samples_leaf": 50}),
    "rf": ModelSpec("rf", "Random Forest", "Ensemble (bagging)",
                    {"n_estimators": 150, "max_depth": 20, "min_samples_leaf": 3, "max_features": "sqrt"}),
    "extratrees": ModelSpec("extratrees", "Extra Trees", "Ensemble (randomised trees)",
                            {"n_estimators": 200, "max_depth": 22, "min_samples_leaf": 2, "max_features": "sqrt"}),
    "hgb": ModelSpec("hgb", "Histogram Gradient Boosting", "Boosting",
                     {"learning_rate": 0.1, "max_iter": 400, "max_leaf_nodes": 31,
                      "min_samples_leaf": 40, "l2_regularization": 0.0}),
    "mlp": ModelSpec("mlp", "MLP (64-32, ReLU)", "Neural network",
                     {"hidden": [64, 32], "alpha": 1e-4, "lr": 1e-3, "batch_size": 1024,
                      "max_epochs": 30, "patience": 4}),
}
