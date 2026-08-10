import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tab_transformer_pytorch import FTTransformer, TabTransformer
from pytorch_tabnet.tab_model import TabNetClassifier as _TabNetClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_categorical_continuous(X, max_categories=20):
    """Split an already-encoded feature matrix into categorical vs continuous
    column indices, for models (FT-Transformer, TabTransformer, TabNet) that embed
    categorical columns separately instead of treating every feature as continuous.

    The rest of the pipeline (model.py) only produces a single numeric matrix
    (one-hot encoded categoricals + raw numeric columns), so categorical columns are
    recovered heuristically: non-negative, integer-valued columns with few unique
    values (e.g. one-hot dummies, FastingBS, Pregnancies) are treated as categorical;
    everything else (Age, Cholesterol, BMI, ...) is treated as continuous.
    """
    X = np.asarray(X, dtype=np.float64)
    categ_idx, cont_idx, cardinalities = [], [], []
    for col in range(X.shape[1]):
        values = X[:, col]
        uniques = np.unique(values)
        is_integer_like = np.allclose(uniques, np.round(uniques))
        if is_integer_like and uniques.min() >= 0 and 2 <= len(uniques) <= max_categories:
            categ_idx.append(col)
            cardinalities.append(int(uniques.max()) + 1)
        else:
            cont_idx.append(col)
    return categ_idx, cont_idx, cardinalities


class _TransformerTabularClassifier(BaseEstimator, ClassifierMixin):
    """Shared sklearn-style fit/predict/predict_proba wrapper around the raw
    torch.nn.Module transformer backbones from tab_transformer_pytorch, which
    otherwise only expose a bare forward(x_categ, x_cont) pass.

    fit() holds out a stratified validation split and does patience-based early
    stopping on validation loss, restoring the best-validation-loss weights
    rather than whatever the last epoch happened to produce — the same
    convergence check (validation split + patience, restore best) used for the
    project's BiLSTM/DistilBERT models, so a fixed epoch count is never silently
    trusted to mean "converged".
    """

    def __init__(self, categ_idx, cont_idx, cardinalities, dim=32, depth=3, heads=4,
                 max_epochs=200, patience=10, min_delta=1e-4, val_size=0.15,
                 batch_size=32, lr=1e-3, weight_decay=1e-5, random_state=42):
        self.categ_idx = categ_idx
        self.cont_idx = cont_idx
        self.cardinalities = cardinalities
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.val_size = val_size
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state

    def _build_backbone(self):
        raise NotImplementedError

    def _encode_categ(self, X):
        if not self.categ_idx:
            return np.zeros((X.shape[0], 0), dtype=np.int64)
        idx = np.round(X[:, self.categ_idx]).astype(np.int64)
        for i, card in enumerate(self.cardinalities):
            idx[:, i] = np.clip(idx[:, i], 0, card - 1)
        return idx

    def _encode_cont(self, X, normalize=True):
        if not self.cont_idx:
            return np.zeros((X.shape[0], 0), dtype=np.float32)
        cont = X[:, self.cont_idx].astype(np.float32)
        if normalize:
            cont = (cont - self.cont_mean_) / self.cont_std_
        return cont

    def _forward(self, x_categ, x_cont):
        return self.model_(x_categ, x_cont).squeeze(-1)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.unique(y)

        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=self.val_size, random_state=self.random_state, stratify=y,
        )

        raw_cont = X_fit[:, self.cont_idx].astype(np.float32) if self.cont_idx else np.zeros((len(X_fit), 0), dtype=np.float32)
        self.cont_mean_ = raw_cont.mean(axis=0)
        self.cont_std_ = raw_cont.std(axis=0) + 1e-6

        x_categ_fit, x_cont_fit = self._encode_categ(X_fit), self._encode_cont(X_fit)
        x_categ_val, x_cont_val = self._encode_categ(X_val), self._encode_cont(X_val)

        self.model_ = self._build_backbone().to(DEVICE)

        n_pos = (y_fit == 1).sum()
        n_neg = (y_fit == 0).sum()
        pos_weight = torch.tensor([n_neg / n_pos], device=DEVICE) if n_pos > 0 else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        dataset = TensorDataset(
            torch.tensor(x_categ_fit, dtype=torch.long),
            torch.tensor(x_cont_fit, dtype=torch.float32),
            torch.tensor(y_fit, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        x_categ_val_t = torch.tensor(x_categ_val, dtype=torch.long, device=DEVICE)
        x_cont_val_t = torch.tensor(x_cont_val, dtype=torch.float32, device=DEVICE)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        self.n_epochs_trained_ = 0
        self.stopped_early_ = False

        for epoch in range(self.max_epochs):
            self.model_.train()
            for xb_categ, xb_cont, yb in loader:
                xb_categ, xb_cont, yb = xb_categ.to(DEVICE), xb_cont.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self._forward(xb_categ, xb_cont), yb)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = criterion(self._forward(x_categ_val_t, x_cont_val_t), y_val_t).item()
            self.n_epochs_trained_ = epoch + 1

            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    self.stopped_early_ = True
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_val_loss_ = best_val_loss
        return self

    def _predict_pos_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        x_categ = self._encode_categ(X)
        x_cont = self._encode_cont(X)
        self.model_.eval()
        with torch.no_grad():
            logits = self._forward(
                torch.tensor(x_categ, dtype=torch.long, device=DEVICE),
                torch.tensor(x_cont, dtype=torch.float32, device=DEVICE),
            )
            return torch.sigmoid(logits).cpu().numpy()

    def predict_proba(self, X):
        pos = self._predict_pos_proba(X)
        return np.stack([1 - pos, pos], axis=1)

    def predict(self, X):
        return (self._predict_pos_proba(X) >= 0.5).astype(int)


class FTTransformerClassifier(_TransformerTabularClassifier):
    def _build_backbone(self):
        return FTTransformer(
            categories=tuple(self.cardinalities), num_continuous=len(self.cont_idx),
            dim=self.dim, depth=self.depth, heads=self.heads, dim_out=1,
            attn_dropout=0.1, ff_dropout=0.1,
        )


class TabTransformerClassifier(_TransformerTabularClassifier):
    def _build_backbone(self):
        return TabTransformer(
            categories=tuple(self.cardinalities), num_continuous=len(self.cont_idx),
            dim=self.dim, depth=self.depth, heads=self.heads, dim_out=1,
            attn_dropout=0.1, ff_dropout=0.1, mlp_act=nn.ReLU(),
        )


class TabNetClassifier(_TabNetClassifier):
    """pytorch-tabnet's TabNetClassifier only accepts raw numpy arrays, while the
    rest of the pipeline passes pandas DataFrames/Series — coerce at the boundary
    so it drops into the same generic fit/predict/predict_proba call sites."""

    def fit(self, X, y, **kwargs):
        return super().fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), **kwargs)

    def predict(self, X):
        return super().predict(np.asarray(X, dtype=np.float32))

    def predict_proba(self, X):
        return super().predict_proba(np.asarray(X, dtype=np.float32))


def make_tabnet(categ_idx, cardinalities, random_state=42):
    return TabNetClassifier(cat_idxs=categ_idx, cat_dims=cardinalities, cat_emb_dim=1,
                             seed=random_state, verbose=0)


def fit_tabnet_with_early_stopping(model, X, y, max_epochs=300, patience=20,
                                    batch_size=32, virtual_batch_size=16,
                                    val_size=0.15, random_state=42):
    """TabNetClassifier has native eval_set/patience support, but it's a no-op
    unless an eval_set is actually passed (pytorch-tabnet just runs the full
    max_epochs otherwise, silently). This carves out the validation split and
    wires it in, so TabNet gets the same "stop once validation loss stops
    improving" convergence check as the other two deep models instead of a
    fixed epoch budget."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state, stratify=y,
    )
    model.fit(
        X_fit, y_fit, eval_set=[(X_val, y_val)], eval_name=["val"], eval_metric=["logloss"],
        max_epochs=max_epochs, patience=patience, batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
    )
    model.n_epochs_trained_ = len(model.history["loss"])
    model.stopped_early_ = model.n_epochs_trained_ < max_epochs
    return model


def manual_cv_f1(model_factory, X, y, cv, fit_fn=None):
    """StratifiedKFold F1 scoring via manually-rebuilt estimators, for models
    (like TabNetClassifier) whose __init__ mutates its own params and therefore
    breaks sklearn.base.clone(), which cross_val_score relies on internally."""
    X = np.asarray(X)
    y = np.asarray(y)
    fit_fn = fit_fn or (lambda model, X_tr, y_tr: model.fit(X_tr, y_tr))
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        model = model_factory()
        fit_fn(model, X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        scores.append(f1_score(y[val_idx], preds))
    return np.array(scores)
