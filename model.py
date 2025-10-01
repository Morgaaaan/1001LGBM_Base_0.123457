# model_LGBM.py
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


class ModelLGBM:
    """
    與你的訓練/推論流程相容：
      - fit(X_train, y_train, X_valid=None, y_valid=None, eval_metric="auc", early_stopping_rounds=200, verbose=50)
      - predict_proba(X) -> np.ndarray (shape [n,])
      - predict(X, threshold=0.5) -> {0,1}
      - save(path) / load(path)
    特色：
      - 若未指定 scale_pos_weight，fit() 會依訓練集自動計算 (#neg/#pos)
      - 自動記錄 best_iteration_ 供推論使用
    """

    def __init__(
        self,
        objective: str = "binary",
        learning_rate: float = 0.05,
        n_estimators: int = 5000,        # 配合 early stopping
        num_leaves: int = 63,
        max_depth: int = -1,
        min_child_samples: int = 100,    # 大資料抑制過擬合
        colsample_bytree: float = 0.7,
        subsample: float = 0.8,
        subsample_freq: int = 1,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        max_bin: int = 127,
        scale_pos_weight: float | None = None,  # None 時 fit() 會自動估
        random_state: int = 42,
        n_jobs: int = -1,
        device: str | None = None,       # "gpu" 或 None
    ):
        self.params = dict(
            objective=objective,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            subsample_freq=subsample_freq,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            max_bin=max_bin,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        if scale_pos_weight is not None:
            self.params["scale_pos_weight"] = scale_pos_weight
        if device is not None:
            self.params["device"] = device

        self.model = LGBMClassifier(**self.params)
        self.best_iteration_: int | None = None

    @staticmethod
    def _auto_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
        y = np.asarray(y)
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        return neg / max(1, pos)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
        eval_metric: str = "auc",
        early_stopping_rounds: int = 200,
        verbose: int = 50,
        **_,
    ):
        # 若未指定 scale_pos_weight，依訓練集自動估
        if "scale_pos_weight" not in self.model.get_params():
            spw = self._auto_scale_pos_weight(y_train)
            self.model.set_params(scale_pos_weight=spw)

        eval_set = None
        if X_valid is not None and y_valid is not None:
            eval_set = [(X_valid, y_valid)]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds if eval_set else None,
            callbacks=[],
            verbose=verbose if eval_set else -1,
        )

        # 記錄 best_iteration_
        if getattr(self.model, "best_iteration_", None):
            self.best_iteration_ = int(self.model.best_iteration_)
        else:
            self.best_iteration_ = int(self.model.n_estimators_)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # LightGBM sklearn API：predict_proba 回傳 [n,2]，第二欄是正類機率
        proba = self.model.predict_proba(X, num_iteration=self.best_iteration_)
        return proba[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            sklearn_model=self.model,
            best_iteration_=self.best_iteration_,
            params=self.params,
        )
        joblib.dump(payload, path)

    def load(self, path: str | Path):
        payload = joblib.load(path)
        self.model = payload["sklearn_model"]
        self.best_iteration_ = payload.get("best_iteration_", None)
        self.params = payload.get("params", self.params)
        return self
