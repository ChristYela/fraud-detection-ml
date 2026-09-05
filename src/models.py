"""
models.py
==========
Definição e treinamento de modelos para detecção de fraudes.

Suporta:
    - Modelos supervisionados (LogisticRegression, RandomForest, XGBoost)
    - Modelos não-supervisionados (IsolationForest, LOF, Autoencoder)
    - Ensemble híbrido (IsolationForest + XGBoost em cascata)

Uso:
    from src.models import FraudDetector

    detector = FraudDetector(model_type="xgboost")
    detector.fit(X_train, y_train)
    y_pred = detector.predict(X_test)
    y_prob = detector.predict_proba(X_test)
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List
import warnings

warnings.filterwarnings("ignore")

# sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# xgboost
from xgboost import XGBClassifier

# tensorflow
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping


class FraudDetector:
    """
    Wrapper unificado para treinamento e predição de modelos
    de detecção de fraudes.

    Parameters
    ----------
    model_type : str
        Tipo de modelo:
        - "logistic": LogisticRegression
        - "random_forest": RandomForestClassifier
        - "xgboost": XGBClassifier
        - "isolation_forest": IsolationForest
        - "lof": LocalOutlierFactor
        - "one_class_svm": OneClassSVM
        - "autoencoder": Autoencoder (Keras)
        - "hybrid": IsolationForest + XGBoost em cascata
    contamination : float, default=0.01
        Proporção esperada de anomalias (para modelos não-supervisionados).
    random_state : int, default=42
    **kwargs : dict
        Parâmetros adicionais passados ao modelo base.

    Attributes
    ----------
    model_ : object
        Instância do modelo treinado.
    is_fitted_ : bool
        Flag indicando se o modelo foi treinado.
    """

    VALID_MODELS = [
        "logistic",
        "random_forest",
        "xgboost",
        "isolation_forest",
        "lof",
        "one_class_svm",
        "autoencoder",
        "hybrid",
    ]

    def __init__(
        self,
        model_type: str = "xgboost",
        contamination: float = 0.01,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        if model_type not in self.VALID_MODELS:
            raise ValueError(
                f"model_type deve ser um de: {self.VALID_MODELS}"
            )

        self.model_type = model_type
        self.contamination = contamination
        self.random_state = random_state
        self.kwargs = kwargs

        self.model_: Optional[Any] = None
        self.secondary_model_: Optional[Any] = None  # Para hybrid
        self.is_fitted_: bool = False
        self._threshold: Optional[float] = None
        self._input_dim: Optional[int] = None

    def _build_model(self, scale_pos_weight: Optional[float] = None) -> Any:
        """Constrói a instância do modelo base."""

        if self.model_type == "logistic":
            return LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state,
                **self.kwargs,
            )

        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.kwargs.get("n_estimators", 200),
                max_depth=self.kwargs.get("max_depth", 10),
                class_weight="balanced_subsample",
                random_state=self.random_state,
                n_jobs=-1,
            )

        elif self.model_type == "xgboost":
            return XGBClassifier(
                n_estimators=self.kwargs.get("n_estimators", 300),
                max_depth=self.kwargs.get("max_depth", 6),
                learning_rate=self.kwargs.get("learning_rate", 0.1),
                scale_pos_weight=scale_pos_weight or 1.0,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
                use_label_encoder=False,
            )

        elif self.model_type == "isolation_forest":
            return IsolationForest(
                contamination=self.contamination,
                n_estimators=self.kwargs.get("n_estimators", 300),
                random_state=self.random_state,
                n_jobs=-1,
            )

        elif self.model_type == "lof":
            return LocalOutlierFactor(
                n_neighbors=self.kwargs.get("n_neighbors", 20),
                contamination=self.contamination,
                novelty=True,
                n_jobs=-1,
            )

        elif self.model_type == "one_class_svm":
            return OneClassSVM(
                gamma=self.kwargs.get("gamma", "auto"),
                nu=self.contamination,
            )

        elif self.model_type == "autoencoder":
            return None  # Construído no fit

        elif self.model_type == "hybrid":
            return IsolationForest(
                contamination=self.contamination,
                n_estimators=200,
                random_state=self.random_state,
                n_jobs=-1,
            )

    def _build_autoencoder(self, input_dim: int) -> Sequential:
        """Constrói a arquitetura do autoencoder."""
        encoding_dim = self.kwargs.get("encoding_dim", 8)
        dropout = self.kwargs.get("dropout", 0.2)

        model = Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(16, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(encoding_dim, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(input_dim, activation="linear"),
        ])

        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        validation_split: float = 0.1,
        epochs: int = 80,
        batch_size: int = 256,
        verbose: int = 0,
    ) -> "FraudDetector":
        """
        Treina o modelo.

        Parameters
        ----------
        X_train : np.ndarray
            Dados de treino.
        y_train : np.ndarray, optional
            Labels (necessário para supervisionados).
        validation_split : float, default=0.1
            Para autoencoder.
        epochs : int, default=80
            Para autoencoder.
        batch_size : int, default=256
            Para autoencoder.
        verbose : int, default=0
            Verbosity do treinamento.

        Returns
        -------
        FraudDetector
            Instância ajustada.
        """
        print(f"🔹 Treinando {self.model_type}...")

        # Modelos supervisionados
        if self.model_type in ["logistic", "random_forest", "xgboost"]:
            if y_train is None:
                raise ValueError(f"{self.model_type} requer y_train")

            if self.model_type == "xgboost":
                scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
                self.model_ = self._build_model(scale_pos_weight=scale_pos)
            else:
                self.model_ = self._build_model()

            self.model_.fit(X_train, y_train)

        # Modelos não-supervisionados
        elif self.model_type in ["isolation_forest", "lof", "one_class_svm"]:
            self.model_ = self._build_model()
            self.model_.fit(X_train)

        # Autoencoder
        elif self.model_type == "autoencoder":
            self._input_dim = X_train.shape[1]
            self.model_ = self._build_autoencoder(self._input_dim)

            # Treinar apenas com dados normais
            if y_train is not None:
                X_normal = X_train[y_train == 0]
            else:
                X_normal = X_train

            self.model_.fit(
                X_normal,
                X_normal,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=verbose,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            )

        # Hybrid: Isolation Forest + XGBoost
        elif self.model_type == "hybrid":
            if y_train is None:
                raise ValueError("Hybrid requer y_train")

            # Passo 1: Isolation Forest filtra normais óbvios
            self.model_ = self._build_model()
            self.model_.fit(X_train)

            iso_scores = self.model_.score_samples(X_train)
            # Manter apenas as amostras mais suspeitas para o XGBoost
            threshold_iso = np.percentile(iso_scores, 20)  # Bottom 20%
            mask_suspect = iso_scores <= threshold_iso

            X_suspect = X_train[mask_suspect]
            y_suspect = y_train[mask_suspect]

            print(f"   🎯 Hybrid: {mask_suspect.sum()} amostras suspeitas para XGBoost")

            # Passo 2: XGBoost nas suspeitas
            scale_pos = (y_suspect == 0).sum() / max((y_suspect == 1).sum(), 1)
            self.secondary_model_ = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
                use_label_encoder=False,
            )
            self.secondary_model_.fit(X_suspect, y_suspect)

        self.is_fitted_ = True
        print(f"✅ {self.model_type} treinado com sucesso!")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna predições binárias (0 = normal, 1 = fraude).

        Parameters
        ----------
        X : np.ndarray
            Dados de entrada.

        Returns
        -------
        np.ndarray
            Vetor de predições.
        """
        if not self.is_fitted_:
            raise RuntimeError("Chame fit() antes de predict()")

        if self.model_type in ["logistic", "random_forest", "xgboost"]:
            return self.model_.predict(X)

        elif self.model_type in ["isolation_forest", "lof", "one_class_svm"]:
            return (self.model_.predict(X) == -1).astype(int)

        elif self.model_type == "autoencoder":
            reconstructions = self.model_.predict(X, verbose=0)
            mse = np.mean((X - reconstructions) ** 2, axis=1)
            if self._threshold is None:
                self._threshold = np.percentile(mse, 99)
            return (mse > self._threshold).astype(int)

        elif self.model_type == "hybrid":
            iso_scores = self.model_.score_samples(X)
            threshold_iso = np.percentile(iso_scores, 20)
            mask_suspect = iso_scores <= threshold_iso

            y_pred = np.zeros(len(X), dtype=int)
            if mask_suspect.sum() > 0:
                y_pred[mask_suspect] = self.secondary_model_.predict(X[mask_suspect])
            return y_pred

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Retorna scores/probabilidades de fraude.

        Parameters
        ----------
        X : np.ndarray
            Dados de entrada.

        Returns
        -------
        np.ndarray ou None
            Scores de anomalia (quanto maior, mais anômalo).
        """
        if not self.is_fitted_:
            raise RuntimeError("Chame fit() antes de predict_proba()")

        if self.model_type in ["logistic", "random_forest", "xgboost"]:
            return self.model_.predict_proba(X)[:, 1]

        elif self.model_type == "isolation_forest":
            return -self.model_.score_samples(X)

        elif self.model_type == "lof":
            return -self.model_.score_samples(X)

        elif self.model_type == "one_class_svm":
            return -self.model_.score_samples(X)

        elif self.model_type == "autoencoder":
            reconstructions = self.model_.predict(X, verbose=0)
            return np.mean((X - reconstructions) ** 2, axis=1)

        elif self.model_type == "hybrid":
            # Retorna score combinado
            iso_scores = -self.model_.score_samples(X)
            proba = np.zeros(len(X))
            mask_suspect = iso_scores >= np.percentile(iso_scores, 80)
            if mask_suspect.sum() > 0:
                proba[mask_suspect] = self.secondary_model_.predict_proba(X[mask_suspect])[:, 1]
            return proba

        return None

    def get_model(self) -> Any:
        """Retorna a instância do modelo base."""
        return self.model_


# ============================================================
# EXEMPLO DE USO
# ============================================================
if __name__ == "__main__":
    from src.data_generator import generate_dataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Dados
    df = generate_dataset(n_samples=5000, fraud_ratio=0.02, random_state=42)
    X = df.drop("fraude", axis=1)
    y = df["fraude"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Testar 3 modelos
    for model_name in ["xgboost", "isolation_forest", "autoencoder"]:
        print(f"
{'='*50}")
        detector = FraudDetector(model_type=model_name, contamination=0.02)
        detector.fit(X_train_s, y_train.values, epochs=30, verbose=0)
        y_pred = detector.predict(X_test_s)
        print(f"Predições: {np.bincount(y_pred)}")
