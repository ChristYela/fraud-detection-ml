"""
preprocessing.py
==================
Pipeline de pré-processamento para dados de transações financeiras.

Inclui:
    - Divisão treino/teste estratificada
    - Normalização (StandardScaler / RobustScaler)
    - Feature engineering (ratios, agregações temporais)
    - Detecção e tratamento de outliers

Uso:
    from src.preprocessing import PreprocessingPipeline

    pipe = PreprocessingPipeline(scaler_type="robust")
    X_train, X_test, y_train, y_test = pipe.fit_transform(df, target_col="fraude")
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from typing import Optional, Tuple, List, Union
import warnings

warnings.filterwarnings("ignore")


class PreprocessingPipeline:
    """
    Pipeline completo de pré-processamento para detecção de fraudes.

    Parameters
    ----------
    test_size : float, default=0.2
        Proporção do conjunto de teste.
    random_state : int, default=42
        Semente para reprodutibilidade.
    scaler_type : str, default="standard"
        Tipo de scaler: "standard", "robust" ou "minmax".
    stratify : bool, default=True
        Se True, mantém proporção de classes no split.
    apply_feature_engineering : bool, default=True
        Se True, cria features derivadas.
    outlier_method : str, optional
        Método para tratar outliers: "iqr", "zscore" ou None.
    outlier_threshold : float, default=3.0
        Threshold para remoção de outliers (apenas para zscore).

    Attributes
    ----------
    scaler_ : object
        Instância do scaler ajustado.
    feature_names_ : list
        Lista final de nomes das features após engineering.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        scaler_type: str = "standard",
        stratify: bool = True,
        apply_feature_engineering: bool = True,
        outlier_method: Optional[str] = None,
        outlier_threshold: float = 3.0,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.scaler_type = scaler_type
        self.stratify = stratify
        self.apply_feature_engineering = apply_feature_engineering
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold

        self.scaler_: Optional[Union[StandardScaler, RobustScaler, MinMaxScaler]] = None
        self.feature_names_: List[str] = []
        self.target_col_: Optional[str] = None
        self._fitted: bool = False

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features derivadas (feature engineering).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame original.

        Returns
        -------
        pd.DataFrame
            DataFrame com novas features.
        """
        df = df.copy()

        # 1. Razão valor_transacao / valor_medio_30d
        if "valor_transacao" in df.columns and "valor_medio_30d" in df.columns:
            df["razao_valor_medio"] = df["valor_transacao"] / (
                df["valor_medio_30d"] + 1e-6
            )

        # 2. Velocidade de transações (transações por dia)
        if "num_transacoes_7d" in df.columns:
            df["velocidade_transacoes"] = df["num_transacoes_7d"] / 7.0

        # 3. Score comportamental: score_credito normalizado invertido
        if "score_credito" in df.columns:
            df["risco_credito_invertido"] = 850 - df["score_credito"]

        # 4. Razão dívida / renda
        if "divida_total" in df.columns and "renda_estimada" in df.columns:
            df["razao_divida_renda"] = df["divida_total"] / (
                df["renda_estimada"] + 1e-6
            )

        # 5. Transação fora do horário comercial (22h - 6h)
        if "hora_dia" in df.columns:
            df["horario_risco"] = (
                (df["hora_dia"] >= 22) | (df["hora_dia"] <= 6)
            ).astype(int)

        # 6. Score combinado de risco
        risk_cols = ["pais_risco", "merchant_risco"]
        if all(col in df.columns for col in risk_cols):
            df["score_risco_combinado"] = df[risk_cols].mean(axis=1)

        # 7. Interação: valor alto + dispositivo novo
        if "valor_transacao" in df.columns and "dispositivo_novo" in df.columns:
            valor_q90 = df["valor_transacao"].quantile(0.90)
            df["alto_valor_dispositivo_novo"] = (
                (df["valor_transacao"] > valor_q90) & (df["dispositivo_novo"] == 1)
            ).astype(int)

        # 8. Sessão muito curta (< 30 segundos)
        if "tempo_sessao_seg" in df.columns:
            df["sessao_muito_curta"] = (df["tempo_sessao_seg"] < 30).astype(int)

        return df

    def _remove_outliers(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Remove outliers do conjunto de treino.

        Parameters
        ----------
        X : pd.DataFrame
            Features.
        y : pd.Series
            Target.

        Returns
        -------
        tuple
            (X_filtrado, y_filtrado)
        """
        if self.outlier_method is None:
            return X, y

        mask = pd.Series(True, index=X.index)

        if self.outlier_method == "iqr":
            for col in X.select_dtypes(include=[np.number]).columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                mask &= (X[col] >= lower) & (X[col] <= upper)

        elif self.outlier_method == "zscore":
            from scipy import stats

            z_scores = np.abs(stats.zscore(X.select_dtypes(include=[np.number])))
            mask = (z_scores < self.outlier_threshold).all(axis=1)

        removed = (~mask).sum()
        if removed > 0:
            print(f"   🧹 Outliers removidos: {removed} amostras ({removed/len(X)*100:.2f}%)")

        return X[mask], y[mask]

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "fraude",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Executa o pipeline completo de pré-processamento.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame com features e target.
        target_col : str, default="fraude"
            Nome da coluna target.

        Returns
        -------
        tuple
            (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        print("=" * 60)
        print("🔧 PIPELINE DE PRÉ-PROCESSAMENTO")
        print("=" * 60)

        self.target_col_ = target_col

        # 1. Separar target
        if target_col not in df.columns:
            raise ValueError(f"Coluna target '{target_col}' não encontrada no DataFrame")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        print(f"📊 Dataset original: {len(X)} amostras, {len(X.columns)} features")

        # 2. Feature engineering
        if self.apply_feature_engineering:
            X = self._create_features(X)
            print(f"✨ Features criadas: {len(X.columns)} total")

        # 3. Divisão treino/teste
        stratify_param = y if self.stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param,
        )
        print(f"📦 Split: treino={len(X_train)}, teste={len(X_test)}")

        # 4. Remover outliers do treino (se configurado)
        X_train, y_train = self._remove_outliers(X_train, y_train)

        # 5. Normalização
        if self.scaler_type == "standard":
            self.scaler_ = StandardScaler()
        elif self.scaler_type == "robust":
            self.scaler_ = RobustScaler()
        elif self.scaler_type == "minmax":
            self.scaler_ = MinMaxScaler()
        else:
            raise ValueError("scaler_type deve ser 'standard', 'robust' ou 'minmax'")

        X_train_scaled = self.scaler_.fit_transform(X_train)
        X_test_scaled = self.scaler_.transform(X_test)

        self.feature_names_ = list(X_train.columns)
        self._fitted = True

        print(f"⚖️  Fraudas no treino: {y_train.sum()} / {len(y_train)} ({y_train.mean()*100:.2f}%)")
        print(f"⚖️  Fraudas no teste:  {y_test.sum()} / {len(y_test)} ({y_test.mean()*100:.2f}%)")
        print(f"✅ Scaler: {self.scaler_type}")
        print("=" * 60)

        return X_train_scaled, X_test_scaled, y_train.values, y_test.values

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Aplica o pré-processamento em novos dados (após fit).

        Parameters
        ----------
        df : pd.DataFrame
            Novos dados com as mesmas features.

        Returns
        -------
        np.ndarray
            Dados normalizados.
        """
        if not self._fitted:
            raise RuntimeError("Chame fit_transform() antes de transform()")

        df = df.copy()

        if self.apply_feature_engineering:
            df = self._create_features(df)

        # Garantir mesmas colunas
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0

        df = df[self.feature_names_]
        return self.scaler_.transform(df)

    def get_feature_names(self) -> List[str]:
        """Retorna a lista de nomes das features após pré-processamento."""
        if not self._fitted:
            raise RuntimeError("Chame fit_transform() primeiro")
        return self.feature_names_


# ============================================================
# EXEMPLO DE USO
# ============================================================
if __name__ == "__main__":
    from src.data_generator import generate_dataset

    # Gerar dados
    df = generate_dataset(n_samples=5000, fraud_ratio=0.02, random_state=42)

    # Pipeline completo
    pipe = PreprocessingPipeline(
        test_size=0.2,
        scaler_type="robust",
        apply_feature_engineering=True,
        outlier_method="iqr",
    )

    X_train, X_test, y_train, y_test = pipe.fit_transform(df, target_col="fraude")

    print(f"
📋 Features finais ({len(pipe.feature_names_)}):")
    for i, name in enumerate(pipe.feature_names_, 1):
        print(f"   {i:2d}. {name}")
