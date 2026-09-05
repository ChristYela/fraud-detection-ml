"""
explainability.py
==================
Módulo de explicabilidade usando SHAP para modelos de detecção de fraudes.

Inclui:
    - Summary plot (importância global)
    - Waterfall plot (explicação individual)
    - Force plot (impacto interativo)
    - Dependence plot (relação feature vs impacto)
    - Beeswarm plot (distribuição de SHAP values)

Uso:
    from src.explainability import ShapExplainer

    explainer = ShapExplainer(model=xgb_model, model_type="tree")
    explainer.fit(X_train)
    explainer.summary_plot(X_test, feature_names)
    explainer.waterfall_plot(X_test, feature_names, instance_idx=0)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Optional, List, Union, Any
import warnings

warnings.filterwarnings("ignore")


class ShapExplainer:
    """
    Wrapper para geração de explicações SHAP em modelos de fraude.

    Parameters
    ----------
    model : object
        Modelo treinado (XGBoost, RandomForest, etc.).
    model_type : str, default="tree"
        Tipo de explainer: "tree" (TreeExplainer), "kernel" (KernelExplainer)
        ou "deep" (DeepExplainer para redes neurais).
    background_data : np.ndarray, optional
        Dados de background para KernelExplainer.

    Attributes
    ----------
    explainer_ : shap.Explainer
        Instância do explainer SHAP ajustado.
    shap_values_ : np.ndarray
        Valores SHAP calculados (após explain()).
    """

    def __init__(
        self,
        model: Any,
        model_type: str = "tree",
        background_data: Optional[np.ndarray] = None,
    ) -> None:
        self.model = model
        self.model_type = model_type
        self.background_data = background_data

        self.explainer_: Optional[Any] = None
        self.shap_values_: Optional[np.ndarray] = None
        self.expected_value_: Optional[float] = None
        self._fitted: bool = False

    def fit(self, X_background: Optional[np.ndarray] = None) -> "ShapExplainer":
        """
        Inicializa o explainer SHAP.

        Parameters
        ----------
        X_background : np.ndarray, optional
            Amostra de background. Necessário para KernelExplainer.

        Returns
        -------
        ShapExplainer
            Instância ajustada.
        """
        print(f"🔍 Inicializando SHAP ({self.model_type})...")

        if self.model_type == "tree":
            self.explainer_ = shap.TreeExplainer(self.model)
            self.expected_value_ = self.explainer_.expected_value

        elif self.model_type == "kernel":
            if X_background is None and self.background_data is None:
                raise ValueError("KernelExplainer requer background_data")
            bg = X_background if X_background is not None else self.background_data
            self.explainer_ = shap.KernelExplainer(self.model.predict, bg)
            self.expected_value_ = self.explainer_.expected_value

        elif self.model_type == "deep":
            if X_background is None and self.background_data is None:
                raise ValueError("DeepExplainer requer background_data")
            bg = X_background if X_background is not None else self.background_data
            self.explainer_ = shap.DeepExplainer(self.model, bg)
            self.expected_value_ = self.explainer_.expected_value

        else:
            raise ValueError("model_type deve ser 'tree', 'kernel' ou 'deep'")

        self._fitted = True
        print("✅ Explainer SHAP pronto!")
        return self

    def explain(
        self,
        X: np.ndarray,
        max_samples: int = 500,
    ) -> np.ndarray:
        """
        Calcula os SHAP values para um conjunto de dados.

        Parameters
        ----------
        X : np.ndarray
            Dados a serem explicados.
        max_samples : int, default=500
            Número máximo de amostras para calcular (performance).

        Returns
        -------
        np.ndarray
            Matriz de SHAP values.
        """
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de explain()")

        X_sample = X[:max_samples] if len(X) > max_samples else X

        print(f"🧮 Calculando SHAP values para {len(X_sample)} amostras...")

        if self.model_type == "tree":
            self.shap_values_ = self.explainer_.shap_values(X_sample)
            # Para classificação binária, TreeExplainer retorna lista
            if isinstance(self.shap_values_, list):
                self.shap_values_ = self.shap_values_[1]  # classe positiva
        else:
            self.shap_values_ = self.explainer_.shap_values(X_sample)

        print(f"✅ SHAP values calculados! Shape: {self.shap_values_.shape}")
        return self.shap_values_

    def summary_plot(
        self,
        X: np.ndarray,
        feature_names: List[str],
        max_display: int = 15,
        show: bool = True,
    ) -> None:
        """
        Gera summary plot (importância global das features).

        Parameters
        ----------
        X : np.ndarray
            Dados de entrada.
        feature_names : list
            Nomes das features.
        max_display : int, default=15
            Número máximo de features no plot.
        show : bool, default=True
            Se True, exibe o plot.
        """
        if self.shap_values_ is None:
            self.explain(X)

        X_sample = X[:len(self.shap_values_)]

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values_,
            X_sample,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        plt.title("Importância Global das Features (SHAP)", fontweight="bold", fontsize=14, pad=20)
        plt.tight_layout()
        if show:
            plt.show()

    def waterfall_plot(
        self,
        X: np.ndarray,
        feature_names: List[str],
        instance_idx: int = 0,
        show: bool = True,
    ) -> None:
        """
        Gera waterfall plot para uma instância específica.

        Parameters
        ----------
        X : np.ndarray
            Dados de entrada.
        feature_names : list
            Nomes das features.
        instance_idx : int, default=0
            Índice da instância a ser explicada.
        show : bool, default=True
            Se True, exibe o plot.
        """
        if self.shap_values_ is None:
            self.explain(X)

        if instance_idx >= len(self.shap_values_):
            raise ValueError(f"instance_idx {instance_idx} fora do range")

        plt.figure(figsize=(12, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values_[instance_idx],
                base_values=self.expected_value_,
                data=X[instance_idx],
                feature_names=feature_names,
            ),
            show=False,
        )
        plt.title(f"Explicação SHAP - Transação #{instance_idx}", fontweight="bold", fontsize=13, pad=20)
        plt.tight_layout()
        if show:
            plt.show()

    def force_plot(
        self,
        X: np.ndarray,
        feature_names: List[str],
        instance_idx: int = 0,
        show: bool = True,
    ) -> None:
        """
        Gera force plot para uma instância específica.

        Parameters
        ----------
        X : np.ndarray
            Dados de entrada.
        feature_names : list
            Nomes das features.
        instance_idx : int, default=0
            Índice da instância.
        show : bool, default=True
            Se True, exibe o plot.
        """
        if self.shap_values_ is None:
            self.explain(X)

        plt.figure(figsize=(16, 4))
        shap.force_plot(
            self.expected_value_,
            self.shap_values_[instance_idx],
            X[instance_idx],
            feature_names=feature_names,
            show=False,
            matplotlib=True,
        )
        plt.title(f"Force Plot - Transação #{instance_idx}", fontweight="bold", fontsize=13)
        plt.tight_layout()
        if show:
            plt.show()

    def dependence_plot(
        self,
        feature: Union[str, int],
        X: np.ndarray,
        feature_names: List[str],
        interaction_index: Optional[Union[str, int]] = None,
        show: bool = True,
    ) -> None:
        """
        Gera dependence plot para uma feature específica.

        Parameters
        ----------
        feature : str ou int
            Nome ou índice da feature.
        X : np.ndarray
            Dados de entrada.
        feature_names : list
            Nomes das features.
        interaction_index : str ou int, optional
            Feature para colorir o scatter.
        show : bool, default=True
            Se True, exibe o plot.
        """
        if self.shap_values_ is None:
            self.explain(X)

        X_sample = X[:len(self.shap_values_)]

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature,
            self.shap_values_,
            X_sample,
            feature_names=feature_names,
            interaction_index=interaction_index,
            show=False,
        )
        plt.title(f"Dependência: {feature}", fontweight="bold", fontsize=13)
        plt.tight_layout()
        if show:
            plt.show()

    def beeswarm_plot(
        self,
        X: np.ndarray,
        feature_names: List[str],
        max_display: int = 15,
        show: bool = True,
    ) -> None:
        """
        Gera beeswarm plot (alternativa ao summary plot).

        Parameters
        ----------
        X : np.ndarray
            Dados de entrada.
        feature_names : list
            Nomes das features.
        max_display : int, default=15
            Número máximo de features.
        show : bool, default=True
            Se True, exibe o plot.
        """
        if self.shap_values_ is None:
            self.explain(X)

        X_sample = X[:len(self.shap_values_)]

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values_,
            X_sample,
            feature_names=feature_names,
            max_display=max_display,
            plot_type="dot",  # beeswarm
            show=False,
        )
        plt.title("Distribuição dos SHAP Values (Beeswarm)", fontweight="bold", fontsize=14, pad=20)
        plt.tight_layout()
        if show:
            plt.show()

    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Retorna DataFrame com importância das features (média do |SHAP value|).

        Parameters
        ----------
        feature_names : list
            Nomes das features.

        Returns
        -------
        pd.DataFrame
            Tabela ordenada por importância.
        """
        if self.shap_values_ is None:
            raise RuntimeError("Chame explain() primeiro")

        importance = np.abs(self.shap_values_).mean(axis=0)
        df = pd.DataFrame({
            "feature": feature_names,
            "importancia": importance,
        }).sort_values("importancia", ascending=False).reset_index(drop=True)

        return df

    def explain_instance_text(
        self,
        X: np.ndarray,
        feature_names: List[str],
        instance_idx: int = 0,
    ) -> str:
        """
        Gera explicação em texto para uma instância específica.

        Parameters
        ----------
        X : np.ndarray
            Dados de entrada.
        feature_names : list
            Nomes das features.
        instance_idx : int, default=0
            Índice da instância.

        Returns
        -------
        str
            Texto explicativo.
        """
        if self.shap_values_ is None:
            self.explain(X)

        shap_vals = self.shap_values_[instance_idx]
        data_vals = X[instance_idx]

        # Top 5 features que empurram para fraude
        top_positive = np.argsort(shap_vals)[-5:][::-1]
        # Top 5 features que empurram para normal
        top_negative = np.argsort(shap_vals)[:5]

        lines = [f"📋 Explicação da Transação #{instance_idx}", "=" * 50]
        lines.append(f"
🔴 Fatores que INDICAM FRAUDE (top 5):")
        for idx in top_positive:
            lines.append(f"   • {feature_names[idx]}: {data_vals[idx]:.2f} (impacto: +{shap_vals[idx]:.4f})")

        lines.append(f"
🟢 Fatores que INDICAM NORMALIDADE (top 5):")
        for idx in top_negative:
            lines.append(f"   • {feature_names[idx]}: {data_vals[idx]:.2f} (impacto: {shap_vals[idx]:.4f})")

        return "
".join(lines)


# ============================================================
# EXEMPLO DE USO
# ============================================================
if __name__ == "__main__":
    from src.data_generator import generate_dataset
    from src.models import FraudDetector
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Dados
    df = generate_dataset(n_samples=3000, fraud_ratio=0.02, random_state=42)
    X = df.drop("fraude", axis=1)
    y = df["fraude"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Treinar XGBoost
    detector = FraudDetector(model_type="xgboost")
    detector.fit(X_train_s, y_train.values)

    # Explicar
    explainer = ShapExplainer(model=detector.get_model(), model_type="tree")
    explainer.fit()
    explainer.explain(X_test_s, max_samples=200)

    feature_names = list(X.columns)

    print("
" + explainer.explain_instance_text(X_test_s, feature_names, instance_idx=0))
