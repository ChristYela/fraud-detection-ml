"""
evaluation.py
==============
Funções de avaliação e comparação de modelos de detecção de fraudes.

Inclui:
    - Métricas especializadas para dados desbalanceados
    - Matriz de confusão com custo de erro
    - Curvas ROC e Precision-Recall
    - Tabela comparativa de modelos

Uso:
    from src.evaluation import ModelEvaluator

    evaluator = ModelEvaluator()
    evaluator.add_result("XGBoost", y_test, y_pred, y_prob)
    evaluator.add_result("Isolation Forest", y_test, y_pred_iso, y_score_iso)
    evaluator.print_comparison()
    evaluator.plot_comparison()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List, Tuple, Any
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
import warnings

warnings.filterwarnings("ignore")


class ModelEvaluator:
    """
    Avaliador unificado para comparação de múltiplos modelos
    de detecção de fraudes.

    Parameters
    ----------
    custo_fn : float, default=100.0
        Custo de um falso negativo (fraude não detectada).
    custo_fp : float, default=5.0
        Custo de um falso positivo (alarme falso).

    Attributes
    ----------
    resultados_ : dict
        Dicionário com métricas de cada modelo.
    """

    def __init__(
        self,
        custo_fn: float = 100.0,
        custo_fp: float = 5.0,
    ) -> None:
        self.custo_fn = custo_fn
        self.custo_fp = custo_fp
        self.resultados_: Dict[str, Dict[str, Any]] = {}

    def add_result(
        self,
        nome_modelo: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> None:
        """
        Adiciona os resultados de um modelo para comparação.

        Parameters
        ----------
        nome_modelo : str
            Nome identificador do modelo.
        y_true : np.ndarray
            Labels reais.
        y_pred : np.ndarray
            Predições binárias.
        y_prob : np.ndarray, optional
            Scores/probabilidades (necessário para AUC).
        """
        # Métricas básicas
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        # AUC (se scores disponíveis)
        auc_roc = 0.5
        auc_pr = float(np.mean(y_true))

        if y_prob is not None and len(np.unique(y_prob)) > 1:
            try:
                auc_roc = roc_auc_score(y_true, y_prob)
                auc_pr = average_precision_score(y_true, y_prob)
            except ValueError:
                pass

        # Custo de erro
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        custo_total = fn * self.custo_fn + fp * self.custo_fp

        self.resultados_[nome_modelo] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": acc,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "custo": custo_total,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    def get_summary(self) -> pd.DataFrame:
        """
        Retorna DataFrame com resumo comparativo de todos os modelos.

        Returns
        -------
        pd.DataFrame
            Tabela ordenada por AUC-PR (decrescente).
        """
        if not self.resultados_:
            raise RuntimeError("Nenhum resultado adicionado. Use add_result() primeiro.")

        data = []
        for nome, res in self.resultados_.items():
            data.append({
                "modelo": nome,
                "precision": res["precision"],
                "recall": res["recall"],
                "f1": res["f1"],
                "accuracy": res["accuracy"],
                "auc_roc": res["auc_roc"],
                "auc_pr": res["auc_pr"],
                "custo_r$": res["custo"],
                "tp": res["tp"],
                "fp": res["fp"],
                "tn": res["tn"],
                "fn": res["fn"],
            })

        df = pd.DataFrame(data)
        df = df.sort_values("auc_pr", ascending=False).reset_index(drop=True)
        return df

    def print_comparison(self) -> None:
        """Imprime tabela comparativa formatada no console."""
        df = self.get_summary()

        print("=" * 100)
        print(f"{'Modelo':<28} {'Precisão':>9} {'Recall':>9} {'F1':>9} {'AUC-ROC':>9} {'AUC-PR':>9} {'Custo(R$)':>11}")
        print("=" * 100)

        for _, row in df.iterrows():
            print(
                f"{row['modelo']:<28} "
                f"{row['precision']:>9.4f} "
                f"{row['recall']:>9.4f} "
                f"{row['f1']:>9.4f} "
                f"{row['auc_roc']:>9.4f} "
                f"{row['auc_pr']:>9.4f} "
                f"{row['custo_r$']:>11.0f}"
            )

        print("=" * 100)

        melhor = df.iloc[0]
        print(f"
🏆 Melhor modelo: {melhor['modelo']} (AUC-PR: {melhor['auc_pr']:.4f})")

    def plot_comparison(self, figsize: Tuple[int, int] = (16, 12)) -> None:
        """
        Gera visualizações comparativas de todos os modelos.

        Parameters
        ----------
        figsize : tuple, default=(16, 12)
            Tamanho da figura matplotlib.
        """
        if not self.resultados_:
            raise RuntimeError("Nenhum resultado adicionado.")

        df = self.get_summary()

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # 1. AUC-PR
        ax = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
        bars = ax.barh(df["modelo"], df["auc_pr"], color=colors, edgecolor="black")
        ax.set_xlabel("AUC-PR (Average Precision)")
        ax.set_title("AUC-PR por Modelo", fontweight="bold", fontsize=12)
        ax.set_xlim(0, 1)
        for bar, val in zip(bars, df["auc_pr"]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)

        # 2. Precision vs Recall
        ax = axes[0, 1]
        ax.scatter(df["recall"], df["precision"], s=200, c="coral", edgecolors="black", zorder=3)
        for _, row in df.iterrows():
            ax.annotate(row["modelo"], (row["recall"], row["precision"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision vs Recall", fontweight="bold", fontsize=12)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.3)

        # 3. Custo
        ax = axes[0, 2]
        bars = ax.barh(df["modelo"], df["custo_r$"], color="crimson", edgecolor="black", alpha=0.8)
        ax.set_xlabel("Custo Estimado (R$)")
        ax.set_title("Custo de Erros", fontweight="bold", fontsize=12)
        for bar, val in zip(bars, df["custo_r$"]):
            ax.text(val + max(df["custo_r$"])*0.01, bar.get_y() + bar.get_height()/2,
                    f"R${val:,.0f}", va="center", fontsize=8)

        # 4. Curvas Precision-Recall
        ax = axes[1, 0]
        for nome, res in self.resultados_.items():
            y_prob = res["y_prob"]
            if y_prob is not None and len(np.unique(y_prob)) > 1:
                prec_curve, rec_curve, _ = precision_recall_curve(res["y_true"], y_prob)
                ax.plot(rec_curve, prec_curve, label=nome, linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Curvas Precision-Recall", fontweight="bold", fontsize=12)
        ax.legend(loc="lower left", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # 5. Curvas ROC
        ax = axes[1, 1]
        for nome, res in self.resultados_.items():
            y_prob = res["y_prob"]
            if y_prob is not None and len(np.unique(y_prob)) > 1:
                fpr, tpr, _ = roc_curve(res["y_true"], y_prob)
                ax.plot(fpr, tpr, label=f"{nome} (AUC={res['auc_roc']:.3f})", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Aleatório")
        ax.set_xlabel("Taxa de Falso Positivo (FPR)")
        ax.set_ylabel("Taxa de Verdadeiro Positivo (TPR)")
        ax.set_title("Curvas ROC", fontweight="bold", fontsize=12)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # 6. Matriz de confusão do melhor modelo
        ax = axes[1, 2]
        melhor_nome = df.iloc[0]["modelo"]
        res = self.resultados_[melhor_nome]
        cm = np.array([[res["tn"], res["fp"]], [res["fn"], res["tp"]]])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Fraude"],
                    yticklabels=["Normal", "Fraude"])
        ax.set_title(f"Matriz de Confusão: {melhor_nome}", fontweight="bold", fontsize=12)
        ax.set_ylabel("Real")
        ax.set_xlabel("Predito")

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, nome_modelo: str, figsize: Tuple[int, int] = (6, 5)) -> None:
        """
        Plota matriz de confusão de um modelo específico.

        Parameters
        ----------
        nome_modelo : str
            Nome do modelo.
        figsize : tuple, default=(6, 5)
            Tamanho da figura.
        """
        if nome_modelo not in self.resultados_:
            raise ValueError(f"Modelo '{nome_modelo}' não encontrado.")

        res = self.resultados_[nome_modelo]
        cm = np.array([[res["tn"], res["fp"]], [res["fn"], res["tp"]]])

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
                    xticklabels=["Normal", "Fraude"],
                    yticklabels=["Normal", "Fraude"])
        ax.set_title(f"Matriz de Confusão: {nome_modelo}", fontweight="bold", fontsize=13)
        ax.set_ylabel("Real")
        ax.set_xlabel("Predito")
        plt.tight_layout()
        plt.show()

    def get_best_model(self, metric: str = "auc_pr") -> str:
        """
        Retorna o nome do melhor modelo segundo uma métrica.

        Parameters
        ----------
        metric : str, default="auc_pr"
            Métrica para ranqueamento.

        Returns
        -------
        str
            Nome do melhor modelo.
        """
        df = self.get_summary()
        return str(df.iloc[0]["modelo"])

    def classification_report(self, nome_modelo: str) -> str:
        """
        Retorna o classification report de um modelo.

        Parameters
        ----------
        nome_modelo : str
            Nome do modelo.

        Returns
        -------
        str
            Classification report formatado.
        """
        if nome_modelo not in self.resultados_:
            raise ValueError(f"Modelo '{nome_modelo}' não encontrado.")

        res = self.resultados_[nome_modelo]
        return classification_report(
            res["y_true"], res["y_pred"],
            target_names=["Normal", "Fraude"]
        )


# ============================================================
# EXEMPLO DE USO
# ============================================================
if __name__ == "__main__":
    from src.data_generator import generate_dataset
    from src.models import FraudDetector
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

    # Avaliar múltiplos modelos
    evaluator = ModelEvaluator(custo_fn=100, custo_fp=5)

    for model_name in ["logistic", "xgboost", "isolation_forest"]:
        detector = FraudDetector(model_type=model_name, contamination=0.02)
        detector.fit(X_train_s, y_train.values, epochs=20, verbose=0)
        y_pred = detector.predict(X_test_s)
        y_prob = detector.predict_proba(X_test_s)
        evaluator.add_result(model_name, y_test.values, y_pred, y_prob)

    evaluator.print_comparison()
