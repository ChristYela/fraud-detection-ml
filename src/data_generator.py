"""
data_generator.py
=================
Módulo para geração de dados sintéticos de transações financeiras
com foco em detecção de anomalias (fraude).

Uso:
    from src.data_generator import TransactionDataGenerator

    gen = TransactionDataGenerator(n_samples=10000, fraud_ratio=0.01, random_state=42)
    df = gen.generate()
    gen.save_to_csv("data/transacoes.csv")

Autor: Seu Nome
Data: 2026
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from typing import Optional, List, Tuple
import os


class TransactionDataGenerator:
    """
    Gerador de dados sintéticos de transações financeiras para
    detecção de fraudes.

    Parameters
    ----------
    n_samples : int, default=10000
        Número total de transações a serem geradas.
    n_features : int, default=20
        Número de features (colunas) por transação.
    fraud_ratio : float, default=0.01
        Proporção de transações fraudulentas (classe minoritária).
        Valor entre 0.0 e 1.0.
    random_state : int, optional
        Semente para reprodutibilidade.
    class_sep : float, default=1.5
        Quanto maior, mais separáveis são as classes.
    noise : float, default=0.03
        Proporção de labels trocados aleatoriamente (ruído).

    Attributes
    ----------
    df_ : pd.DataFrame
        DataFrame com os dados gerados (após chamar `generate()`).
    feature_names_ : list
        Nomes das features geradas.
    """

    FEATURE_NAMES: List[str] = [
        "valor_transacao",
        "hora_dia",
        "dia_semana",
        "idade_conta_dias",
        "num_transacoes_24h",
        "num_transacoes_7d",
        "valor_medio_30d",
        "desvio_valor_30d",
        "pais_risco",
        "vpn_detectado",
        "dispositivo_novo",
        "tentativas_senha",
        "tempo_sessao_seg",
        "latencia_ms",
        "score_credito",
        "renda_estimada",
        "divida_total",
        "num_cartoes",
        "transacao_internacional",
        "merchant_risco",
    ]

    def __init__(
        self,
        n_samples: int = 10000,
        n_features: int = 20,
        fraud_ratio: float = 0.01,
        random_state: Optional[int] = 42,
        class_sep: float = 1.5,
        noise: float = 0.03,
    ) -> None:
        if not (0.0 < fraud_ratio < 1.0):
            raise ValueError("fraud_ratio deve estar entre 0.0 e 1.0")
        if n_features < 5:
            raise ValueError("n_features deve ser pelo menos 5")

        self.n_samples = n_samples
        self.n_features = n_features
        self.fraud_ratio = fraud_ratio
        self.random_state = random_state
        self.class_sep = class_sep
        self.noise = noise

        self.df_: Optional[pd.DataFrame] = None
        self.feature_names_ = self.FEATURE_NAMES[:n_features]

    def _generate_raw(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera os dados brutos usando sklearn.datasets.make_classification.

        Returns
        -------
        X : np.ndarray
            Matriz de features.
        y : np.ndarray
            Vetor de labels (0 = normal, 1 = fraude).
        """
        n_informative = max(2, int(self.n_features * 0.75))
        n_redundant = self.n_features - n_informative

        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=2,
            weights=[1 - self.fraud_ratio, self.fraud_ratio],
            flip_y=self.noise,
            class_sep=self.class_sep,
            random_state=self.random_state,
        )
        return X, y

    def _apply_realistic_scales(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajusta as escalas das features para valores realistas de transações
        bancárias.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame com dados brutos.

        Returns
        -------
        pd.DataFrame
            DataFrame com escalas ajustadas.
        """
        # Ajustes determinísticos baseados no nome da feature
        scale_map = {
            "valor_transacao": lambda x: np.abs(x) * 800 + 15,
            "hora_dia": lambda x: np.clip((x + 3) * 4, 0, 23).astype(int),
            "dia_semana": lambda x: np.clip((x + 3.5), 0, 6).astype(int),
            "idade_conta_dias": lambda x: np.abs(x) * 400 + 30,
            "num_transacoes_24h": lambda x: np.clip(np.abs(x) * 5, 0, 50).astype(int),
            "num_transacoes_7d": lambda x: np.clip(np.abs(x) * 15, 0, 200).astype(int),
            "valor_medio_30d": lambda x: np.abs(x) * 600 + 20,
            "desvio_valor_30d": lambda x: np.abs(x) * 300 + 5,
            "pais_risco": lambda x: np.clip((x + 2) * 25, 0, 100),
            "vpn_detectado": lambda x: (x > 0.5).astype(int),
            "dispositivo_novo": lambda x: (x > 0.7).astype(int),
            "tentativas_senha": lambda x: np.clip(np.abs(x) * 3, 0, 10).astype(int),
            "tempo_sessao_seg": lambda x: np.abs(x) * 300 + 30,
            "latencia_ms": lambda x: np.abs(x) * 100 + 20,
            "score_credito": lambda x: np.clip((x + 2) * 140 + 300, 300, 850).astype(int),
            "renda_estimada": lambda x: np.abs(x) * 5000 + 1500,
            "divida_total": lambda x: np.abs(x) * 10000 + 500,
            "num_cartoes": lambda x: np.clip(np.abs(x) * 3 + 1, 1, 10).astype(int),
            "transacao_internacional": lambda x: (x > 0.3).astype(int),
            "merchant_risco": lambda x: np.clip((x + 2) * 25, 0, 100),
        }

        for col in df.columns:
            if col in scale_map and col != "fraude":
                df[col] = scale_map[col](df[col])

        return df

    def _inject_fraud_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona padrões artificiais nas transações fraudulentas para tornar
        o dataset mais realista (outliers intencionais).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame com dados escalados.

        Returns
        -------
        pd.DataFrame
            DataFrame com padrões de fraude injetados.
        """
        fraud_mask = df["fraude"] == 1
        n_frauds = fraud_mask.sum()

        if n_frauds == 0:
            return df

        rng = np.random.default_rng(self.random_state)

        # Aumentar valor da transação em fraudes
        df.loc[fraud_mask, "valor_transacao"] *= rng.uniform(1.5, 5.0, size=n_frauds)

        # Mais tentativas de senha em fraudes
        df.loc[fraud_mask, "tentativas_senha"] += rng.integers(1, 5, size=n_frauds)
        df.loc[fraud_mask, "tentativas_senha"] = df.loc[fraud_mask, "tentativas_senha"].clip(0, 10)

        # Sessão mais curta em fraudes (robôs / scripts)
        df.loc[fraud_mask, "tempo_sessao_seg"] *= rng.uniform(0.1, 0.5, size=n_frauds)

        # VPN mais comum em fraudes
        df.loc[fraud_mask, "vpn_detectado"] = rng.choice([0, 1], size=n_frauds, p=[0.3, 0.7])

        # Dispositivo novo mais comum
        df.loc[fraud_mask, "dispositivo_novo"] = rng.choice([0, 1], size=n_frauds, p=[0.2, 0.8])

        return df

    def generate(self, inject_patterns: bool = True) -> pd.DataFrame:
        """
        Executa o pipeline completo de geração de dados.

        Parameters
        ----------
        inject_patterns : bool, default=True
            Se True, injeta padrões artificiais nas fraudes.

        Returns
        -------
        pd.DataFrame
            DataFrame final com transações sintéticas.
        """
        # 1. Gerar dados brutos
        X, y = self._generate_raw()

        # 2. Criar DataFrame
        df = pd.DataFrame(X, columns=self.feature_names_)
        df["fraude"] = y

        # 3. Aplicar escalas realistas
        df = self._apply_realistic_scales(df)

        # 4. Injetar padrões de fraude
        if inject_patterns:
            df = self._inject_fraud_patterns(df)

        self.df_ = df
        return df

    def get_summary(self) -> dict:
        """
        Retorna um resumo estatístico do dataset gerado.

        Returns
        -------
        dict
            Dicionário com estatísticas do dataset.
        """
        if self.df_ is None:
            raise RuntimeError("Chame generate() antes de get_summary()")

        return {
            "n_total": len(self.df_),
            "n_normais": int((self.df_["fraude"] == 0).sum()),
            "n_fraudes": int(self.df_["fraude"].sum()),
            "fraud_ratio": float(self.df_["fraude"].mean()),
            "n_features": len(self.feature_names_),
            "features": self.feature_names_,
        }

    def print_summary(self) -> None:
        """Imprime um resumo formatado do dataset no console."""
        summary = self.get_summary()
        print("=" * 60)
        print("📊 RESUMO DO DATASET SINTÉTICO")
        print("=" * 60)
        print(f"Total de transações:    {summary['n_total']:,}")
        print(f"Transações normais:     {summary['n_normais']:,} ({(1-summary['fraud_ratio'])*100:.2f}%)")
        print(f"Transações fraudulentas: {summary['n_fraudes']:,} ({summary['fraud_ratio']*100:.2f}%)")
        print(f"Número de features:     {summary['n_features']}")
        print("=" * 60)

    def save_to_csv(self, filepath: str, index: bool = False) -> None:
        """
        Salva o DataFrame gerado em um arquivo CSV.

        Parameters
        ----------
        filepath : str
            Caminho do arquivo de saída.
        index : bool, default=False
            Se True, inclui o índice do DataFrame.
        """
        if self.df_ is None:
            raise RuntimeError("Chame generate() antes de save_to_csv()")

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self.df_.to_csv(filepath, index=index)
        print(f"✅ Dataset salvo em: {filepath}")

    def save_to_parquet(self, filepath: str) -> None:
        """
        Salva o DataFrame gerado em um arquivo Parquet (mais eficiente).

        Parameters
        ----------
        filepath : str
            Caminho do arquivo de saída.
        """
        if self.df_ is None:
            raise RuntimeError("Chame generate() antes de save_to_parquet()")

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self.df_.to_parquet(filepath, index=False)
        print(f"✅ Dataset salvo em: {filepath}")

    def get_train_test_split(
        self,
        test_size: float = 0.2,
        stratify: bool = True,
        scaler_type: str = "standard",
    ) -> Tuple:
        """
        Divide os dados em treino/teste e aplica normalização.

        Parameters
        ----------
        test_size : float, default=0.2
            Proporção do conjunto de teste.
        stratify : bool, default=True
            Se True, mantém a proporção de classes.
        scaler_type : str, default="standard"
            Tipo de scaler: "standard" ou "robust".

        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test, scaler)
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, RobustScaler

        if self.df_ is None:
            raise RuntimeError("Chame generate() antes de get_train_test_split()")

        X = self.df_.drop("fraude", axis=1)
        y = self.df_["fraude"]

        stratify_param = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify_param
        )

        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError("scaler_type deve ser 'standard' ou 'robust'")

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def generate_dataset(
    n_samples: int = 10000,
    fraud_ratio: float = 0.01,
    output_path: Optional[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Função de conveniência para gerar e opcionalmente salvar um dataset.

    Parameters
    ----------
    n_samples : int, default=10000
        Número de transações.
    fraud_ratio : float, default=0.01
        Proporção de fraudes.
    output_path : str, optional
        Caminho para salvar o CSV. Se None, não salva.
    random_state : int, default=42
        Semente para reprodutibilidade.

    Returns
    -------
    pd.DataFrame
        DataFrame com os dados gerados.
    """
    gen = TransactionDataGenerator(
        n_samples=n_samples,
        fraud_ratio=fraud_ratio,
        random_state=random_state,
    )
    df = gen.generate()
    gen.print_summary()

    if output_path:
        gen.save_to_csv(output_path)

    return df


# ============================================================
# EXEMPLO DE USO (executar diretamente)
# ============================================================
if __name__ == "__main__":
    print("🚀 Gerando dataset sintético de transações...")
    print()

    # Exemplo 1: Gerar e visualizar
    df = generate_dataset(n_samples=5000, fraud_ratio=0.02, random_state=42)
    print("
📋 Primeiras 5 linhas:")
    print(df.head())

    print("
📈 Estatísticas descritivas:")
    print(df.describe())

    # Exemplo 2: Usar a classe completa com split
    print("
" + "=" * 60)
    print("🧪 Exemplo com train/test split:")
    print("=" * 60)

    gen = TransactionDataGenerator(n_samples=10000, fraud_ratio=0.01, random_state=42)
    gen.generate()
    gen.print_summary()

    X_train, X_test, y_train, y_test, scaler = gen.get_train_test_split(
        test_size=0.2, stratify=True, scaler_type="standard"
    )

    print(f"
📦 Conjunto de treino: {X_train.shape}")
    print(f"📦 Conjunto de teste:  {X_test.shape}")
    print(f"⚖️  Fraudas no treino: {y_train.sum()} / {len(y_train)}")
    print(f"⚖️  Fraudas no teste:  {y_test.sum()} / {len(y_test)}")
