"""
test_models.py
===============
Testes unitários para o pipeline de detecção de fraudes.

Executar com:
    pytest tests/test_models.py -v
    pytest tests/test_models.py -v --cov=src

Requer:
    pytest>=7.4.0
    pytest-cov>=4.1.0
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Importar módulos do projeto
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generator import TransactionDataGenerator, generate_dataset
from src.preprocessing import PreprocessingPipeline
from src.models import FraudDetector
from src.evaluation import ModelEvaluator


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def sample_data():
    """Fixture: dataset sintético pequeno para testes rápidos."""
    gen = TransactionDataGenerator(n_samples=1000, fraud_ratio=0.05, random_state=42)
    df = gen.generate()
    return df


@pytest.fixture(scope="module")
def sample_data_scaled(sample_data):
    """Fixture: dados pré-processados (X_train, X_test, y_train, y_test)."""
    pipe = PreprocessingPipeline(
        test_size=0.2,
        scaler_type="standard",
        apply_feature_engineering=False,
        outlier_method=None,
    )
    X_train, X_test, y_train, y_test = pipe.fit_transform(sample_data, target_col="fraude")
    return X_train, X_test, y_train, y_test


# ============================================================
# TESTES: data_generator.py
# ============================================================

class TestDataGenerator:
    """Testes para o módulo de geração de dados."""

    def test_generate_returns_dataframe(self):
        """Verifica se generate() retorna um DataFrame."""
        gen = TransactionDataGenerator(n_samples=100, random_state=42)
        df = gen.generate()
        assert isinstance(df, pd.DataFrame)

    def test_correct_shape(self):
        """Verifica se o DataFrame tem o shape correto."""
        gen = TransactionDataGenerator(n_samples=500, n_features=20, random_state=42)
        df = gen.generate()
        assert len(df) == 500
        assert len(df.columns) == 21  # 20 features + fraude

    def test_fraud_ratio(self):
        """Verifica se a proporção de fraudes está próxima do esperado."""
        gen = TransactionDataGenerator(n_samples=2000, fraud_ratio=0.05, random_state=42)
        df = gen.generate()
        ratio = df["fraude"].mean()
        assert 0.03 <= ratio <= 0.07  # tolerância de ±2%

    def test_summary(self):
        """Verifica se get_summary() retorna estatísticas corretas."""
        gen = TransactionDataGenerator(n_samples=1000, fraud_ratio=0.02, random_state=42)
        gen.generate()
        summary = gen.get_summary()
        assert summary["n_total"] == 1000
        assert summary["n_features"] == 20
        assert "features" in summary

    def test_invalid_fraud_ratio(self):
        """Verifica se fraud_ratio inválido gera erro."""
        with pytest.raises(ValueError):
            TransactionDataGenerator(fraud_ratio=1.5)

    def test_train_test_split(self):
        """Verifica se get_train_test_split retorna 5 objetos."""
        gen = TransactionDataGenerator(n_samples=500, random_state=42)
        gen.generate()
        result = gen.get_train_test_split(test_size=0.2)
        assert len(result) == 5
        X_train, X_test, y_train, y_test, scaler = result
        assert len(X_train) + len(X_test) == 500
        assert scaler is not None


# ============================================================
# TESTES: preprocessing.py
# ============================================================

class TestPreprocessing:
    """Testes para o módulo de pré-processamento."""

    def test_fit_transform_returns_arrays(self, sample_data):
        """Verifica se fit_transform retorna 4 arrays."""
        pipe = PreprocessingPipeline(test_size=0.2, apply_feature_engineering=False)
        result = pipe.fit_transform(sample_data, target_col="fraude")
        assert len(result) == 4
        X_train, X_test, y_train, y_test = result
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)

    def test_stratify_maintains_ratio(self, sample_data):
        """Verifica se a proporção de fraudes é mantida no split."""
        pipe = PreprocessingPipeline(test_size=0.2, stratify=True)
        X_train, X_test, y_train, y_test = pipe.fit_transform(sample_data, target_col="fraude")
        original_ratio = sample_data["fraude"].mean()
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - original_ratio) < 0.02
        assert abs(test_ratio - original_ratio) < 0.02

    def test_feature_engineering_creates_features(self, sample_data):
        """Verifica se feature engineering cria novas colunas."""
        pipe = PreprocessingPipeline(apply_feature_engineering=True)
        X_train, _, _, _ = pipe.fit_transform(sample_data, target_col="fraude")
        assert len(pipe.feature_names_) > 20  # mais que as originais

    def test_scaler_types(self, sample_data):
        """Verifica se todos os tipos de scaler funcionam."""
        for scaler_type in ["standard", "robust", "minmax"]:
            pipe = PreprocessingPipeline(scaler_type=scaler_type)
            X_train, _, _, _ = pipe.fit_transform(sample_data, target_col="fraude")
            assert X_train is not None

    def test_transform_after_fit(self, sample_data):
        """Verifica se transform() funciona após fit_transform()."""
        pipe = PreprocessingPipeline()
        pipe.fit_transform(sample_data, target_col="fraude")
        X_new = pipe.transform(sample_data.drop("fraude", axis=1).head(10))
        assert X_new.shape[0] == 10

    def test_invalid_scaler_type(self, sample_data):
        """Verifica se scaler_type inválido gera erro."""
        pipe = PreprocessingPipeline(scaler_type="invalid")
        with pytest.raises(ValueError):
            pipe.fit_transform(sample_data, target_col="fraude")


# ============================================================
# TESTES: models.py
# ============================================================

class TestModels:
    """Testes para o módulo de modelos."""

    @pytest.mark.parametrize("model_type", [
        "logistic",
        "random_forest",
        "xgboost",
        "isolation_forest",
        "lof",
    ])
    def test_model_fit_and_predict(self, sample_data_scaled, model_type):
        """Verifica se cada modelo treina e prediz corretamente."""
        X_train, X_test, y_train, y_test = sample_data_scaled

        detector = FraudDetector(model_type=model_type, contamination=0.05)
        detector.fit(X_train, y_train, epochs=10, verbose=0)

        y_pred = detector.predict(X_test)
        assert len(y_pred) == len(X_test)
        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_xgboost_predict_proba(self, sample_data_scaled):
        """Verifica se XGBoost retorna probabilidades."""
        X_train, X_test, y_train, _ = sample_data_scaled
        detector = FraudDetector(model_type="xgboost")
        detector.fit(X_train, y_train)
        y_prob = detector.predict_proba(X_test)
        assert y_prob is not None
        assert len(y_prob) == len(X_test)
        assert np.all((y_prob >= 0) & (y_prob <= 1))

    def test_autoencoder_fit_and_predict(self, sample_data_scaled):
        """Verifica se o autoencoder treina e prediz."""
        X_train, X_test, y_train, _ = sample_data_scaled
        detector = FraudDetector(model_type="autoencoder")
        detector.fit(X_train, y_train, epochs=10, verbose=0)
        y_pred = detector.predict(X_test)
        assert len(y_pred) == len(X_test)

    def test_hybrid_model(self, sample_data_scaled):
        """Verifica se o modelo híbrido funciona."""
        X_train, X_test, y_train, _ = sample_data_scaled
        detector = FraudDetector(model_type="hybrid", contamination=0.05)
        detector.fit(X_train, y_train, epochs=10, verbose=0)
        y_pred = detector.predict(X_test)
        assert len(y_pred) == len(X_test)

    def test_invalid_model_type(self):
        """Verifica se model_type inválido gera erro."""
        with pytest.raises(ValueError):
            FraudDetector(model_type="invalid_model")

    def test_predict_before_fit(self, sample_data_scaled):
        """Verifica se predict() antes de fit() gera erro."""
        _, X_test, _, _ = sample_data_scaled
        detector = FraudDetector(model_type="logistic")
        with pytest.raises(RuntimeError):
            detector.predict(X_test)

    def test_supervised_requires_y(self, sample_data_scaled):
        """Verifica se modelos supervisionados exigem y_train."""
        X_train, _, _, _ = sample_data_scaled
        detector = FraudDetector(model_type="xgboost")
        with pytest.raises(ValueError):
            detector.fit(X_train)  # sem y_train


# ============================================================
# TESTES: evaluation.py
# ============================================================

class TestEvaluation:
    """Testes para o módulo de avaliação."""

    def test_add_result(self, sample_data_scaled):
        """Verifica se add_result funciona corretamente."""
        X_train, X_test, y_train, y_test = sample_data_scaled

        detector = FraudDetector(model_type="logistic")
        detector.fit(X_train, y_train)
        y_pred = detector.predict(X_test)
        y_prob = detector.predict_proba(X_test)

        evaluator = ModelEvaluator()
        evaluator.add_result("Logistic", y_test, y_pred, y_prob)

        assert "Logistic" in evaluator.resultados_
        assert evaluator.resultados_["Logistic"]["auc_pr"] >= 0

    def test_get_summary(self, sample_data_scaled):
        """Verifica se get_summary retorna DataFrame ordenado."""
        X_train, X_test, y_train, y_test = sample_data_scaled

        evaluator = ModelEvaluator()
        for name in ["logistic", "xgboost"]:
            detector = FraudDetector(model_type=name)
            detector.fit(X_train, y_train)
            y_pred = detector.predict(X_test)
            y_prob = detector.predict_proba(X_test)
            evaluator.add_result(name, y_test, y_pred, y_prob)

        df = evaluator.get_summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "auc_pr" in df.columns
        # Verificar ordenação decrescente
        assert df["auc_pr"].is_monotonic_decreasing or df["auc_pr"].iloc[0] >= df["auc_pr"].iloc[-1]

    def test_get_best_model(self, sample_data_scaled):
        """Verifica se get_best_model retorna string."""
        X_train, X_test, y_train, y_test = sample_data_scaled

        evaluator = ModelEvaluator()
        detector = FraudDetector(model_type="logistic")
        detector.fit(X_train, y_train)
        y_pred = detector.predict(X_test)
        y_prob = detector.predict_proba(X_test)
        evaluator.add_result("Logistic", y_test, y_pred, y_prob)

        best = evaluator.get_best_model()
        assert isinstance(best, str)
        assert best == "Logistic"

    def test_empty_evaluator_raises(self):
        """Verifica se get_summary() vazio gera erro."""
        evaluator = ModelEvaluator()
        with pytest.raises(RuntimeError):
            evaluator.get_summary()

    def test_custo_calculo(self):
        """Verifica se o custo é calculado corretamente."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])  # 1 FP, 1 FN, 1 TP, 1 TN

        evaluator = ModelEvaluator(custo_fn=100, custo_fp=5)
        evaluator.add_result("Test", y_true, y_pred)
        res = evaluator.resultados_["Test"]
        assert res["custo"] == 105  # 1*100 + 1*5
        assert res["tp"] == 1
        assert res["fp"] == 1
        assert res["tn"] == 1
        assert res["fn"] == 1


# ============================================================
# TESTE DE INTEGRAÇÃO
# ============================================================

class TestIntegration:
    """Testes de integração do pipeline completo."""

    def test_pipeline_completo(self):
        """Testa o fluxo completo: geração → pré-processamento → modelo → avaliação."""
        # 1. Gerar dados
        df = generate_dataset(n_samples=500, fraud_ratio=0.05, random_state=42)
        assert "fraude" in df.columns

        # 2. Pré-processar
        pipe = PreprocessingPipeline(test_size=0.2, apply_feature_engineering=True)
        X_train, X_test, y_train, y_test = pipe.fit_transform(df, target_col="fraude")
        assert X_train.shape[1] > 20  # features criadas

        # 3. Treinar modelo
        detector = FraudDetector(model_type="xgboost")
        detector.fit(X_train, y_train)
        y_pred = detector.predict(X_test)
        y_prob = detector.predict_proba(X_test)
        assert len(y_pred) == len(y_test)

        # 4. Avaliar
        evaluator = ModelEvaluator()
        evaluator.add_result("XGBoost", y_test, y_pred, y_prob)
        summary = evaluator.get_summary()
        assert len(summary) == 1
        assert summary.iloc[0]["modelo"] == "XGBoost"

        print("
✅ Pipeline de integração executado com sucesso!")
