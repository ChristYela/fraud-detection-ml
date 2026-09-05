"""
Fraud Detection ML
==================
Pacote Python para detecção de anomalias em transações financeiras.

Módulos:
    data_generator    - Geração de dados sintéticos
    preprocessing     - Pré-processamento e feature engineering
    models            - Definição e treinamento de modelos
    evaluation        - Avaliação e métricas
    explainability    - SHAP e explicabilidade

Exemplo:
    from src.data_generator import TransactionDataGenerator
    from src.models import FraudDetector
    from src.evaluation import evaluate_all_models
"""

__version__ = "0.1.0"
__author__ = "Seu Nome"

from .data_generator import TransactionDataGenerator
from .preprocessing import PreprocessingPipeline
from .models import FraudDetector
from .evaluation import ModelEvaluator
from .explainability import ShapExplainer

__all__ = [
    "TransactionDataGenerator",
    "PreprocessingPipeline",
    "FraudDetector",
    "ModelEvaluator",
    "ShapExplainer",
]
