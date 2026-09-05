# 🔍 Detecção de Anomalias em Transações Financeiras

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.13+-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/XGBoost-2.0+-green.svg" alt="XGBoost">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<p align="center">
  <b>Pipeline completo de detecção de fraudes em transações bancárias</b><br>
  Dados sintéticos → EDA → Balanceamento → Modelos Avançados → Explicabilidade SHAP
</p>

---

## 📋 Sumário

- [Sobre o Projeto](#-sobre-o-projeto)
- [Arquitetura](#-arquitetura)
- [Tecnologias](#-tecnologias)
- [Como Executar](#-como-executar)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Resultados](#-resultados)
- [Modelos Implementados](#-modelos-implementados)
- [Explicabilidade](#-explicabilidade)
- [Próximos Passos](#-próximos-passos)
- [Licença](#-licença)

---

## 🎯 Sobre o Projeto

Este projeto demonstra um pipeline completo de **detecção de anomalias em transações financeiras**, utilizando dados sintéticos gerados aleatoriamente para simular um cenário realista de fraude bancária.

### Características principais:

- ✅ **Dataset sintético** de 10.000 transações com 20 features e desbalanceamento realista (~1% de fraudes)
- ✅ **4 técnicas de balanceamento** comparadas: SMOTE, Undersampling, SMOTEENN e ADASYN
- ✅ **6 modelos** treinados: supervisionados, não-supervisionados e híbridos
- ✅ **Avaliação robusta** com AUC-PR, custo de erro e matrizes de confusão
- ✅ **Explicabilidade com SHAP** para transparência e auditoria
- ✅ **Notebook 100% funcional** no Google Colab / Jupyter

> ⚠️ **Aviso:** Os dados utilizados são **sintéticos** e gerados para fins educacionais. Em produção, substitua por dados reais anonimizados.

---

## 🏗️ Arquitetura

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Geração de     │────▶│  Pré-processa-   │────▶│  Balanceamento  │
│  Dados          │     │  mento           │     │  de Classes     │
│  (make_classif) │     │  (StandardScaler)│     │  (SMOTE, etc.)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Explicabilidade│◀────│  Avaliação       │◀────│  Modelos        │
│  (SHAP)         │     │  (AUC-PR, Custo) │     │  (XGB, AE, IF)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

---

## 🛠️ Tecnologias

| Categoria | Bibliotecas |
|-----------|-------------|
| **Manipulação de Dados** | `pandas`, `numpy` |
| **Machine Learning** | `scikit-learn`, `xgboost`, `imbalanced-learn` |
| **Deep Learning** | `tensorflow` / `keras` |
| **Explicabilidade** | `shap` |
| **Visualização** | `matplotlib`, `seaborn`, `plotly` |
| **Ambiente** | Google Colab / Jupyter Notebook |

---

## 🚀 Como Executar

### Opção 1: Google Colab (Recomendado)

Clique no badge abaixo para abrir o notebook diretamente no Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChristYela/fraud-detection-ml/blob/main/notebooks/fraud_detection_pipeline.ipynb)

> Substitua `SEU_USUARIO` pelo seu nome de usuário no GitHub.

### Opção 2: Localmente

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/SEU_USUARIO/fraud-detection-ml.git
   cd fraud-detection-ml
   ```

2. **Crie um ambiente virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou: venv\Scripts\activate  # Windows
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute o notebook:**
   ```bash
   jupyter notebook notebooks/fraud_detection_pipeline.ipynb
   ```

---

## 📁 Estrutura do Repositório

```
fraud-detection-ml/
├── 📂 notebooks/
│   └── fraud_detection_pipeline.ipynb    # Notebook principal (este projeto)
├── 📂 src/
│   ├── __init__.py
│   ├── data_generator.py                 # Geração de dados sintéticos
│   ├── preprocessing.py                  # Pré-processamento e normalização
│   ├── models.py                         # Definição dos modelos
│   ├── evaluation.py                     # Funções de avaliação
│   └── explainability.py               # SHAP e visualizações
├── 📂 tests/
│   └── test_models.py                    # Testes unitários
├── 📄 requirements.txt                   # Dependências do projeto
├── 📄 README.md                          # Este arquivo
└── 📄 LICENSE                            # Licença MIT
```

---

## 📊 Resultados

### Dataset

| Métrica | Valor |
|---------|-------|
| Total de transações | 10.000 |
| Transações normais | ~9.900 (99%) |
| Transações fraudulentas | ~100 (1%) |
| Features | 20 |

### Modelos Avaliados

| Modelo | Tipo | Destaque |
|--------|------|----------|
| Logistic Regression | Supervisionado | Baseline interpretável |
| Random Forest | Supervisionado | Ensemble robusto |
| **XGBoost + SMOTE** | Supervisionado | Melhor AUC-PR |
| Isolation Forest | Não-supervisionado | Detecção de outliers |
| Local Outlier Factor | Não-supervisionado | Baseado em densidade |
| Autoencoder | Não-supervisionado | Deep learning para anomalias |

### Métricas de Avaliação

> ⚠️ Em dados desbalanceados, **AUC-PR (Average Precision)** é mais confiável que AUC-ROC.

As métricas incluem:
- **Precision**: Fraudes corretamente identificadas / Total marcado como fraude
- **Recall**: Fraudes detectadas / Total real de fraudes
- **F1-Score**: Média harmônica entre Precision e Recall
- **AUC-PR**: Área sob a curva Precision-Recall
- **Custo Estimado**: Cada FN = R$100, cada FP = R$5

---

## 🤖 Modelos Implementados

### Supervisionados
- **Logistic Regression** com `class_weight='balanced'`
- **Random Forest** com `class_weight='balanced_subsample'`
- **XGBoost** com `scale_pos_weight` e dados balanceados via SMOTE

### Não-supervisionados
- **Isolation Forest**: Isola anomalias medindo o número de divisões necessárias
- **Local Outlier Factor (LOF)**: Detecta anomalias pela densidade local dos dados
- **Autoencoder**: Rede neural que aprende a reconstruir transações normais; alto erro de reconstrução indica anomalia

### Estratégia Híbrida (Recomendada para Produção)
1. **Isolation Forest** filtra transações "claramente normais" (~90% do volume)
2. **XGBoost** analisa apenas as transações suspeitas restantes
3. Reduz latência e custo computacional em produção

---

## 🔍 Explicabilidade

O projeto utiliza **SHAP (SHapley Additive exPlanations)** para tornar as decisões do modelo transparentes:

- **Summary Plot**: Importância global das features
- **Waterfall Plot**: Explicação individual de uma transação específica
- **Force Plot**: Visualização interativa do impacto de cada feature
- **Dependence Plot**: Relação entre uma feature e seu impacto no modelo

> A explicabilidade é essencial para **compliance**, **auditoria** e **aprovação de modelos** em instituições financeiras.

---

## 🔮 Próximos Passos

- [ ] Substituir dados sintéticos pelo dataset [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/c/ieee-fraud-detection)
- [ ] Implementar pipeline com `sklearn.pipeline` e `GridSearchCV` para otimização de hiperparâmetros
- [ ] Criar API REST com **FastAPI** para predição em tempo real
- [ ] Adicionar monitoramento de **data drift** em produção
- [ ] Implementar ensemble híbrido em cascata (Isolation Forest → XGBoost)
- [ ] Adicionar testes de carga e benchmark de latência
- [ ] Criar dashboard com Streamlit para visualização de fraudes em tempo real

---

## 🤝 Contribuição

Contribuições são bem-vindas! Siga os passos:

1. Faça um **fork** do projeto
2. Crie uma **branch** (`git checkout -b feature/nova-funcionalidade`)
3. Faça o **commit** (`git commit -m 'Adiciona nova funcionalidade'`)
4. Faça o **push** (`git push origin feature/nova-funcionalidade`)
5. Abra um **Pull Request**

---

## 📚 Referências

- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [SHAP: A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [Autoencoders for Anomaly Detection](https://www.deeplearningbook.org/)

---

## 📄 Licença

Este projeto está licenciado sob a licença **MIT**. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

<p align="center">
  Desenvolvido com 💜 para fins educacionais e de portfólio.
</p>
