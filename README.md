# SISCA — Stock prediction example

Repositório pequeno para extrair features de séries temporais de ações (TSLA), treinar modelos simples e visualizar features.

**Estrutura relevante**

- Modelo/: código fonte para geração de features, treino e plots
  - Modelo/use_features.py — gera `data/TSLA_features.csv` a partir de `data/TSLA.csv`
  - Modelo/model_dev.py — treina `LinearRegression` e `RandomForestRegressor` e imprime métricas
  - Modelo/model_comparison.py — compara 6+ modelos com 8 combinações de features (novo!)
  - Modelo/plot_features_TSLA.py — plota preços, MAs, RSI, ATR e volatilidade
  - Modelo/realtime_polling.py — polling com yfinance para atualizar dados em tempo real
- data/: ficheiros `TSLA.csv` (raw) e `TSLA_features.csv` (gerado)

Requisitos

- Python 3.8+
- Instalar dependências:

```bash
pip install -r requirements.txt
```

Como usar

**0. Verificar estado dos dados (opcional mas recomendado):**

```bash
python Modelo/check_data.py
```

1. Gerar features (executar uma vez ou quando `data/TSLA.csv` for atualizado):

```bash
python Modelo/use_features.py
```

2. Treinar e avaliar modelos (baseline: LinearRegression + RandomForest):

```bash
python Modelo/model_dev.py
```

3. **Comparar modelos com diferentes combinações de features (novo!)**

```bash
python Modelo/model_comparison.py
```

Saída:

- Imprime tabela comparativa com métricas (MAE, RMSE, R²)
- Guarda resultados detalhados em `data/model_comparison_results.csv`
- Gera plots comparativos em `data/model_comparison_plots.png`

4. Gerar gráficos das features:

```bash
python Modelo/plot_features_TSLA.py
```

5. Atualizar dados em tempo real (polling com yfinance):

```bash
python Modelo/realtime_polling.py --ticker TSLA --interval 60 --once
```
