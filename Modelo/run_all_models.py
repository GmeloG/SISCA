from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import pmdarima as pm
from prophet import Prophet 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 1. Carregar dataset
# =========================

def load_feature_dataset(ticker: str = "TSLA") -> pd.DataFrame:
    """
    Lê o ficheiro XXX_features.csv (ou gera a partir do XXX.csv se for preciso)
    e faz uma limpeza básica (ordenar por data, remover NaN).
    """
    base_dir = Path(__file__).resolve().parents[1]  # .../SISCA
    data_dir = base_dir / "data"

    features_path = data_dir / f"{ticker}_features.csv"
    raw_path = data_dir / f"{ticker}.csv"

    if features_path.exists():
        df = pd.read_csv(features_path, parse_dates=["Date"])
    else:
        # fallback: tentar gerar a partir do CSV bruto
        from features import build_feature_dataframe

        if not raw_path.exists():
            raise FileNotFoundError(
                f"Nem {features_path} nem {raw_path} existem. "
                f"Coloca os ficheiros na pasta data/."
            )
        df_raw = pd.read_csv(raw_path)
        df = build_feature_dataframe(df_raw)
        df.to_csv(features_path, index=False)
        print(f"Gerado ficheiro de features em: {features_path}")

    df = df.sort_values("Date")
    df = df.dropna().reset_index(drop=True)  # garantir que não há NaN nas features
    return df


# =========================
# 2. Construir X, y e split
# =========================

def make_supervised_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Cria target: Close do dia seguinte, remove linhas sem target
    e devolve X (features), y (target) e lista de nomes das features.
    """
    df = df.copy()
    df["target_close_next"] = df["Close"].shift(-1)

    # remover a última linha (não tem target)
    df = df.dropna(subset=["target_close_next"]).reset_index(drop=True)

    # colunas a excluir de X
    drop_cols = ["Date", "target_close_next"]

    # se existirem colunas "brutas" que não queres usar:
    for col in ["raw_close", "change_percent", "avg_vol_20d", "Adj Close"]:
        if col in df.columns:
            drop_cols.append(col)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df["target_close_next"]

    # sanity check: remover qualquer NaN residual
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    return X, y, feature_cols


def time_series_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int]:
    """
    Split temporal 80/20 (por default) sem baralhar.
    Devolve também o índice de split.
    """
    n = len(X)
    split_idx = int(n * (1 - test_size))

    X_train = X.iloc[:split_idx, :].copy()
    X_test = X.iloc[split_idx:, :].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test, split_idx


# =========================
# 3. Seleção de features
# =========================

def rank_features_by_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
    n_top: int = 10,
) -> List[Tuple[str, float]]:
    """
    Usa RandomForestRegressor para calcular importâncias de features
    e devolve o top n_top ordenado.
    """
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    top_features = [(feature_cols[i], float(importances[i])) for i in indices[:n_top]]
    return top_features


# =========================
# 4. Métricas e plot
# =========================

def evaluate_regression_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n=== Resultados {model_name} ===")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")


def plot_predictions(
    dates: pd.Series,
    prices: pd.Series,
    split_idx: int,
    y_test_pred: np.ndarray,
    title: str,
    save_path: Path | None = None,
) -> None:
    """
    Faz o plot no formato dos gráficos que mostraste:
    - azul: dados de treino
    - laranja: dados reais na janela de teste
    - verde: previsões na janela de teste
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates.iloc[:split_idx], prices.iloc[:split_idx], label="Model Training Data")
    plt.plot(dates.iloc[split_idx:], prices.iloc[split_idx:], label="Actual Data")
    plt.plot(dates.iloc[split_idx:], y_test_pred, label="Predicted Data")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.show()


# =========================
# 5. Modelos
# =========================
def create_lstm_sequences(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria janelas temporais para LSTM.
    Cada amostra usa 'lookback' dias de features para prever o preço do dia seguinte.
    """
    X_seq = []
    y_seq = []
    for i in range(lookback - 1, len(X_arr)):
        X_seq.append(X_arr[i - lookback + 1 : i + 1, :])
        y_seq.append(y_arr[i])
    return np.array(X_seq), np.array(y_seq)

def run_moving_average_baseline(
    df_sup: pd.DataFrame,
    split_idx: int,
    window: int = 50,
) -> np.ndarray:
    """
    Baseline simples: previsão = SMA(window).
    Assume que df_sup já não tem NaN na SMA.
    """
    close = df_sup["Close"]
    # se já existir SMA correspondente, usa; se não, calcula
    sma_col = f"SMA_{window}"
    if sma_col in df_sup.columns:
        sma = df_sup[sma_col]
    else:
        sma = close.rolling(window=window, min_periods=window).mean()

    y_pred_test = sma.iloc[split_idx:].to_numpy()
    return y_pred_test


def run_linear_regression(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_subset: List[str] | None = None,
) -> np.ndarray:
    """
    Regressão linear com StandardScaler.
    Se feature_subset for dado, usa só essas colunas.
    """
    if feature_subset is not None:
        X_train = X_train[feature_subset]
        X_test = X_test[feature_subset]

    linreg_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    print("\nTreinar LinearRegression...")
    linreg_pipeline.fit(X_train, y_train)
    y_pred = linreg_pipeline.predict(X_test)
    evaluate_regression_model(y_test, y_pred, "LinearRegression")
    return y_pred


def run_knn(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_subset: List[str] | None = None,
    n_neighbors: int = 10,
) -> np.ndarray:
    """
    K-Nearest Neighbors com scaling (muito importante para KNN).
    """
    if feature_subset is not None:
        X_train = X_train[feature_subset]
        X_test = X_test[feature_subset]

    knn_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=n_neighbors)),
        ]
    )

    print("\nTreinar KNeighborsRegressor...")
    knn_pipeline.fit(X_train, y_train)
    y_pred = knn_pipeline.predict(X_test)
    evaluate_regression_model(y_test, y_pred, "K-Nearest Neighbors")
    return y_pred


def run_auto_arima(
    close_series: pd.Series,
    split_idx: int,
) -> np.ndarray:
    """
    Auto ARIMA univariado sobre a série Close.
    Treina na parte de treino e faz forecast multi-step = tamanho da parte de teste.
    """
    train_close = close_series.iloc[:split_idx]
    test_close = close_series.iloc[split_idx:]

    print("\nTreinar Auto ARIMA...")
    model = pm.auto_arima(
        train_close,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )

    y_pred = model.predict(n_periods=len(test_close))
    evaluate_regression_model(test_close.to_numpy(), y_pred, "Auto ARIMA")
    return y_pred


def run_prophet(
    dates: pd.Series,
    close_series: pd.Series,
    split_idx: int,
) -> np.ndarray:
    """
    Prophet univariado sobre Close.
    Treina até split_idx e prevê até ao fim dos dados.
    """
    df_prophet = pd.DataFrame({"ds": dates, "y": close_series})
    train_prophet = df_prophet.iloc[:split_idx]

    print("\nTreinar Prophet...")
    m = Prophet(daily_seasonality=True)
    m.fit(train_prophet)

    # datas até ao final do dataset -> últimas len(test) entradas são a janela de teste
    future = df_prophet[["ds"]]
    forecast = m.predict(future)

    yhat = forecast["yhat"].to_numpy()
    y_pred_test = yhat[split_idx:]

    test_close = close_series.iloc[split_idx:]
    evaluate_regression_model(test_close.to_numpy(), y_pred_test, "Prophet")

    return y_pred_test

def run_lstm(
    X: pd.DataFrame,
    y: pd.Series,
    split_idx: int,
    feature_subset: List[str] | None = None,
    lookback: int = 60,
) -> np.ndarray:
    """
    Modelo LSTM para prever o Close do dia seguinte.
    Usa janelas temporais de 'lookback' dias e devolve as previsões
    no período de teste (em escala original, $).
    """
    # selecionar features
    if feature_subset is not None:
        X_sel = X[feature_subset].to_numpy()
    else:
        X_sel = X.to_numpy()

    y_arr = y.to_numpy().reshape(-1, 1)

    # escalar X e y para [0,1]
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = x_scaler.fit_transform(X_sel)
    y_scaled = y_scaler.fit_transform(y_arr).flatten()

    # criar sequências para o LSTM
    X_seq, y_seq = create_lstm_sequences(X_scaled, y_scaled, lookback=lookback)

    # número de amostras originais
    n = len(y)
    # índice de split em termos de sequências
    # (primeira sequência tem target no índice lookback-1)
    split_idx_seq = split_idx - (lookback - 1)
    if split_idx_seq <= 0:
        raise ValueError(
            f"split_idx_seq <= 0: reduz o lookback (lookback={lookback}, split_idx={split_idx})"
        )

    X_train_seq = X_seq[:split_idx_seq]
    X_test_seq = X_seq[split_idx_seq:]
    y_train_seq = y_seq[:split_idx_seq]
    y_test_seq = y_seq[split_idx_seq:]

    n_features = X_train_seq.shape[2]

    print("\nTreinar LSTM...")
    model = Sequential()
    model.add(
        LSTM(
            units=50,
            activation="tanh",
            input_shape=(lookback, n_features),
        )
    )
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train_seq,
        y_train_seq,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[es],
        verbose=0,
    )

    y_pred_scaled = model.predict(X_test_seq, verbose=0).flatten()

    # voltar à escala original ($)
    y_test_unscaled = y_scaler.inverse_transform(
        y_test_seq.reshape(-1, 1)
    ).flatten()
    y_pred_unscaled = y_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).flatten()

    evaluate_regression_model(y_test_unscaled, y_pred_unscaled, "LSTM")

    return y_pred_unscaled

# =========================
# 6. main()
# =========================

def main() -> None:
    ticker = "TSLA"  # depois podes mudar para "MSFT", etc.
    print(f"A carregar dataset de features para {ticker}...")
    df = load_feature_dataset(ticker)

    # construir X, y para modelos supervisionados (LR, KNN, etc.)
    X, y, feature_cols = make_supervised_dataset(df)
    dates = df["Date"].iloc[: len(X)]      # alinhar datas com X/y
    close = df["Close"].iloc[: len(X)]     # idem

    X_train, X_test, y_train, y_test, split_idx = time_series_train_test_split(
        X, y, test_size=0.2
    )

    print(f"Número total de amostras: {len(X)}")
    print(f"Tamanho treino: {len(X_train)}, teste: {len(X_test)}")
    print(f"Número de features: {len(feature_cols)}")

    # -------------------------
    # Seleção de features
    # -------------------------
    print("\nA calcular importâncias de features com RandomForest...")
    top_features = rank_features_by_importance(X_train, y_train, feature_cols, n_top=10)
    print("\nTop 10 features mais importantes:")
    for name, score in top_features:
        print(f"  {name:<20s}  importance = {score:.4f}")

    best_feature_names = [name for name, _ in top_features]

    # DataFrame "supervisionado" para baseline e plots
    df_supervised = pd.DataFrame(
        {
            "Date": dates.reset_index(drop=True),
            "Close": close.reset_index(drop=True),
        }
    )
    for col in ["SMA_10", "SMA_20", "SMA_50", "SMA_200"]:
        if col in df.columns:
            df_supervised[col] = df[col].iloc[: len(X)].reset_index(drop=True)

    # -------------------------
    # 1) Baseline Moving Average
    # -------------------------
    y_pred_ma = run_moving_average_baseline(df_supervised, split_idx, window=50)
    evaluate_regression_model(y_test.to_numpy(), y_pred_ma, "Moving Average (SMA_50)")
    plot_predictions(
        df_supervised["Date"],
        df_supervised["Close"],
        split_idx,
        y_pred_ma,
        title="Stock Price Prediction by Moving Averages",
    )

    # -------------------------
    # 2) Linear Regression
    # -------------------------
    y_pred_lin = run_linear_regression(
        X_train, X_test, y_train, y_test, feature_subset=best_feature_names
    )
    plot_predictions(
        df_supervised["Date"],
        df_supervised["Close"],
        split_idx,
        y_pred_lin,
        title="Stock Price Prediction by Linear Regression",
    )

    # -------------------------
    # 3) K-Nearest Neighbors
    # -------------------------
    y_pred_knn = run_knn(
        X_train, X_test, y_train, y_test, feature_subset=best_feature_names, n_neighbors=10
    )
    plot_predictions(
        df_supervised["Date"],
        df_supervised["Close"],
        split_idx,
        y_pred_knn,
        title="Stock Price Prediction by K-Nearest Neighbors",
    )

    # -------------------------
    # 4) Auto ARIMA (univariado)
    # -------------------------
    y_pred_arima = run_auto_arima(close, split_idx)
    plot_predictions(
        df_supervised["Date"],
        df_supervised["Close"],
        split_idx,
        y_pred_arima,
        title="Stock Price Prediction by Auto ARIMA",
    )

    # -------------------------
    # 5) Prophet (univariado)
    # -------------------------
    y_pred_prophet = run_prophet(dates, close, split_idx)
    plot_predictions(
        df_supervised["Date"],
        df_supervised["Close"],
        split_idx,
        y_pred_prophet,
        title="Stock Price Prediction by FB Prophet",
    )

    # -------------------------
    # 6) LSTM
    # -------------------------
    y_pred_lstm = run_lstm(
        X,
        y,
        split_idx=split_idx,
        feature_subset=best_feature_names,  # usa top features da RandomForest
        lookback=60,                        # podes ajustar (30, 50, 60, ...)
    )
    plot_predictions(
        df_supervised["Date"],
        df_supervised["Close"],
        split_idx,
        y_pred_lstm,
        title="Stock Price Prediction by LSTM",
    )

if __name__ == "__main__":
    main()
