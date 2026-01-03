from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def load_feature_dataset() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parents[1]
    features_path = base_dir / "data" / "TSLA_features.csv"
    df = pd.read_csv(features_path, parse_dates=["Date"])
    df = df.sort_values("Date")
    df = df.dropna().reset_index(drop=True)
    return df


def make_features_and_target(df: pd.DataFrame):
    df = df.copy()
    df["target_close_next"] = df["Close"].shift(-1)
    df = df.dropna(subset=["target_close_next"]).reset_index(drop=True)
    
    drop_cols = ["Date", "target_close_next"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df["target_close_next"]
    
    return X, y, feature_cols


def train_test_split_time_series(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    n = len(X)
    split_idx = int(n * (1 - test_size))
    
    X_train = X.iloc[:split_idx, :].copy()
    X_test = X.iloc[split_idx:, :].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    
    return X_train, X_test, y_train, y_test


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R²": r2}


def get_feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    """Define feature combinations for testing."""
    groups = {
        "All": feature_cols,
        "Price_Only": [c for c in feature_cols if c in ["Open", "High", "Low", "Close", "Volume"]],
        "Technical": [c for c in feature_cols if any(x in c for x in ["SMA", "EMA", "RSI", "ATR", "volatility"])],
        "Momentum": [c for c in feature_cols if any(x in c for x in ["return", "RSI", "volatility"])],
        "MovingAvg": [c for c in feature_cols if any(x in c for x in ["SMA", "EMA"])],
        "Volume": [c for c in feature_cols if "Volume" in c],
        "Time": [c for c in feature_cols if c in ["year", "month", "day", "dayofweek", "is_month_start", "is_month_end"]],
        "Tech_Volume": [c for c in feature_cols if any(x in c for x in ["SMA", "EMA", "RSI", "ATR", "volatility", "Volume"])],
    }
    
    # Filter to keep only groups with existing columns
    return {k: v for k, v in groups.items() if v}


def build_models() -> dict:
    """Define models to compare."""
    models = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.1, max_iter=10000)),
        ]),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=100, gamma="scale")),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }
    
    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        )
    
    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1
        )
    
    return models


def main() -> None:
    print("Loading data...")
    df = load_feature_dataset()
    print(f"Dataset shape: {df.shape}")
    
    X, y, feature_cols = make_features_and_target(df)
    print(f"Features: {len(feature_cols)}, Samples: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y, test_size=0.2)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    feature_groups = get_feature_groups(feature_cols)
    models = build_models()
    
    results = []
    
    for group_name, group_features in feature_groups.items():
        X_train_group = X_train[group_features]
        X_test_group = X_test[group_features]
        
        print(f"\n{'='*60}")
        print(f"Feature Group: {group_name} ({len(group_features)} features)")
        print(f"{'='*60}")
        
        for model_name, model in models.items():
            try:
                model.fit(X_train_group, y_train)
                y_pred = model.predict(X_test_group)
                metrics = evaluate_model(y_test.values, y_pred)
                
                result = {
                    "Feature_Group": group_name,
                    "Model": model_name,
                    "Num_Features": len(group_features),
                    **metrics,
                }
                results.append(result)
                
                print(f"{model_name:20s} | MAE: {metrics['MAE']:8.4f} | RMSE: {metrics['RMSE']:8.4f} | R²: {metrics['R²']:7.4f}")
            except Exception as e:
                print(f"{model_name:20s} | Error: {str(e)[:50]}")
    
    results_df = pd.DataFrame(results)
    
    # Save detailed results
    base_dir = Path(__file__).resolve().parents[1]
    results_path = base_dir / "data" / "model_comparison_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n\nDetailed results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("BEST MODELS BY METRIC")
    print("="*60)
    best_r2 = results_df.loc[results_df["R²"].idxmax()]
    best_rmse = results_df.loc[results_df["RMSE"].idxmin()]
    best_mae = results_df.loc[results_df["MAE"].idxmin()]
    
    print(f"\nBest R² (Explained Variance):")
    print(f"  Model: {best_r2['Model']}, Feature Group: {best_r2['Feature_Group']}, R² = {best_r2['R²']:.4f}")
    
    print(f"\nBest RMSE (Prediction Error):")
    print(f"  Model: {best_rmse['Model']}, Feature Group: {best_rmse['Feature_Group']}, RMSE = {best_rmse['RMSE']:.4f}")
    
    print(f"\nBest MAE (Mean Absolute Error):")
    print(f"  Model: {best_mae['Model']}, Feature Group: {best_mae['Feature_Group']}, MAE = {best_mae['MAE']:.4f}")
    
    # Plot comparisons
    plot_model_comparison(results_df)


def plot_model_comparison(results_df: pd.DataFrame) -> None:
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Model Comparison Across Feature Groups", fontsize=16)
    
    # Plot 1: R² by Model and Feature Group
    ax = axes[0, 0]
    pivot_r2 = results_df.pivot_table(values="R²", index="Model", columns="Feature_Group", aggfunc="mean")
    pivot_r2.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("R² Score by Model and Feature Group")
    ax.set_ylabel("R²")
    ax.legend(title="Feature Group", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    
    # Plot 2: RMSE by Model
    ax = axes[0, 1]
    pivot_rmse = results_df.pivot_table(values="RMSE", index="Model", columns="Feature_Group", aggfunc="mean")
    pivot_rmse.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("RMSE by Model and Feature Group")
    ax.set_ylabel("RMSE")
    ax.legend(title="Feature Group", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    
    # Plot 3: Best R² per Feature Group
    ax = axes[1, 0]
    best_per_group = results_df.loc[results_df.groupby("Feature_Group")["R²"].idxmax()]
    x_pos = np.arange(len(best_per_group))
    ax.bar(x_pos, best_per_group["R²"].values, color="steelblue")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(best_per_group["Feature_Group"].values, rotation=45, ha="right")
    ax.set_title("Best R² per Feature Group")
    ax.set_ylabel("R²")
    ax.grid(axis="y", alpha=0.3)
    
    # Plot 4: Model Performance Heatmap (R²)
    ax = axes[1, 1]
    heatmap_data = results_df.pivot_table(values="R²", index="Model", columns="Feature_Group", aggfunc="mean")
    im = ax.imshow(heatmap_data.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha="right")
    ax.set_yticklabels(heatmap_data.index)
    ax.set_title("R² Heatmap")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("R²")
    
    plt.tight_layout()
    
    base_dir = Path(__file__).resolve().parents[1]
    plot_path = base_dir / "data" / "model_comparison_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plots saved to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
