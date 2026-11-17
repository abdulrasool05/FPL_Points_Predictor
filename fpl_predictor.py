import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CORE_STATS = [
    "assists", "bonus", "clean_sheets", "goals_conceded",
    "goals_scored", "minutes", "saves", "total_points",
]
OPTIONAL_COLUMNS = {"was_home": False, "opponent_team": np.nan}
POSITIONS = ["DEF", "FWD", "GK", "MID"]
TARGET = "total_points"


def load_data(file_pattern="data/*.csv"):
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files matched: '{file_pattern}'")

    frames = []
    for filepath in files:
        season = Path(filepath).stem.split("_")[-1]
        df = pd.read_csv(filepath, encoding="latin1")
        df["season"] = season
        frames.append(df)
        log.info("Loaded %s  (%s rows)", Path(filepath).name, len(df))

    combined = pd.concat(frames, ignore_index=True)
    log.info("Combined: %s rows × %s columns", *combined.shape)
    return combined


def preprocess(df):
    # Keep only the columns needed for modelling
    keep = ["name", "position", "GW", "season"] + CORE_STATS
    for col, fallback in OPTIONAL_COLUMNS.items():
        if col not in df.columns:
            df[col] = fallback
        keep.append(col)
    df = df[keep].copy()

    df["was_home"] = df["was_home"].astype(bool).astype(int)
    df["opponent_team"] = pd.to_numeric(df["opponent_team"], errors="coerce")
    df = df.sort_values(["name", "season", "GW"], ignore_index=True)

    # 1. Previous season points (with fallbacks to 2 seasons ago, then position average)
    df["season_points"] = df.groupby(["name", "season"])[TARGET].transform("sum")

    season_totals = df.groupby(["name", "season"])["season_points"].first().reset_index()
    season_totals = season_totals.sort_values(["name", "season"])
    season_totals["prev_season_points"] = season_totals.groupby("name")["season_points"].shift(1)
    season_totals["prev2_season_points"] = season_totals.groupby("name")["season_points"].shift(2)

    merge_cols = ["name", "season", "prev_season_points", "prev2_season_points"]
    df = df.merge(season_totals[merge_cols], on=["name", "season"], how="left")

    df["prev_season_points"] = df["prev_season_points"].fillna(df["prev2_season_points"])
    position_avg = df.groupby("position")["season_points"].mean()
    df["prev_season_points"] = df["prev_season_points"].fillna(df["position"].map(position_avg))
    df["prev_season_points"] = df["prev_season_points"].round(1)
    df = df.drop(columns=["prev2_season_points", "season_points"])

    # 2. Opponent strength: average points conceded to each position
    opp_strength = (
        df.groupby(["opponent_team", "position"])[TARGET]
        .mean()
        .rename("opp_strength")
        .reset_index()
    )
    df = df.merge(opp_strength, on=["opponent_team", "position"], how="left")
    df["opp_strength"] = df["opp_strength"].fillna(df["opp_strength"].median())
    df = df.drop(columns=["opponent_team"])

    # 3. Rolling form averages (last 3 and 5 gameweeks)
    for window in (3, 5):
        for col in CORE_STATS:
            grouped = df.groupby(["name", "season"])[col]
            # shift(1) prevents leakage of the current game's stats
            df[f"{col}_roll{window}"] = grouped.transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )

    # 4. Cumulative season points-per-game so far
    points_grouped = df.groupby(["name", "season"])[TARGET]
    df["season_ppg"] = points_grouped.transform(lambda s: s.shift(1).expanding().mean())

    # 5. Position one-hot encoding
    for pos in POSITIONS:
        df[f"pos_{pos}"] = (df["position"] == pos).astype(int)
    df = df.drop(columns=["position"])

    # 6. Drop raw match stats (model now uses rolling features)
    raw_stats_to_drop = [c for c in CORE_STATS if c != TARGET]
    df = df.drop(columns=raw_stats_to_drop)
    df = df.dropna().reset_index(drop=True)

    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(2)

    # Ensure target is the last column
    feature_cols = [c for c in df.columns if c != TARGET]
    df = df[feature_cols + [TARGET]]
    log.info("Preprocessed: %s rows × %s columns", *df.shape)
    return df


def split_features_target(df):
    id_cols = ["name", "season", "GW"]
    feature_cols = [c for c in df.columns if c not in id_cols and c != TARGET]

    ids = df[id_cols].reset_index(drop=True)
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    return ids, X, y, feature_cols


def temporal_split(df, val_fraction=0.2):
    # Sort seasons chronologically to evaluate on the most recent data
    seasons = sorted(df["season"].unique(), key=lambda s: int(s.split("-")[0]))
    num_val_seasons = max(1, int(len(seasons) * val_fraction))
    val_seasons = set(seasons[-num_val_seasons:])

    train = df[~df["season"].isin(val_seasons)].copy()
    val = df[df["season"].isin(val_seasons)].copy()

    log.info("Train seasons: %s", sorted(set(seasons) - val_seasons))
    log.info("Val seasons:   %s", sorted(val_seasons))
    return train, val


def train_model(X_train, y_train, X_val, y_val, feature_names=None):
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    log.info("Validation — MSE: %.2f | MAE: %.2f | R²: %.3f | Best trees: %d", mse, mae, r2, model.best_iteration)

    # Feature importance: shows which variables drove the model's decisions
    if feature_names:
        importances = pd.Series(model.feature_importances_, index=feature_names)
        top10 = importances.sort_values(ascending=False).head(10)
        log.info("Top 10 features by importance:")
        for name, score in top10.items():
            log.info("  %-25s %.4f", name, score)

    return model


def prepare_next_gw(df, season, gw):
    prev_gw = gw - 1
    prev_data = df[(df["GW"] == prev_gw) & (df["season"] == season)]

    if prev_data.empty:
        available = sorted(df[df["season"] == season]["GW"].unique())
        raise ValueError(f"No data for GW {prev_gw} in season {season}. Available GWs: {available}")

    next_gw = prev_data.copy()
    next_gw["GW"] = gw
    next_gw[TARGET] = 0

    extended = pd.concat([df, next_gw], ignore_index=True)
    return extended, next_gw


def generate_predictions(df, season="2025-26", gw=7, top_n=20):
    log.info("Predicting GW %d, season %s", gw, season)
    extended, next_gw = prepare_next_gw(df, season, gw)

    # Exclude the target gameweek from training/validation
    train_val_data = extended[~((extended["GW"] == gw) & (extended["season"] == season))]
    train_data, val_data = temporal_split(train_val_data)

    ids_train, X_train, y_train, feature_names = split_features_target(train_data)
    ids_val, X_val, y_val, _ = split_features_target(val_data)
    ids_gw, X_gw, _, _ = split_features_target(next_gw)

    model = train_model(X_train, y_train, X_val, y_val, feature_names=feature_names)

    raw_preds = model.predict(X_gw)
    y_pred = np.clip(raw_preds, 0, None).round(1)

    predictions = pd.DataFrame({
        "name": ids_gw["name"].values,
        "predicted_points": y_pred,
    })
    predictions = predictions.sort_values("predicted_points", ascending=False).reset_index(drop=True)
    predictions.index += 1
    predictions.index.name = "rank"

    print(f"\nTop {top_n} predicted points — GW {gw}, {season}")
    print(predictions.head(top_n).to_string())
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Predict FPL player points for a given gameweek.")
    parser.add_argument("--season", default="2025-26", help="Season (e.g. 2025-26)")
    parser.add_argument("--gw", type=int, default=7, help="Target Gameweek")
    parser.add_argument("--top", type=int, default=20, help="Number of players to display")
    parser.add_argument("--data", default="data/*.csv", help="Pattern for data files")
    args = parser.parse_args()

    raw = load_data(args.data)
    processed = preprocess(raw)
    generate_predictions(processed, season=args.season, gw=args.gw, top_n=args.top)


if __name__ == "__main__":
    main()
