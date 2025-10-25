import glob

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def preprocess_data(file_pattern="data/*.csv"):
    """Loads, combines, cleans, and feature-engineers player data from CSV files."""
    files = glob.glob(file_pattern)
    all_dfs = []

    for file in files:
        season = file.split("_")[-1].replace(".csv", "")
        df = pd.read_csv(file, encoding="latin1")
        df["season"] = season
        all_dfs.append(df)

    dfs = pd.concat(all_dfs, ignore_index=True)

    features = [
        'name', 'position', 'assists', 'bonus', 'clean_sheets',
        'goals_conceded', 'goals_scored', 'minutes',
        'saves', 'total_points', 'GW', 'season'
    ]
    dfs = dfs[features]

    dfs['season_points'] = dfs.groupby(['name', 'season'])['total_points'].transform('sum')

    # Compute previous season totals
    season_df = dfs.groupby(['name', 'season'])['season_points'].first().reset_index()
    season_df['prev_season_points'] = season_df.groupby('name')['season_points'].shift(1)
    season_df['prev2_season_points'] = season_df.groupby('name')['season_points'].shift(2)

    dfs = dfs.merge(
        season_df[['name', 'season', 'prev_season_points', 'prev2_season_points']],
        on=['name', 'season'], how="left"
    )

    # Fill missing previous points
    dfs['prev_season_points'] = dfs['prev_season_points'].fillna(dfs['prev2_season_points'])
    dfs.drop(columns=['prev2_season_points'], inplace=True)

    position_avg = dfs.groupby('position')['season_points'].mean()

    dfs['prev_season_points'] = (
        dfs['prev_season_points']
        .fillna(dfs['position'].map(position_avg))
        .round(1)
    )

    dfs.sort_values(['name', 'season', 'GW'], inplace=True, ignore_index=True)

    rolling_stats = ['assists', 'bonus', 'clean_sheets', 'goals_conceded',
                 'goals_scored', 'minutes', 'saves', 'total_points']

    for col in rolling_stats:
        dfs[f'{col}_rolling3'] = (
            dfs.groupby(['name', 'season'])[col]
            .transform(lambda x: x.shift(1).rolling(3).mean())
        )

    dfs[ [f'{c}_rolling3' for c in rolling_stats] ] = dfs[
        [f'{c}_rolling3' for c in rolling_stats]
    ].round(2)


    dfs = dfs.drop(
        ['position', 'assists', 'bonus', 'clean_sheets',
         'goals_conceded', 'goals_scored', 'minutes',
         'saves', 'season_points'],
        axis=1
    )

    dfs = dfs.dropna().reset_index(drop=True)
    dfs = dfs[[c for c in dfs if c != 'total_points'] + ['total_points']]
    return dfs


def scale_dataset(dataframe):
    """Scale features, return identifiers and arrays for model input."""
    exclude_cols = ['name', 'season', 'GW']
    feature_cols = [c for c in dataframe.columns[:-1] if c not in exclude_cols]
    X = dataframe[feature_cols].values
    y = dataframe[dataframe.columns[-1]].values
    ids = dataframe[exclude_cols].reset_index(drop=True)
    return ids, X, y


def prepare_next_gw_data(dfs, season, gw):
    """Duplicate previous GW data to create placeholders for the next GW."""
    prev_gw = gw - 1
    prev_gw_data = dfs[(dfs['GW'] == prev_gw) & (dfs['season'] == season)].copy()

    if prev_gw_data.empty:
        raise ValueError(f"No data found for previous GW ({prev_gw}) in {season}.")

    next_gw_data = prev_gw_data.copy()
    next_gw_data['GW'] = gw
    next_gw_data['total_points'] = 0 

    dfs_extended = pd.concat([dfs, next_gw_data], ignore_index=True)
    return dfs_extended, next_gw_data


def train_model(X_train, y_train, X_valid, y_valid):
    """Train and validate XGBoost model."""
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    y_valid_pred = model.predict(X_valid)
    mse_valid = mean_squared_error(y_valid, y_valid_pred)
    r2_valid = r2_score(y_valid, y_valid_pred)

    print(f"Validation MSE: {mse_valid:.2f} | R²: {r2_valid:.2f}")
    return model


def generate_prediction(dfs, season="2025-26", gw=7):
    """Generate player point predictions for a given GW and season."""
    # Prepare next GW data
    dfs_extended, next_gw_data = prepare_next_gw_data(dfs, season, gw)

    # Training/validation split (excluding current GW)
    train_valid_data = dfs_extended[~((dfs_extended['GW'] == gw) & (dfs_extended['season'] == season))]
    train_data, valid_data = train_test_split(train_valid_data, test_size=0.2, random_state=42)

    ids_train, X_train, y_train = scale_dataset(train_data)
    ids_valid, X_valid, y_valid = scale_dataset(valid_data)
    ids_gw, X_gw, _ = scale_dataset(next_gw_data)

    # Train model
    model = train_model(X_train, y_train, X_valid, y_valid)

    # Predict next GW
    y_pred = np.clip(model.predict(X_gw), 0, None)

    predictions = pd.DataFrame({
        'name': ids_gw['name'],
        'season': ids_gw['season'],
        'GW': ids_gw['GW'],
        'predicted_points': y_pred
    }).sort_values('predicted_points', ascending=False).reset_index(drop=True)

    return predictions


dfs = preprocess_data("data/*.csv")
predictions = generate_prediction(dfs, season="2025-26", gw=7)
print(predictions.head(20))
