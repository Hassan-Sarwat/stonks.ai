from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from src.strategies.base import IStrategy
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class XGBoostStrategy(IStrategy):
    """
    ML-based strategy using XGBoost to predict future returns.
    Uses walk-forward training to avoid look-ahead bias.
    """

    def __init__(
        self,
        train_split: float = 0.7,
        lookback_window: int = 60,
        prediction_horizon: int = 5,
        return_threshold: float = 0.001,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
    ):
        """
        Args:
            train_split: Fraction of data to use for initial training (0.7 = 70%)
            lookback_window: Number of periods to look back for features
            prediction_horizon: Number of periods ahead to predict
            return_threshold: Minimum predicted return to generate buy signal
            n_estimators: XGBoost n_estimators parameter
            max_depth: XGBoost max_depth parameter
            learning_rate: XGBoost learning_rate parameter
        """
        super().__init__()
        self.train_split = train_split
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.return_threshold = return_threshold
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        # Store trained models per symbol
        self.models: Dict[str, xgb.XGBRegressor] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: Dict[str, list] = {}

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical features from OHLCV data.
        Returns DataFrame with feature columns.
        """
        df = df.copy()

        # Drop datetime columns if they exist (we don't want datetime strings in features)
        # Store them separately if needed for indexing later
        datetime_cols = ["open_time", "close_time"]
        for col in datetime_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Price-based features
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Volatility features
        df["volatility_5"] = df["returns"].rolling(window=5).std()
        df["volatility_20"] = df["returns"].rolling(window=20).std()

        # Price momentum
        df["rsi_14"] = self._calculate_rsi(df["close"], 14)
        df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
        df["momentum_20"] = df["close"] / df["close"].shift(20) - 1

        # Moving averages
        df["sma_5"] = df["close"].rolling(window=5).mean()
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

        # Price position relative to range
        df["high_low_ratio"] = df["high"] / df["low"]
        df["close_position"] = (df["close"] - df["low"]) / (
            df["high"] - df["low"] + 1e-8
        )

        # Volume features
        df["volume_ma_5"] = df["volume"].rolling(window=5).mean()
        df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_ma_20"] + 1e-8)

        # Order flow features (if available)
        if "taker_buy_base" in df.columns:
            taker_sell_base = df["volume"] - df["taker_buy_base"]
            df["ofi"] = df["taker_buy_base"] - taker_sell_base
            df["ofi_normalized"] = df["ofi"] / (df["volume"] + 1e-8)
            df["buy_pressure"] = df["taker_buy_base"] / (df["volume"] + 1e-8)
        else:
            df["ofi"] = 0
            df["ofi_normalized"] = 0
            df["buy_pressure"] = 0.5

        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f"returns_lag_{lag}"] = df["returns"].shift(lag)
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag)

        # Bollinger Bands
        df["bb_sma"] = df["close"].rolling(window=20).mean()
        df["bb_std"] = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_sma"] + 2 * df["bb_std"]
        df["bb_lower"] = df["bb_sma"] - 2 * df["bb_std"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"] + 1e-8
        )

        return df

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Create target variable: future return over prediction_horizon periods.
        """
        future_return = df["close"].shift(-self.prediction_horizon) / df["close"] - 1
        return future_return

    def initialize(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Train XGBoost models for each symbol using walk-forward approach.
        """
        self.data = data
        print("Training XGBoost models for each symbol...")

        for symbol, df in data.items():
            print(f"  Processing {symbol}...")

            # Ensure data is sorted by time
            if "open_time" in df.columns:
                df = df.sort_values("open_time").reset_index(drop=True)
                time_col = "open_time"
            else:
                df = df.sort_index()
                time_col = None

            # Create features
            df_features = self._create_features(df)

            # Create target
            target = self._create_target(df_features)

            # Select feature columns (exclude non-feature columns)
            exclude_cols = [
                "open_time",
                "close_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "taker_buy_base",
                "taker_buy_quote",
                "trades",
            ]
            feature_cols = [
                col
                for col in df_features.columns
                if col not in exclude_cols and not col.startswith("target")
            ]

            # Safety check: ensure datetime columns are never included
            datetime_cols = ["open_time", "close_time"]
            feature_cols = [col for col in feature_cols if col not in datetime_cols]

            # Ensure all feature columns are numeric (not datetime/string)
            numeric_cols = []
            for col in feature_cols:
                if col in df_features.columns:
                    # Check if column is numeric
                    if pd.api.types.is_numeric_dtype(df_features[col]):
                        numeric_cols.append(col)
                    else:
                        # Try to convert to numeric
                        try:
                            pd.to_numeric(df_features[col], errors="raise")
                            numeric_cols.append(col)
                        except (ValueError, TypeError):
                            # Only warn for unexpected non-numeric columns (not datetime columns we already excluded)
                            if col not in ["open_time", "close_time"]:
                                print(
                                    f"    Warning: Skipping non-numeric column '{col}' for {symbol}"
                                )
            feature_cols = numeric_cols

            # Remove rows with NaN (from rolling calculations)
            valid_mask = ~(df_features[feature_cols].isna().any(axis=1) | target.isna())

            # Get valid rows and ensure all columns are numeric
            X_df = df_features[feature_cols][valid_mask].copy()
            y_series = target[valid_mask].copy()

            # Convert all feature columns to numeric, coercing errors to NaN
            for col in feature_cols:
                if col in X_df.columns:
                    X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

            # Drop rows that became NaN after numeric conversion
            final_valid = ~X_df.isna().any(axis=1)
            X_df_clean = X_df[final_valid]
            y_clean = y_series[final_valid]

            # Convert to numpy arrays
            X = X_df_clean.values.astype(np.float64)
            y = y_clean.values.astype(np.float64)

            if len(X) < self.lookback_window + self.prediction_horizon:
                print(f"    Warning: {symbol} has insufficient data ({len(X)} rows)")
                continue

            # Split into train/test
            split_idx = int(len(X) * self.train_split)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            if len(X_train) < 100:
                print(
                    f"    Warning: {symbol} has insufficient training data ({len(X_train)} rows)"
                )
                continue

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train XGBoost model
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )

            model.fit(X_train_scaled, y_train)

            # Store model and scaler
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.feature_names[symbol] = feature_cols

            # Evaluate on test set
            y_pred = model.predict(X_test_scaled)
            mse = np.mean((y_pred - y_test) ** 2)
            print(
                f"    Trained model for {symbol}: Test MSE = {mse:.6f}, "
                f"Train samples = {len(X_train)}, Test samples = {len(X_test)}"
            )

    def generate_signals(self) -> Dict[str, pd.DataFrame]:
        """
        Generate signals using trained XGBoost models.
        """
        signals = {}

        for symbol, df in self.data.items():
            if symbol not in self.models:
                # No model trained, return all zeros
                df = df.copy()
                df["signal"] = 0
                signals[symbol] = df[["signal"]]
                continue

            # Ensure data is sorted
            if "open_time" in df.columns:
                df = df.sort_values("open_time").reset_index(drop=True)
            else:
                df = df.sort_index().reset_index(drop=True)

            # Create features
            df_features = self._create_features(df)

            # Get feature columns
            feature_cols = self.feature_names[symbol]

            # Prepare data for prediction
            valid_mask = ~df_features[feature_cols].isna().any(axis=1)
            X = df_features[feature_cols][valid_mask].values

            if len(X) == 0:
                df["signal"] = 0
                signals[symbol] = df[["signal"]]
                continue

            # Scale and predict
            scaler = self.scalers[symbol]
            X_scaled = scaler.transform(X)
            predictions = self.models[symbol].predict(X_scaled)

            # Create signal column initialized to 0
            df["signal"] = 0

            # Create a predictions array aligned with the full dataframe
            # Initialize with NaN for invalid rows, then fill valid rows with predictions
            predictions_full = np.full(len(df), np.nan, dtype=np.float64)
            predictions_full[valid_mask] = predictions

            # Store predicted returns for debugging (optional)
            df["predicted_return"] = predictions_full

            # Generate signals based on predicted returns
            # Buy if predicted return > threshold (only for valid rows)
            # Sell if predicted return < -threshold (only for valid rows)
            buy_mask = valid_mask & (predictions_full > self.return_threshold)
            sell_mask = valid_mask & (predictions_full < -self.return_threshold)

            df.loc[buy_mask, "signal"] = 1
            df.loc[sell_mask, "signal"] = -1

            # Keep only signal column for return
            signals[symbol] = df[["signal"]]

        return signals

    def on_tick(
        self,
        timestamp: pd.Timestamp,
        prices: Dict[str, float],
        rows: Dict[str, pd.Series],
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Event-driven signal generation (for live trading compatibility).
        Not used in vectorized backtesting but kept for interface compliance.
        """
        signals = {}

        for symbol, row in rows.items():
            if symbol not in self.models:
                continue

            # This would require real-time feature calculation
            # For now, return None (use generate_signals instead)
            pass

        return signals if signals else None
