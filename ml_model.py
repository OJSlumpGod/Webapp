import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import talib

from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

# Optional: remove TSFresh if not using it.
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

# PyTorch device setup (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import your config/hyperparameters
from config import PYTORCH_HIDDEN_SIZE, PYTORCH_LEARNING_RATE, PYTORCH_EPOCHS, MODEL_CONFIG

# Import your strategy classes
from strategies import (
    TrendFollowingStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    MomentumStrategy,
    RangeBoundStrategy,
    ADXTrendStrategy,
    BollingerSqueezeStrategy,
    CrossoverStrategy,
    ScalpingStrategy
)

class ForexNet(nn.Module):
    """
    An enhanced feed-forward PyTorch model for binary classification.
    Includes hidden layers + ReLU + optional dropout/regularization if desired.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(ForexNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # First layer
        out = self.fc1(x)
        out = self.relu1(out)

        # Second layer
        out = self.fc2(out)
        out = self.relu2(out)

        # Output layer
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


class MLModel:
    """
    MLModel manages:
      - Feature engineering (TA-Lib + multi-strategy signals + Rolling / TSFresh),
      - Classical models (SGD, RF, GB) + a Voting ensemble,
      - A PyTorch model (ForexNet),
      - Saving/loading artifacts (scaler, PCA, models),
      - Training and evaluation routines.

    Typical usage:
      1) Initialize MLModel().
      2) data = ml_model.prepare_features(price_data)
      3) ml_model.train(X_train, y_train, X_val, y_val)
      4) predictions = ml_model.predict(X_test)
    """

    def __init__(self):
        # ---------------------------------------------------------------------
        # 1) Logging Setup
        # ---------------------------------------------------------------------
        self.logger = logging.getLogger('ml_model')
        self.logger.setLevel(logging.INFO)

        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # File-based logging
        file_handler = logging.FileHandler(os.path.join('logs', 'ml_model.log'))
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console logging
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(file_formatter)
        self.logger.addHandler(console_handler)

        self.logger.info("[MLModel] Initializing model pipeline...")

        # ---------------------------------------------------------------------
        # 2) Scaler & PCA
        # ---------------------------------------------------------------------
        # Adjust 'n_components' if you want fewer principal components.
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=15)

        # ---------------------------------------------------------------------
        # 3) PyTorch Hyperparameters
        # ---------------------------------------------------------------------
        self.pytorch_hidden_size = PYTORCH_HIDDEN_SIZE
        self.pytorch_epochs = PYTORCH_EPOCHS
        self.pytorch_lr = PYTORCH_LEARNING_RATE

        # ---------------------------------------------------------------------
        # 4) Model Artifact Paths
        # ---------------------------------------------------------------------
        self.models_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        self.model_files = {
            "sgd":      os.path.join(self.models_dir, "sgd_model.pkl"),
            "rf":       os.path.join(self.models_dir, "rf_model.pkl"),
            "gb":       os.path.join(self.models_dir, "gb_model.pkl"),
            "voting":   os.path.join(self.models_dir, "voting_model.pkl"),
            "pytorch":  os.path.join(self.models_dir, "pytorch_model.pth"),
            "scaler":   os.path.join(self.models_dir, "scaler.pkl"),
            "pca":      os.path.join(self.models_dir, "pca.pkl")
        }

        # ---------------------------------------------------------------------
        # 5) Classical Models
        # ---------------------------------------------------------------------
        # Use default or loaded models (SGD, RF, GB)
        self.models = {
            "sgd": self._load_model("sgd", default_model=SGDClassifier(max_iter=1000, tol=1e-3)),
            "rf":  self._load_model("rf",  default_model=RandomForestClassifier(n_estimators=100)),
            "gb":  self._load_model("gb",  default_model=GradientBoostingClassifier(n_estimators=100))
        }

        # Voting ensemble (hard voting by default)
        self.voting_model = VotingClassifier(
            estimators=[
                ('sgd', self.models["sgd"]),
                ('rf',  self.models["rf"]),
                ('gb',  self.models["gb"])
            ],
            voting='hard'
        )

        # Load existing Voting model if available
        if os.path.exists(self.model_files["voting"]):
            try:
                self.logger.info("[MLModel] Loading existing Voting classifier.")
                self.voting_model = joblib.load(self.model_files["voting"])
            except Exception as e:
                self.logger.error(f"[MLModel] Failed to load existing voting model: {e}")

        # ---------------------------------------------------------------------
        # 6) PyTorch Model Build/Load
        # ---------------------------------------------------------------------
        self.pytorch_model = self._build_pytorch_model()
        if os.path.exists(self.model_files["pytorch"]):
            self.logger.info("[MLModel] Loading existing PyTorch model state.")
            try:
                self.pytorch_model.load_state_dict(
                    torch.load(self.model_files["pytorch"], map_location=device)
                )
                self.pytorch_model.to(device)
            except Exception as ex:
                self.logger.error(f"[MLModel] Error loading PyTorch model: {ex}")
                self.logger.info("[MLModel] Re-initializing a fresh PyTorch model.")
                self.pytorch_model = self._build_pytorch_model()
        else:
            self.logger.info("[MLModel] No existing PyTorch model found; using a fresh model.")

        # ---------------------------------------------------------------------
        # 7) Load Scaler/PCA if available
        # ---------------------------------------------------------------------
        self._load_scaler_pca()

        # ---------------------------------------------------------------------
        # 8) Track Trading/Success Stats
        # ---------------------------------------------------------------------
        self.successful_trades = 0
        self.failed_trades = 0

        # ---------------------------------------------------------------------
        # 9) Strategy List (Multi-Strategy Signals)
        # ---------------------------------------------------------------------
        # Define all possible strategies in a fixed order
        self.all_strategies = [
            "TrendFollowingStrategy",
            "MeanReversionStrategy",
            "BreakoutStrategy",
            "MomentumStrategy",
            "RangeBoundStrategy",
            "ADXTrendStrategy",
            "BollingerSqueezeStrategy",
            "CrossoverStrategy",
            "ScalpingStrategy"
        ]

        # Initialize strategies as enabled by default
        self.enabled_strategies = set(self.all_strategies)  # This will be updated via settings

        # Initialize strategy instances
        self.strategies = [
            TrendFollowingStrategy(),
            MeanReversionStrategy(),
            BreakoutStrategy(),
            MomentumStrategy(),
            RangeBoundStrategy(),
            ADXTrendStrategy(),
            BollingerSqueezeStrategy(),
            CrossoverStrategy(),
            ScalpingStrategy()
        ]
        self.logger.info("[MLModel] Strategy classes loaded for feature engineering.")

    # --------------------------------------------------------------------------
    # HELPER METHODS: Load/Save Models, Build PyTorch, etc.
    # --------------------------------------------------------------------------
    def _load_model(self, model_name, default_model):
        filepath = self.model_files[model_name]
        if os.path.exists(filepath):
            try:
                self.logger.info(f"[MLModel] Loading {model_name.upper()} from {filepath}")
                return joblib.load(filepath)
            except Exception as e:
                self.logger.error(f"[MLModel] Could not load {model_name.upper()} from {filepath}: {e}")
                return default_model
        else:
            self.logger.info(f"[MLModel] No saved {model_name.upper()} found; using default.")
            return default_model

    def _save_model(self, model_name, model):
        filepath = self.model_files[model_name]
        try:
            joblib.dump(model, filepath)
            self.logger.info(f"[MLModel] Saved {model_name.upper()} model to {filepath}")
        except Exception as e:
            self.logger.error(f"[MLModel] Failed to save {model_name.upper()} model: {e}")

    def _build_pytorch_model(self):
        input_size = self.pca.n_components
        hidden_size = self.pytorch_hidden_size
        output_size = 1
        self.logger.info(
            f"[MLModel] Building PyTorch model: "
            f"input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}"
        )
        return ForexNet(input_size, hidden_size, output_size)

    def _load_scaler_pca(self):
        """
        Attempt to load existing scaler/pca artifacts; warn if missing.
        """
        try:
            scaler_path = self.model_files["scaler"]
            pca_path = self.model_files["pca"]
            if os.path.exists(scaler_path) and os.path.exists(pca_path):
                self.scaler = joblib.load(scaler_path)
                self.pca = joblib.load(pca_path)
                self.logger.info("[MLModel] Scaler and PCA loaded successfully.")
            else:
                self.logger.warning("[MLModel] No saved scaler/pca found; will fit anew during training.")
        except Exception as e:
            self.logger.error(f"[MLModel] Error loading scaler/pca: {e}")

    def _save_scaler_pca(self):
        """
        Save the scaler and PCA for future inference.
        """
        try:
            joblib.dump(self.scaler, self.model_files["scaler"])
            joblib.dump(self.pca, self.model_files["pca"])
            self.logger.info("[MLModel] Saved Scaler & PCA successfully.")
        except Exception as e:
            self.logger.error(f"[MLModel] Could not save scaler/pca: {e}")

    def is_ready(self):
        """
        Check if scaler & PCA are fitted by verifying standard attributes.
        """
        return (hasattr(self.scaler, 'scale_') and 
                hasattr(self.pca, 'components_'))

    # --------------------------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------------------------
    def prepare_features(self, price_data, training=False, window_size=5):
        """
        Comprehensive feature engineering pipeline:
          1) Convert OANDA candle data to arrays,
          2) Compute TA-Lib indicators,
          3) Filter out NaN rows,
          4) Apply multiple trading strategies for buy/sell signals,
          5) Extract rolling window features,
          6) Scale & apply PCA for dimensionality reduction.

        :param price_data: OANDA candle data (dict).
        :param training:   Boolean indicating if pipeline is for training (fit PCA/scaler) or inference (transform only).
        :param window_size: Rolling window size for additional features.
        :return: Numpy array of final feature set.
        """
        self.logger.info("[MLModel] Starting feature engineering with rolling window features.")
        try:
            # (1) Convert candle data
            candles = price_data.get('candles', [])
            if not candles:
                self.logger.warning("[prepare_features] No candle data provided.")
                return np.empty((0, 0))

            closes = np.array([float(c['mid']['c']) for c in candles])
            highs  = np.array([float(c['mid']['h']) for c in candles])
            lows   = np.array([float(c['mid']['l']) for c in candles])
            vols   = np.array([float(c['volume'])   for c in candles])

            # Ensure there are enough data points
            required_length = max(50, 500)  # As per SMA50 and H1 candles count
            if len(closes) < required_length:
                self.logger.warning(f"[prepare_features] Insufficient candle data length: {len(closes)}. Required: {required_length}")
                return np.empty((0, 0))

            # (2) TA-Lib indicators
            sma_10 = talib.SMA(closes, timeperiod=10)
            sma_50 = talib.SMA(closes, timeperiod=50)
            rsi    = talib.RSI(closes, timeperiod=14)
            ub, mb, lb = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2)
            atr    = talib.ATR(highs, lows, closes, timeperiod=14)
            macd, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
            slowk, slowd = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            cci   = talib.CCI(highs, lows, closes, timeperiod=14)
            adx   = talib.ADX(highs, lows, closes, timeperiod=14)

            # Combine base features
            base_feats = np.column_stack([
                sma_10, sma_50, rsi, ub, lb,
                atr, macd_hist, slowk, slowd, cci, adx
            ])

            # (3) Filter out rows with NaNs
            valid_indices = ~np.isnan(base_feats).any(axis=1)
            base_feats = base_feats[valid_indices]
            closes_filtered = closes[valid_indices]
            highs_filtered = highs[valid_indices]
            lows_filtered = lows[valid_indices]
            vols_filtered = vols[valid_indices]

            self.logger.debug(
                f"After NaN filtering: base_feats.shape={base_feats.shape}, "
                f"valid_indices.sum()={valid_indices.sum()}"
            )

            # (4) Multi-strategy signals
            # Initialize an array to hold strat_signals with shape (n_samples, 2 * n_strategies)
            num_strategies = len(self.strategies)
            strat_signals = np.zeros((base_feats.shape[0], 2 * num_strategies))

            for i in range(base_feats.shape[0]):
                sub_closes = closes_filtered[:i+1]
                sub_highs  = highs_filtered[:i+1]
                sub_lows   = lows_filtered[:i+1]
                sub_vols   = vols_filtered[:i+1]

                if len(sub_closes) < 2:
                    self.logger.debug(f"[prepare_features] Insufficient data for strategy signals at index {i}.")
                    continue  # strat_signals[i] remains [0, 0, ...] for all strategies

                for s_idx, strat_name in enumerate(self.all_strategies):
                    try:
                        # Find the strategy instance by name
                        strat = next((s for s in self.strategies if s.__class__.__name__ == strat_name), None)
                        if strat and strat_name in self.enabled_strategies:
                            signals = strat.calculate_signals(sub_closes, sub_highs, sub_lows, sub_vols)
                            buy  = 1.0 if signals.get("buy", False) else 0.0
                            sell = 1.0 if signals.get("sell", False) else 0.0
                        else:
                            # Strategy is disabled; set signals to 0
                            buy = 0.0
                            sell = 0.0
                        strat_signals[i, 2*s_idx]     = buy
                        strat_signals[i, 2*s_idx + 1] = sell
                    except Exception as e:
                        self.logger.error(
                            f"[prepare_features] Strategy {strat_name} error at index {i}: {e}"
                        )
                        # Defaults to 0.0 for buy and sell on error
                        strat_signals[i, 2*s_idx]     = 0.0
                        strat_signals[i, 2*s_idx + 1] = 0.0

            # Concatenate base_feats with strat_signals
            final_features = np.hstack([base_feats, strat_signals])
            self.logger.info(f"[MLModel] Built feature matrix shape: {final_features.shape}")

            if final_features.shape[0] == 0:
                self.logger.warning("[MLModel] No valid features after combining signals.")
                return np.empty((0, final_features.shape[1]))

            # (5) Rolling Window Features
            rolling_out = self._extract_rolling_features(closes_filtered, window_size=window_size)
            if rolling_out.size > 0:
                if rolling_out.shape[0] == final_features.shape[0]:
                    final_features = np.hstack((final_features, rolling_out))
                    self.logger.info(
                        f"[MLModel] Rolling window features appended. New shape={final_features.shape}"
                    )
                else:
                    self.logger.warning("[MLModel] Rolling window row mismatch. Skipping rolling features.")
                    self.logger.debug(f"final_features.shape[0]={final_features.shape[0]}, "
                                      f"rolling_out.shape[0]={rolling_out.shape[0]}")
            else:
                self.logger.warning("[MLModel] No rolling window features extracted.")

            # (6) PCA & Scaler
            if training:
            # Fit and transform if training
                pca_out = self.pca.fit_transform(final_features)
                scaled = self.scaler.fit_transform(pca_out)
                self.logger.info("[MLModel] PCA and Scaler fitted & transformed (training mode).")
            else:
                # Transform only if inference
                if not self.is_ready():
                    self.logger.error("[MLModel] PCA or Scaler not fitted yet.")
                    return np.empty((0, self.pca.n_components))
                # Check for dimension mismatch
                if final_features.shape[1] != self.pca.n_components:
                    self.logger.error(
                        f"Feature dimension {final_features.shape[1]} does not match PCA components {self.pca.n_components}."
                    )
                    self.logger.info("Refitting PCA and Scaler due to dimension mismatch.")
                    pca_out = self.pca.fit_transform(final_features)
                    scaled = self.scaler.fit_transform(pca_out)
                    self.logger.info("[MLModel] PCA and Scaler refitted during inference.")
                else:
                    pca_out = self.pca.transform(final_features)
                    scaled = self.scaler.transform(pca_out)
                    self.logger.info("[MLModel] PCA & Scaler transformed (inference mode).")

            return scaled
        
        except Exception as e:
            self.logger.error(f"[MLModel] Error in feature engineering: {e}")
            return np.empty((0, self.pca.n_components))

    def _extract_rolling_features(self, closes, window_size=5):
            """
            Extract rolling window features from closing prices.

            :param closes: Numpy array of closing prices.
            :param window_size: Size of the rolling window.
            :return: Numpy array of rolling features.
            """
            try:
                if len(closes) < window_size:
                    self.logger.warning(f"[_extract_rolling_features] Not enough data for window_size={window_size}")
                    return np.empty((0, 0))

                # Example rolling features: rolling mean and rolling std
                rolling_mean = talib.SMA(closes, timeperiod=window_size)
                rolling_std = talib.STDDEV(closes, timeperiod=window_size, nbdev=1)

                # Combine rolling features
                rolling_feats = np.column_stack([
                    rolling_mean,
                    rolling_std
                ])

                # Filter out NaNs resulted from rolling calculations
                valid = ~np.isnan(rolling_feats).any(axis=1)
                rolling_feats = rolling_feats[valid]

                self.logger.debug(f"Extracted rolling features shape: {rolling_feats.shape}")
                return rolling_feats

            except Exception as e:
                self.logger.error(f"[_extract_rolling_features] Error: {e}")
                return np.empty((0, 0))

    # --------------------------------------------------------------------------
    # TRAINING ROUTINES
    # --------------------------------------------------------------------------
    def train_classical_models(self, X_train, y_train):
        """
        Train SGD, RF, GB, then re-fit the Voting ensemble.
        """
        self.logger.info("[MLModel] Training classical models.")
        try:
            # (1) SGD
            self.logger.info("[MLModel] Training SGDClassifier.")
            self.models["sgd"].fit(X_train, y_train)
            self._save_model("sgd", self.models["sgd"])

            # (2) RF
            self.logger.info("[MLModel] Training RandomForestClassifier.")
            self.models["rf"].fit(X_train, y_train)
            self._save_model("rf", self.models["rf"])

            # (3) GB
            self.logger.info("[MLModel] Training GradientBoostingClassifier.")
            self.models["gb"].fit(X_train, y_train)
            self._save_model("gb", self.models["gb"])

            # (4) Update & train Voting
            self.logger.info("[MLModel] Training VotingClassifier.")
            self.voting_model.fit(X_train, y_train)
            self._save_model("voting", self.voting_model)

            self.logger.info("[MLModel] Classical models trained and saved successfully.")
        except Exception as e:
            self.logger.error(f"[MLModel] Error training classical: {e}")

    def _train_pytorch_model(self, X_train, y_train, X_val, y_val):
        """
        Enhanced PyTorch training loop with early stopping & best-model tracking.
        """
        self.logger.info("[MLModel] Starting PyTorch training.")
        patience = 5
        best_val_loss = float('inf')
        patience_counter = 0

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.pytorch_model.parameters(), lr=self.pytorch_lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)

        # Convert data to tensors
        xt = torch.tensor(X_train, dtype=torch.float32).to(device)
        yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        xv = torch.tensor(X_val, dtype=torch.float32).to(device)
        yv = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

        self.pytorch_model.to(device)
        self.pytorch_model.train()

        for epoch in range(self.pytorch_epochs):
            optimizer.zero_grad()
            output = self.pytorch_model(xt)
            loss = criterion(output, yt)
            loss.backward()
            optimizer.step()

            # Validation step
            self.pytorch_model.eval()
            with torch.no_grad():
                val_out = self.pytorch_model(xv)
                val_loss = criterion(val_out, yv)

            self.logger.info(
                f"[MLModel] Epoch {epoch+1}: "
                f"Train Loss={loss.item():.6f}, Val Loss={val_loss.item():.6f}"
            )

            # Best model check
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                # Save the best model state
                torch.save(self.pytorch_model.state_dict(), self.model_files["pytorch"])
                self.logger.info(f"[MLModel] New best PyTorch model saved at epoch {epoch+1}.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"[MLModel] Early stopping triggered at epoch {epoch+1}.")
                    break

            self.pytorch_model.train()
            scheduler.step(val_loss)

    def _save_pytorch_model(self):
        """
        Save the final PyTorch model state.
        """
        try:
            torch.save(self.pytorch_model.state_dict(), self.model_files["pytorch"])
            self.logger.info("[MLModel] PyTorch model saved successfully.")
        except Exception as e:
            self.logger.error(f"[MLModel] Error saving PyTorch model: {e}")

    def train_pytorch(self, X_train, y_train, X_val, y_val):
        """
        Public method to train the PyTorch model & save state.
        """
        self._train_pytorch_model(X_train, y_train, X_val, y_val)
        self._save_pytorch_model()

    def train(self, X_train, y_train, X_val, y_val):
        """
        Master training function for classical and PyTorch models.
        """
        if X_train.size == 0 or y_train.size == 0:
            self.logger.warning("[MLModel] Received empty training set. Skipping.")
            return
        try:
            # 1) Train classical
            self.train_classical_models(X_train, y_train)

            # 2) Train PyTorch
            self.train_pytorch(X_train, y_train, X_val, y_val)

            # 3) Save final scaler/pca
            self._save_scaler_pca()

            self.logger.info("[MLModel] All models trained successfully.")
        except Exception as e:
            self.logger.error(f"[MLModel] Training error: {e}")

    # --------------------------------------------------------------------------
    # PREDICTION METHODS
    # --------------------------------------------------------------------------
    def _predict_classical(self, X):
        """
        Get predictions from SGD, RF, GB, and the Voting classifier.
        Returns a tuple of predictions, e.g. (sgd, rf, gb, voting).
        """
        try:
            sgd_pred = self.models["sgd"].predict(X)
            rf_pred  = self.models["rf"].predict(X)
            gb_pred  = self.models["gb"].predict(X)
            vote_pred = self.voting_model.predict(X)
            return sgd_pred, rf_pred, gb_pred, vote_pred
        except Exception as e:
            self.logger.error(f"[MLModel] Error in classical predict: {e}")
            return None, None, None, None

    def _predict_pytorch(self, X):
        """
        Forward pass for PyTorch model -> binary predictions (0/1).
        """
        if X.shape[0] == 0:
            self.logger.error("[MLModel] No input features for PyTorch predict.")
            return None

        self.pytorch_model.eval()
        try:
            with torch.no_grad():
                xt = torch.tensor(X, dtype=torch.float32).to(device)
                preds = self.pytorch_model(xt)
                # Binarize at 0.5 threshold
                binary_preds = (preds > 0.5).float().cpu().numpy()
            return binary_preds
        except Exception as e:
            self.logger.error(f"[MLModel] PyTorch predict error: {e}")
            return None

    def predict(self, X):
        if X.shape[0] == 0:
            self.logger.error("[MLModel] Empty features for prediction.")
            return None, None

        # 1) Classical predictions
        sgd_pred, rf_pred, gb_pred, vote_pred = self._predict_classical(X)
        if None in (sgd_pred, rf_pred, gb_pred, vote_pred):
            self.logger.error("[MLModel] One or more classical model predictions failed.")
            return None, None

        # 2) PyTorch predictions
        pytorch_pred = self._predict_pytorch(X)
        if pytorch_pred is None:
            ensemble = np.vstack((sgd_pred, rf_pred, gb_pred, vote_pred))
        else:
            ensemble = np.vstack((sgd_pred, rf_pred, gb_pred, vote_pred, pytorch_pred.flatten()))

        num_models = ensemble.shape[0]
        final_pred = np.apply_along_axis(
                lambda row: np.bincount(row.astype(int)).argmax(),
                axis=0, arr=ensemble
            )
        final_pred = final_pred.astype(int)
        # Compute confidence for each sample as the proportion of models agreeing with the majority
        confidence = np.mean(ensemble == final_pred, axis=0)

        return final_pred, confidence

    # --------------------------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------------------------
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate all models and log:
          - Accuracy, Precision, Recall, F1-score, ROC AUC.
          - Confusion matrices in debug logs.
        """
        self.logger.info("[MLModel] Evaluating on test set.")
        try:
            # Evaluate classical models
            for name, model in self.models.items():
                preds = model.predict(X_test)
                acc  = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds, zero_division=0)
                rec  = recall_score(y_test, preds, zero_division=0)
                f1_  = f1_score(y_test, preds, zero_division=0)
                if hasattr(model, "predict_proba"):
                    roc_ = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
                else:
                    roc_ = 'N/A'

                self.logger.info(
                    f"{name.upper()} -> Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, "
                    f"F1={f1_:.4f}, ROC AUC={roc_}"
                )
                self.logger.debug(
                    f"Confusion Matrix for {name.upper()}:\n{confusion_matrix(y_test, preds)}"
                )

            # Voting ensemble
            vote_preds = self.voting_model.predict(X_test)
            acc_v  = accuracy_score(y_test, vote_preds)
            prec_v = precision_score(y_test, vote_preds, zero_division=0)
            rec_v  = recall_score(y_test, vote_preds, zero_division=0)
            f1_v   = f1_score(y_test, vote_preds, zero_division=0)
            if hasattr(self.voting_model, "predict_proba"):
                roc_v = roc_auc_score(y_test, self.voting_model.predict_proba(X_test)[:,1])
            else:
                roc_v = 'N/A'

            self.logger.info(
                f"VOTING -> Acc={acc_v:.4f}, Prec={prec_v:.4f}, Rec={rec_v:.4f}, "
                f"F1={f1_v:.4f}, ROC AUC={roc_v}"
            )
            self.logger.debug(
                f"Confusion Matrix for Voting Ensemble:\n{confusion_matrix(y_test, vote_preds)}"
            )

            # PyTorch
            pytorch_preds = self._predict_pytorch(X_test)
            if pytorch_preds is not None:
                pytorch_preds = pytorch_preds.flatten()
                acc_pt  = accuracy_score(y_test, pytorch_preds)
                prec_pt = precision_score(y_test, pytorch_preds, zero_division=0)
                rec_pt  = recall_score(y_test, pytorch_preds, zero_division=0)
                f1_pt   = f1_score(y_test, pytorch_preds, zero_division=0)
                # If binary, we can treat predictions as scores or adapt for probability
                # For a more robust approach, implement a forward pass with .sigmoid() output for actual probabilities.
                roc_pt  = roc_auc_score(y_test, pytorch_preds)
                self.logger.info(
                    f"PYTORCH -> Acc={acc_pt:.4f}, Prec={prec_pt:.4f}, Rec={rec_pt:.4f}, "
                    f"F1={f1_pt:.4f}, ROC AUC={roc_pt:.4f}"
                )
                self.logger.debug(
                    f"Confusion Matrix for PyTorch:\n{confusion_matrix(y_test, pytorch_preds)}"
                )
            else:
                self.logger.warning("[MLModel] PyTorch predictions unavailable for evaluation.")

        except Exception as e:
            self.logger.error(f"[MLModel] Evaluation error: {e}")

    # --------------------------------------------------------------------------
    # STATS
    # --------------------------------------------------------------------------
    def get_success_rate(self):
        """
        Calculate success rate of tracked trades if you're logging them internally.
        """
        total = self.successful_trades + self.failed_trades
        if total == 0:
            return 0.0
        return round((self.successful_trades / total) * 100, 2)

    def get_trade_stats(self):
        """
        Return a dict of trade stats, e.g. for UI or monitoring.
        """
        return {
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "success_rate": self.get_success_rate()
        }
