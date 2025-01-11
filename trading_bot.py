import requests
import logging
import os
import json
import sqlite3
import time
import threading
import numpy as np
import talib

from datetime import datetime
from sklearn.model_selection import train_test_split

# Local imports
from config import OANDA_API_KEY, OANDA_ACCOUNT_ID, BASE_URL, TRADE_INSTRUMENT
from ml_model import MLModel

# Ensure logs folder exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


class TradingBot:
    def __init__(self, ml_model=None):
        # -----------------------------
        # Logging Setup
        # -----------------------------
        self.logger = logging.getLogger('TradingBot')
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'trading_bot.log'))
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        self.logger.info("Initializing TradingBot...")

        # Credentials / ML model
        self.api_key = OANDA_API_KEY
        self.account_id = OANDA_ACCOUNT_ID
        self.ml_model = ml_model if ml_model else MLModel()

        # Basic thresholds
        self.retrain_threshold = 1000
        self.retry_count = 3

        # Default settings
        self.risk_level = "medium"
        self.max_trades = 5
        self.trade_interval_hours = 24  # Can be adjusted as needed
        self.stop_loss_percentage = 2.0
        self.take_profit_percentage = 5.0
        self.trade_cooldown = 60  # Seconds
        self.position_units = 1000
        self.trailing_atr_multiplier = 1.5
        self.adjust_atr_multiplier = 1.5

        # API headers
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

        # Account
        self.account_balance = 0.0
        self.initial_balance = 0.0
        self.load_account_balance()
        self.initial_balance = self.account_balance

        # Internal state
        self.trades = []
        self.trade_count = 0
        self.profit_loss = 0.0
        self.success_rate = 0.0
        self.new_data_count = 0
        self.historical_metrics_file = 'historical_metrics.json'
        # If you want multiple concurrent trades, these two boolean/state variables
        # track only the FIRST open position (still optional to keep).
        self.open_position = False
        self.current_side = None

        self.entry_price = None
        self.take_profit = None
        self.stop_loss = None

        self.running = False
        self.last_trade_time = None
        self.start_time = None
        self.accuracy_history = []
        self.performance_threshold = 0.6
        self.positions = []  # Local list tracking ALL open positions
        self.last_candle_time = None

        self.lock = threading.Lock()

        self.logger.info("TradingBot initialized successfully.")
        self.setup_logging()
        self.load_state()
        self.load_historical_metrics()

        # MLModel readiness check
        if not self.ml_model.is_ready():
            self.logger.info("[TradingBot] MLModel not ready. Performing initial training.")
            initial_price_data = self.get_prices(count=500, timeframe="H1")
            if initial_price_data:
                X_train, y_train, X_val, y_val = self.prepare_training_data(initial_price_data)
                if X_train.size > 0 and y_train.size > 0 and X_val.size > 0 and y_val.size > 0:
                    self.ml_model.train(X_train, y_train, X_val, y_val)
                    self.logger.info("[TradingBot] Initial training completed successfully.")
                else:
                    self.logger.error("[TradingBot] Insufficient data for initial training.")
            else:
                self.logger.error("[TradingBot] Failed to fetch price data for initial training.")

    def setup_logging(self):
        """
        Optional method for additional or custom logging logic.
        Currently unused.
        """
        pass

    # ----------------------------------------------------------------------
    # ACCOUNT + METRICS
    # ----------------------------------------------------------------------
    def load_account_balance(self):
        """
        Retrieve and set the OANDA account's current balance, retrying if needed.
        """
        for _ in range(self.retry_count):
            try:
                url = f"{BASE_URL}/accounts/{self.account_id}/summary"
                self.logger.info(f"[load_account_balance] GET {url}")
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                data = response.json()
                self.account_balance = float(data['account']['balance'])
                self.logger.info(f"[load_account_balance] Account balance: {self.account_balance}")
                return
            except Exception as e:
                self.logger.error(f"[load_account_balance] Exception: {e}")
                time.sleep(1)

    def load_historical_metrics(self):
        """
        Load historical trade counts and profit_loss from a local SQLite DB.
        """
        try:
            conn = sqlite3.connect('metrics.db')
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY,
                    trade_count INTEGER,
                    profit_loss REAL
                )
            """)
            cursor.execute("SELECT trade_count, profit_loss FROM metrics WHERE id=1")
            row = cursor.fetchone()
            if row:
                self.trade_count, self.profit_loss = row
                self.logger.info(f"[load_historical_metrics] Found trade_count={self.trade_count}, "
                                 f"profit_loss={self.profit_loss}")
            else:
                cursor.execute("INSERT INTO metrics (id, trade_count, profit_loss) VALUES (1, 0, 0.0)")
                conn.commit()
                self.logger.info("[load_historical_metrics] Initialized new metrics in DB.")
            conn.close()
        except Exception as e:
            self.logger.error(f"[load_historical_metrics] DB error: {e}")

    def save_historical_metrics(self):
        """
        Persist updated trade_count and profit_loss to the SQLite DB.
        """
        try:
            conn = sqlite3.connect('metrics.db')
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY,
                    trade_count INTEGER,
                    profit_loss REAL
                )
            """)
            cursor.execute("UPDATE metrics SET trade_count=?, profit_loss=? WHERE id=1",
                           (self.trade_count, self.profit_loss))
            conn.commit()
            conn.close()
            self.logger.info(f"[save_historical_metrics] Stored trade_count={self.trade_count}, "
                             f"profit_loss={self.profit_loss}")
        except Exception as e:
            self.logger.error(f"[save_historical_metrics] Error: {e}")

    # ----------------------------------------------------------------------
    # UTILITIES
    # ----------------------------------------------------------------------
    def get_prices(self, count=500, timeframe="H1"):
        """
        Fetch candle data from OANDA for a given timeframe and candle count.
        Retries self.retry_count times on transient errors.
        """
        for _ in range(self.retry_count):
            try:
                url = f"{BASE_URL}/instruments/{TRADE_INSTRUMENT}/candles"
                params = {"granularity": timeframe, "count": count}
                self.logger.info(f"[get_prices] Fetching candles from {url} with params {params}")
                r = requests.get(url, headers=self.headers, params=params, timeout=5)
                r.raise_for_status()
                data = r.json()
                if "candles" not in data or not data["candles"]:
                    self.logger.error("[get_prices] No candle data returned from OANDA.")
                    return None
                return data
            except requests.exceptions.RequestException as e:
                self.logger.error(f"[get_prices] RequestException: {e}")
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"[get_prices] Unexpected error: {e}")
                time.sleep(1)
        self.logger.error("[get_prices] Failed to get prices after retries.")
        return None

    def get_current_price(self):
        """
        Return the latest mid close price for the instrument by fetching 1 candle (M1).
        """
        price_data = self.get_prices(count=1, timeframe="M1")
        if price_data and "candles" in price_data:
            try:
                return float(price_data["candles"][-1]["mid"]["c"])
            except Exception as e:
                self.logger.error(f"[get_current_price] Data extraction error: {e}")
                return 0.0
        self.logger.error("[get_current_price] Price data missing or invalid.")
        return 0.0

    def can_trade(self):
        """
        Enforce a cooldown between trades to avoid too frequent entries.
        """
        if self.last_trade_time is None:
            return True
        elapsed = (datetime.now() - self.last_trade_time).total_seconds()
        if elapsed < self.trade_cooldown:
            self.logger.info("[can_trade] Cooldown active, skipping new trade.")
            return False
        return True

    # ----------------------------------------------------------------------
    # MULTI-TIMEFRAME STRATEGY AGGREGATION
    # ----------------------------------------------------------------------
    def apply_strategies_to_data(self, price_data):
        """
        Returns total buy/sell signals by applying all strategies from the ML model.
        """
        if not price_data or "candles" not in price_data:
            return 0, 0

        closes = np.array([float(c['mid']['c']) for c in price_data['candles']])
        highs = np.array([float(c['mid']['h']) for c in price_data['candles']])
        lows = np.array([float(c['mid']['l']) for c in price_data['candles']])
        vols = np.array([float(c['volume']) for c in price_data['candles']])

        buy_signals = 0
        sell_signals = 0

        # Use the same strategies loaded in the ML model
        for strat in self.ml_model.strategies:
            try:
                signals = strat.calculate_signals(closes, highs, lows, vols)
                if signals.get("buy", False):
                    buy_signals += 1
                if signals.get("sell", False):
                    sell_signals += 1
            except Exception as e:
                self.logger.error(f"[apply_strategies_to_data] Strategy {strat.__class__.__name__} error: {e}")
        return buy_signals, sell_signals

    def multi_timeframe_analysis(self):
        """
        Combine signals from multiple timeframes (M15, H4) for a final total.
        """
        lower_tf_data = self.get_prices(count=100, timeframe="M15")
        higher_tf_data = self.get_prices(count=100, timeframe="H4")

        if not lower_tf_data or not higher_tf_data:
            self.logger.error("[multi_timeframe_analysis] Failed to fetch multi-timeframe data.")
            return 0, 0

        lower_buy, lower_sell = self.apply_strategies_to_data(lower_tf_data)
        higher_buy, higher_sell = self.apply_strategies_to_data(higher_tf_data)

        total_buy = lower_buy + higher_buy
        total_sell = lower_sell + higher_sell
        self.logger.info(f"[multi_timeframe_analysis] Combined: Buy={total_buy}, Sell={total_sell}")
        return total_buy, total_sell

    # ----------------------------------------------------------------------
    # TRADING LOGIC
    # ----------------------------------------------------------------------
    @staticmethod
    def calculate_atr(highs, lows, closes, period=14):
        # Change condition to require at least 'period' candles instead of 'period + 1'
        if len(closes) < period:
            return np.nan
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        initial_atr = np.mean(trs[:period])
        atr = initial_atr
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period
        return atr
    
    def compute_pivot_points(self, highs, lows, closes):
        # Use the most recent period's high, low, and close to calculate pivot points
        pp = (highs[-1] + lows[-1] + closes[-1]) / 3.0
        r1 = 2 * pp - lows[-1]  # First resistance
        s1 = 2 * pp - highs[-1]  # First support
        return {"PP": pp, "R1": r1, "S1": s1}

    def calculate_dynamic_units(self, confidence):
        # Define risk multipliers for different risk levels
        risk_multipliers = {"low": 1.0, "medium": 2.0, "high": 4.0}
        risk_multiplier = risk_multipliers.get(self.risk_level.lower(), 1.0)

        # Scale the base units based on account balance relative to the initial balance
        scale_factor = self.account_balance / self.initial_balance if self.initial_balance > 0 else 1.0
        adjusted_base = self.position_units * scale_factor * risk_multiplier

        # Incorporate model confidence into the unit calculation
        factor = max(0.5, min(1.5, confidence))
        dynamic_units = int(adjusted_base * factor)
        self.logger.info(f"[calculate_dynamic_units] Confidence: {confidence}, "
                        f"Adjusted base units: {adjusted_base}, Dynamic units: {dynamic_units}")
        return dynamic_units
    
    def update_trailing_stop_losses(self):
        # Fetch recent candle data for ATR calculation
        price_data = self.get_prices(count=100, timeframe="M15")
        if not price_data:
            return

        # Filter only complete candles from the fetched data
        candles = [c for c in price_data.get('candles', []) if c.get('complete', False)]
        if len(candles) < 14:
            self.logger.warning("Not enough complete candles for ATR calculation.")
            return

        # Use the most recent 14 complete candles for ATR calculation
        recent_candles = candles[-14:]
        closes = np.array([float(c['mid']['c']) for c in recent_candles])
        highs = np.array([float(c['mid']['h']) for c in recent_candles])
        lows = np.array([float(c['mid']['l']) for c in recent_candles])

        atr_values = talib.ATR(highs, lows, closes, timeperiod=14)
        atr_value = atr_values[-1] if len(atr_values) > 0 else np.nan

        # If TA-Lib returns NaN, use custom ATR calculation
        if np.isnan(atr_value):
            self.logger.warning("ATR value is NaN. Using custom ATR calculation.")
            atr_value = self.calculate_atr(highs, lows, closes, period=14)
            if np.isnan(atr_value):
                self.logger.warning("Custom ATR calculation returned NaN. Skipping trailing update.")
                return

        trailing_distance = atr_value * self.trailing_atr_multiplier
        current_price = closes[-1]

        for pos in self.positions:
            if pos["side"] == "buy":
                # Initialize and update highest price reached for BUY
                pos.setdefault("max_price", current_price)
                if current_price > pos["max_price"]:
                    pos["max_price"] = current_price

                new_stop = pos["max_price"] - trailing_distance
                if np.isnan(new_stop):
                    self.logger.warning(f"Calculated new_stop for BUY trade {pos['trade_id']} is NaN. Skipping update.")
                    continue

                # Move stop loss up only if the new stop is higher than the current stop
                if pos["stop_loss"] is None or new_stop > pos["stop_loss"]:
                    pos["stop_loss"] = new_stop
                    self.logger.info(f"[update_trailing_stop_losses] Updated SL for BUY trade {pos['trade_id']} to {new_stop}")
                    try:
                        update_url = f"{BASE_URL}/accounts/{self.account_id}/trades/{pos['trade_id']}/orders"
                        payload = {"stopLoss": {"price": f"{new_stop:.5f}"}}
                        update_resp = requests.put(update_url, headers=self.headers, json=payload, timeout=5)
                        update_resp.raise_for_status()
                    except Exception as e:
                        self.logger.error(f"Failed to update stop loss on OANDA for trade {pos['trade_id']}: {e}")

            elif pos["side"] == "sell":
                # Initialize and update lowest price reached for SELL
                pos.setdefault("min_price", current_price)
                if current_price < pos["min_price"]:
                    pos["min_price"] = current_price

                new_stop = pos["min_price"] + trailing_distance
                if np.isnan(new_stop):
                    self.logger.warning(f"Calculated new_stop for SELL trade {pos['trade_id']} is NaN. Skipping update.")
                    continue

                # Move stop loss down only if the new stop is lower than the current stop
                if pos["stop_loss"] is None or new_stop < pos["stop_loss"]:
                    pos["stop_loss"] = new_stop
                    self.logger.info(f"[update_trailing_stop_losses] Updated SL for SELL trade {pos['trade_id']} to {new_stop}")
                    try:
                        update_url = f"{BASE_URL}/accounts/{self.account_id}/trades/{pos['trade_id']}/orders"
                        payload = {"stopLoss": {"price": f"{new_stop:.5f}"}}
                        update_resp = requests.put(update_url, headers=self.headers, json=payload, timeout=5)
                        update_resp.raise_for_status()
                    except Exception as e:
                        self.logger.error(f"Failed to update stop loss on OANDA for trade {pos['trade_id']}: {e}")

    def manage_position(self, current_price, ml_prediction, confidence, combined_buy, combined_sell):
        """
        Manage a single open position based on strategy signals and ML prediction.
        Operates under Single Position Mode: only one open trade at a time.
        """
        # Apply ML weighting
        if ml_prediction == 1:
            combined_buy += 2
        elif ml_prediction == 0:
            combined_sell += 2

        threshold = 3

        # If no open position, attempt to open a new one
        if not self.open_position:
            if combined_buy >= threshold:
                self.logger.info(f"[manage_position] No open position. Opening BUY trade. Score={combined_buy}")
                sl = current_price - (current_price * (self.stop_loss_percentage / 100))
                # Fetch recent data to calculate pivot points
                price_data_for_pp = self.get_prices(count=2, timeframe="H1")
                if price_data_for_pp and len(price_data_for_pp.get('candles', [])) >= 2:
                    candles = price_data_for_pp['candles']
                    prev_candle = candles[-2]  # Use the previous candle for pivot calculation
                    high = float(prev_candle['mid']['h'])
                    low = float(prev_candle['mid']['l'])
                    close = float(prev_candle['mid']['c'])
                    pivots = self.compute_pivot_points(np.array([high]), np.array([low]), np.array([close]))
                    tp = pivots["R1"]
                else:
                    tp = current_price + (current_price * (self.take_profit_percentage / 100))  # fallback
                units = self.calculate_dynamic_units(confidence)
                self.make_trade("buy", units, current_price, sl, tp)

            elif combined_sell >= threshold:
                self.logger.info(f"[manage_position] No open position. Opening SELL trade. Score={combined_sell}")
                sl = current_price + (current_price * (self.stop_loss_percentage / 100))
                # Fetch recent data to calculate pivot points
                price_data_for_pp = self.get_prices(count=2, timeframe="H1")
                if price_data_for_pp and len(price_data_for_pp.get('candles', [])) >= 2:
                    candles = price_data_for_pp['candles']
                    prev_candle = candles[-2]  # Use the previous candle for pivot calculation
                    high = float(prev_candle['mid']['h'])
                    low = float(prev_candle['mid']['l'])
                    close = float(prev_candle['mid']['c'])
                    pivots = self.compute_pivot_points(np.array([high]), np.array([low]), np.array([close]))
                    tp = pivots["S1"]
                else:
                    tp = current_price - (current_price * (self.take_profit_percentage / 100))  # fallback
                units = self.calculate_dynamic_units(confidence)
                self.make_trade("sell", units, current_price, sl, tp)
            else:
                self.logger.info("[manage_position] No strong signals to open new position.")

        # If a position is already open
        else:
            # Check if opposite signal is strong enough to warrant closing current position
            if (self.current_side == "buy" and combined_sell >= threshold) or \
               (self.current_side == "sell" and combined_buy >= threshold):
                self.logger.info("[manage_position] Opposing signal detected. Closing current position to switch direction.")
                # Close entire existing position before opening new one
                self.close_position_fifo(self.positions[0]["units"])
            else:
                self.logger.info("[manage_position] Existing position continues. No opposing strong signal detected.")

            # Check SL/TP triggers regardless of signal changes
            if self.stop_loss and self.take_profit:
                if self.current_side == "buy":
                    if current_price <= self.stop_loss or current_price >= self.take_profit:
                        self.logger.info("[manage_position] SL/TP triggered for BUY => closing.")
                        self.close_position_fifo(self.positions[0]["units"])
                else:  # SELL side
                    if current_price >= self.stop_loss or current_price <= self.take_profit:
                        self.logger.info("[manage_position] SL/TP triggered for SELL => closing.")
                        self.close_position_fifo(self.positions[0]["units"])

    def make_trade(self, side, units, current_price, stop_loss, take_profit):
        """
        Place a market order with SL/TP using OPEN_ONLY to allow multiple positions
        in hedging mode. In single position mode, account restrictions will apply.
        """
        if not isinstance(stop_loss, (float, int)) or not isinstance(take_profit, (float, int)):
            self.logger.error("[make_trade] Invalid SL/TP values.")
            return

        order_data = {
            "order": {
                "units": str(units) if side == "buy" else str(-units),
                "instrument": TRADE_INSTRUMENT,
                "timeInForce": "FOK",
                "type": "MARKET",
                "positionFill": "DEFAULT",  # Use DEFAULT for compatibility with hedging mode
                "stopLossOnFill": {
                    "price": f"{stop_loss:.5f}"
                },
                "takeProfitOnFill": {
                    "price": f"{take_profit:.5f}"
                }
            }
        }

        for attempt in range(1, self.retry_count + 1):
            try:
                url = f"{BASE_URL}/accounts/{self.account_id}/orders"
                self.logger.info(f"[make_trade] Attempt {attempt}: {json.dumps(order_data)}")
                resp = requests.post(url, headers=self.headers, json=order_data, timeout=5)
                if resp.status_code in [200, 201]:
                    data = resp.json()
                    if "orderCancelTransaction" in data:
                        cancel_reason = data["orderCancelTransaction"].get("reason", "Unknown reason")
                        self.logger.error(f"[make_trade] Order canceled due to {cancel_reason}. No trade opened.")
                        return

                    trade_id = data.get("orderFillTransaction", {}).get("tradeOpened", {}).get("tradeID")
                    if not trade_id:
                        self.logger.error(f"[make_trade] Trade ID missing. Response: {data}")
                        return

                    self.log_trade(data)
                    self.load_account_balance()
                    self.positions.append({
                        "side": side,
                        "units": units,
                        "entry_price": current_price,
                        "trade_id": trade_id,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "sl_order_id": None,
                        "tp_order_id": None
                    })
                    self.open_position = True
                    self.current_side = side
                    self.entry_price = current_price
                    self.stop_loss = stop_loss
                    self.take_profit = take_profit
                    self.last_trade_time = datetime.now()

                    self.logger.info(
                        f"[make_trade] {side.upper()} at {current_price:.5f}, "
                        f"SL={stop_loss:.5f}, TP={take_profit:.5f}"
                    )
                    return
                else:
                    self.logger.error(f"[make_trade] OANDA error: {resp.text}")
                    time.sleep(1)
            except Exception as e:
                self.logger.error(f"[make_trade] Exception on attempt {attempt}: {e}")
                time.sleep(1)
        self.logger.error("[make_trade] Could not place trade after retries.")

    def close_position_fifo(self, units_to_close):
        """
        Close positions in FIFO order to comply with US regulations.
        If partial closes are needed, always start from the oldest in self.positions.
        """
        remaining = units_to_close
        # Always close from the beginning of self.positions (oldest first)
        while remaining > 0 and self.positions:
            oldest = self.positions[0]  # The oldest position
            if oldest["units"] <= remaining:
                # Close the entire oldest position
                pos = self.positions.pop(0)
                self._close_single_position(pos)
                remaining -= pos["units"]
            else:
                # Partially close the oldest position
                oldest["units"] -= remaining
                partial_pos = {
                    "side": oldest["side"],
                    "units": remaining,
                    "entry_price": oldest["entry_price"],
                    "trade_id": oldest["trade_id"],
                    "stop_loss": oldest["stop_loss"],
                    "take_profit": oldest["take_profit"],
                    "sl_order_id": oldest.get("sl_order_id"),
                    "tp_order_id": oldest.get("tp_order_id")
                }
                self._close_single_position(partial_pos)
                remaining = 0

        # If no positions remain, reset single-position flags
        if not self.positions:
            self.open_position = False
            self.current_side = None
            self.entry_price = None
            self.stop_loss = None
            self.take_profit = None

    def _close_single_position(self, pos):
        """
        Close one position (full or partial). Make a call to OANDA and track realized P/L.
        """
        cp = self.get_current_price()
        realized = 0.0
        if pos["side"] == "buy":
            realized = (cp - pos["entry_price"]) * pos["units"]
        else:
            realized = (pos["entry_price"] - cp) * pos["units"]

        self.profit_loss += realized
        self.logger.info(
            f"[_close_single_position] Closed {pos['units']} {pos['side'].upper()} "
            f"at {cp:.5f}, Realized={realized:.2f}, Total P/L={self.profit_loss:.2f}"
        )

        # Actually close on OANDA
        self.close_trade(pos["trade_id"], pos["units"])
        self.save_historical_metrics()

    def close_trade(self, trade_id, units):
        """
        Close a trade on OANDA by ID, partial or full, respecting FIFO rules.
        """
        try:
            url = f"{BASE_URL}/accounts/{self.account_id}/trades/{trade_id}/close"
            data = {"units": str(units)}
            self.logger.info(f"[close_trade] Closing trade {trade_id} with data={data}")
            resp = requests.put(url, headers=self.headers, json=data, timeout=5)
            resp.raise_for_status()
            self.logger.info(f"[close_trade] Closed trade={trade_id}, units={units}")
            self.load_account_balance()
        except Exception as e:
            self.logger.error(f"[close_trade] Error: {e}")

    def close_positions_by_side(self, side):
        """
        Close all open positions of the specified side (e.g., 'buy' or 'sell') in FIFO order.
        """
        trades_to_close = [pos for pos in self.positions if pos["side"] == side]
        if not trades_to_close:
            self.logger.info(f"[close_positions_by_side] No {side.upper()} trades to close.")
            return

        # Close each trade from oldest to newest
        for pos in trades_to_close:
            if pos in self.positions:
                self.positions.remove(pos)
                self._close_single_position(pos)

        # If no positions remain after closing, reset single-position flags
        if not self.positions:
            self.logger.info(f"[close_positions_by_side] All {side.upper()} trades closed; no open trades left.")
            self.open_position = False
            self.current_side = None
            self.entry_price = None
            self.stop_loss = None
            self.take_profit = None

    def count_positions_by_side(self, side):
        """
        Return how many open positions are currently on the given side.
        """
        return sum(1 for pos in self.positions if pos["side"] == side)

    def check_sl_tp_triggers(self, current_price):
        """
        Check if any open position has hit its SL or TP, and close if triggered.
        """
        to_remove = []
        for pos in self.positions:
            if pos["side"] == "buy":
                # SL/TP check for BUY
                if (pos["stop_loss"] and current_price <= pos["stop_loss"]) or \
                   (pos["take_profit"] and current_price >= pos["take_profit"]):
                    self.logger.info("[check_sl_tp_triggers] SL/TP triggered for BUY => closing.")
                    self._close_single_position(pos)
                    to_remove.append(pos)
            else:  # SELL side
                if (pos["stop_loss"] and current_price >= pos["stop_loss"]) or \
                   (pos["take_profit"] and current_price <= pos["take_profit"]):
                    self.logger.info("[check_sl_tp_triggers] SL/TP triggered for SELL => closing.")
                    self._close_single_position(pos)
                    to_remove.append(pos)

        # Remove closed positions from self.positions
        for closed_pos in to_remove:
            if closed_pos in self.positions:
                self.positions.remove(closed_pos)

    # ----------------------------------------------------------------------
    # OPEN TRADES SYNC
    # ----------------------------------------------------------------------
    def get_open_trades(self):
        """
        Fetch open trades from OANDA for local synchronization.
        """
        try:
            url = f"{BASE_URL}/accounts/{self.account_id}/openTrades"
            self.logger.info(f"[get_open_trades] GET {url}")
            resp = requests.get(url, headers=self.headers, timeout=5)
            resp.raise_for_status()
            trades = resp.json().get("trades", [])
            for t in trades:
                t["price"] = float(t.get("price", 0))
                t["currentUnits"] = float(t.get("currentUnits", 0))
                t["tradeID"] = t.get("id")
            return trades
        except Exception as e:
            self.logger.error(f"[get_open_trades] Error: {e}")
            return []

    def synchronize_positions(self):
        """
        Refresh local self.positions to match OANDA's open trades in FIFO order.
        """
        open_trades = self.get_open_trades()
        if not open_trades:
            self.logger.info("[synchronize_positions] No open trades => Clearing local positions.")
            self.positions = []
            self.open_position = False
            return

        synced = []
        for t in open_trades:
            side = "buy" if t["currentUnits"] > 0 else "sell"
            synced.append({
                "side": side,
                "units": abs(int(t["currentUnits"])),
                "entry_price": t["price"],
                "trade_id": t["tradeID"],
                "stop_loss": None,
                "take_profit": None,
                "sl_order_id": None,
                "tp_order_id": None
            })

        # OANDA typically returns trades in FIFO order, but if not certain,
        # sort them by trade creation or id as needed:
        # synced.sort(key=lambda x: x["trade_id"] or <some creation time> )

        self.positions = synced
        self.logger.info(f"[synchronize_positions] Positions in sync: {len(self.positions)}")

        # Update single-position flags if needed
        if self.positions:
            self.open_position = True
            self.current_side = self.positions[0]["side"]
            self.entry_price = self.positions[0]["entry_price"]
            self.stop_loss = self.positions[0]["stop_loss"]
            self.take_profit = self.positions[0]["take_profit"]
        else:
            self.open_position = False
            self.current_side = None
            self.entry_price = None
            self.stop_loss = None
            self.take_profit = None

    # ----------------------------------------------------------------------
    # MAIN LOOP AND CONTROL
    # ----------------------------------------------------------------------
    def run_trading_loop(self):
        """
        The main trading loop that continuously checks for trading opportunities.
        Runs in a separate thread.
        """
        self.logger.info("[run_trading_loop] Trading loop started.")
        self.start_time = datetime.now()
        while self.running:
            try:
                # Synchronize positions at the start
                self.synchronize_positions()

                # Multi-timeframe signals
                buy_signals, sell_signals = self.multi_timeframe_analysis()

                # ML model prediction
                price_data = self.get_prices(count=500, timeframe="H1")
                if not price_data:
                    self.logger.info("[run_trading_loop] No price data available.")
                    time.sleep(60)  # Wait 5 minutes before retrying
                    continue

                feats = self.ml_model.prepare_features(price_data)
                if feats.shape[0] == 0:
                    self.logger.info("[run_trading_loop] No valid ML features.")
                    time.sleep(60)
                    continue

                # Get prediction and confidence from ML model
                pred, confidence = self.ml_model.predict(feats[-1].reshape(1, -1))
                if pred is None or confidence is None:
                    self.logger.info("[run_trading_loop] ML returned None prediction or confidence.")
                    time.sleep(60)
                    continue

                current_price = self.get_current_price()
                if current_price == 0:
                    self.logger.info("[run_trading_loop] Invalid current price=0.")
                    time.sleep(60)
                    continue

                # Manage positions if cooldown allows
                if self.can_trade():
                    self.manage_position(current_price, pred[0], confidence[0], buy_signals, sell_signals)

                # Update trailing stop losses after managing positions
                self.update_trailing_stop_losses()

                # Check if retraining is needed based on trade count
                if self.trade_count >= self.retrain_threshold:
                    self.logger.info("[run_trading_loop] Retrain threshold reached.")
                    self.retrain_model()

                # Sleep for a defined interval before next check
                time.sleep(60)  # 5 minutes

            except Exception as e:
                self.logger.error(f"[run_trading_loop] Error: {e}")
                time.sleep(60)  # Wait before retrying in case of unexpected errors

    def start(self):
        """
        Start the trading bot by initiating the main trading loop in a separate thread.
        """
        with self.lock:
            if self.running:
                self.logger.warning("[start] TradingBot is already running.")
                return
            self.running = True
            self.start_time = datetime.now()
            self.logger.info("[start] TradingBot has started.")
            # Start the trading loop in a new thread
            self.trading_thread = threading.Thread(target=self.run_trading_loop, daemon=True)
            self.trading_thread.start()

    def stop(self):
        """
        Stop the bot, mark running=False, clear start_time, and shut down trading loop.
        """
        with self.lock:
            if not self.running:
                self.logger.warning("[stop] TradingBot is not running.")
                return

            self.running = False
            self.start_time = None
            self.save_state()
            self.logger.info("[stop] TradingBot stopping...")

            # Wait for the trading loop thread to finish
            if hasattr(self, 'trading_thread') and self.trading_thread.is_alive():
                self.trading_thread.join()
                self.logger.info("[stop] TradingBot stopped successfully.")
            else:
                self.logger.info("[stop] No active trading thread found.")

    # ----------------------------------------------------------------------
    # RETRAINING
    # ----------------------------------------------------------------------
    def retrain_model(self):
        """
        Retrain the ML model with fresh data.
        """
        self.logger.info("[retrain_model] Retraining ML model with fresh data.")
        price_data = self.get_prices(count=500, timeframe="H1")
        if not price_data:
            self.logger.error("[retrain_model] Could not get price data for retraining.")
            return

        X_train, y_train, X_val, y_val = self.prepare_training_data(price_data)
        if X_train.size > 0 and y_train.size > 0 and X_val.size > 0 and y_val.size > 0:
            self.ml_model.train(X_train, y_train, X_val, y_val)
            self.logger.info("[retrain_model] Retraining completed successfully.")
            # Reset retrain threshold
            self.trade_count = 0
        else:
            self.logger.error("[retrain_model] Insufficient data to retrain.")

    # ----------------------------------------------------------------------
    # DATA PREP
    # ----------------------------------------------------------------------
    def apply_settings(self, settings):
        try:
            self.risk_level = settings.get("riskLevel", self.risk_level)
            self.max_trades = int(settings.get("maxTrades", self.max_trades))
            self.trade_interval_hours = float(settings.get("tradeIntervalHours", self.trade_interval_hours))
            self.stop_loss_percentage = float(settings.get("stopLossPercentage", self.stop_loss_percentage))
            self.take_profit_percentage = float(settings.get("takeProfitPercentage", self.take_profit_percentage))
            self.trailing_atr_multiplier = float(settings.get("trailingATRMultiplier", self.trailing_atr_multiplier))
            self.adjust_atr_multiplier = float(settings.get("adjustATRMultiplier", self.adjust_atr_multiplier))
            self.trade_cooldown = int(settings.get("tradeCooldown", self.trade_cooldown))

            # Update enabled strategies
            enabled_strats = settings.get("enabledStrategies", [])
            self.ml_model.enabled_strategies = set(enabled_strats)

            # Scalping-specific parameters (ensure defaults are set if not provided)
            self.scalping_ema_period = settings.get("scalpingEmaPeriod", 5)
            self.scalping_rsi_period = settings.get("scalpingRsiPeriod", 5)
            self.scalping_atr_period = settings.get("scalpingAtrPeriod", 14)
            self.scalping_bb_period = settings.get("scalpingBbPeriod", 20)
            self.scalping_threshold_multiplier = settings.get("scalpingThresholdMultiplier", 0.5)

            # Update ScalpingStrategy parameters if enabled
            if "ScalpingStrategy" in self.ml_model.enabled_strategies:
                # Find the ScalpingStrategy instance and update its parameters
                for strat in self.ml_model.strategies:
                    if strat.__class__.__name__ == "ScalpingStrategy":
                        strat.update_parameters(
                            ema_period=self.scalping_ema_period,
                            rsi_period=self.scalping_rsi_period,
                            atr_period=self.scalping_atr_period,
                            bb_period=self.scalping_bb_period,
                            threshold_multiplier=self.scalping_threshold_multiplier
                        )
                        self.logger.info("[apply_settings] Updated ScalpingStrategy parameters.")
                        break
            else:
                self.logger.info("[apply_settings] ScalpingStrategy is disabled.")

            self.logger.info("TradingBot: Settings applied successfully.")
        except Exception as e:
            self.logger.error(f"TradingBot: Failed to apply settings - {e}")

    def prepare_training_data(self, price_data):
        """
        Convert OANDA data -> X, y arrays for ML training.
        Simple approach: label is 1 if price goes up next candle, else 0.
        """
        # Force MLModel to 'training' mode => ensures PCA/scaler fit
        features = self.ml_model.prepare_features(price_data, training=True)
        closes = [float(c['mid']['c']) for c in price_data['candles']]
        closes = closes[-len(features):]

        # Basic labeling
        labels = []
        for i in range(1, len(closes)):
            labels.append(1 if closes[i] > closes[i - 1] else 0)

        X = features[1:]
        y = np.array(labels)
        if len(X) != len(y):
            self.logger.error("[prepare_training_data] Feature/label mismatch.")
            return np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,))

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        self.logger.info(f"[prepare_training_data] Prepared training data: X_train={X_train.shape}, "
                         f"X_val={X_val.shape}")
        return X_train, y_train, X_val, y_val

    def get_oanda_trade_history(self, count=50):
        """
        Fetches the trade history from the OANDA API.
        :param count: Number of recent trades to retrieve.
        :return: List of trade dictionaries or empty list on failure.
        """
        endpoint = f"{BASE_URL}/accounts/{self.account_id}/trades"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        params = {
            "count": count,  # Number of trades to retrieve
            "state": "CLOSED"  # Fetch only closed trades
        }

        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()
            trades_data = response.json()

            trades = trades_data.get('trades', [])
            self.logger.info(f"[get_oanda_trade_history] Retrieved {len(trades)} trades.")
            return trades

        except requests.exceptions.HTTPError as http_err:
            self.logger.error(f"[get_oanda_trade_history] HTTP error occurred: {http_err}")
        except Exception as err:
            self.logger.error(f"[get_oanda_trade_history] Other error occurred: {err}")

        return []

    # ----------------------------------------------------------------------
    # STATE MANAGEMENT
    # ----------------------------------------------------------------------
    def load_state(self):
        """
        Load run state from the SQLite DB (running, start_time).
        On startup, assume the bot is not running regardless of the DB state.
        """
        try:
            conn = sqlite3.connect('metrics.db')
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bot_state (
                    id INTEGER PRIMARY KEY,
                    running BOOLEAN,
                    start_time TEXT
                )
            ''')
            cursor.execute('SELECT running, start_time FROM bot_state WHERE id=1')
            row = cursor.fetchone()
            if row:
                # Override the running state to False on startup
                self.running = False
                self.start_time = None
                cursor.execute(
                    'UPDATE bot_state SET running=?, start_time=? WHERE id=1',
                    (self.running, self.start_time.isoformat() if self.start_time else None)
                )
                conn.commit()
                self.logger.info("[load_state] Bot state loaded as NOT running on startup.")
            else:
                cursor.execute(
                    'INSERT INTO bot_state (id, running, start_time) VALUES (1, ?, ?)',
                    (self.running, self.start_time.isoformat() if self.start_time else None)
                )
                conn.commit()
                self.logger.info("[load_state] Initialized new bot state in DB as NOT running.")
            conn.close()
        except Exception as e:
            self.logger.error(f"[load_state] DB error: {e}")

    def save_state(self):
        """
        Persist the run state to the SQLite DB.
        """
        try:
            conn = sqlite3.connect('metrics.db')
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bot_state (
                    id INTEGER PRIMARY KEY,
                    running BOOLEAN,
                    start_time TEXT
                )
            ''')
            cursor.execute(
                'UPDATE bot_state SET running=?, start_time=? WHERE id=1',
                (self.running, self.start_time.isoformat() if self.start_time else None)
            )
            conn.commit()
            conn.close()
            self.logger.info("[save_state] Bot state saved.")
        except Exception as e:
            self.logger.error(f"[save_state] Error: {e}")

    # ----------------------------------------------------------------------
    # LOGGING TRADES
    # ----------------------------------------------------------------------
    def log_trade(self, resp_data):
        """
        Record a newly opened trade from OANDA's orderFillTransaction, if present.
        """
        if 'orderFillTransaction' in resp_data:
            fill = resp_data['orderFillTransaction']
            trade_id = fill.get("tradeOpened", {}).get("tradeID")
            if not trade_id:
                self.logger.error("[log_trade] No trade ID in fill transaction.")
                return

            record = {
                'id': trade_id,
                'time': fill.get('time'),
                'instrument': fill.get('instrument'),
                'units': fill.get('units'),
                'price': float(fill.get('price', 0.0)),
                'sl_order_id': None,
                'tp_order_id': None
            }
            self.trades.append(record)
            self.trade_count += 1
            self.logger.info(f"[log_trade] New trade recorded: {record}")
            self.save_historical_metrics()

    def get_trades(self):
        """
        Return local list of recorded trades.
        """
        return self.trades

    def get_metrics(self):
        """
        Return key metrics (accountBalance, tradeCount, profitLoss, timeElapsed).
        """
        elapsed = str(datetime.now() - self.start_time) if self.start_time else "00:00:00"
        return {
            "accountBalance": self.account_balance,
            "tradeCount": len(self.trades),
            "profitLoss": self.profit_loss,
            "timeElapsed": elapsed
        }

    def calculate_profit_loss(self):
        """
        Optionally recalc P/L from self.trades if needed.
        """
        return sum(t.get("profit", 0) for t in self.trades)
