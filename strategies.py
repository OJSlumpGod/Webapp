import numpy as np
import talib





class TrendFollowingStrategy:
    def __init__(self, fast_period=20, slow_period=50):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def calculate_signals(self, closes, highs, lows, volumes):
        ema_fast = talib.EMA(closes, timeperiod=self.fast_period)
        ema_slow = talib.EMA(closes, timeperiod=self.slow_period)
        sar = talib.SAR(highs, lows, acceleration=0.02, maximum=0.2)

        buy = (ema_fast[-1] > ema_slow[-1]) and (closes[-1] > sar[-1])
        sell = (ema_fast[-1] < ema_slow[-1]) and (closes[-1] < sar[-1])
        return {"buy": buy, "sell": sell}


class MeanReversionStrategy:
    def __init__(self, rsi_period=14, bb_period=20):
        self.rsi_period = rsi_period
        self.bb_period = bb_period

    def calculate_signals(self, closes, highs, lows, volumes):
        rsi = talib.RSI(closes, timeperiod=self.rsi_period)
        upper, mid, lower = talib.BBANDS(closes, timeperiod=self.bb_period)

        buy = (rsi[-1] < 30) and (closes[-1] < lower[-1])
        sell = (rsi[-1] > 70) and (closes[-1] > upper[-1])
        return {"buy": buy, "sell": sell}


class BreakoutStrategy:
    def __init__(self, lookback=2):
        self.lookback = lookback

    def calculate_signals(self, closes, highs, lows, volumes):
        recent_high = highs[-self.lookback]
        recent_low = lows[-self.lookback]
        avg_volume = np.mean(volumes[-10:])

        buy = (closes[-1] > recent_high) and (volumes[-1] > avg_volume)
        sell = (closes[-1] < recent_low) and (volumes[-1] > avg_volume)
        return {"buy": buy, "sell": sell}


class MomentumStrategy:
    def calculate_signals(self, closes, highs, lows, volumes):
        macd, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
        stoch_k, stoch_d = talib.STOCH(highs, lows, closes)
        willr = talib.WILLR(highs, lows, closes, timeperiod=14)

        buy = (macd[-1] > macd_signal[-1]) and (stoch_k[-1] > stoch_d[-1]) and (willr[-1] < -80)
        sell = (macd[-1] < macd_signal[-1]) and (stoch_k[-1] < stoch_d[-1]) and (willr[-1] > -20)
        return {"buy": buy, "sell": sell}


class RangeBoundStrategy:
    def __init__(self, rsi_period=14):
        self.rsi_period = rsi_period

    def calculate_signals(self, closes, highs, lows, volumes):
        rsi = talib.RSI(closes, timeperiod=self.rsi_period)
        upper, mid, lower = talib.BBANDS(closes, timeperiod=20)
        in_range = (40 <= rsi[-1] <= 60)

        buy = in_range and (closes[-1] < mid[-1])
        sell = in_range and (closes[-1] > mid[-1])
        return {"buy": buy, "sell": sell}


class ADXTrendStrategy:
    def __init__(self):
        self.adx_period = 14
        self.ema_fast = 12
        self.ema_slow = 26

    def calculate_signals(self, closes, highs, lows, volumes):
        adx = talib.ADX(highs, lows, closes, timeperiod=self.adx_period)
        ema_f = talib.EMA(closes, timeperiod=self.ema_fast)
        ema_s = talib.EMA(closes, timeperiod=self.ema_slow)

        strong_trend = (adx[-1] > 25)
        buy = strong_trend and (ema_f[-1] > ema_s[-1])
        sell = strong_trend and (ema_f[-1] < ema_s[-1])
        return {"buy": buy, "sell": sell}


class BollingerSqueezeStrategy:
    def __init__(self):
        self.bb_period = 20

    def calculate_signals(self, closes, highs, lows, volumes):
        upper, mid, lower = talib.BBANDS(closes, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2)
        band_width = upper[-1] - lower[-1]
        avg_band_width = np.mean(upper - lower)
        in_squeeze = band_width < (0.8 * avg_band_width if avg_band_width != 0 else band_width)

        buy = in_squeeze and (closes[-1] > upper[-1])
        sell = in_squeeze and (closes[-1] < lower[-1])
        return {"buy": buy, "sell": sell}

class CrossoverStrategy:
    def __init__(self, short_window=50, long_window=200):
        self.short_window = short_window
        self.long_window = long_window

    def calculate_signals(self, closes, highs, lows, volumes):
        sma_short = talib.SMA(closes, timeperiod=self.short_window)
        sma_long = talib.SMA(closes, timeperiod=self.long_window)

        buy = (sma_short[-1] > sma_long[-1]) and (sma_short[-2] <= sma_long[-2])
        sell = (sma_short[-1] < sma_long[-1]) and (sma_short[-2] >= sma_long[-2])
        return {"buy": buy, "sell": sell}
    
class ScalpingStrategy:
    def __init__(self, ema_period=5, rsi_period=5, atr_period=14, bb_period=20, threshold_multiplier=0.5):
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.threshold_multiplier = threshold_multiplier

    def update_parameters(self, ema_period=None, rsi_period=None, atr_period=None, bb_period=None, threshold_multiplier=None):
        if ema_period is not None:
            self.ema_period = ema_period
        if rsi_period is not None:
            self.rsi_period = rsi_period
        if atr_period is not None:
            self.atr_period = atr_period
        if bb_period is not None:
            self.bb_period = bb_period
        if threshold_multiplier is not None:
            self.threshold_multiplier = threshold_multiplier

    def calculate_signals(self, closes, highs, lows, volumes):
        """
        Calculate buy/sell signals based on EMA, RSI, ATR, and Bollinger Bands.
        
        Buy Signal Conditions:
          - Price is significantly below EMA (based on ATR).
          - RSI indicates oversold conditions (< 30).
          - Price is below the lower Bollinger Band.
        
        Sell Signal Conditions:
          - Price is significantly above EMA (based on ATR).
          - RSI indicates overbought conditions (> 70).
          - Price is above the upper Bollinger Band.
        """
        # Ensure there are enough data points for indicators
        required_length = max(self.ema_period, self.rsi_period, self.atr_period, self.bb_period)
        if len(closes) < required_length:
            return {"buy": False, "sell": False}

        # Calculate technical indicators
        ema = talib.EMA(closes, timeperiod=self.ema_period)
        rsi = talib.RSI(closes, timeperiod=self.rsi_period)
        atr = talib.ATR(highs, lows, closes, timeperiod=self.atr_period)
        upper_band, middle_band, lower_band = talib.BBANDS(closes, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2)

        # Validate indicator outputs
        if np.isnan(ema[-1]) or np.isnan(rsi[-1]) or np.isnan(atr[-1]) or np.isnan(upper_band[-1]) or np.isnan(lower_band[-1]):
            return {"buy": False, "sell": False}

        # Dynamic threshold based on volatility
        threshold = atr[-1] * self.threshold_multiplier
        current_price = closes[-1]
        price_diff = current_price - ema[-1]

        # Define conditions for BUY and SELL signals
        buy_signal = (price_diff < -threshold) and (rsi[-1] < 30) and (current_price < lower_band[-1])
        sell_signal = (price_diff > threshold) and (rsi[-1] > 70) and (current_price > upper_band[-1])

        return {"buy": buy_signal, "sell": sell_signal}
