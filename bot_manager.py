import logging
import os
import json
from datetime import datetime
from trading_bot import TradingBot

# Constants
STATE_FILE = "bot_state.json"
LOG_DIR = "logs"

# Ensure logs folder exists
os.makedirs(LOG_DIR, exist_ok=True)

class BotManager:
    """
    BotManager handles the lifecycle of the TradingBot, including starting, stopping,
    applying settings, and providing metrics to the Flask application.
    """

    def __init__(self, trading_bot):
        """
        Initialize the BotManager with a TradingBot instance.

        :param trading_bot: An instance of TradingBot.
        """
        # Initialize logger for BotManager
        self.logger = logging.getLogger('BotManager')
        self.logger.setLevel(logging.INFO)

        # File handler for BotManager logs
        file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'bot_manager.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler for BotManager logs
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Assign the passed TradingBot instance
        self.trading_bot = trading_bot
        self.running = self.trading_bot.running
        self.progress = "Initialized."

        # State management
        self.start_time = None
        self.load_state()

    def apply_settings(self, settings):
        """
        Apply user-provided settings to the TradingBot.

        :param settings: Dictionary of settings.
        """
        try:
            self.trading_bot.apply_settings(settings)
            self.logger.info("BotManager: Settings applied to TradingBot.")
        except Exception as e:
            self.logger.error(f"BotManager: Failed to apply settings - {e}")

    def start_bot(self):
        """
        Start the TradingBot.
        """
        if not self.running:
            self.trading_bot.start()
            self.running = self.trading_bot.running
            self.start_time = datetime.now()
            self.progress = "Bot started."
            self.logger.info("BotManager: TradingBot started.")
            self.save_state()
        else:
            self.logger.info("BotManager: TradingBot is already running.")
            self.progress = "Already running."

    def stop_bot(self):
        """
        Stop the TradingBot.
        """
        if self.running:
            self.trading_bot.stop()
            self.running = self.trading_bot.running
            self.progress = "Bot stopped."
            self.logger.info("BotManager: TradingBot stopped.")
            self.save_state()
        else:
            self.logger.warning("BotManager: TradingBot is not running.")
            self.progress = "Not running."

    def reset_bot(self):
        """
        Reset the TradingBot by stopping it and re-initializing.
        """
        self.stop_bot()
        # Re-initialize TradingBot if necessary
        # For example, you might want to clear states or reconfigure
        self.trading_bot = TradingBot()
        self.running = self.trading_bot.running
        self.progress = "Bot reset."
        self.logger.info("BotManager: TradingBot reset.")
        self.save_state()

    def get_oanda_trade_history(self, count=50):
        """
        Fetch trade history from the TradingBot.

        :param count: Number of recent trades to retrieve.
        :return: List of trade dictionaries or empty list on failure.
        """
        return self.trading_bot.get_oanda_trade_history(count)

    def get_metrics(self):
        """
        Retrieve metrics from the TradingBot and include the elapsed time.

        :return: Dictionary of metrics.
        """
        try:
            metrics = self.trading_bot.get_metrics()
            if self.start_time:
                time_elapsed = str(datetime.now() - self.start_time)
            else:
                time_elapsed = "00:00:00"
            metrics["timeElapsed"] = time_elapsed
            return metrics
        except Exception as e:
            self.logger.error(f"BotManager: Failed to get metrics - {e}")
            return {
                "accountBalance": 0.0,
                "tradeCount": 0,
                "profitLoss": 0.0,
                "timeElapsed": "00:00:00",
                "error": str(e)
            }

    def save_state(self):
        """
        Save the current state of the BotManager to a JSON file.
        """
        state = {
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "running": self.running
        }
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=4)
            self.logger.info("BotManager: State saved successfully.")
        except Exception as e:
            self.logger.error(f"BotManager: Failed to save state - {e}")

    def load_state(self):
        """
        Load the BotManager state from a JSON file, if it exists.
        """
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    state = json.load(f)
                    start_time_str = state.get("start_time")
                    self.start_time = datetime.fromisoformat(start_time_str) if start_time_str else None
                    self.running = state.get("running", False)
                self.logger.info("BotManager: State loaded successfully.")
                if self.running:
                    self.start_bot()
            except Exception as e:
                self.logger.error(f"BotManager: Failed to load state - {e}")
        else:
            self.logger.info("BotManager: No existing state file found. Starting fresh.")

    def get_progress(self):
        """
        Get the current progress/status of the BotManager.

        :return: Progress string.
        """
        return self.progress

    def log_trade(self, trade, result):
        """
        Update TradingBot's metrics based on the outcome of a trade.

        :param trade: Trade information.
        :param result: Result of the trade.
        """
        try:
            self.trading_bot.trade_count += 1
            self.trading_bot.profit_loss += result['profit_loss']
            self.logger.info(f"BotManager: Trade logged - P/L: {result['profit_loss']:.2f}")
            self.trading_bot.save_historical_metrics()
        except Exception as e:
            self.logger.error(f"BotManager: Failed to log trade - {e}")
