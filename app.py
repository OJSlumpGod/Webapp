from flask import Flask, render_template, jsonify, request, Response
from bot_manager import BotManager  # Ensure BotManager is correctly implemented
from ml_model import MLModel
from trading_bot import TradingBot
import json
import sqlite3
import time
from datetime import datetime
import logging
import os

app = Flask(__name__, static_folder="static", template_folder="templates")

# Configure Logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "flask_app.log")  # Adjust as needed

# Configuration
SETTINGS_FILE = os.path.join("config", "settings.json")  # Unified settings file path

# Configure Flask Logger
flask_logger = logging.getLogger('flask_app')
flask_logger.setLevel(logging.INFO)

# File handler for Flask logs
flask_file_handler = logging.FileHandler(LOG_FILE_PATH)
flask_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
flask_file_handler.setFormatter(flask_formatter)
flask_logger.addHandler(flask_file_handler)

# Console handler for Flask logs
console_handler = logging.StreamHandler()
console_handler.setFormatter(flask_formatter)
flask_logger.addHandler(console_handler)

def load_settings():
    """Load settings from file."""
    try:
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        flask_logger.warning("Settings file not found. Creating default settings.")
        default_settings = {
            "riskLevel": "medium",
            "maxTrades": 5,
            "tradeIntervalHours": 24,
            "stopLossPercentage": 2.0,
            "takeProfitPercentage": 5.0,
            "rsiPeriod": 14,
            "emaFastPeriod": 12,
            "emaSlowPeriod": 26,
            "bbandsPeriod": 20,
            "trailingATRMultiplier": 1.5,
            "adjustATRMultiplier": 1.5,
            "tradeCooldown": 60
        }
        save_settings(default_settings)
        return default_settings
    except json.JSONDecodeError as e:
        flask_logger.error(f"JSON decode error in settings file: {e}")
        return {
            "riskLevel": "medium",
            "maxTrades": 5,
            "tradeIntervalHours": 24,
            "stopLossPercentage": 2.0,
            "takeProfitPercentage": 5.0,
            "rsiPeriod": 14,
            "emaFastPeriod": 12,
            "emaSlowPeriod": 26,
            "bbandsPeriod": 20,
            "trailingATRMultiplier": 1.5,
            "adjustATRMultiplier": 1.5,
            "tradeCooldown": 60
        }

def save_settings(data):
    """Save settings to file with backup."""
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        # Backup existing settings
        if os.path.exists(SETTINGS_FILE):
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            backup_file = f"{SETTINGS_FILE}.{timestamp}.bak"
            os.rename(SETTINGS_FILE, backup_file)
            flask_logger.info(f"Backup created at {backup_file}")
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        flask_logger.info("Settings saved successfully.")
    except Exception as e:
        flask_logger.error(f"Failed to save settings: {e}")

def initialize_bot():
    """Initialize bot with settings from file."""
    settings = load_settings()
    bot_manager.apply_settings(settings)
    flask_logger.info("Bot initialized with settings.")

# Initialize MLModel instance
ml_model = MLModel()

# Initialize TradingBot instance with MLModel
trading_bot = TradingBot(ml_model=ml_model)

# Initialize BotManager with TradingBot instance
bot_manager = BotManager(trading_bot=trading_bot)

# Initialize bot settings on startup
initialize_bot()

@app.route("/")
def index():
    """Render the main overview page."""
    return render_template("overview.html")

@app.route("/metrics_stream")  # SSE endpoint
def metrics_stream():
    """
    SSE endpoint to stream the bot’s metrics (trade count, P/L, time elapsed, etc.).
    """
    def event_stream():
        while True:
            try:
                metrics = bot_manager.get_metrics()
                yield f"data: {json.dumps(metrics)}\n\n"
                time.sleep(1)
            except GeneratorExit:
                flask_logger.info("Client disconnected from metrics_stream stream.")
                break
            except Exception as e:
                flask_logger.error(f"Error in metrics_stream stream: {e}")
                break
    return Response(event_stream(), mimetype="text/event-stream")

@app.route("/history_data", methods=["GET"])
def history_data():
    """
    Endpoint to fetch trade history from OANDA.
    """
    try:
        trades = bot_manager.get_oanda_trade_history(count=50)
        if not trades:
            return jsonify({"message": "No trades found or failed to retrieve trades."}), 200

        # Format trades as needed for the front-end
        formatted_trades = []
        for trade in trades:
            formatted_trades.append({
                "id": trade.get("id"),
                "time": trade.get("openTime"),
                "instrument": trade.get("instrument"),
                "units": trade.get("currentUnits"),
                "price": trade.get("price"),
                "state": trade.get("state"),
                # Add other relevant fields as needed
            })

        return jsonify({"trades": formatted_trades}), 200

    except Exception as e:
        flask_logger.error(f"Error in history_data: {e}")
        return jsonify({"error": f"Failed to retrieve trade history: {str(e)}"}), 500

@app.route("/open_positions", methods=["GET"])
def open_positions():
    """
    Return current open trades from OANDA.
    """
    try:
        open_positions = bot_manager.trading_bot.get_open_trades()
        return jsonify(open_positions), 200
    except Exception as e:
        flask_logger.error(f"Error in open_positions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/overview")
def overview():
    """Render the overview page."""
    return render_template("overview.html")

@app.route("/positions")
def positions():
    """Render the positions page."""
    return render_template("positions.html")

@app.route("/history")
def history():
    """Render the history page."""
    return render_template("history.html")

@app.route("/settings", methods=["GET"])
def settings_page():
    """Render the settings page."""
    return render_template("settings.html")

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        # Retrieve current settings
        try:
            with open(SETTINGS_FILE, 'r') as settings_file:
                settings = json.load(settings_file)
            return jsonify(settings), 200
        except FileNotFoundError:
            flask_logger.error("Settings file not found.")
            return jsonify({"error": "Settings file not found."}), 404
        except json.JSONDecodeError:
            flask_logger.error("Error decoding settings file.")
            return jsonify({"error": "Error decoding settings file."}), 500
        except Exception as e:
            flask_logger.error(f"Unexpected error: {e}")
            return jsonify({"error": "An unexpected error occurred."}), 500

    elif request.method == "POST":
        # Update settings
        new_settings = request.get_json()
        flask_logger.info(f"Received new settings: {new_settings}")

        # Provide default for riskLevel if missing or empty
        if "riskLevel" not in new_settings or not new_settings["riskLevel"]:
            new_settings["riskLevel"] = "medium"
            flask_logger.info("riskLevel missing or empty. Set to default 'medium'.")

        required_fields = ["riskLevel", "maxTrades", "tradeIntervalHours",
                           "stopLossPercentage", "takeProfitPercentage",
                           "rsiPeriod", "emaFastPeriod", "emaSlowPeriod",
                           "bbandsPeriod", "trailingATRMultiplier",
                           "adjustATRMultiplier", "tradeCooldown"]

        # Validate incoming data
        missing_fields = [field for field in required_fields if field not in new_settings]
        if missing_fields:
            error_message = f"Missing field(s): {', '.join(missing_fields)}"
            flask_logger.error(f"Settings validation error: {error_message}")
            return jsonify({"error": f"Missing field(s): {', '.join(missing_fields)}"}), 400

        # Additional validation: Ensure riskLevel is one of the expected values
        valid_risk_levels = ["low", "medium", "high"]
        if new_settings["riskLevel"].lower() not in valid_risk_levels:
            error_message = f"Invalid value for riskLevel. Expected one of {valid_risk_levels}."
            flask_logger.error(f"Settings validation error: {error_message}")
            return jsonify({"error": error_message}), 400

        # Additional validation can be added here (e.g., value ranges)

        try:
            with open(SETTINGS_FILE, 'w') as settings_file:
                json.dump(new_settings, settings_file, indent=4)
            flask_logger.info("Settings updated successfully.")
            bot_manager.apply_settings(new_settings)  # Apply new settings to the bot
            return jsonify({"status": "success", "message": "Settings updated successfully."}), 200
        except Exception as e:
            flask_logger.error(f"Error writing settings file: {e}")
            return jsonify({"error": "Failed to save settings."}), 500
        
@app.route("/api/logs", methods=["GET"])
def api_combined_logs():
    """
    Fetch and return combined logs from all relevant log files.
    """
    log_files = [
        os.path.join(LOG_DIR, "flask_app.log"),
        os.path.join(LOG_DIR, "trading_bot.log"),
        os.path.join(LOG_DIR, "ml_model.log"),
    ]
    all_logs = ""
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    log_content = f.read()
                    if log_content.strip():  # Ensure the file isn't empty
                        all_logs += f"--- {os.path.basename(log_file)} ---\n{log_content}\n\n"
                    else:
                        all_logs += f"--- {os.path.basename(log_file)} ---\n[Log file is empty]\n\n"
            except Exception as e:
                all_logs += f"--- {os.path.basename(log_file)} ---\n[Failed to read log file: {e}]\n\n"
        else:
            all_logs += f"--- {os.path.basename(log_file)} ---\n[Log file not found]\n\n"
    
    return all_logs, 200, {"Content-Type": "text/plain"}

@app.route("/start_bot", methods=["POST"])
def start_bot():
    """
    Start the trading bot if not already running.
    """
    try:
        if not bot_manager.running:
            bot_manager.start_bot()
            return jsonify({"success": True, "message": "Bot started successfully."}), 200
        return jsonify({"success": False, "message": "Bot is already running."}), 200
    except Exception as e:
        flask_logger.error(f"Error starting bot: {e}")
        return jsonify({"success": False, "message": f"Failed to start bot: {str(e)}"}), 500

@app.route("/stop_bot", methods=["POST"])
def stop_bot():
    """
    Stop the trading bot if it's currently running.
    """
    try:
        if bot_manager.running:
            bot_manager.stop_bot()
            return jsonify({"success": True, "message": "Bot stopped successfully."}), 200
        return jsonify({"success": False, "message": "Bot is not running."}), 200
    except Exception as e:
        flask_logger.error(f"Error stopping bot: {e}")
        return jsonify({"success": False, "message": f"Failed to stop bot: {str(e)}"}), 500

@app.route("/reset_bot", methods=["POST"])
def reset_bot():
    """
    Reset the trading bot.
    """
    try:
        bot_manager.reset_bot()  # Ensure this method is correctly implemented in BotManager
        return jsonify({"status": "success", "message": "Bot reset successfully!"}), 200
    except Exception as e:
        flask_logger.error(f"Error resetting bot: {e}")
        return jsonify({"status": "error", "message": f"Failed to reset bot: {str(e)}"}), 500

@app.route("/feature_engineering_status", methods=["GET"])
def feature_engineering_status():
    """
    Return the latest feature engineering status or errors.
    """
    # Implement logic to retrieve the latest status. For simplicity, return last log message.
    # In production, consider using a more robust logging mechanism or status tracking.
    try:
        with open(LOG_FILE_PATH, 'r') as log_file:
            lines = log_file.readlines()
            if lines:
                last_log = lines[-1]
            else:
                last_log = "No logs available."
        return jsonify({"status": "OK", "message": last_log.strip()}), 200
    except Exception as e:
        flask_logger.error(f"Error fetching feature engineering status: {e}")
        return jsonify({"status": "error", "message": f"Failed to fetch status: {str(e)}"}), 500

@app.route("/bot_status", methods=["GET"])
def bot_status():
    """
    Return the current running status of the bot.
    """
    try:
        status = bot_manager.running
        return jsonify({"running": status}), 200
    except Exception as e:
        flask_logger.error(f"Error fetching bot status: {e}")
        return jsonify({"error": f"Failed to fetch bot status: {str(e)}"}), 500
    
@app.route("/logs", methods=["GET"])
def logs():
    """
    Render the Logs page with all logs (Flask app, trading bot, ML, etc.).
    """
    try:
        # List of log files to include
        log_files = [
            os.path.join(LOG_DIR, "flask_app.log"),
            os.path.join(LOG_DIR, "trading_bot.log"),
            os.path.join(LOG_DIR, "ml_model.log"),
        ]

        all_logs = ""
        for log_file in log_files:
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    log_content = f.read()
                    all_logs += f"--- {os.path.basename(log_file)} ---\n{log_content}\n\n"
            else:
                all_logs += f"--- {os.path.basename(log_file)} ---\nNo logs available.\n\n"

        return render_template("logs.html", logs=all_logs)

    except Exception as e:
        flask_logger.error(f"Error rendering logs: {e}")
        error_message = f"Error displaying logs: {e}"
        return render_template("logs.html", logs=error_message)

if __name__ == "__main__":
    # Ensure metrics table exists
    try:
        conn = sqlite3.connect('metrics.db')
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_count INT,
                profit_loss FLOAT
            )
        """)
        conn.commit()
        conn.close()
        flask_logger.info("Metrics table ensured in DB.")
    except Exception as e:
        flask_logger.error(f"Error ensuring metrics table: {e}")

    # Run Flask app
    app.run(debug=True, host='127.0.0.1', port=5000)
