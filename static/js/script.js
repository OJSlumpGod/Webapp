document.addEventListener("DOMContentLoaded", function () {
  // Initialize button elements
  const startBtn = document.getElementById("startBot");
  const stopBtn = document.getElementById("stopBot");
  const resetBtn = document.getElementById("resetBot");
  const saveSettingsBtn = document.getElementById("saveSettingsButton");
  const refreshLogsBtn = document.getElementById("refreshLogsButton");
  const mlLogsPre = document.getElementById("mlLogs");

  /**
   * Displays feedback messages to the user.
   * @param {string} message - The message to display.
   * @param {string} type - The type of alert ('success', 'danger', 'warning', 'info').
   */
  function showFeedback(message, type) {
      const feedbackEl = document.getElementById("feedback");
      if (!feedbackEl) {
          console.warn("Feedback element with id 'feedback' not found.");
          return;
      }
      feedbackEl.textContent = message || "An unexpected error occurred.";
      feedbackEl.className = `alert alert-${type}`;
      feedbackEl.style.display = "block";
      setTimeout(() => {
          feedbackEl.style.display = "none";
      }, 5000);
  }

  /**
   * Handles bot control actions (start, stop, reset) by sending POST requests.
   * @param {string} endpoint - The API endpoint to hit.
   * @param {string} successMessage - Message to display on success.
   * @param {string} errorMessage - Message to display on error.
   * @param {HTMLElement} btnToDisable - Button to disable after action.
   * @param {HTMLElement} btnToEnable - Button to enable after action.
   */
  function handleBotControl(endpoint, successMessage, errorMessage, btnToDisable, btnToEnable) {
      fetch(endpoint, { method: "POST" })
          .then(response => response.json())
          .then(data => {
              const isSuccess = data.success || data.status === "success";
              const type = isSuccess ? "success" : "danger";
              const message = data.message || (isSuccess ? successMessage : errorMessage);
              showFeedback(message, type);

              if (isSuccess) {
                  if (btnToDisable) btnToDisable.disabled = true;
                  if (btnToEnable) btnToEnable.disabled = false;
              }
          })
          .catch(() => {
              showFeedback(errorMessage, "danger");
          });
  }

  // Attach event listeners to control buttons if they exist
  if (startBtn) {
      startBtn.addEventListener("click", () =>
          handleBotControl(
              "/start_bot",
              "Bot started successfully!",
              "Failed to start bot.",
              startBtn,
              stopBtn
          )
      );
  } else {
      console.warn("Start Bot button with id 'startBot' not found.");
  }

  if (stopBtn) {
      stopBtn.addEventListener("click", () =>
          handleBotControl(
              "/stop_bot",
              "Bot stopped successfully!",
              "Failed to stop bot.",
              stopBtn,
              startBtn
          )
      );
  } else {
      console.warn("Stop Bot button with id 'stopBot' not found.");
  }

  if (resetBtn) {
      resetBtn.addEventListener("click", () =>
          handleBotControl(
              "/reset_bot",
              "Bot reset successfully!",
              "Failed to reset bot.",
              resetBtn,
              startBtn
          )
      );
  } else {
      console.warn("Reset Bot button with id 'resetBot' not found.");
  }

  /**
   * Fetches the current status of the bot and updates button states accordingly.
   */
  function updateBotStatus() {
      fetch("/bot_status", { method: "GET" })
          .then(response => response.json())
          .then(data => {
              // Ensure buttons exist before trying to disable/enable them
              if (data.running) {
                  if (startBtn) startBtn.disabled = true;
                  if (stopBtn) stopBtn.disabled = false;
              } else {
                  if (startBtn) startBtn.disabled = false;
                  if (stopBtn) stopBtn.disabled = true;
              }
          })
          .catch(error => {
              console.error("Error fetching bot status:", error);
              showFeedback("Failed to fetch bot status.", "danger");
          });
  }

  // Initial status check
  updateBotStatus();

  // Periodically update status (every 5 seconds)
  setInterval(updateBotStatus, 5000);

  /**
   * Parses the trade interval input.
   * @param {string} value - The trade interval value (e.g., '1h', '30m').
   * @returns {number} - The trade interval in hours.
   */
  function parseTradeInterval(value) {
      if (typeof value !== 'string') {
          console.warn("Trade interval value is not a string:", value);
          return 1;
      }

      if (value.endsWith('m')) {
          return parseFloat(value) / 60;
      } else if (value.endsWith('h')) {
          return parseFloat(value);
      }
      return 1;
  }

  /**
   * Loads settings from the server and populates the settings form.
   */
  function loadSettings() {
      fetch("/api/settings", { method: "GET" })
          .then(response => {
              if (!response.ok) throw new Error("Failed to load settings");
              return response.json();
          })
          .then(settings => {
              console.log("Loaded Settings:", settings);

              // Helper function to safely set element values
              const setElementValue = (id, value) => {
                  const el = document.getElementById(id);
                  if (el) {
                      el.value = value;
                  } else {
                      console.warn(`Element with id '${id}' not found.`);
                  }
              };

              setElementValue("riskLevel", settings.riskLevel || "medium");
              setElementValue("maxTrades", settings.maxTrades || 5);
              setElementValue("tradeInterval", 
                  settings.tradeIntervalHours <= 1 ? "1h" : `${settings.tradeIntervalHours * 60}m`);
              setElementValue("stopLossPercentage", 
                  settings.stopLossPercentage !== undefined ? settings.stopLossPercentage : 2.0);
              setElementValue("takeProfitPercentage", 
                  settings.takeProfitPercentage !== undefined ? settings.takeProfitPercentage : 5.0);
              setElementValue("rsiPeriod", 
                  settings.rsiPeriod !== undefined ? settings.rsiPeriod : 14);
              setElementValue("emaFastPeriod", 
                  settings.emaFastPeriod !== undefined ? settings.emaFastPeriod : 12);
              setElementValue("emaSlowPeriod", 
                  settings.emaSlowPeriod !== undefined ? settings.emaSlowPeriod : 26);
              setElementValue("bbandsPeriod", 
                  settings.bbandsPeriod !== undefined ? settings.bbandsPeriod : 20);
              setElementValue("trailingATRMultiplier", 
                  settings.trailingATRMultiplier !== undefined ? settings.trailingATRMultiplier : 1.5);
              setElementValue("adjustATRMultiplier", 
                  settings.adjustATRMultiplier !== undefined ? settings.adjustATRMultiplier : 1.5);
              setElementValue("tradeCooldown", 
                  settings.tradeCooldown !== undefined ? settings.tradeCooldown : 60);
              setElementValue("scalpingEmaPeriod", settings.scalpingEmaPeriod || 5);
              setElementValue("scalpingRsiPeriod", settings.scalpingRsiPeriod || 5);
              setElementValue("scalpingAtrPeriod", settings.scalpingAtrPeriod || 14);
              setElementValue("scalpingBbPeriod", settings.scalpingBbPeriod || 20);
              setElementValue("scalpingThresholdMultiplier", settings.scalpingThresholdMultiplier || 0.5); 
          })
          .catch(error => {
              console.error("Error loading settings:", error);
              showFeedback("Failed to load settings.", "danger");
          });
  }

  /**
 * Saves settings by sending them to the server.
 */
  function saveSettings() {
    const tradeIntervalValue = document.getElementById("tradeInterval").value;
    const tradeIntervalHours = parseTradeInterval(tradeIntervalValue);

    const getElementValue = (id, parser = v => v) => {
        const el = document.getElementById(id);
        if (el) {
            return parser(el.value);
        } else {
            console.warn(`Element with id '${id}' not found.`);
            return null;
        }
    };

    // Collect settings values
    const riskLevelValue = getElementValue("riskLevel");
    if (!riskLevelValue || !["low", "medium", "high"].includes(riskLevelValue.toLowerCase())) {
        showFeedback("Please select a valid risk level.", "warning");
        return;
    }

    const settings = {
        riskLevel: riskLevelValue.toLowerCase(),
        maxTrades: getElementValue("maxTrades", val => parseInt(val, 10)),
        tradeIntervalHours: tradeIntervalHours,
        stopLossPercentage: getElementValue("stopLossPercentage", parseFloat),
        takeProfitPercentage: getElementValue("takeProfitPercentage", parseFloat),
        rsiPeriod: getElementValue("rsiPeriod", val => parseInt(val, 10)),
        emaFastPeriod: getElementValue("emaFastPeriod", val => parseInt(val, 10)),
        emaSlowPeriod: getElementValue("emaSlowPeriod", val => parseInt(val, 10)),
        bbandsPeriod: getElementValue("bbandsPeriod", val => parseInt(val, 10)),
        trailingATRMultiplier: getElementValue("trailingATRMultiplier", parseFloat),
        adjustATRMultiplier: getElementValue("adjustATRMultiplier", parseFloat),
        tradeCooldown: getElementValue("tradeCooldown", val => parseInt(val, 10)),
        scalpingEmaPeriod: getElementValue("scalpingEmaPeriod", v => parseInt(v, 10)),
        scalpingRsiPeriod: getElementValue("scalpingRsiPeriod", v => parseInt(v, 10)),
        scalpingAtrPeriod: getElementValue("scalpingAtrPeriod", v => parseInt(v, 10)),
        scalpingBbPeriod: getElementValue("scalpingBbPeriod", v => parseInt(v, 10)),
        scalpingThresholdMultiplier: getElementValue("scalpingThresholdMultiplier", parseFloat),
    };

    // Collect enabled strategies from checkboxes
    const selectedStrategies = Array.from(document.querySelectorAll('input[name="strategy"]:checked'))
                                    .map(checkbox => checkbox.value);
    settings.enabledStrategies = selectedStrategies;

    // Validate required fields
    const requiredStringFields = ["riskLevel"];
    const requiredNumericFields = ["maxTrades", "tradeIntervalHours",
                                   "stopLossPercentage", "takeProfitPercentage",
                                   "trailingATRMultiplier", "adjustATRMultiplier",
                                   "tradeCooldown"];
    for (let field of requiredStringFields) {
        if (!settings[field] || typeof settings[field] !== 'string') {
            showFeedback(`Invalid or missing value for ${field}.`, "warning");
            return;
        }
    }
    for (let field of requiredNumericFields) {
        if (settings[field] === null || isNaN(settings[field])) {
            showFeedback(`Invalid or missing value for ${field}.`, "warning");
            return;
        }
    }

    fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showFeedback(data.error, "danger");
        } else if (data.status === "success" || data.message) {
            showFeedback(data.message || "Settings updated successfully.", "success");
        } else {
            showFeedback("Unknown response from server.", "warning");
        }
    })
    .catch(error => {
        console.error("Error saving settings:", error);
        showFeedback("Failed to save settings.", "danger");
    });
}

  // Attach event listener to Save Settings button if it exists
  if (saveSettingsBtn) {
      saveSettingsBtn.addEventListener("click", saveSettings);
  } else {
      console.warn("Save Settings button with id 'saveSettingsButton' not found.");
  }

  /**
   * Fetches and displays open positions in the designated table.
   */
  function fetchOpenPositions() {
      const openPositionsTableBody = document.getElementById("openPositionsTable");
      if (!openPositionsTableBody) {
          console.warn("[fetchOpenPositions] Table body with id 'openPositionsTable' not found.");
          return;
      }

      // Display loading spinner
      openPositionsTableBody.innerHTML = `
          <tr>
              <td colspan="6" class="text-center">
                  <div class="spinner-border text-light" role="status">
                      <span class="visually-hidden">Loading...</span>
                  </div>
              </td>
          </tr>
      `;

      fetch("/open_positions")
          .then(response => {
              if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
              return response.json();
          })
          .then(positions => {
              openPositionsTableBody.innerHTML = ""; // Clear existing content

              if (Array.isArray(positions) && positions.length > 0) {
                  positions.forEach(pos => {
                      // Safely access pos.side
                      let side = pos.side;
                      if (typeof side === 'string') {
                          side = side.toUpperCase();
                      } else {
                          side = "UNKNOWN";
                          console.warn("Trade side is undefined or not a string:", pos);
                      }

                      // Safely access other properties with defaults
                      const instrument = pos.instrument || "N/A";
                      const units = pos.units !== undefined ? pos.units : "N/A";
                      const entryPrice = pos.entry_price !== undefined ? pos.entry_price.toFixed(5) : "N/A";
                      const currentPrice = pos.current_price !== undefined ? pos.current_price.toFixed(5) : "N/A";

                      // Format profit/loss
                      let profitLossFormatted = pos.profit_loss !== undefined ? pos.profit_loss.toFixed(2) : "N/A";
                      let profitLossClass = "text-white";
                      if (pos.profit_loss !== undefined) {
                          profitLossClass = pos.profit_loss >= 0 ? "text-success" : "text-danger";
                          profitLossFormatted = `$${profitLossFormatted}`;
                      }

                      // Construct table row
                      const row = document.createElement("tr");

                      row.innerHTML = `
                          <td>${instrument}</td>
                          <td>${side}</td>
                          <td>${units}</td>
                          <td>${entryPrice}</td>
                          <td>${currentPrice}</td>
                          <td class="${profitLossClass}">${profitLossFormatted}</td>
                      `;

                      openPositionsTableBody.appendChild(row);
                  });
              } else {
                  // Display message when no open positions are available
                  openPositionsTableBody.innerHTML = `
                      <tr>
                          <td colspan="6" class="text-center">No open positions available.</td>
                      </tr>`;
              }
          })
          .catch(error => {
              console.error("[fetchOpenPositions] error:", error);
              openPositionsTableBody.innerHTML = `
                  <tr>
                      <td colspan="6" class="text-center">Failed to fetch open positions.</td>
                  </tr>`;
              showFeedback("Failed to fetch open positions.", "danger");
          });
  }

  /**
   * Fetches and displays trade history in the designated tables.
   */
  function fetchTradeHistory() {
      const oandaTableBody = document.getElementById("oandaTradesTable");
      const localTableBody = document.getElementById("localTradesTable");

      if (!oandaTableBody) {
          console.warn("[fetchTradeHistory] Table body with id 'oandaTradesTable' not found.");
      }

      if (!localTableBody) {
          console.warn("[fetchTradeHistory] Table body with id 'localTradesTable' not found.");
      }

      fetch("/history_data")
          .then(res => {
              if (!res.ok) throw new Error(`HTTP error! Status: ${res.status}`);
              return res.json();
          })
          .then(data => {
              // Populate OANDA Trades Table
              if (oandaTableBody) {
                  oandaTableBody.innerHTML = "";
                  if (data.trades && data.trades.length > 0) {  // Adjusted based on previous Flask response
                      data.trades.forEach(trade => {
                          // Safely access properties with defaults
                          const id = trade.id || "N/A";
                          const time = trade.time || "N/A";
                          const instrument = trade.instrument || "N/A";
                          const side = trade.units > 0 ? "BUY" : "SELL";
                          const units = Math.abs(trade.units);
                          const price = trade.price !== undefined ? parseFloat(trade.price).toFixed(5) : "N/A";

                          // Construct table row
                          const row = document.createElement("tr");

                          row.innerHTML = `
                              <td>${id}</td>
                              <td>${time}</td>
                              <td>${instrument}</td>
                              <td>${side}</td>
                              <td>${units}</td>
                              <td>${price}</td>
                          `;

                          oandaTableBody.appendChild(row);
                      });
                  } else {
                      oandaTableBody.innerHTML = `
                          <tr>
                              <td colspan="6" class="text-center">No trades found.</td>
                          </tr>`;
                  }
              }

              // Populate Local Trades Table (if applicable)
              if (localTableBody) {
                  localTableBody.innerHTML = "";
                  // Example: Iterate over local trades if provided
                  if (data.local_trades && data.local_trades.length > 0) {
                      data.local_trades.forEach(trade => {
                          // Safely access properties with defaults
                          const id = trade.id || "N/A";
                          const time = trade.time || "N/A";
                          const instrument = trade.instrument || "N/A";
                          const side = trade.units > 0 ? "BUY" : "SELL";
                          const units = Math.abs(trade.units);
                          const price = trade.price !== undefined ? parseFloat(trade.price).toFixed(5) : "N/A";

                          // Construct table row
                          const row = document.createElement("tr");

                          row.innerHTML = `
                              <td>${id}</td>
                              <td>${time}</td>
                              <td>${instrument}</td>
                              <td>${side}</td>
                              <td>${units}</td>
                              <td>${price}</td>
                          `;

                          localTableBody.appendChild(row);
                      });
                  } else {
                      localTableBody.innerHTML = `
                          <tr>
                              <td colspan="6" class="text-center">No local trades found.</td>
                          </tr>`;
                  }
              }
          })
          .catch(error => {
              console.error("[fetchTradeHistory] error:", error);
              showFeedback("Failed to fetch trade history.", "danger");
          });
  }

  /**
   * Fetches the status of feature engineering and displays it.
   */
  function fetchFeatureEngineeringStatus() {
      fetch("/feature_engineering_status", { method: "GET" })  // Ensure this endpoint is implemented
          .then(response => response.json())
          .then(data => {
              const statusPre = document.getElementById("featureEngineeringStatus");
              if (!statusPre) {
                  console.warn("Element with id 'featureEngineeringStatus' not found.");
                  return;
              }
              statusPre.textContent = JSON.stringify(data, null, 2);
          })
          .catch(error => {
              console.error("[fetchFeatureEngineeringStatus] error:", error);
              showFeedback("Failed to fetch feature engineering status.", "danger");
          });
  }

  /**
   * Fetches and displays the latest logs from the server.
   */
  function fetchLogs() {
      if (!mlLogsPre) {
          console.warn("Logs pre element with id 'mlLogs' not found.");
          return;
      }

      // Show loading message
      mlLogsPre.textContent = "Fetching logs...";

      fetch("/api/logs")
          .then(response => {
              if (!response.ok) {
                  throw new Error("Failed to fetch logs.");
              }
              return response.text();
          })
          .then(logs => {
              if (logs.trim()) {
                  mlLogsPre.textContent = logs || "No logs available.";
               } else {
                  mlLogsPre.textContent = "No logs available.";
              }
              mlLogsPre.scrollTop = mlLogsPre.scrollHeight; // Auto-scroll to the bottom
          })
          .catch(error => {
              console.error("Error fetching logs:", error);
              mlLogsPre.textContent = "Failed to fetch logs. Please check the server or log files.";
          });
  }

  // Attach event listener to Refresh Logs button if it exists
  if (refreshLogsBtn) {
      refreshLogsBtn.addEventListener("click", fetchLogs);
  } else {
      console.warn("Refresh Logs button with id 'refreshLogsButton' not found.");
  }

  /**
   * Initializes Server-Sent Events (SSE) to receive real-time metrics.
   */
  function initializeMetricsSSE() {
      const source = new EventSource("/metrics_stream"); 
      source.onmessage = (event) => {
          try {
              const data = JSON.parse(event.data);

              // Retrieve metric elements
              const tradeCountEl = document.getElementById("tradeCount");
              const profitLossEl = document.getElementById("profitLoss");
              const accountBalanceEl = document.getElementById("accountBalance");
              const timeElapsedEl = document.getElementById("timeElapsed");

              // Update trade count
              if (tradeCountEl) {
                  tradeCountEl.innerText = data.tradeCount !== undefined ? data.tradeCount : 0;
              } else {
                  console.warn("Element with id 'tradeCount' not found.");
              }

              // Update profit/loss
              if (profitLossEl) {
                  const profitLoss = parseFloat(data.profitLoss);
                  profitLossEl.innerText = !isNaN(profitLoss) ? `$${profitLoss.toFixed(2)}` : "$0.00";
              } else {
                  console.warn("Element with id 'profitLoss' not found.");
              }

              // Update account balance
              if (accountBalanceEl) {
                  const balance = parseFloat(data.accountBalance);
                  accountBalanceEl.innerText = !isNaN(balance) ? `$${balance.toFixed(2)}` : "$0.00";
              } else {
                  console.warn("Element with id 'accountBalance' not found.");
              }

              // Update time elapsed
              if (timeElapsedEl) {
                  timeElapsedEl.innerText = data.timeElapsed || "00:00:00";
              } else {
                  console.warn("Element with id 'timeElapsed' not found.");
              }
          } catch (error) {
              console.error("SSE parse error:", error);
          }
      };

      source.onerror = (err) => {
          console.error("SSE connection error:", err);
          showFeedback("Lost connection to metrics. Reconnecting...", "warning");
          source.close();
          setTimeout(() => initializeMetricsSSE(), 5000);
      };
  }

  /**
   * Determine the current page and initialize relevant functionalities.
   */
  const currentPath = window.location.pathname;
  if (currentPath === "/overview") {
      initializeMetricsSSE();
      fetchFeatureEngineeringStatus();
      updateBotStatus();  // Initial status check
  } else if (currentPath === "/positions") {
      fetchOpenPositions();
  } else if (currentPath === "/history") {
      fetchTradeHistory();
  } else if (currentPath === "/settings") { // Ensure both routes are covered
        loadSettings();
  } else if (currentPath === "/logs") {
      fetchLogs(); // Corrected: Added parentheses to invoke the function
  } else {
      console.warn("Unknown page path. No specific initialization performed.");
  }
});
