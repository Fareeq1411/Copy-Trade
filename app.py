

from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import joblib
import mysql.connector
from mysql.connector import Error
import json
import yagmail
import time
import requests
#import talib as ta
#import tensorflow as tf
#from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import random
from flask import Response
from bs4 import BeautifulSoup
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
from selenium.webdriver.common.by import By
import re
import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
import threading
import pytz



app = Flask(__name__)

# Database and bot configuration
hostloc = 'sbsmanagerdb.c1gcwi06wgyq.ap-southeast-5.rds.amazonaws.com'
username = 'admin'
password_user = 'sbstraderhq123'
database_name = 'sbsdb'

bot_token = '7960870475:AAEyRnwx4ObsSXOcwrnXcO6GmXmVDUmWrMQ'
chat_id = '-1002270494735'

email_sender = 'fareeq1411@gmail.com'
email_password = 'skrndeebrjtbmner'
yag = yagmail.SMTP(email_sender, email_password)

@app.route('/')
def home():
    return 'Connected', 200  # 200 ensures the message is displayed


# @app.route('/predict', methods=['POST'])
# def predict2():
#     raw_data = request.data
#     cleaned_data = raw_data.rstrip(b'\x00')

#     try:
#         data = json.loads(cleaned_data)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

#     data_candles = data.get("candles", [])
    
#     # Reverse the order of the candles to ensure the most recent complete candle is last
#     data_candles.reverse()

#     # No exclusion of the last candle, instead process all the candles
#     test_data = []
#     for item in data_candles:
#         test_data.append([item["open"], item["high"], item["low"], item["close"], item["volume"]])

#     if any(pd.isna(value) for sublist in test_data for value in sublist):
#         return jsonify({"error": "Input data contains NaN values"}), 400

#     df = pd.DataFrame(test_data, columns=["open", "high", "low", "close", "volume"])

#     # Calculate technical indicators
#     df['RSI_6'] = ta.RSI(df['close'], timeperiod=6)
#     macd, macd_signal, macd_hist = ta.MACD(df['close'], fastperiod=6, slowperiod=13, signalperiod=9)
#     df['MACD'] = macd
#     df['MACD_Signal'] = macd_signal
#     df['MACD_Hist'] = macd_hist
#     upper_band, middle_band, lower_band = ta.BBANDS(df['close'], timeperiod=10, nbdevup=2, nbdevdn=2)
#     df['Upper_BB'] = upper_band
#     df['Middle_BB'] = middle_band
#     df['Lower_BB'] = lower_band
#     df['VWMA_9'] = (df['close'] * df['volume']).rolling(window=9).sum() / df['volume'].rolling(window=9).sum()

#     df.dropna(inplace=True)

#     if len(df) < 5:
#         return jsonify({"error": "Insufficient data for prediction"}), 400

#     feature_columns = ['open', 'high', 'low', 'close', 'volume', 'RSI_6', 'MACD', 'MACD_Signal',
#                        'MACD_Hist', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'VWMA_9']
#     df['label'] = np.where(df['close'] - df['open'] > 0, 1, 0)
#     n_candles = 5

#     def create_sequences(data, n_candles):
#         X, y = [], []
#         for i in range(len(data) - n_candles):
#             X.append(data.iloc[i:i + n_candles][feature_columns].values)
#             y.append(data.iloc[i + n_candles]['label'])
#         return np.array(X), np.array(y)

#     scaler = MinMaxScaler()
#     df[feature_columns] = scaler.fit_transform(df[feature_columns])

#     X_test_seq, y_test_seq = create_sequences(df, n_candles)

#     model = tf.keras.models.load_model('rl_candle_model_swing.h5')
#     predictions = model.predict(X_test_seq)
#     predicted_labels = [1 if pred > 0.5 else 0 for pred in predictions]

#     action_name = 'Buy' if predicted_labels[-1] == 1 else 'Sell'
#     confidence = predictions[-1][0] if predicted_labels[-1] == 1 else 1 - predictions[-1][0]
#     confidence_percentage = round(confidence * 100, 2)

#     # Retrieve the last completed candle (most recent one after reversing)
#     last_candle = data_candles[-1]  # The last completed candle after reversing

#     print(last_candle)
#     print(data_candles)
#     TheDate = last_candle["date"]
#     TheTime = last_candle["time"]

#     # If TheTime isn't in the format HH:MM, convert it (only needed if the format is different)
#     if len(TheTime) == 5:  # If time is in HH:MM (without seconds)
#         TheTime = TheTime + ":00"  # Append ":00" if seconds are missing

#     Price = last_candle["close"]

#     # Step 1: Fetch the previous signal
#     conn = check_connection()
#     try:
#         cursor = conn.cursor(dictionary=True)
#         cursor.execute("SELECT id, action, date, time FROM signal_trade2 ORDER BY id DESC LIMIT 1")
#         previous_signal = cursor.fetchone()

#         # Step 2: Derive action from the previous candle (use -2 for previous candle)
#         if previous_signal:
#             prev_id = previous_signal['id']
#             prev_action = previous_signal['action']
#             prev_date = previous_signal['date']
#             prev_time = previous_signal['time']

#             # Derive previous action
#             previous_close = last_candle['close']
#             previous_open = last_candle['open']
#             derived_action = 'Buy' if (previous_close - previous_open) > 0 else 'Sell'

#             # Calculate the accuracy
#             is_correct = 1 if derived_action == prev_action else 0
#             result = f"**CORRECT ✅**" if is_correct else f"**WRONG ❌**"

#             # Update accuracy column in the database
#             update_query = "UPDATE signal_trade2 SET accuracy = %s WHERE id = %s"
#             cursor.execute(update_query, (is_correct, prev_id))
#             conn.commit()

#             # Send result to Telegram
#             message = (f"Previous Signal:\nDate: {prev_date} Time: {prev_time}\n"
#                        f"Action: {prev_action}\nDerived Action: {derived_action}\nResult: {result}")
#             tele_bot(message)
#             print("Previous signal and accuracy updated in database.")

#         # Step 3: Insert new signal into the database
#         # Ensure confidence is converted to float
#         confidence_percentage = float(confidence_percentage)  # Convert to native Python float
#         insert_query = "INSERT INTO signal_trade2 (date, time, action, confidence) VALUES (%s, %s, %s, %s)"
#         cursor.execute(insert_query, (TheDate, TheTime, action_name, confidence_percentage))
#         conn.commit()

#         # Send new signal with confidence to Telegram
#         current_message = (f"New Signal:\n"
#                            f"Date: {TheDate} Time: {TheTime}\n"
#                            f"Action: {action_name}\n"
#                            f"Price: {Price}\n"
#                            f"Confidence: {confidence_percentage}%")

#         tele_bot(current_message)
#         print("New signal and confidence sent to Telegram.")

#         # Send separator
#         tele_bot("...")
#         print("Separator sent to Telegram.")

#     except Error as e:
#         print(f"ERROR: {e}")
#     finally:
#         cursor.close()
#         conn.close()

#     return '', 204


def check_connection():
    try:
        conn = mysql.connector.connect(
            host=hostloc, user=username, password=password_user, database=database_name
        )
        if conn.is_connected():
            return conn
    except Error as e:
        print(f"ERROR: {e}")
    return None

@app.route('/signal', methods=['GET'])
def get_signal():
    # Open a connection to the database
    conn = check_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        cursor = conn.cursor(dictionary=True)

        # Fetch the last 3 non-null accuracy and decision values for rolling accuracy calculation
        cursor.execute('''
            SELECT accuracy, decision
            FROM signal_trade2
            WHERE accuracy IS NOT NULL
            ORDER BY id DESC
            LIMIT 3
        ''')
        accuracy_rows = cursor.fetchall()

        print(accuracy_rows)

        # If there are fewer than 3 accuracy values, return a default "hold" signal
        if len(accuracy_rows) < 3:
            return jsonify({"signal": "hold"}), 200
        
        accuracy_string = ''.join(str(row['accuracy']) for row in accuracy_rows)
    
        print("Accuracy string:", accuracy_string)

        # Calculate rolling accuracy (last 3 trades' accuracy)
        accuracy_values = [row['accuracy'] for row in accuracy_rows]
    
        rolling_accuracy = sum(accuracy_values[:3]) / 3  # Last 3 trades' accuracy
        print(rolling_accuracy)

        # Concatenate the decision values to create a string
        decision_string = ''.join([row['decision'] if row['decision'] else "None" for row in accuracy_rows])

        print("Decision string:", decision_string)

        # Check if the last three accuracy values are "111", or if the rolling accuracy is less than 0.60, or if the decision string contains "NoneNoneHold"
        if accuracy_string == "111" or 'nonohold' in decision_string or rolling_accuracy < 0.60:
            # Force hold if the condition is met, and ensure "001" doesn't trigger hold
            if accuracy_string != "100":  # Do not hold if the accuracy is "001"
                cursor.execute('''
                    UPDATE signal_trade2 
                    SET decision = 'hold'
                    WHERE decision IS NULL
                    ORDER BY id DESC
                    LIMIT 1
                ''')
                conn.commit()
                print("HOLD")

                print("HOLD")

                # Get the current action from the database (i.e., the most recent action before the hold decision)
                cursor.execute('''
                    SELECT action
                    FROM signal_trade2
                    WHERE accuracy IS NULL
                    ORDER BY id DESC
                    LIMIT 1
                ''')
                action_row = cursor.fetchone()
                
                # Reverse the action
                action = action_row['action'] if action_row else "hold"
                reversed_action = "Sell" if action == "Buy" else "Buy"  # Reverse Buy/Sell actions

                print("reversed")
                
                # Return the reversed action to MT5 in the response
                return jsonify({"signal": reversed_action}), 200  # Send reversed action to MT5

        # Fetch the action from the most recent signal (accuracy is NULL)
        cursor.execute('''
            SELECT action
            FROM signal_trade2
            WHERE accuracy IS NULL
            ORDER BY id DESC
            LIMIT 1
        ''')
        action_row = cursor.fetchone()

        # Return the action (Buy, Sell, etc.) based on the signal action
        action = action_row['action'] if action_row else "hold"

        print(action)

        # Update the "decision" for the current signal if it's not "hold"
        cursor.execute('''
            UPDATE signal_trade2
            SET decision = 'no'
            WHERE action = %s AND decision IS NULL
            ORDER BY id DESC
            LIMIT 1
        ''', (action,))
        conn.commit()

        return jsonify({"signal": action}), 200

    except Error as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()




# ---------------------------------------------------------
#  Helper: generate_unique_magic_num()
#  Return an 8-digit random that doesn't exist in trade_logs
# ---------------------------------------------------------
def generate_unique_magic_num():
    while True:
        magic_num = random.randint(10000000, 99999999)  # Generate 8-digit number
        conn = check_connection()
        if not conn:
            return None  # Return None if DB fails

        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM trade_logs WHERE magic_number = %s LIMIT 1", (magic_num,))
        exists = cursor.fetchone() is not None
        conn.close()

        if not exists:
            return magic_num  # Return only if unique


# ---------------------------------------------------------
#  MASTER EA ROUTE: /check_trade
#  Purpose : Checks if a given trade_ticket is already in DB
#            for the specified bot_id
# ---------------------------------------------------------
@app.route('/check_trade', methods=['POST'])
def check_trade():
    data = request.get_json()
    trade_ticket = data.get('ticket')
    bot_id = data.get('bot_id')

    if not bot_id:
        return jsonify({"error": "Missing bot_id"}), 400

    conn = check_connection()
    if not conn:
        return jsonify({"error": "DB Connection Failed"}), 500

    cursor = conn.cursor()
    try:
        if not trade_ticket:
            # Example: If no trade_ticket provided, just respond
            return jsonify({"status": "no_ticket"}), 400

        # Check if the trade ticket exists for this bot_id
        sql = """SELECT 1 FROM trade_logs 
                 WHERE trade_ticket = %s 
                   AND bot_id = %s 
                 LIMIT 1"""
        cursor.execute(sql, (trade_ticket, bot_id))
        trade_exists = cursor.fetchone() is not None

        return jsonify({"exists": trade_exists})

    except mysql.connector.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# ---------------------------------------------------------
#  MASTER EA ROUTE: /add_position
#  Purpose : Insert a new trade into trade_logs with 
#            a unique magic_number, assigned "New" status.
# ---------------------------------------------------------
@app.route('/add_position', methods=['POST'])
def add_position():
    data = request.get_json()
    print("📩 Received JSON:", data)

    # We expect these keys: "bot_id", "action", "action_type", "symbol", ...
    required_keys = ["bot_id","action","action_type","symbol","lot","price","type","ticket"]
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        return jsonify({"error": "Missing required keys", "missing": missing_keys}), 400

    bot_id = data["bot_id"]
    trade_ticket = data["ticket"]

    conn = check_connection()
    if not conn:
        return jsonify({"error": "DB Connection Failed"}), 500

    cursor = conn.cursor()
    try:
        # Check if trade_ticket already exists for this bot_id
        cursor.execute("""
            SELECT 1 
            FROM trade_logs 
            WHERE bot_id = %s
              AND trade_ticket = %s
            LIMIT 1
        """, (bot_id, trade_ticket))

        if cursor.fetchone():
            # If found => no need to re-insert
            return jsonify({"status": "exists"}), 200

        # Generate a unique magic_number
        magic_num = generate_unique_magic_num()
        if magic_num is None:
            return jsonify({"error": "Failed to generate magic number"}), 500

        # Insert new row
        sql = """
            INSERT INTO trade_logs (
                bot_id,
                trade_action,
                trade_type,
                asset,
                volume,
                entry_price,
                trade_direction,
                trade_ticket,
                magic_number,
                trade_status
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'New')
        """
        values = (
            bot_id,
            data['action'],
            data['action_type'],
            data['symbol'],
            data['lot'],
            data['price'],
            data['type'],
            trade_ticket,
            magic_num
        )
        cursor.execute(sql, values)
        conn.commit()

        return jsonify({"status": "success", "magic_number": magic_num}), 200

    except mysql.connector.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# ---------------------------------------------------------
#  MASTER EA ROUTE: /check_closed_trades
#  Purpose : The Master EA sends a list of active trade tickets
#            for a given bot_id. Anything in DB but not in that
#            list => we mark as 'Closed' in the DB
# ---------------------------------------------------------
@app.route('/check_closed_trades', methods=['POST'])
def check_closed_trades():
    data = request.get_json()
    bot_id = data.get("bot_id")
    mt4_active_tickets = set(data.get("tickets", []))

    if not bot_id:
        return jsonify({"error": "Missing bot_id"}), 400

    conn = check_connection()
    if not conn:
        return jsonify({"error": "DB Connection Failed"}), 500

    cursor = conn.cursor()

    try:
        # 1) Get all active trades for this bot_id from DB
        sql = """
            SELECT trade_ticket 
            FROM trade_logs 
            WHERE trade_status != 'Closed'
              AND bot_id = %s
        """
        cursor.execute(sql, (bot_id,))
        db_active_tickets = {row[0] for row in cursor.fetchall()}

        # 2) closed_tickets = those in DB but not in MT4 list
        closed_tickets = db_active_tickets - mt4_active_tickets
        if closed_tickets:
            # Mark them as 'Closed'
            format_strings = ','.join(['%s'] * len(closed_tickets))
            update_sql = f"""
                UPDATE trade_logs
                SET trade_status = 'Closed'
                WHERE trade_ticket IN ({format_strings})
                  AND bot_id = %s
            """
            cursor.execute(update_sql, (*closed_tickets, bot_id))
            conn.commit()

            return jsonify({"status": "updated", "closed_tickets": list(closed_tickets)})

        return jsonify({"status": "no_changes", "message": "No trades to close"}), 204

    except mysql.connector.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

      # ---------------------------------------------------------
#  CLIENT EA ROUTE: /get_trade_signal
#  Purpose : Returns the next "New" trade from trade_logs 
#            for the specified bot_id that the client has 
#            not yet executed.
# ---------------------------------------------------------
@app.route('/get_trade_signal', methods=['POST'])
def get_trade_signal():
    data = request.get_json()
    client_id = data.get('client_id')
    bot_id = data.get('bot_id')

    if not client_id or not bot_id:
        return jsonify({"error": "Missing client_id or bot_id"}), 400

    conn = check_connection()
    if not conn:
        return jsonify({"error": "DB Connection Failed"}), 500

    cursor = conn.cursor(dictionary=True)
    try:
        # 1. Find all "New" trades for this bot_id
        cursor.execute("""
            SELECT magic_number
            FROM trade_logs
            WHERE trade_status = 'New'
              AND bot_id = %s
        """, (bot_id,))
        new_magic_numbers = {row["magic_number"] for row in cursor.fetchall()}

        # 2. Find all trades already executed by this client for this bot_id
        cursor.execute("""
            SELECT magic_number
            FROM client_trade_logs
            WHERE client_id = %s
              AND bot_id = %s
        """, (client_id, bot_id))
        executed_magic_numbers = {row["magic_number"] for row in cursor.fetchall()}

        # 3. The difference => trades not yet executed
        available_magic_numbers = new_magic_numbers - executed_magic_numbers
        if not available_magic_numbers:
            return jsonify({}), 204  # no content

        # 4. Return details for the first unexecuted trade
        magic_number = list(available_magic_numbers)[0]
        cursor.execute("""
            SELECT asset AS symbol,
                   volume AS lot,
                   entry_price,
                   trade_direction AS type,
                   magic_number
            FROM trade_logs
            WHERE magic_number = %s
              AND bot_id = %s
        """, (magic_number, bot_id))
        trade = cursor.fetchone()

        return jsonify(trade or {})

    except mysql.connector.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()


# ---------------------------------------------------------
#  CLIENT EA ROUTE: /log_client_execution
#  Purpose : When the client executes a new trade, it logs 
#            the execution to client_trade_logs for that bot_id
# ---------------------------------------------------------
@app.route('/log_client_execution', methods=['POST'])
def log_client_execution():
    data = request.json
    client_id = data.get('client_id')
    bot_id = data.get('bot_id')
    magic_number = data.get('magic_number')
    trade_ticket = data.get('trade_ticket')
    execution_status = data.get('execution_status', 'new')  # default "new"

    if not client_id or not bot_id or not magic_number:
        return jsonify({"error": "Missing required parameters"}), 400

    conn = check_connection()
    if not conn:
        return jsonify({"error": "DB Connection Failed"}), 500

    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO client_trade_logs (
                client_id,
                bot_id,
                trade_ticket,
                executed_at,
                trade_type,
                magic_number,
                execution_status
            )
            VALUES (%s, %s, %s, NOW(), 'TRADE', %s, %s)
        """, (client_id, bot_id, trade_ticket, magic_number, execution_status))
        conn.commit()

        return jsonify({"message": "Trade logged successfully."}), 200
    except mysql.connector.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()


# ---------------------------------------------------------
#  CLIENT EA ROUTE: /check_closed_trades_client
#  Purpose : Tells the client if any of its "new" trades 
#            have been closed in `trade_logs` for this bot_id
# ---------------------------------------------------------
@app.route('/check_closed_trades_client', methods=['POST'])
def check_closed_trades_client():
    data = request.json
    client_id = data.get('client_id')
    bot_id = data.get('bot_id')

    if not client_id or not bot_id:
        return jsonify({}), 400

    conn = check_connection()
    if not conn:
        return jsonify({"error": "DB Connection Failed"}), 500

    cursor = conn.cursor(dictionary=True)
    try:
        # 1. Get "new" trades from client_trade_logs
        cursor.execute("""
            SELECT magic_number 
            FROM client_trade_logs
            WHERE client_id = %s
              AND bot_id = %s
              AND execution_status = 'new'
        """, (client_id, bot_id))
        client_trades = {row["magic_number"] for row in cursor.fetchall()}

        # 2. Check which of those are now 'Closed' in trade_logs
        cursor.execute("""
            SELECT magic_number
            FROM trade_logs
            WHERE bot_id = %s
              AND trade_status = 'Closed'
        """, (bot_id,))
        closed_magic_numbers = {row["magic_number"] for row in cursor.fetchall()}

        # Intersection => trades that were open for this client, but now closed
        closed_magic = client_trades & closed_magic_numbers
        if closed_magic:
            # Return the first closed magic_number
            return jsonify({"magic_number": list(closed_magic)[0]})

        return jsonify({}), 204

    except mysql.connector.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()


# ---------------------------------------------------------
#  CLIENT EA ROUTE: /log_closed_trade
#  Purpose : When the client closes a trade locally, it 
#            updates that trade’s status in client_trade_logs
# ---------------------------------------------------------
@app.route('/log_closed_trade', methods=['POST'])
def log_closed_trade():
    data = request.json
    client_id = data.get('client_id')
    bot_id = data.get('bot_id')
    magic_number = data.get('magic_number')

    if not client_id or not bot_id or not magic_number:
        return jsonify({"error": "Missing required parameters"}), 400

    conn = check_connection()
    if not conn:
        return jsonify({"error": "DB Connection Failed"}), 500

    cursor = conn.cursor()
    try:
        # Mark the trade as closed
        cursor.execute("""
            UPDATE client_trade_logs
            SET execution_status = 'Closed'
            WHERE client_id = %s
              AND bot_id = %s
              AND magic_number = %s
              AND execution_status = 'new'
        """, (client_id, bot_id, magic_number))
        conn.commit()

        return jsonify({"message": "Trade marked as closed successfully."}), 200
    except mysql.connector.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()


# ---------------------------------------------------------
#  CLIENT EA ROUTE: /log_today_trade_history
#  Purpose : Client can push its local trade history 
#            (usually closed trades) to server for logging
# ---------------------------------------------------------
@app.route('/log_today_trade_history', methods=['POST'])
def log_today_trade_history():
    try:
        data = request.json
        mt5id = data.get("mt5id")
        trades = data.get("trades", [])

        print(f"Received MT5 ID: {mt5id}")
        print(f"Received Trades: {trades}")

        if not mt5id:
            return jsonify({"error": "Missing mt5id"}), 400

        if not trades:
            return jsonify({"status": "success", "message": "No trades to insert"}), 200

        conn = check_connection()
        if not conn:
            return jsonify({"error": "DB Connection Failed"}), 500

        cursor = conn.cursor()
        # Fetch existing tickets for this mt5id
        cursor.execute("SELECT order_ticket FROM trade_history WHERE mt5id = %s", (mt5id,))
        existing = cursor.fetchall()
        if existing:
            existing = {row[0] for row in existing}
        else:
            existing = set()

        new_trades = []
        for t in trades:
            # Skip if already in DB
            if t["order_ticket"] in existing:
                continue

            # Convert times to strings if needed
            open_time = str(t.get("open_time"))
            close_time = str(t.get("close_time", open_time))

            new_trades.append({
                "mt5id": mt5id,
                "order_ticket": t["order_ticket"],
                "open_time": open_time,
                "type": t["type"],
                "lot_size": t["lot_size"],
                "symbol": t["symbol"],
                "open_price": t["open_price"],
                "stop_loss": t["stop_loss"],
                "take_profit": t["take_profit"],
                "close_time": close_time,
                "close_price": t["close_price"],
                "swap": t["swap"],
                "profit": t["profit"],
            })

        if new_trades:
            query = """
                INSERT INTO trade_history (
                    mt5id, order_ticket, open_time, type, lot_size, symbol,
                    open_price, stop_loss, take_profit,
                    close_time, close_price, swap, profit
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = [
                (
                    t["mt5id"],
                    t["order_ticket"],
                    t["open_time"],
                    t["type"],
                    t["lot_size"],
                    t["symbol"],
                    t["open_price"],
                    t["stop_loss"],
                    t["take_profit"],
                    t["close_time"],
                    t["close_price"],
                    t["swap"],
                    t["profit"]
                )
                for t in new_trades
            ]
            cursor.executemany(query, values)
            conn.commit()
            message = f"{len(new_trades)} trades logged successfully"
        else:
            message = "All trades already exist or had invalid data"

        conn.close()
        return jsonify({"status": "success", "message": message}), 201

    except Exception as e:
        error_message = str(e)
        return jsonify({
            "status": "error",
            "message": "An error occurred while logging trade history",
            "error": error_message
        }), 500


# ---------------------------------------------------------
#  CLIENT EA ROUTE: /update_bot_status
#  Purpose : When the client EA is removed/deinitialized, it 
#            sets its bot_status to "Disconnected" in bot_auth
# ---------------------------------------------------------
@app.route('/update_bot_status', methods=['POST'])
def update_bot_status():
    data = request.json
    mt5id = data.get("mt5id")
    bot_status = data.get("bot_status")

    if not mt5id or not bot_status:
        return "Invalid request", 400

    conn = check_connection()
    if not conn:
        return jsonify({"error": "DB Connection Failed"}), 500

    cursor = conn.cursor()
    try:
        # set bot_status in bot_auth
        cursor.execute("UPDATE bot_auth SET bot_status = %s WHERE mt5id = %s", (bot_status, mt5id))
        conn.commit()
        return "Bot status updated successfully", 200
    except mysql.connector.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()


@app.route('/check-permission', methods=['POST'])
def check_permission():
    data = request.json
    mt5id = data.get("mt5id")

    # 1) Validate input
    if not mt5id:
        return jsonify({"error": "Missing mt5id"}), 400

    # 2) Connect to DB
    conn = check_connection()
    if not conn:
        return jsonify({"error": "DB Connection Failed"}), 500

    cursor = conn.cursor()
    try:
        # 3) Look up the record in bot_auth
        sql = """
            SELECT bot_id, permission, bot_status, switch
            FROM bot_auth
            WHERE mt5id = %s
        """
        cursor.execute(sql, (mt5id,))
        result = cursor.fetchone()
        if not result:
            # No record => Not allowed
            return "Not Allowed", 403

        bot_id, permission, bot_status, switch = result

        # 4) If permission="No" or switch="Off", mark as Disconnected and deny
        if permission == "No" or switch == "Off":
            cursor.execute("""
                UPDATE bot_auth
                SET bot_status = %s
                WHERE mt5id = %s
            """, ("Disconnected", mt5id))
            conn.commit()
            return "Not Allowed", 403

        # 5) Otherwise, mark as Connected and return "Allowed" + bot_id
        cursor.execute("""
            UPDATE bot_auth
            SET last_con = NOW(),
                bot_status = %s
            WHERE mt5id = %s
        """, ("Connected", mt5id))
        conn.commit()

        # Return JSON: e.g. {"status":"Allowed","bot_id":"XYZ"}
        return jsonify({"status": "Allowed", "bot_id": bot_id}), 200

    except mysql.connector.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()


@app.route('/get-lot-size', methods=['POST'])
def get_lot_size():
    data = request.json
    mt5_id = data.get("mt5_id")

    if not mt5_id:
        return jsonify({"error": "Missing mt5_id"}), 400

    connection = check_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("SELECT lot_size FROM client_accounts WHERE acc_id = %s", (mt5_id,))
        row = cursor.fetchone()
        
        if row:
            return jsonify({"lot_size": row[0]})
        else:
            return jsonify({"error": "Lot size not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        connection.close()

@app.route('/goldenline-auth', methods=['GET'])
def goldenline_auth():
    # Get the account id from the GET request query parameters.
    account_id = int(request.args.get('account'))
    if not account_id:
        return Response("NO", status=400)
    
    try:
        mt5id = int(account_id)
    except ValueError:
        return Response("NO", status=400)
    
    conn = check_connection()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT permission FROM gl_auth WHERE mt5id = %s"
            cursor.execute(sql, (mt5id,))
            result = cursor.fetchone()
            # permission = result["permission"]

            # if permission_value == "YES":
            #     return "OK"
            # else:
            #     return "NO"

            # Build the response. You can include both the permission_value and the full result.
            if result[0] == "YES":
                return "OK"
            else:
                return "NO"


    except Exception as e:
        print("Error:", e)

        return Response("hm", status=500)
    finally:
        conn.close()




@app.route("/news-getter")
def news_getter():
    try:
        conn = check_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name, time, date FROM news ORDER BY id DESC LIMIT 4")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # Create a one-liner response: name|time|date,name|time|date,...
        items = []
        for row in rows:
            items.append(f"{row[0]}|{row[1]}|{row[2]}")
        
        # Join with comma and return
        return ",".join(items), 200, {"Content-Type": "text/plain"}

    except Exception as e:
        print("News route error:", e)
        return "", 500, {"Content-Type": "text/plain"}


@app.route('/get_strength', methods=['GET'])
def get_strength():
    try:
        # Retrieve the symbol from the request query parameter
        symbol = request.args.get("symbol", None)
        if symbol is None:
            return "no", 400, {"Content-Type": "text/plain"}
        symbol = symbol.strip().upper()  # standardize the symbol format

        conn = check_connection()
        cursor = conn.cursor()

        # Fetch strength values for the given symbol and timeframe
        cursor.execute("SELECT strength FROM current_strength WHERE tf = %s AND symbol = %s", ("H1", symbol))
        row_h1 = cursor.fetchone()

        cursor.execute("SELECT strength FROM current_strength WHERE tf = %s AND symbol = %s", ("H4", symbol))
        row_h4 = cursor.fetchone()

        cursor.execute("SELECT strength FROM current_strength WHERE tf = %s AND symbol = %s", ("D1", symbol))
        row_d1 = cursor.fetchone()

        cursor.close()
        conn.close()

        # Validate existence; if any of the timeframes is missing a value, signal "no"
        if not row_h1 or not row_h4 or not row_d1:
            return "no", 200, {"Content-Type": "text/plain"}

        # Normalization helper: correct typo "strong_dow" to "strong_down"
        def normalize_strength(value):
            value = value.strip().lower()
            if value == "strong_dow":
                return "strong_down"
            return value

        strength_h1 = normalize_strength(row_h1[0])
        strength_h4 = normalize_strength(row_h4[0])
        strength_d1 = normalize_strength(row_d1[0])

        # Determine overall signal based on the strength logic:
        # - When D1 is strong_up but both H1 and H4 are strong_down then send "both".
        # - When D1 is strong_down but both H1 and H4 are strong_up then send "both".
        # - Otherwise, follow D1.
        if strength_d1 == "strong_up":
            if strength_h1 == "strong_down" and strength_h4 == "strong_down":
                return "both", 200, {"Content-Type": "text/plain"}
            else:
                return "up", 200, {"Content-Type": "text/plain"}
        elif strength_d1 == "strong_down":
            if strength_h1 == "strong_up" and strength_h4 == "strong_up":
                return "both", 200, {"Content-Type": "text/plain"}
            else:
                return "down", 200, {"Content-Type": "text/plain"}
        # If D1 doesn't signal strong direction, return "no"
        return "no", 200, {"Content-Type": "text/plain"}

    except Exception as e:
        print("Error in /get_strength:", e)
        return "error", 500, {"Content-Type": "text/plain"}



@app.route('/db_check', methods=['GET'])
def db_check():
    try:
        # Get the symbol from the query parameter; expect something like "xauusd"
        symbol = request.args.get("symbol", None)
        if symbol is None:
            return jsonify({"error": "Missing symbol parameter"}), 400
        symbol = symbol.strip().upper()  # standardize the symbol format

        conn = check_connection()
        cursor = conn.cursor()

        # Fetch the strength values for the provided symbol for timeframes H1, H4, and D1
        query = "SELECT tf, strength FROM current_strength WHERE symbol = %s AND tf IN ('H1', 'H4', 'D1')"
        cursor.execute(query, (symbol,))
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        # Build a JSON-friendly dictionary for the results.
        result_data = {}
        # If no rows were returned, inform the caller.
        if not rows:
            return jsonify({"error": "No records found for symbol: " + symbol}), 404

        for row in rows:
            tf, strength = row
            # Ensure the strength is a string and clean it if necessary
            result_data[tf] = strength.strip() if strength else None

        return jsonify(result_data), 200

    except Exception as e:
        print("Error in /db_check:", e)
        return jsonify({"error": "Internal error", "message": str(e)}), 500



@app.route('/send_strength', methods=['POST'])
def send_strength():
    try:
        data = request.get_json(force=True)
        if not data or "timeframe" not in data or "direction" not in data or "symbol" not in data:
            return jsonify({"error": "Invalid data. 'timeframe', 'direction', and 'symbol' are required."}), 400

        tf = data["timeframe"]
        new_strength = data["direction"].strip().lower()
        symbol = data["symbol"].strip().upper()  # Standardize the symbol to uppercase

        # Accept the full new set of strength directions
        valid_strengths = {
            "strong", "weak", "stalling", "turning_down", "unknown",
            "turning_up", "strong_up", "strong_down", "reversing_from_top", "reversing_from_bottom"
        }

        if new_strength not in valid_strengths:
            return jsonify({"error": "Invalid strength value"}), 400

        # Connect to the database
        conn = check_connection()
        cursor = conn.cursor()

        # Query based on both timeframe and symbol
        cursor.execute("SELECT strength FROM current_strength WHERE tf = %s AND symbol = %s", (tf, symbol))
        row = cursor.fetchone()

        if row:
            old_strength = row[0].strip().lower()
            if old_strength == new_strength:
                cursor.close()
                conn.close()
                return jsonify({"status": "no change"}), 200
            else:
                cursor.execute("UPDATE current_strength SET strength = %s WHERE tf = %s AND symbol = %s", 
                               (new_strength, tf, symbol))
                conn.commit()
                cursor.close()
                conn.close()
                return jsonify({"status": "updated"}), 200
        else:
            cursor.execute("INSERT INTO current_strength (tf, symbol, strength) VALUES (%s, %s, %s)", 
                           (tf, symbol, new_strength))
            conn.commit()
            cursor.close()
            conn.close()
            return jsonify({"status": "created"}), 201

    except Exception as e:
        print("Error in /send_strength:", e)
        return jsonify({"error": str(e)}), 500




@app.route("/time-to-news", methods=["GET"])
def time_to_news():
    try:
        conn = check_connection()
        cursor = conn.cursor()
        # Get the last row (most recent) from the news table
        cursor.execute("SELECT time, date FROM news ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not row:
            return "N/A", 200, {"Content-Type": "text/plain"}
        
        # row[0] is expected to be a time string like "00:50:00+02:00" or "00:50:00"
        # row[1] is the date string (which we ignore for this calculation)
        # We'll extract only the HH:MM:SS portion (ignoring any timezone part)
        # Split at the '+' if present:
        time_str_raw = row[0].split('+')[0].strip()  # e.g. "00:50:00"
        
        # Parse the time from the string
        event_time_obj = datetime.strptime(time_str_raw, "%H:%M:%S")
        # Convert to total minutes since midnight
        event_minutes = event_time_obj.hour * 60 + event_time_obj.minute
        
        # Get the current time in UTC+2
        tz = pytz.FixedOffset(120)
        now = datetime.now(tz)
        # Use only the time portion of now
        current_minutes = now.hour * 60 + now.minute

        # Calculate difference in minutes:
        diff = event_minutes - current_minutes
        # If negative, assume the event is on the next day.
        if diff < 0:
            diff += 24 * 60
        
        return str(diff), 200, {"Content-Type": "text/plain"}
    except Exception as e:
        print("Time-to-news route error:", e)
        return "Error", 500, {"Content-Type": "text/plain"}


@app.route('/acc_balance', methods=['POST'])
def acc_balance():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400

    mt5id = data.get("mt5id")
    balance = data.get("balance")
    total_deposits = data.get("total_deposits")
    total_withdrawals = data.get("total_withdrawals")
    
    try:
        conn = check_connection()
        cursor = conn.cursor()

        # --- Account Balance ---
        # Check if a row with the same client_id, balance, and today's date already exists.
        cursor.execute(
            "SELECT COUNT(*) FROM account_balance WHERE client_id = %s AND balance = %s AND DATE(last_updated) = CURDATE()",
            (mt5id, balance)
        )
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                "INSERT INTO account_balance (client_id, balance, last_updated) VALUES (%s, %s, NOW())",
                (mt5id, balance)
            )

        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"status": "success"}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Function to Send Telegram Alerts
def tele_bot(message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url)

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5001)  # Flask running on port 5000

