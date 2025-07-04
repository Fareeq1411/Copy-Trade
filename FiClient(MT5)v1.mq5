//+------------------------------------------------------------------+
//|                    FiBridge Client MT5                           |
//|   Syncs Trades from Master EA via Flask API in Real Time          |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, FiBridge"
#property link      "https://www.fiscacrm.com"
#property version   "1.00"
#property strict

// ** Global variables **
#define FlaskURL "https://server.fiscacrm.com"  // Replace with your actual server URL
string g_BotID = "";             // Stores the assigned bot ID
double g_ClientLotSize = 0.0;    // Client-specific lot size from server

//+------------------------------------------------------------------+
//| GetAccountID: returns the current account login value            |
//+------------------------------------------------------------------+
long GetAccountID()
{
   return AccountInfoInteger(ACCOUNT_LOGIN);
}

//+------------------------------------------------------------------+
//| SendToFlask: handles HTTP communication with the server          |
//+------------------------------------------------------------------+
bool SendToFlask(string endpoint, string jsonData, string &response)
{
   uchar data[], result[];
   string headers = "Content-Type: application/json\r\n";
   int timeout = 5000;
   string resultHeaders;
   
   // Convert JSON string to uchar array using UTF-8 encoding
   StringToCharArray(jsonData, data, 0, StringLen(jsonData), CP_UTF8);
   
   // Send WebRequest with proper parameters
   int responseSize = WebRequest(
      "POST",
      FlaskURL + endpoint,
      headers,
      timeout,
      data,
      result,
      resultHeaders
   );
   
   if(responseSize > 0)
   {
      response = CharArrayToString(result, 0, ArraySize(result), CP_UTF8);
      return true;
   }
   
   response = "";
   Print("❌ WebRequest Error:", GetLastError(), " at ", endpoint);
   return false;
}

//+------------------------------------------------------------------+
//| ExtractJSONValue: parses a JSON string for a given key           |
//+------------------------------------------------------------------+
string ExtractJSONValue(string json, string key)
{
   // Trim whitespace from both ends (in-place)
   StringTrimLeft(json);
   StringTrimRight(json);
   
   string searchKey = "\"" + key + "\":";
   int keyPos = StringFind(json, searchKey);
   if(keyPos == -1)
      return "";
   
   keyPos += StringLen(searchKey);
   while(StringGetCharacter(json, keyPos) == ' ' || StringGetCharacter(json, keyPos) == ':')
      keyPos++;
   
   int valueStart = keyPos;
   int valueEnd = -1;
   
   if(StringGetCharacter(json, valueStart) == '"')
   {
      valueStart++;
      valueEnd = StringFind(json, "\"", valueStart);
   }
   else
   {
      valueEnd = StringFind(json, ",", valueStart);
      if(valueEnd == -1)
         valueEnd = StringFind(json, "}", valueStart);
   }
   
   if(valueEnd == -1)
      return "";
   
   return StringSubstr(json, valueStart, valueEnd - valueStart);
}

//+------------------------------------------------------------------+
//| ExecuteTrade: executes a trade using MQL5 trade structures         |
//+------------------------------------------------------------------+
int ExecuteTrade(string symbol, double lotSize, double price, string type, int magicNumber)
{
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action    = TRADE_ACTION_DEAL;
   request.symbol    = symbol;
   request.volume    = NormalizeDouble(lotSize, 2);
   request.type      = (type == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   request.price     = NormalizeDouble(price, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS));
   request.deviation = 10;
   request.magic     = magicNumber;
   
   if(!OrderSend(request, result))
   {
      Print("❌ Trade Failed | Error:", GetLastError());
      return -1;
   }
   
   Print("✅ Trade Executed | Symbol:", symbol, " | Type:", type, " | Lot:", lotSize);
   return (int)result.order;
}

//+------------------------------------------------------------------+
//| CheckPermission: checks access rights with the server            |
//+------------------------------------------------------------------+
bool CheckPermission()
{
   long account = GetAccountID();
   string jsonData = StringFormat("{\"mt5id\":%I64d}", account);
   Print("Sending JSON for permission check: ", jsonData);
   
   string response;
   if(!SendToFlask("/check-permission", jsonData, response))
   {
      Print("❌ Error calling /check-permission");
      return false;
   }
   
   Print("Raw permission response: ", response);
   if(StringFind(response, "Not Allowed") >= 0)
   {
      Print("❌ Access Denied for ", account);
      return false;
   }
   
   string statusVal = ExtractJSONValue(response, "status");
   if(statusVal == "Allowed")
   {
      g_BotID = ExtractJSONValue(response, "bot_id");
      Print("✅ Permission Granted. Bot ID: ", g_BotID);
      return true;
   }
   
   Print("❓ Unexpected permission response: ", response);
   return false;
}

//+------------------------------------------------------------------+
//| FetchAndUpdateLotSize: retrieves the lot size from the server      |
//+------------------------------------------------------------------+
void FetchAndUpdateLotSize()
{
   long account = GetAccountID();
   string jsonData = StringFormat("{\"mt5_id\":%I64d}", account);
   string response;
   
   if(!SendToFlask("/get-lot-size", jsonData, response))
   {
      Print("❌ Failed to fetch lot size from Flask.");
      return;
   }
   
   string lotSizeStr = ExtractJSONValue(response, "lot_size");
   if(lotSizeStr != "")
   {
      double newLotSize = StringToDouble(lotSizeStr);
      if(newLotSize > 0 && newLotSize != g_ClientLotSize)
      {
         g_ClientLotSize = newLotSize;
         Print("📊 Lot size updated: ", g_ClientLotSize);
      }
   }
   else
   {
      Print("⚠️ No lot size received, using previous: ", g_ClientLotSize);
   }
}

//+------------------------------------------------------------------+
//| SendTodayTradeHistory: sends today's trade history to the server   |
//+------------------------------------------------------------------+
void SendTodayTradeHistory()
{
   long account = GetAccountID();
   Print("📤 Sending account history trades for account: ", account);
   datetime startTime = iTime(_Symbol, PERIOD_D1, 0);
   datetime endTime   = TimeCurrent();
   HistorySelect(startTime, endTime);
   
   // Build JSON payload including required fields with placeholders (0.0 if not available)
   string jsonData = "{\"mt5id\": " + StringFormat("%I64d", account) + ", \"trades\": [";
   int count = 0;
   
   int totalDeals = HistoryDealsTotal();
   for(int i = 0; i < totalDeals; i++)
   {
      ulong dealTicket = HistoryDealGetTicket(i);
      if(HistoryDealGetInteger(dealTicket, DEAL_ENTRY) == DEAL_ENTRY_OUT)
      {
         if(count > 0)
            jsonData += ",";
         
         string symbol = HistoryDealGetString(dealTicket, DEAL_SYMBOL);
         double volume = HistoryDealGetDouble(dealTicket, DEAL_VOLUME);
         double price  = HistoryDealGetDouble(dealTicket, DEAL_PRICE);
         double profit = HistoryDealGetDouble(dealTicket, DEAL_PROFIT);
         datetime time = (datetime)HistoryDealGetInteger(dealTicket, DEAL_TIME);
         
         // Include required fields: open_price, stop_loss, take_profit, swap (all 0.0 as placeholders)
         jsonData += StringFormat(
            "{\"order_ticket\": %d, \"open_price\": %.5f, \"stop_loss\": %.5f, \"take_profit\": %.5f, \"swap\": %.5f, \"close_time\": \"%s\", \"type\": \"%s\", \"lot_size\": %.2f, \"symbol\": \"%s\", \"close_price\": %.5f, \"profit\": %.2f}",
            (int)dealTicket, 0.0, 0.0, 0.0, 0.0, TimeToString(time),
            (HistoryDealGetInteger(dealTicket, DEAL_TYPE) == DEAL_TYPE_BUY ? "buy" : "sell"),
            volume, symbol, price, profit
         );
         count++;
      }
   }
   jsonData += "]}";
   
   if(count > 0)
   {
      string response;
      if(SendToFlask("/log_today_trade_history", jsonData, response))
         Print("✅ Sent ", count, " orders to Flask. Server says: ", response);
      else
         Print("❌ Failed to send account history to Flask.");
   }
   else
      Print("✅ No new trades to send from history.");
}

//+------------------------------------------------------------------+
//| CheckAndExecuteTrades: processes trade signals from the server     |
//+------------------------------------------------------------------+
void CheckAndExecuteTrades()
{
   long account = GetAccountID();
   string jsonData = StringFormat("{\"client_id\":%I64d, \"bot_id\":\"%s\"}", account, g_BotID);
   string response;
   
   if(!SendToFlask("/get_trade_signal", jsonData, response) || StringLen(response) < 5)
      return;
   
   string symbol = ExtractJSONValue(response, "symbol");
   double price = StringToDouble(ExtractJSONValue(response, "entry_price"));
   double signalLotSize = StringToDouble(ExtractJSONValue(response, "lot"));
   string orderType = ExtractJSONValue(response, "type");
   int magic = (int)StringToInteger(ExtractJSONValue(response, "magic_number"));
   
   // If the signal price differs from the current market price, override it.
   if(symbol != "")
   {
      double currentPrice = (orderType == "BUY") ? SymbolInfoDouble(symbol, SYMBOL_ASK) : SymbolInfoDouble(symbol, SYMBOL_BID);
      if(MathAbs(price - currentPrice) > 0.00001)
      {
         Print("Overriding signal price (", price, ") with current market price (", currentPrice, ") for ", symbol);
         price = currentPrice;
      }
   }
   
   // Determine the lot size: if g_ClientLotSize equals 999, use the signal's lot size.
   double lotSizeToUse = (g_ClientLotSize == 999) ? signalLotSize : g_ClientLotSize;
   
   if(symbol != "" && price > 0 && orderType != "" && magic > 0 && lotSizeToUse > 0)
   {
      int ticket = ExecuteTrade(symbol, lotSizeToUse, price, orderType, magic);
      if(ticket > 0)
      {
         // Log trade execution including extra fields as placeholders.
         string jsonExec = StringFormat(
            "{\"client_id\":%I64d, \"bot_id\":\"%s\", \"trade_ticket\":%d, \"magic_number\":%d, \"open_price\":%.5f, \"stop_loss\":%.5f, \"take_profit\":%.5f, \"swap\":%.5f}",
            account, g_BotID, ticket, magic, 0.0, 0.0, 0.0, 0.0
         );
         string logResponse;
         SendToFlask("/log_client_execution", jsonExec, logResponse);
      }
   }
}

//+------------------------------------------------------------------+
//| CheckAndCloseTrades: closes positions based on server signals      |
//+------------------------------------------------------------------+
void CheckAndCloseTrades()
{
   long account = GetAccountID();
   string jsonData = StringFormat("{\"client_id\":%I64d, \"bot_id\":\"%s\"}", account, g_BotID);
   string response;
   
   if(!SendToFlask("/check_closed_trades_client", jsonData, response) || StringLen(response) < 5)
      return;
   
   int magic = (int)StringToInteger(ExtractJSONValue(response, "magic_number"));
   if(magic <= 0)
      return;
   
   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket) && PositionGetInteger(POSITION_MAGIC) == magic)
      {
         MqlTradeRequest request = {};
         MqlTradeResult result = {};
         request.action = TRADE_ACTION_DEAL;
         request.position = ticket;
         request.symbol = PositionGetString(POSITION_SYMBOL);
         request.volume = PositionGetDouble(POSITION_VOLUME);
         request.type = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
         request.price = SymbolInfoDouble(request.symbol, (request.type == ORDER_TYPE_BUY) ? SYMBOL_ASK : SYMBOL_BID);
         
         if(OrderSend(request, result))
         {
            Print("✅ Closed position | Magic:", magic);
            // Log closed trade including extra fields as placeholders.
            string jsonClose = StringFormat(
               "{\"client_id\":%I64d, \"bot_id\":\"%s\", \"magic_number\":%d, \"open_price\":%.5f, \"stop_loss\":%.5f, \"take_profit\":%.5f, \"swap\":%.5f}",
               account, g_BotID, magic, 0.0, 0.0, 0.0, 0.0
            );
            string closeResp;
            SendToFlask("/log_closed_trade", jsonClose, closeResp);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Timer function for periodic updates (fires every second)         |
//+------------------------------------------------------------------+
void OnTimer()
{
   if(CheckPermission())
   {
      FetchAndUpdateLotSize();
      CheckAndExecuteTrades();
      CheckAndCloseTrades();
      SendTodayTradeHistory();
   }
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("🚀 EA Initialized | Account: ", GetAccountID());
   // Set the timer to fire every 1 second
   EventSetTimer(1);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   long account = GetAccountID();
   if(g_BotID != "")
   {
      string jsonData = StringFormat("{\"mt5id\":%I64d, \"bot_status\":\"Disconnected\"}", account);
      string response;
      SendToFlask("/update_bot_status", jsonData, response);
   }
   EventKillTimer();
   Print("🛑 EA Deinitialized");
}
//+------------------------------------------------------------------+
