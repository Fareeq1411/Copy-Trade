//+------------------------------------------------------------------+
//|                        FiBridge MT4 (Master)                    |
//+------------------------------------------------------------------+
#property strict

//--- Input field for Bot ID
input string BotID = "DefaultBotID"; 

//--- Flask Server URL
#define FlaskURL "https://server.fiscacrm.com"

//--- Store list of currently open positions (tickets)
ulong openPositionsList[];

/*
// ------------------------------------------------------------------
//  Function: SendToFlask
//  Purpose : Send a JSON string (with bot_id included) to Flask
// ------------------------------------------------------------------
*/
bool SendToFlask(string endpoint, string jsonData) {
   char result[];
   char data[];
   string headers = "Content-Type: application/json\r\n";
   int timeout = 5000;

   ArrayResize(data, StringToCharArray(jsonData, data, 0, WHOLE_ARRAY, CP_UTF8) - 1);

   ResetLastError();
   int responseSize = WebRequest("POST", FlaskURL + endpoint, headers, timeout, data, result, headers);

   if (responseSize > 0) {
      string response = CharArrayToString(result);
      Print("Response from Flask (", endpoint, "): ", response);
      return true;
   }

   Print("❌ WebRequest Error:", GetLastError(), " when sending to ", endpoint);
   return false;
}

/*
// ------------------------------------------------------------------
//  Function: IsTradeRecorded
//  Purpose : Check if a trade with a certain ticket & bot_id is recorded
// ------------------------------------------------------------------
*/
bool IsTradeRecorded(ulong ticket) {
   string url = FlaskURL + "/check_trade";
   string jsonData = StringFormat("{\"ticket\":%d,\"bot_id\":\"%s\"}", ticket, BotID);

   char result[];
   char data[];
   string headers = "Content-Type: application/json\r\n";
   int timeout = 5000;

   ArrayResize(data, StringToCharArray(jsonData, data, 0, WHOLE_ARRAY, CP_UTF8) - 1);

   ResetLastError();
   int responseSize = WebRequest("POST", url, headers, timeout, data, result, headers);

   if (responseSize > 0) {
      string response = CharArrayToString(result);
      Print("Response from Flask (check_trade): ", response);

      if (StringFind(response, "\"exists\": true") != -1) {
         return true; 
      }

      Print("Trade with ticket ", ticket, " not found in DB for bot_id=", BotID);
      return false;
   } else {
      Print("❌ WebRequest Error:", GetLastError(), " in IsTradeRecorded()");
      return false;
   }
}

/*
// ------------------------------------------------------------------
//  Function: CreateTradeJSON
//  Purpose : Builds JSON for active trades to send to Flask
// ------------------------------------------------------------------
*/
string CreateTradeJSON(string action, string action_type, string symbol, double lot, double price, string type, ulong ticket) {
   string json = "{";
   json += StringFormat("\"bot_id\":\"%s\",", BotID);
   json += StringFormat("\"action\":\"%s\",", action);
   json += StringFormat("\"action_type\":\"%s\",", action_type);
   json += StringFormat("\"symbol\":\"%s\",", symbol);
   json += StringFormat("\"lot\":%.2f,", lot);
   json += StringFormat("\"price\":%.5f,", price);
   json += StringFormat("\"type\":\"%s\",", type);
   json += StringFormat("\"ticket\":%d", ticket);
   json += "}";

   return json;
}

/*
// ------------------------------------------------------------------
//  Function: CheckOpenPositions
//  Purpose : Logs only active trades (BUY/SELL) to Flask
// ------------------------------------------------------------------
*/
void CheckOpenPositions() {
   ulong updatedTickets[];

   for (int i = OrdersTotal() - 1; i >= 0; i--) {
      if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {
         ulong ticket = OrderTicket();
         string symbol = OrderSymbol();
         int orderTypeNum = OrderType();
         string orderType;

         // ** Only process active market trades (BUY/SELL) **
         if (orderTypeNum == OP_BUY) {
            orderType = "BUY";
         } else if (orderTypeNum == OP_SELL) {
            orderType = "SELL";
         } else {
            Print("⏩ Ignoring pending order: ", OrderType());
            continue; // Ignore pending orders
         }

         double lots = OrderLots();
         double openPrice = OrderOpenPrice();

         // Check if trade is already recorded
         if (!IsTradeRecorded(ticket)) {
            string jsonData = CreateTradeJSON(orderType, "New Trade", symbol, lots, openPrice, orderType, ticket);
            bool success = SendToFlask("/add_position", jsonData);
            if (success) {
               Print("✅ Logged new trade (bot_id=", BotID, "): ", jsonData);
            } else {
               Print("❌ Failed logging new trade ticket=", ticket);
            }
         }

         // Add ticket to updatedTickets array
         ArrayResize(updatedTickets, ArraySize(updatedTickets) + 1);
         updatedTickets[ArraySize(updatedTickets) - 1] = ticket;
      }
   }

   // Update global openPositionsList
   ArrayResize(openPositionsList, ArraySize(updatedTickets));
   for (int i = 0; i < ArraySize(updatedTickets); i++) {
      openPositionsList[i] = updatedTickets[i];
   }
}

/*
// ------------------------------------------------------------------
//  Function: CheckClosedPositions
//  Purpose : Sends open trade list to Flask (ignores pending orders)
// ------------------------------------------------------------------
*/
void CheckClosedPositions() {
   string jsonData = "{\"bot_id\":\"" + BotID + "\",\"tickets\":[";
   bool hasActiveTrades = false;

   for (int i = OrdersTotal() - 1; i >= 0; i--) {
      if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {
         ulong ticket = OrderTicket();
         int orderTypeNum = OrderType();

         // ** Only add active market trades (BUY/SELL) **
         if (orderTypeNum == OP_BUY || orderTypeNum == OP_SELL) {
            if (hasActiveTrades) {
               jsonData += ",";
            }
            jsonData += IntegerToString(ticket);
            hasActiveTrades = true;
         }
      }
   }

   jsonData += "]}";

   // Send to /check_closed_trades
   bool success = SendToFlask("/check_closed_trades", jsonData);
   if (success) {
      Print("✅ Sent active trades list to Flask (bot_id=", BotID, "): ", jsonData);
   } else {
      Print("❌ Failed to send active trades list.");
   }
}

/*
// ------------------------------------------------------------------
//  Event: OnTimer
//  Purpose : Runs every second, calls check functions
// ------------------------------------------------------------------
*/
void OnTimer() {
   CheckOpenPositions();
   CheckClosedPositions();
}

/*
// ------------------------------------------------------------------
//  Event: OnInit
//  Purpose : Initializes EA and sets timer
// ------------------------------------------------------------------
*/
int OnInit() {
   Print("🚀 Master EA Initialized (bot_id=", BotID, "). Running every second.");
   EventSetTimer(1);
   return(INIT_SUCCEEDED);
}

/*
// ------------------------------------------------------------------
//  Event: OnDeinit
//  Purpose : Stops the timer when EA is removed
// ------------------------------------------------------------------
*/
void OnDeinit(const int reason) {
   EventKillTimer();
   Print("🛑 Master EA Deinitialized (bot_id=", BotID, ").");
}
//+------------------------------------------------------------------+
