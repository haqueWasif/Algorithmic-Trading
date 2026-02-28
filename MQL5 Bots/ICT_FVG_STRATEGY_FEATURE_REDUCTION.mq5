#property copyright "Shanto_Quant_AI"
#property version   "18.1" // V18.1: Variable Sync Patch
#property strict

input bool   InpUseNewsFilter  = true;
input int    InpMinsBeforeNews = 30; 
input int    InpMinsAfterNews  = 30; 

#property tester_file "fvg_model_GBPUSD_v4.onnx"
#property tester_file "fvg_mfe_regressor_GBPUSD_v5.onnx"

#include <Trade\Trade.mqh>

input long   InpBaseMagic  = 1000000; 
input double InpLot        = 0.1;
input double InpThreshold  = 0.55; 
input string InpDxySymbol  = "DXY"; 

input double InpNYRiskMult     = 1.00; 
input double InpLondonRiskMult = 0.75; 
input double InpAsiaRiskMult   = 0.50; 

// --- V18: ORDER FLOW DECAY & VOLATILITY CONTAINMENT ---
input int    InpOrderFlowLookbackBars = 10;   
input int    InpMaxUnmitigatedFVGs    = 0;    
input int    InpMaxH1PullbackBars     = 12;   
input int    InpStdDevPeriod          = 20;   
input double InpMaxStdDevMultiplier   = 2.0;  

input double InpMinFillPct        = 0.50; 
input double InpMaxFillPct        = 0.85; 

input double InpATRMultiplier = 1.0; 
input double InpMinSLPips     = 5.0;  
input double InpMaxSLPips     = 40.0;
input double InpMaxRR         = 4.0;  

input double InpMaxSpreadPips     = 2.5;  
input double InpBEPipsBuffer      = 2.0;  
input bool   InpUseTrailingManager = true; 

long   classifier_handle = INVALID_HANDLE;
long   regressor_handle  = INVALID_HANDLE;
bool   g_dxy_ok = false; 

// --- H1 FVG STATE MEMORY ---
double g_active_bull_top = 0, g_active_bull_bottom = 0, g_active_bull_size = 0, g_active_bull_ratio = 0;
double g_active_bear_bottom = 0, g_active_bear_top = 0, g_active_bear_size = 0, g_active_bear_ratio = 0;
datetime g_h1_formation_time = 0;
double g_h1_rsi = 50.0, g_h1_dist = 0.0, g_h1_dist_pdh = 0.0, g_h1_dist_pdl = 0.0;

// --- ARM & FIRE STATE MACHINE ---
bool g_armed_bull = false;
bool g_armed_bear = false;
double g_latest_m15_opp_fvg_top = 0;
double g_latest_m15_opp_fvg_bot = 0;
double g_bull_max_pen = 0, g_bear_max_pen = 0;

CTrade trade;

//+------------------------------------------------------------------+
//| CORE UTILITIES                                                   |
//+------------------------------------------------------------------+
struct NewsEvent { datetime blackout_start; datetime blackout_end; };
NewsEvent CachedNews[]; int CachedNewsCount = 0;

void LoadNewsIntoMemory() {
   if(!InpUseNewsFilter) return;
   int h = FileOpen("High_Impact_News.csv", FILE_READ|FILE_CSV|FILE_COMMON|FILE_ANSI, ',');
   if(h == INVALID_HANDLE) return;
   string base_curr = StringSubstr(_Symbol, 0, 3); string quote_curr = StringSubstr(_Symbol, 3, 3);
   FileReadString(h); FileReadString(h); FileReadString(h); 
   while(!FileIsEnding(h)) {
      string time_str = FileReadString(h); string currency = FileReadString(h); string event = FileReadString(h);
      if(time_str == "") break;
      if(currency == base_curr || currency == quote_curr || currency == "ALL") {
         StringReplace(time_str, "-", ".");
         datetime news_time = StringToTime(time_str);
         ArrayResize(CachedNews, CachedNewsCount + 1);
         CachedNews[CachedNewsCount].blackout_start = news_time - (InpMinsBeforeNews * 60);
         CachedNews[CachedNewsCount].blackout_end   = news_time + (InpMinsAfterNews * 60);
         CachedNewsCount++;
      }
   }
   FileClose(h);
}

bool IsNewsTradingAllowed() {
   if(!InpUseNewsFilter || CachedNewsCount == 0) return true;
   datetime current_utc = TimeCurrent() - (2 * 3600); 
   for(int i = 0; i < CachedNewsCount; i++) {
      if(current_utc >= CachedNews[i].blackout_start && current_utc <= CachedNews[i].blackout_end) return false; 
   }
   return true;
}

bool IsMyMagic(long m) { return (m >= InpBaseMagic && m < InpBaseMagic + 500000); }

int CountMyPositions() {
   int c = 0;
   for(int i=0; i<PositionsTotal(); i++) {
      ulong t = PositionGetTicket(i);
      if(PositionSelectByTicket(t)) { 
         if(PositionGetString(POSITION_SYMBOL) == _Symbol && IsMyMagic(PositionGetInteger(POSITION_MAGIC))) c++; 
      }
   }
   return c;
}

double CalculatePythonEMA(string sym, ENUM_TIMEFRAMES tf, int period, int shift) {
   double alpha = 2.0 / (double)(period + 1.0);
   double res = iClose(sym, tf, shift + period);
   if(res == 0) return 0;
   for(int i = shift + period - 1; i >= shift; i--) res = (iClose(sym, tf, i) * alpha) + (res * (1.0 - alpha));
   return res;
}

double CalculatePythonRSI(string sym, ENUM_TIMEFRAMES tf, int period, int shift) {
   double gains = 0, losses = 0;
   for(int i = shift; i < shift + period; i++) {
      double diff = iClose(sym, tf, i) - iClose(sym, tf, i+1);
      if(diff > 0) gains += diff; else losses -= diff;
   }
   double avg_gain = gains / (double)period; double avg_loss = losses / (double)period;
   if(avg_loss == 0) return 100.0;
   return 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)));
}

double CalculatePythonATR(string sym, ENUM_TIMEFRAMES tf, int period, int shift) {
   double tr_sum = 0;
   for(int i = shift; i < shift + period; i++) {
      double h = iHigh(sym, tf, i); double l = iLow(sym, tf, i); double pc = iClose(sym, tf, i+1);
      tr_sum += MathMax(h-l, MathMax(MathAbs(h-pc), MathAbs(l-pc)));
   }
   return tr_sum / (double)period;
}

double CalculateStdDev(string sym, ENUM_TIMEFRAMES tf, int period, int shift) {
   double sum = 0, sum_sq = 0;
   for(int i = shift; i < shift + period; i++) sum += iClose(sym, tf, i);
   double mean = sum / period;
   for(int i = shift; i < shift + period; i++) sum_sq += MathPow(iClose(sym, tf, i) - mean, 2);
   return MathSqrt(sum_sq / period);
}

double GetEquityDrawdownMultiplier() {
   static double peak_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(current_equity > peak_equity) peak_equity = current_equity; 
   if(peak_equity <= 0) return 1.0;
   
   double drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100.0;
   if(drawdown_pct >= 4.0) return 0.25; 
   else if(drawdown_pct >= 2.0) return 0.50; 
   return 1.0; 
}

//+------------------------------------------------------------------+
//| V18.1 TIERED SCALE-OUT MANAGER                                   |
//+------------------------------------------------------------------+
void ManageOpenPositions() {
   if(PositionsTotal() == 0 || !InpUseTrailingManager) return; 

   double pip_val = (_Digits == 3 || _Digits == 5) ? 10.0 * _Point : _Point;
   double be_buffer = InpBEPipsBuffer * pip_val;
   double min_stop = ((int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL)) * _Point;
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK); 
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket) && PositionGetString(POSITION_SYMBOL) == _Symbol) {
         long magic = PositionGetInteger(POSITION_MAGIC);
         if(!IsMyMagic(magic)) continue;

         datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
         if(TimeCurrent() - open_time >= (96 * 15 * 60)) { 
             Print("Time Stop Triggered: 24 Hours Expired. Closing position.");
             trade.PositionClose(ticket);
             continue;
         }

         long offset = magic - InpBaseMagic;
         int tier = (int)(offset / 100000); 
         int risk_pts = (int)(offset % 100000); 
         double risk_dist = risk_pts * _Point;

         double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
         double current_sl = PositionGetDouble(POSITION_SL);
         double current_tp = PositionGetDouble(POSITION_TP);
         double volume     = PositionGetDouble(POSITION_VOLUME);
         long   type       = PositionGetInteger(POSITION_TYPE);
         
         if(current_tp == 0 || risk_dist == 0) continue;

         double current_rr = (type == POSITION_TYPE_BUY) ? (bid - open_price) / risk_dist : (open_price - ask) / risk_dist;
         if(current_rr < 1.0) continue; 

         int target_level = (int)MathFloor(current_rr); 
         if (target_level > 3) target_level = 3; 

         double target_sl_dist = (target_level == 1) ? be_buffer : (target_level - 1) * risk_dist; 
         double target_sl = NormalizeDouble((type == POSITION_TYPE_BUY) ? open_price + target_sl_dist : open_price - target_sl_dist, _Digits);

         bool sl_needs_move = (type == POSITION_TYPE_BUY) ? (current_sl < target_sl - _Point) : (current_sl > target_sl + _Point || current_sl == 0);

         if(sl_needs_move) {
            bool valid_stop = (type == POSITION_TYPE_BUY) ? (bid - target_sl > min_stop) : (target_sl - ask > min_stop);

            if(valid_stop && trade.PositionModify(ticket, target_sl, current_tp)) {
               double pct_to_close = 0.0; 
               
               if(tier == 1) { 
                   if(target_level == 1) pct_to_close = 0.50;      
                   else if(target_level >= 2) pct_to_close = 0.50;  
               }
               else { 
                   if(target_level == 1) pct_to_close = 0.75;      
                   else if(target_level >= 2) pct_to_close = 0.50; 
               }

               if(pct_to_close > 0.0) {
                   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP); 
                   double min_vol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
                   double v_close = MathFloor((volume * pct_to_close) / step) * step;
                   
                   if(v_close >= min_vol && (volume - v_close) >= min_vol) {
                      trade.PositionClosePartial(ticket, v_close);
                      PrintFormat("Scale-Out: %dR Hit! Tier %d. Closed %.2f lots. SL Secured.", target_level, tier, v_close);
                   }
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| INITIALIZATION                                                   |
//+------------------------------------------------------------------+
int OnInit() {
   ResetLastError();
   g_dxy_ok = SymbolSelect(InpDxySymbol, true); 
   LoadNewsIntoMemory();

   long shape_16D[] = {1, 16}; 
   classifier_handle = OnnxCreate("fvg_model_" + _Symbol + "_v4.onnx", ONNX_DEFAULT);
   if(classifier_handle != INVALID_HANDLE) {
      OnnxSetInputShape(classifier_handle, 0, shape_16D);
      if(OnnxGetOutputCount(classifier_handle) >= 2) { long o0[]={1}; long o1[]={1,2}; OnnxSetOutputShape(classifier_handle,0,o0); OnnxSetOutputShape(classifier_handle,1,o1); }
      else { long o0[]={1,2}; OnnxSetOutputShape(classifier_handle,0,o0); }
   } else { PrintFormat("CRITICAL: Failed to load Classifier."); return INIT_FAILED; }

   regressor_handle = OnnxCreate("fvg_mfe_regressor_" + _Symbol + "_v5.onnx", ONNX_DEFAULT);
   if(regressor_handle != INVALID_HANDLE) {
      OnnxSetInputShape(regressor_handle, 0, shape_16D);
      long ro[] = {1, 1}; OnnxSetOutputShape(regressor_handle, 0, ro);
   } else { PrintFormat("CRITICAL: Failed to load Regressor."); return INIT_FAILED; }

   PrintFormat("V18.1 Loaded. Containment & Decay Active. Lookback: %d Hours.", InpOrderFlowLookbackBars);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| MAIN EXECUTION LOOP                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   ManageOpenPositions(); 

   if(Bars(_Symbol, PERIOD_M15) < 100 || Bars(_Symbol, PERIOD_H1) < 50 || Bars(_Symbol, PERIOD_D1) < 5) return;
   double pip_val = (_Digits == 3 || _Digits == 5) ? 10.0 * _Point : _Point;
   if (SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point > (InpMaxSpreadPips * pip_val)) return; 

   static datetime last_h1_bar = 0; 
   datetime h1_time = iTime(_Symbol, PERIOD_H1, 0);
   static datetime last_m15_bar = 0; 
   datetime m15_time = iTime(_Symbol, PERIOD_M15, 0);

   // =======================================================
   // 1. HTF SCANNER & ORDER FLOW EXHAUSTION DECAY
   // =======================================================
   if(h1_time != last_h1_bar) {
       last_h1_bar = h1_time;
       double h1_c1 = iClose(_Symbol, PERIOD_H1, 1);
       double h1_h1 = iHigh(_Symbol, PERIOD_H1, 1);
       double h1_l1 = iLow(_Symbol, PERIOD_H1, 1);
       double h1_py_atr = CalculatePythonATR(_Symbol, PERIOD_H1, 14, 1);
       
       double gap_bull = iLow(_Symbol, PERIOD_H1, 1) - iHigh(_Symbol, PERIOD_H1, 3);
       double gap_bear = iLow(_Symbol, PERIOD_H1, 3) - iHigh(_Symbol, PERIOD_H1, 1);

       bool is_bull_displacement = iClose(_Symbol, PERIOD_H1, 2) > iClose(_Symbol, PERIOD_H1, 3);
       bool is_bear_displacement = iClose(_Symbol, PERIOD_H1, 2) < iClose(_Symbol, PERIOD_H1, 3);

       // --- V18: DYNAMIC ORDER FLOW DECAY ---
       int unmitigated_bulls = 0;
       int unmitigated_bears = 0;
       
       for(int i = 4; i <= InpOrderFlowLookbackBars; i++) { 
           double past_bull_gap = iLow(_Symbol, PERIOD_H1, i-2) - iHigh(_Symbol, PERIOD_H1, i);
           if(past_bull_gap > 0 && iClose(_Symbol, PERIOD_H1, i-1) > iClose(_Symbol, PERIOD_H1, i)) {
               bool is_filled = false;
               for(int k = i-3; k >= 1; k--) {
                   if(iLow(_Symbol, PERIOD_H1, k) < iHigh(_Symbol, PERIOD_H1, i)) { is_filled = true; break; }
               }
               if(!is_filled) unmitigated_bulls++;
           }
           double past_bear_gap = iLow(_Symbol, PERIOD_H1, i) - iHigh(_Symbol, PERIOD_H1, i-2);
           if(past_bear_gap > 0 && iClose(_Symbol, PERIOD_H1, i-1) < iClose(_Symbol, PERIOD_H1, i)) {
               bool is_filled = false;
               for(int k = i-3; k >= 1; k--) {
                   if(iHigh(_Symbol, PERIOD_H1, k) > iLow(_Symbol, PERIOD_H1, i)) { is_filled = true; break; }
               }
               if(!is_filled) unmitigated_bears++;
           }
       }

       // Arm Bull Setup
       if (gap_bull > 0.0 && is_bull_displacement) {
           if(unmitigated_bulls <= InpMaxUnmitigatedFVGs) {
               g_active_bull_top = iLow(_Symbol, PERIOD_H1, 1); 
               g_active_bull_bottom = iHigh(_Symbol, PERIOD_H1, 3);
               g_active_bull_size = gap_bull / pip_val; 
               g_active_bull_ratio = (h1_py_atr > 0) ? (gap_bull / h1_py_atr) : 0.0;
               g_h1_formation_time = h1_time;
               g_armed_bull = false; g_latest_m15_opp_fvg_top = 0.0; g_bull_max_pen = 0.0;
           } else {
               PrintFormat("Bull Veto: %d unmitigated gaps within %d hours.", unmitigated_bulls, InpOrderFlowLookbackBars);
           }
       }
       
       // Arm Bear Setup
       if (gap_bear > 0.0 && is_bear_displacement) {
           if(unmitigated_bears <= InpMaxUnmitigatedFVGs) {
               g_active_bear_bottom = iHigh(_Symbol, PERIOD_H1, 1); 
               g_active_bear_top = iLow(_Symbol, PERIOD_H1, 3);
               g_active_bear_size = gap_bear / pip_val; 
               g_active_bear_ratio = (h1_py_atr > 0) ? (gap_bear / h1_py_atr) : 0.0;
               g_h1_formation_time = h1_time;
               g_armed_bear = false; g_latest_m15_opp_fvg_bot = 0.0; g_bear_max_pen = 0.0;
           } else {
               PrintFormat("Bear Veto: %d unmitigated gaps within %d hours.", unmitigated_bears, InpOrderFlowLookbackBars);
           }
       }
       
       g_h1_rsi = CalculatePythonRSI(_Symbol, PERIOD_H1, 14, 1);
       double h1_ema = CalculatePythonEMA(_Symbol, PERIOD_H1, 50, 1);
       g_h1_dist = (h1_ema > 0) ? (h1_c1 - h1_ema) / pip_val : 0.0;

       double pdh = iHigh(_Symbol, PERIOD_D1, 1);
       double pdl = iLow(_Symbol, PERIOD_D1, 1);
       g_h1_dist_pdh = (h1_h1 - pdh) / pip_val;
       g_h1_dist_pdl = (h1_l1 - pdl) / pip_val;
   }

   // =======================================================
   // 2. LTF ARM & FIRE MANAGER (VOLATILITY CONTAINMENT)
   // =======================================================
   if(m15_time != last_m15_bar) {
       last_m15_bar = m15_time;
       
       double m15_c1 = iClose(_Symbol, PERIOD_M15, 1); double m15_o1 = iOpen(_Symbol, PERIOD_M15, 1);
       double m15_h1 = iHigh(_Symbol, PERIOD_M15, 1);  double m15_l1 = iLow(_Symbol, PERIOD_M15, 1);

       double current_m15_std = CalculateStdDev(_Symbol, PERIOD_M15, InpStdDevPeriod, 1);
       bool execute_bull = false, execute_bear = false;

       // --- BULLISH STATE MACHINE ---
       if(g_active_bull_top > 0) {
           double fvg_midpoint = (g_active_bull_top + g_active_bull_bottom) / 2.0;
           
           if(m15_c1 < g_active_bull_bottom || 
              iBarShift(_Symbol, PERIOD_H1, g_h1_formation_time) > InpMaxH1PullbackBars ||
              MathAbs(m15_c1 - fvg_midpoint) > (InpMaxStdDevMultiplier * current_m15_std)) 
           { 
               g_active_bull_top = 0; 
           } else {
               if (m15_l1 <= g_active_bull_top) {
                   double gap_total = g_active_bull_top - g_active_bull_bottom;
                   double pen = (gap_total > 0) ? (g_active_bull_top - m15_l1) / gap_total : 0;
                   g_bull_max_pen = MathMax(g_bull_max_pen, pen);
                   g_armed_bull = true;
               }
               if (g_bull_max_pen > InpMaxFillPct) g_active_bull_top = 0;
               else if (g_armed_bull) {
                   double m15_bear_gap = iLow(_Symbol, PERIOD_M15, 3) - iHigh(_Symbol, PERIOD_M15, 1);
                   bool m15_bear_disp = iClose(_Symbol, PERIOD_M15, 2) < iClose(_Symbol, PERIOD_M15, 3);
                   if (m15_bear_gap > 0 && m15_bear_disp) g_latest_m15_opp_fvg_top = iLow(_Symbol, PERIOD_M15, 3);
                   
                   if (g_bull_max_pen >= InpMinFillPct && g_latest_m15_opp_fvg_top > 0 && m15_c1 > g_latest_m15_opp_fvg_top) {
                       execute_bull = true;
                   }
               }
           }
       }

       // --- BEARISH STATE MACHINE ---
       if(g_active_bear_bottom > 0) {
           double fvg_midpoint = (g_active_bear_top + g_active_bear_bottom) / 2.0;
           
           if(m15_c1 > g_active_bear_top || 
              iBarShift(_Symbol, PERIOD_H1, g_h1_formation_time) > InpMaxH1PullbackBars ||
              MathAbs(m15_c1 - fvg_midpoint) > (InpMaxStdDevMultiplier * current_m15_std)) 
           { 
               g_active_bear_bottom = 0; 
           } else {
               if (m15_h1 >= g_active_bear_bottom) {
                   double gap_total = g_active_bear_top - g_active_bear_bottom;
                   double pen = (gap_total > 0) ? (m15_h1 - g_active_bear_bottom) / gap_total : 0;
                   g_bear_max_pen = MathMax(g_bear_max_pen, pen);
                   g_armed_bear = true;
               }
               if (g_bear_max_pen > InpMaxFillPct) g_active_bear_bottom = 0;
               else if (g_armed_bear) {
                   double m15_bull_gap = iLow(_Symbol, PERIOD_M15, 1) - iHigh(_Symbol, PERIOD_M15, 3);
                   bool m15_bull_disp = iClose(_Symbol, PERIOD_M15, 2) > iClose(_Symbol, PERIOD_M15, 3);
                   if (m15_bull_gap > 0 && m15_bull_disp) g_latest_m15_opp_fvg_bot = iHigh(_Symbol, PERIOD_M15, 3);
                   
                   if (g_bear_max_pen >= InpMinFillPct && g_latest_m15_opp_fvg_bot > 0 && m15_c1 < g_latest_m15_opp_fvg_bot) {
                       execute_bear = true;
                   }
               }
           }
       }

       if((execute_bull || execute_bear) && IsNewsTradingAllowed()) 
       {
          double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
          double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
          
          double c_rng = m15_h1 - m15_l1;
          double body_pct = 0, u_wick_pct = 0, d_wick_pct = 0;
          if(c_rng > 0) {
              body_pct = MathAbs(m15_o1 - m15_c1) / c_rng;
              u_wick_pct = (m15_h1 - MathMax(m15_o1, m15_c1)) / c_rng;
              d_wick_pct = (MathMin(m15_o1, m15_c1) - m15_l1) / c_rng;
          }
          
          double m15_py_atr = CalculatePythonATR(_Symbol, PERIOD_M15, 14, 1);
          if(m15_py_atr <= 0) return; 

          double h4_ema = CalculatePythonEMA(_Symbol, PERIOD_H4, 50, 1);
          float h4_dist = (h4_ema > 0) ? (float)((iClose(_Symbol, PERIOD_H4, 1) - h4_ema) / pip_val) : 0.0f;
          float h4_atr  = (float)(CalculatePythonATR(_Symbol, PERIOD_H4, 14, 1) / pip_val);

          float dxy_rsi = 50.0f; float dxy_dist = 0.0f;
          if(g_dxy_ok && SymbolInfoDouble(InpDxySymbol, SYMBOL_BID) > 0) {
              double dxy_ema = CalculatePythonEMA(InpDxySymbol, PERIOD_M15, 50, 1);
              if(dxy_ema > 0) {
                  dxy_rsi = (float)CalculatePythonRSI(InpDxySymbol, PERIOD_M15, 14, 1);
                  dxy_dist = (float)(iClose(InpDxySymbol, PERIOD_M15, 1) - dxy_ema);
              }
          }

          double h1_py_atr = CalculatePythonATR(_Symbol, PERIOD_H1, 14, 1);
          double h1_bull_size = (g_active_bull_top - g_active_bull_bottom) / pip_val;
          double h1_bear_size = (g_active_bear_top - g_active_bear_bottom) / pip_val;

          float X_16D[1][16]; 
          X_16D[0][0]  = (float)iTickVolume(_Symbol, PERIOD_M15, 1);
          X_16D[0][1]  = (float)CalculatePythonRSI(_Symbol, PERIOD_M15, 14, 1);
          X_16D[0][2]  = (float)(m15_py_atr / pip_val);
          X_16D[0][3]  = execute_bull ? (float)(h1_bull_size / h1_py_atr) : 0.0f;
          X_16D[0][4]  = execute_bear ? (float)(h1_bear_size / h1_py_atr) : 0.0f;
          X_16D[0][5]  = (float)body_pct;
          X_16D[0][6]  = (float)u_wick_pct;
          X_16D[0][7]  = (float)d_wick_pct;
          X_16D[0][8]  = (float)g_h1_rsi; 
          X_16D[0][9]  = (float)g_h1_dist;
          X_16D[0][10] = (float)g_h1_dist_pdh; 
          X_16D[0][11] = (float)g_h1_dist_pdl; 
          X_16D[0][12] = h4_dist; 
          X_16D[0][13] = h4_atr;
          X_16D[0][14] = dxy_rsi; 
          X_16D[0][15] = dxy_dist;

          float probabilities[1][2];
          if(OnnxGetOutputCount(classifier_handle) >= 2) { long label[1]; OnnxRun(classifier_handle, ONNX_NO_CONVERSION, X_16D, label, probabilities); }
          else { OnnxRun(classifier_handle, ONNX_NO_CONVERSION, X_16D, probabilities); }
          float win_prob = probabilities[0][1];

          if(win_prob >= (float)InpThreshold)
          {
             double atr_pips = m15_py_atr / pip_val;
             double fvg_sz = execute_bull ? h1_bull_size : h1_bear_size;
             
             double sl_pips = fvg_sz + (atr_pips * InpATRMultiplier);
             if(sl_pips < InpMinSLPips) sl_pips = InpMinSLPips; 
             if(sl_pips > InpMaxSLPips) sl_pips = InpMaxSLPips; 

             double tp_pips = sl_pips * InpMaxRR; 

             double min_stop = ((int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL)) * _Point;
             double sl_dist_price = sl_pips * pip_val;
             double tp_dist_price = tp_pips * pip_val;

             if(sl_dist_price >= min_stop && tp_dist_price >= min_stop && CountMyPositions() == 0)
             {
                int tier = 3; if(win_prob >= 0.65) tier = 1; else if(win_prob >= 0.58) tier = 2; 
                
                int risk_pts = (int)MathRound(sl_pips * (pip_val / _Point));
                long encoded_magic = InpBaseMagic + (tier * 100000) + risk_pts;
                trade.SetExpertMagicNumber(encoded_magic);
                
                MqlDateTime dt; TimeCurrent(dt);
                double dynamic_lot = CalculateDynamicLot(sl_pips, tier, dt.hour);

                PrintFormat("V18.1 Execute: Conf %.2f%% | TP: %.1f | SL: %.1f", win_prob*100, tp_pips, sl_pips);
                if(execute_bull) trade.Buy(dynamic_lot, _Symbol, ask, NormalizeDouble(ask - sl_dist_price, _Digits), NormalizeDouble(ask + tp_dist_price, _Digits));
                else if(execute_bear) trade.Sell(dynamic_lot, _Symbol, bid, NormalizeDouble(bid + sl_dist_price, _Digits), NormalizeDouble(bid - tp_dist_price, _Digits));
                
                // Clear state variables using the global names
                if(execute_bull) { 
                    g_active_bull_top = 0; g_active_bull_bottom = 0; 
                    g_armed_bull = false; g_latest_m15_opp_fvg_top = 0; g_bull_max_pen = 0; 
                }
                if(execute_bear) { 
                    g_active_bear_top = 0; g_active_bear_bottom = 0; 
                    g_armed_bear = false; g_latest_m15_opp_fvg_bot = 0; g_bear_max_pen = 0; 
                }
             }
          }
       }
   }
}

//+------------------------------------------------------------------+
//| UTILITIES & RISK MATH                                            |
//+------------------------------------------------------------------+
double CalculateDynamicLot(double sl_distance_pips, int tier, int current_hour) {
   double base_risk = 0.5; 
   if(tier == 1) base_risk = 2.0; 
   else if(tier == 2) base_risk = 1.0; 
   
   double session_mult = InpAsiaRiskMult; 
   if(current_hour >= 13 && current_hour < 22) session_mult = InpNYRiskMult; 
   else if(current_hour >= 8 && current_hour < 13) session_mult = InpLondonRiskMult; 

   double dd_mult = GetEquityDrawdownMultiplier();
   double final_risk = base_risk * session_mult * dd_mult;

   double risk_dollars = AccountInfoDouble(ACCOUNT_BALANCE) * (final_risk / 100.0);
   double tv = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double ts = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double pv = (_Digits == 3 || _Digits == 5) ? 10.0 * _Point : _Point;
   double risk_pts = sl_distance_pips * (pv / _Point);

   if(risk_pts <= 0 || tv <= 0) return InpLot;
   double lot = risk_dollars / (risk_pts * tv * (ts / _Point));
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   return MathMin(MathMax(MathFloor(lot/step)*step, SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN)), SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX));
}

void OnDeinit(const int reason) {
   if(classifier_handle != INVALID_HANDLE) OnnxRelease(classifier_handle);
   if(regressor_handle  != INVALID_HANDLE) OnnxRelease(regressor_handle);
}