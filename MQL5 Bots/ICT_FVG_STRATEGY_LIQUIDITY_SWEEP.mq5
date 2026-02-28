#property copyright "Shanto_Quant_AI"
#property version   "16.2" // V16.2: prob+session risk, smooth DD recovery, 1R scaleout by tier, session thresholds
#property strict

#include <Trade\Trade.mqh>

// -------------------- Inputs --------------------
input bool   InpUseNewsFilter  = true;
input int    InpMinsBeforeNews = 30;
input int    InpMinsAfterNews  = 30;

// --- TESTER DIRECTIVES ---
#property tester_file "fvg_model_EURUSD_v6.onnx"
#property tester_file "fvg_mfe_regressor_EURUSD_v7.onnx"
#property tester_file "fvg_model_GBPUSD_v6.onnx"
#property tester_file "fvg_mfe_regressor_GBPUSD_v7.onnx"
#property tester_file "fvg_model_USDJPY_v6.onnx"
#property tester_file "fvg_mfe_regressor_USDJPY_v7.onnx"

// Meta
#property tester_file "fvg_meta_EURUSD_v6.json"
#property tester_file "fvg_meta_GBPUSD_v6.json"
#property tester_file "fvg_meta_USDJPY_v6.json"

// --- Trading Inputs ---
input long   InpBaseMagic  = 1000000;
input double InpLot        = 0.1;    // fallback if sizing fails
input double InpThreshold  = 0.55;   // fallback if meta missing
input string InpDxySymbol  = "DXY";

// Session multipliers
input double InpNYRiskMult     = 1.00;
input double InpLondonRiskMult = 0.75;
input double InpAsiaRiskMult   = 0.50;

// Probability->Risk (percent of balance)
input double InpRiskTier1Pct = 2.0;   // high prob
input double InpRiskTier2Pct = 1.0;   // mid prob
input double InpRiskTier3Pct = 0.5;   // low prob

// Optional continuous probability risk ramp (multiplies tier risk)
input bool   InpUseProbRiskRamp = true;
input double InpProbRampMin     = 0.80; // at threshold -> 0.80x
input double InpProbRampMax     = 1.20; // near 1.0 -> 1.20x

// Drawdown risk control (smooth)
input bool   InpUseSmoothDDRisk = true;
input double InpDD_MinMult      = 0.25; // floor
input double InpDD_FullRiskDD   = 0.0;  // if dd <= this => 1.0 mult (usually 0)
input double InpDD_MinMultDD    = 6.0;  // if dd >= this => min mult

// Setup filters
input double InpMinFillPct       = 0.50;
input double InpMaxFillPct       = 0.85;
input double InpMaxEntryDistance = 20.0;

// SL/TP
input double InpATRMultiplier = 1.0;
input double InpMinSLPips     = 5.0;
input double InpMaxSLPips     = 40.0;
input double InpMFEDampener   = 0.85;
input double InpMaxRR         = 4.0;

input double InpMaxSpreadPips       = 2.5;
input double InpBEPipsBuffer        = 2.0;
input bool   InpUseTrailingManager  = true;

// Probability thresholds for tiers (calibrated prob)
input double InpTier1Prob = 0.70;
input double InpTier2Prob = 0.62;
input double InpTier3Prob = 0.55;  // usually equal to threshold

// Scale-out settings at RR milestones (by tier)
input double InpSO1_T1 = 0.25; // 1R close %
input double InpSO1_T2 = 0.50;
input double InpSO1_T3 = 0.75;

input double InpSO2_T1 = 0.00; // optional extra at 2R
input double InpSO2_T2 = 0.00;
input double InpSO2_T3 = 0.00;

input double InpSO3_T1 = 0.00; // optional extra at 3R
input double InpSO3_T2 = 0.00;
input double InpSO3_T3 = 0.00;

// -------------------- Globals --------------------
CTrade trade;

long   classifier_handle = INVALID_HANDLE;
long   regressor_handle  = INVALID_HANDLE;
bool   g_dxy_ok = false;

// Meta-driven thresholds
double g_threshold_global = 0.55;
double g_thr_asia   = -1.0;
double g_thr_london = -1.0;
double g_thr_ny     = -1.0;

bool   g_has_calib = false;
double g_calib_a = 0.0;
double g_calib_b = 0.0;

// FVG state
double g_active_bull_top = 0, g_active_bull_bottom = 0, g_active_bull_size = 0, g_active_bull_ratio = 0;
double g_active_bear_bottom = 0, g_active_bear_top = 0, g_active_bear_size = 0, g_active_bear_ratio = 0;

// H1 context
double g_h1_rsi = 50.0, g_h1_dist = 0.0;
double g_h1_dist_pdh = 0.0, g_h1_dist_pdl = 0.0;
double g_h1_swept_pdh = 0.0;
double g_h1_swept_pdl = 0.0;

// -------------------- Helpers --------------------
string GetFxBaseSymbol(const string sym)
{
   if(StringLen(sym) >= 6) return StringSubstr(sym, 0, 6);
   return sym;
}

double PipValue()
{
   return (_Digits == 3 || _Digits == 5) ? (10.0 * _Point) : _Point;
}

double SafeExp(double x)
{
   if(x > 50.0) x = 50.0;
   if(x < -50.0) x = -50.0;
   return MathExp(x);
}

// Correct sigmoid calibration: p = 1/(1+exp(-(a*raw + b)))
double CalibrateProb(double raw_p)
{
   if(!g_has_calib) return raw_p;
   double z = g_calib_a * raw_p + g_calib_b;
   return 1.0 / (1.0 + SafeExp(-z));
}

bool ExtractJsonNumber(const string &json, const string &key, double &out_val)
{
   int k = StringFind(json, "\"" + key + "\"");
   if(k < 0) return false;

   int colon = StringFind(json, ":", k);
   if(colon < 0) return false;

   int start = colon + 1;
   while(start < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, start);
      if(ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') break;
      start++;
   }

   int end = start;
   while(end < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, end);
      if(ch == ',' || ch == '}' || ch == '\n' || ch == '\r') break;
      end++;
   }

   string num = StringSubstr(json, start, end - start);
   StringReplace(num, " ", "");
   StringReplace(num, "\t", "");

   double v = StringToDouble(num);
   // allow 0.0 explicitly
   if(v == 0.0)
   {
      // if string had no digit at all, it's invalid
      bool has_digit = (StringFind(num, "0") >= 0) || (StringFind(num, "1") >= 0) || (StringFind(num, "2") >= 0) ||
                       (StringFind(num, "3") >= 0) || (StringFind(num, "4") >= 0) || (StringFind(num, "5") >= 0) ||
                       (StringFind(num, "6") >= 0) || (StringFind(num, "7") >= 0) || (StringFind(num, "8") >= 0) ||
                       (StringFind(num, "9") >= 0);
      if(!has_digit) return false;
   }

   out_val = v;
   return true;
}

bool LoadMeta(const string fx_base)
{
   g_threshold_global = InpThreshold;
   g_thr_asia = g_thr_london = g_thr_ny = -1.0;
   g_has_calib = false;
   g_calib_a = 0.0; g_calib_b = 0.0;

   string fn = "fvg_meta_" + fx_base + "_v6.json";

   int h = FileOpen(fn, FILE_READ|FILE_TXT|FILE_ANSI);
   if(h == INVALID_HANDLE)
   {
      h = FileOpen(fn, FILE_READ|FILE_TXT|FILE_ANSI|FILE_COMMON);
      if(h == INVALID_HANDLE)
      {
         PrintFormat("Meta not found: %s. Using fallback threshold=%.2f", fn, g_threshold_global);
         return false;
      }
   }

   string json = "";
   while(!FileIsEnding(h)) json += FileReadString(h);
   FileClose(h);

   double t;
   if(ExtractJsonNumber(json, "threshold", t)) g_threshold_global = t;

   // Optional session thresholds if present:
   // threshold_by_session: { "asia":..., "london":..., "ny":... }
   double ta, tl, tn;
   if(ExtractJsonNumber(json, "asia", ta))   g_thr_asia = ta;
   if(ExtractJsonNumber(json, "london", tl)) g_thr_london = tl;
   if(ExtractJsonNumber(json, "ny", tn))     g_thr_ny = tn;

   double a, b;
   bool ok_a = ExtractJsonNumber(json, "a", a);
   bool ok_b = ExtractJsonNumber(json, "b", b);
   if(ok_a && ok_b)
   {
      g_has_calib = true;
      g_calib_a = a; g_calib_b = b;
   }

   PrintFormat("Meta loaded: thr=%.3f (asia=%.3f london=%.3f ny=%.3f) calib=%s a=%.4f b=%.4f",
               g_threshold_global,
               (g_thr_asia   > 0 ? g_thr_asia   : g_threshold_global),
               (g_thr_london > 0 ? g_thr_london : g_threshold_global),
               (g_thr_ny     > 0 ? g_thr_ny     : g_threshold_global),
               (g_has_calib ? "ON" : "OFF"), g_calib_a, g_calib_b);

   return true;
}

// Session flags based on server time
void GetSessionFlags(const datetime t, float &is_asia, float &is_london, float &is_ny, int &hour_out)
{
   MqlDateTime dt;
   TimeToStruct(t, dt);
   int h = dt.hour;
   hour_out = h;
   is_london = (h >= 8  && h < 13) ? 1.0f : 0.0f;
   is_ny     = (h >= 13 && h < 22) ? 1.0f : 0.0f;
   is_asia   = (h < 8   || h >= 22) ? 1.0f : 0.0f;
}

double GetSessionThreshold(float is_asia, float is_london, float is_ny)
{
   if(is_asia > 0.5f && g_thr_asia   > 0) return g_thr_asia;
   if(is_london > 0.5f && g_thr_london > 0) return g_thr_london;
   if(is_ny > 0.5f && g_thr_ny > 0) return g_thr_ny;
   return g_threshold_global;
}

// -------------------- News Filter --------------------
struct NewsEvent { datetime blackout_start_utc; datetime blackout_end_utc; };
NewsEvent CachedNews[];
int CachedNewsCount = 0;

void LoadNewsIntoMemory()
{
   CachedNewsCount = 0;
   ArrayResize(CachedNews, 0);
   if(!InpUseNewsFilter) return;

   int h = FileOpen("High_Impact_News.csv", FILE_READ|FILE_CSV|FILE_COMMON|FILE_ANSI, ',');
   if(h == INVALID_HANDLE)
   {
      Print("News file not found/openable: High_Impact_News.csv (Common Files). Error=", GetLastError());
      return;
   }

   string fx = GetFxBaseSymbol(_Symbol);
   string base_curr  = (StringLen(fx) >= 3) ? StringSubstr(fx, 0, 3) : "";
   string quote_curr = (StringLen(fx) >= 6) ? StringSubstr(fx, 3, 3) : "";

   // header
   if(!FileIsEnding(h)) FileReadString(h);
   if(!FileIsEnding(h)) FileReadString(h);
   if(!FileIsEnding(h)) FileReadString(h);

   datetime server_now = TimeCurrent();
   datetime utc_now    = TimeGMT();
   int offset_sec      = (int)(server_now - utc_now);

   while(!FileIsEnding(h))
   {
      string time_str = FileReadString(h);
      string currency = FileReadString(h);
      string event    = FileReadString(h);

      if(time_str == "") break;

      if(currency == base_curr || currency == quote_curr || currency == "ALL")
      {
         StringReplace(time_str, "-", ".");
         datetime news_time_server = StringToTime(time_str);
         datetime news_time_utc = news_time_server - offset_sec;

         ArrayResize(CachedNews, CachedNewsCount + 1);
         CachedNews[CachedNewsCount].blackout_start_utc = news_time_utc - (InpMinsBeforeNews * 60);
         CachedNews[CachedNewsCount].blackout_end_utc   = news_time_utc + (InpMinsAfterNews  * 60);
         CachedNewsCount++;
      }
   }

   FileClose(h);
   PrintFormat("News loaded: %d blackout windows for %s", CachedNewsCount, _Symbol);
}

bool IsNewsTradingAllowed()
{
   if(!InpUseNewsFilter || CachedNewsCount == 0) return true;

   datetime now_utc = TimeGMT();
   for(int i=0; i<CachedNewsCount; i++)
      if(now_utc >= CachedNews[i].blackout_start_utc && now_utc <= CachedNews[i].blackout_end_utc)
         return false;

   return true;
}

// -------------------- Magic / Stage persistence --------------------
bool IsMyMagic(long m) { return (m >= InpBaseMagic && m < InpBaseMagic + 500000); }

// Encoded magic: Base + tier*100000 + risk_pts
int DecodeTier(long magic)
{
   long offset = magic - InpBaseMagic;
   if(offset < 0) return 3;
   int tier = (int)(offset / 100000);
   if(tier < 1 || tier > 3) tier = 3;
   return tier;
}

double DecodeRiskDist(long magic)
{
   long offset = magic - InpBaseMagic;
   int risk_pts = (int)(offset % 100000);
   return (double)risk_pts * _Point;
}

string StageKey(ulong ticket)
{
   return "SQAI_STAGE_" + IntegerToString((long)ticket);
}

int GetStage(ulong ticket)
{
   string k = StageKey(ticket);
   if(!GlobalVariableCheck(k)) return 0;
   return (int)GlobalVariableGet(k);
}

void SetStage(ulong ticket, int stage)
{
   string k = StageKey(ticket);
   GlobalVariableSet(k, (double)stage);
}

int CountMyPositions()
{
   int c=0;
   for(int i=0; i<PositionsTotal(); i++)
   {
      ulong t = PositionGetTicket(i);
      if(PositionSelectByTicket(t))
         if(PositionGetString(POSITION_SYMBOL) == _Symbol && IsMyMagic(PositionGetInteger(POSITION_MAGIC)))
            c++;
   }
   return c;
}

// -------------------- Indicators (guarded) --------------------
double CalculatePythonEMA(string sym, ENUM_TIMEFRAMES tf, int period, int shift)
{
   if(Bars(sym, tf) < shift + period + 2) return 0.0;
   double alpha = 2.0 / (double)(period + 1.0);

   double res = iClose(sym, tf, shift + period - 1);
   if(res == 0.0) return 0.0;

   for(int i = shift + period - 2; i >= shift; i--)
   {
      double c = iClose(sym, tf, i);
      if(c == 0.0) return 0.0;
      res = (c * alpha) + (res * (1.0 - alpha));
   }
   return res;
}

double CalculatePythonRSI(string sym, ENUM_TIMEFRAMES tf, int period, int shift)
{
   if(Bars(sym, tf) < shift + period + 2) return 50.0;

   double gains=0, losses=0;
   for(int i=shift; i<shift+period; i++)
   {
      double c0 = iClose(sym, tf, i);
      double c1 = iClose(sym, tf, i+1);
      if(c0 == 0.0 || c1 == 0.0) return 50.0;

      double diff = c0 - c1;
      if(diff > 0) gains += diff;
      else losses -= diff;
   }

   double avg_gain = gains / (double)period;
   double avg_loss = losses / (double)period;
   if(avg_loss == 0.0) return 100.0;

   double rs = avg_gain / avg_loss;
   return 100.0 - (100.0 / (1.0 + rs));
}

double CalculatePythonATR(string sym, ENUM_TIMEFRAMES tf, int period, int shift)
{
   if(Bars(sym, tf) < shift + period + 2) return 0.0;

   double tr_sum=0;
   for(int i=shift; i<shift+period; i++)
   {
      double h  = iHigh(sym, tf, i);
      double l  = iLow(sym, tf, i);
      double pc = iClose(sym, tf, i+1);
      if(h == 0.0 || l == 0.0 || pc == 0.0) return 0.0;

      tr_sum += MathMax(h-l, MathMax(MathAbs(h-pc), MathAbs(l-pc)));
   }
   return tr_sum / (double)period;
}

// -------------------- Smooth drawdown risk multiplier --------------------
double GetSmoothDrawdownMult()
{
   if(!InpUseSmoothDDRisk) return 1.0;

   static double peak_equity = 0.0;
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(peak_equity <= 0.0) peak_equity = eq;

   // Update peak always when higher
   if(eq > peak_equity) peak_equity = eq;

   if(peak_equity <= 0.0) return 1.0;

   double dd_pct = ((peak_equity - eq) / peak_equity) * 100.0;
   if(dd_pct <= InpDD_FullRiskDD) return 1.0;
   if(dd_pct >= InpDD_MinMultDD) return InpDD_MinMult;

   // Linear ramp between 1.0 and min mult
   double t = (dd_pct - InpDD_FullRiskDD) / (InpDD_MinMultDD - InpDD_FullRiskDD);
   double mult = 1.0 - t * (1.0 - InpDD_MinMult);
   if(mult < InpDD_MinMult) mult = InpDD_MinMult;
   if(mult > 1.0) mult = 1.0;
   return mult;
}

// -------------------- Money per pip per lot (robust) --------------------
double MoneyPerPipPerLot()
{
   double tv = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double ts = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double pv = PipValue();
   if(tv <= 0.0 || ts <= 0.0 || pv <= 0.0) return 0.0;

   // tick_value is money per tick_size for 1 lot
   // pip money = tick_value * (pip_size / tick_size)
   return tv * (pv / ts);
}

// -------------------- Probability -> Tier + risk % --------------------
int TierFromProb(const double p, const double session_thr)
{
   // must meet session threshold first
   if(p < session_thr) return 0;
   if(p >= InpTier1Prob) return 1;
   if(p >= InpTier2Prob) return 2;
   return 3;
}

double TierRiskPct(const int tier)
{
   if(tier == 1) return InpRiskTier1Pct;
   if(tier == 2) return InpRiskTier2Pct;
   return InpRiskTier3Pct;
}

double ProbRampMult(const double p, const double session_thr)
{
   if(!InpUseProbRiskRamp) return 1.0;
   double denom = (1.0 - session_thr);
   if(denom <= 1e-6) return 1.0;

   double x = (p - session_thr) / denom; // 0..1
   if(x < 0.0) x = 0.0;
   if(x > 1.0) x = 1.0;

   double mult = InpProbRampMin + x * (InpProbRampMax - InpProbRampMin);
   return mult;
}

// -------------------- Dynamic lot sizing (prob + session + DD) --------------------
double CalculateDynamicLot(double sl_pips, int tier, int current_hour, double win_prob, double session_thr)
{
   double base_risk_pct = TierRiskPct(tier);

   // Session multiplier
   double session_mult = InpAsiaRiskMult;
   if(current_hour >= 13 && current_hour < 22) session_mult = InpNYRiskMult;
   else if(current_hour >= 8 && current_hour < 13) session_mult = InpLondonRiskMult;

   // Smooth DD recovery multiplier
   double dd_mult = GetSmoothDrawdownMult();

   // Optional probability ramp multiplier
   double prob_mult = ProbRampMult(win_prob, session_thr);

   double final_risk_pct = base_risk_pct * session_mult * dd_mult * prob_mult;

   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_dollars = balance * (final_risk_pct / 100.0);

   double money_per_pip = MoneyPerPipPerLot();
   if(money_per_pip <= 0.0 || sl_pips <= 0.0) return InpLot;

   double lot = risk_dollars / (sl_pips * money_per_pip);

   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minv = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxv = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   if(step <= 0) step = 0.01;

   lot = MathFloor(lot / step) * step;
   lot = MathMax(lot, minv);
   lot = MathMin(lot, maxv);

   if(lot <= 0.0) lot = InpLot;
   return lot;
}

// -------------------- Trailing / partial manager with 1R tier scaling --------------------
double ScalePctForTier(int tier, int phase)
{
   if(phase == 1)
   {
      if(tier == 1) return InpSO1_T1;
      if(tier == 2) return InpSO1_T2;
      return InpSO1_T3;
   }
   if(phase == 2)
   {
      if(tier == 1) return InpSO2_T1;
      if(tier == 2) return InpSO2_T2;
      return InpSO2_T3;
   }
   if(phase == 3)
   {
      if(tier == 1) return InpSO3_T1;
      if(tier == 2) return InpSO3_T2;
      return InpSO3_T3;
   }
   return 0.0;
}

void TryPartialClose(ulong ticket, double pct)
{
   if(pct <= 0.0) return;

   double volume = PositionGetDouble(POSITION_VOLUME);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minv = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   if(step <= 0) step = 0.01;

   double v_close = MathFloor((volume * pct) / step) * step;
   if(v_close < minv) return;
   if((volume - v_close) < minv) return;

   trade.PositionClosePartial(ticket, v_close);
}

void ManageOpenPositions()
{
   if(PositionsTotal() == 0 || !InpUseTrailingManager) return;

   double pv = PipValue();
   double be_buf = InpBEPipsBuffer * pv;
   double min_stop = ((int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL)) * _Point;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   for(int i=PositionsTotal()-1; i>=0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      long magic = PositionGetInteger(POSITION_MAGIC);
      if(!IsMyMagic(magic)) continue;

      datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
      if(TimeCurrent() - open_time >= (96 * 15 * 60))
      {
         Print("Time Stop: 24h expired. Closing.");
         trade.PositionClose(ticket);
         GlobalVariableDel(StageKey(ticket));
         continue;
      }

      int tier = DecodeTier(magic);
      double risk_dist = DecodeRiskDist(magic); // price distance (not pips)
      if(risk_dist <= 0.0) continue;

      double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
      double current_sl = PositionGetDouble(POSITION_SL);
      double current_tp = PositionGetDouble(POSITION_TP);
      long type = PositionGetInteger(POSITION_TYPE);

      if(current_tp == 0.0) continue;

      double rr = (type == POSITION_TYPE_BUY) ? ((bid - open_price) / risk_dist)
                                             : ((open_price - ask) / risk_dist);

      int stage = GetStage(ticket);

      // Phase 1: >= 1R -> move SL to BE+buffer, partial by tier
      if(rr >= 1.0 && stage < 1)
      {
         double target_sl = (type == POSITION_TYPE_BUY) ? (open_price + be_buf) : (open_price - be_buf);
         target_sl = NormalizeDouble(target_sl, _Digits);

         bool valid = (type == POSITION_TYPE_BUY) ? (bid - target_sl > min_stop) : (target_sl - ask > min_stop);
         if(valid)
         {
            if(trade.PositionModify(ticket, target_sl, current_tp))
            {
               TryPartialClose(ticket, ScalePctForTier(tier, 1));
               SetStage(ticket, 1);
               PrintFormat("Trail P1 (1R): tier=%d SL->BE+buf partial=%.0f%%", tier, ScalePctForTier(tier,1)*100.0);
            }
         }
      }

      // Phase 2: >= 2R -> move SL to +1R, optional partial
      if(rr >= 2.0 && stage < 2)
      {
         double target_sl = (type == POSITION_TYPE_BUY) ? (open_price + risk_dist) : (open_price - risk_dist);
         target_sl = NormalizeDouble(target_sl, _Digits);

         bool valid = (type == POSITION_TYPE_BUY) ? (bid - target_sl > min_stop) : (target_sl - ask > min_stop);
         if(valid)
         {
            if(trade.PositionModify(ticket, target_sl, current_tp))
            {
               TryPartialClose(ticket, ScalePctForTier(tier, 2));
               SetStage(ticket, 2);
               PrintFormat("Trail P2 (2R): tier=%d SL->+1R partial=%.0f%%", tier, ScalePctForTier(tier,2)*100.0);
            }
         }
      }

      // Phase 3: >= 3R -> move SL to +2R, optional partial
      if(rr >= 3.0 && stage < 3)
      {
         double target_sl = (type == POSITION_TYPE_BUY) ? (open_price + (2.0*risk_dist)) : (open_price - (2.0*risk_dist));
         target_sl = NormalizeDouble(target_sl, _Digits);

         bool valid = (type == POSITION_TYPE_BUY) ? (bid - target_sl > min_stop) : (target_sl - ask > min_stop);
         if(valid)
         {
            if(trade.PositionModify(ticket, target_sl, current_tp))
            {
               TryPartialClose(ticket, ScalePctForTier(tier, 3));
               SetStage(ticket, 3);
               PrintFormat("Trail P3 (3R): tier=%d SL->+2R partial=%.0f%%", tier, ScalePctForTier(tier,3)*100.0);
            }
         }
      }
   }
}

// -------------------- Initialization --------------------
int OnInit()
{
   ResetLastError();

   g_dxy_ok = SymbolSelect(InpDxySymbol, true);
   LoadNewsIntoMemory();

   string fx = GetFxBaseSymbol(_Symbol);
   LoadMeta(fx);

   long shape_21D[] = {1, 21};

   classifier_handle = OnnxCreate("fvg_model_" + fx + "_v6.onnx", ONNX_DEFAULT);
   if(classifier_handle == INVALID_HANDLE)
   {
      PrintFormat("CRITICAL: Failed to load classifier for %s err=%d", fx, GetLastError());
      return INIT_FAILED;
   }
   OnnxSetInputShape(classifier_handle, 0, shape_21D);

   if(OnnxGetOutputCount(classifier_handle) >= 2)
   {
      long o0[] = {1};
      long o1[] = {1,2};
      OnnxSetOutputShape(classifier_handle, 0, o0);
      OnnxSetOutputShape(classifier_handle, 1, o1);
   }
   else
   {
      long o0[] = {1,2};
      OnnxSetOutputShape(classifier_handle, 0, o0);
   }

   regressor_handle = OnnxCreate("fvg_mfe_regressor_" + fx + "_v7.onnx", ONNX_DEFAULT);
   if(regressor_handle == INVALID_HANDLE)
   {
      PrintFormat("CRITICAL: Failed to load regressor for %s err=%d", fx, GetLastError());
      return INIT_FAILED;
   }
   OnnxSetInputShape(regressor_handle, 0, shape_21D);
   long ro[] = {1,1};
   OnnxSetOutputShape(regressor_handle, 0, ro);

   PrintFormat("V16.2 Loaded: %s base=%s thr=%.3f calib=%s",
               _Symbol, fx, g_threshold_global, (g_has_calib?"ON":"OFF"));

   return INIT_SUCCEEDED;
}

// -------------------- MAIN --------------------
void OnTick()
{
   ManageOpenPositions();

   if(Bars(_Symbol, PERIOD_M15) < 120 || Bars(_Symbol, PERIOD_H1) < 80 || Bars(_Symbol, PERIOD_D1) < 10) return;

   double pv = PipValue();

   // spread filter
   if(SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point > (InpMaxSpreadPips * pv)) return;

   static datetime last_h1_bar = 0;
   static datetime last_m15_bar = 0;

   datetime h1_time = iTime(_Symbol, PERIOD_H1, 0);
   datetime m15_time = iTime(_Symbol, PERIOD_M15, 0);

   // --- H1 update ---
   if(h1_time != last_h1_bar)
   {
      last_h1_bar = h1_time;

      double h1_c1 = iClose(_Symbol, PERIOD_H1, 1);
      double h1_h1 = iHigh(_Symbol, PERIOD_H1, 1);
      double h1_l1 = iLow(_Symbol, PERIOD_H1, 1);
      double h1_atr = CalculatePythonATR(_Symbol, PERIOD_H1, 14, 1);

      if(g_active_bull_bottom > 0 && h1_c1 < g_active_bull_bottom) { g_active_bull_top = 0; g_active_bull_bottom = 0; }
      if(g_active_bear_top > 0 && h1_c1 > g_active_bear_top) { g_active_bear_bottom = 0; g_active_bear_top = 0; }

      double gap_bull = iLow(_Symbol, PERIOD_H1, 1) - iHigh(_Symbol, PERIOD_H1, 3);
      double gap_bear = iLow(_Symbol, PERIOD_H1, 3) - iHigh(_Symbol, PERIOD_H1, 1);

      bool bull_disp = iClose(_Symbol, PERIOD_H1, 2) > iClose(_Symbol, PERIOD_H1, 3);
      bool bear_disp = iClose(_Symbol, PERIOD_H1, 2) < iClose(_Symbol, PERIOD_H1, 3);

      if(gap_bull > 0.0 && bull_disp)
      {
         g_active_bull_top = iLow(_Symbol, PERIOD_H1, 1);
         g_active_bull_bottom = iHigh(_Symbol, PERIOD_H1, 3);
         g_active_bull_size = gap_bull / pv;
         g_active_bull_ratio = (h1_atr > 0.0) ? (gap_bull / h1_atr) : 0.0;
      }

      if(gap_bear > 0.0 && bear_disp)
      {
         g_active_bear_bottom = iHigh(_Symbol, PERIOD_H1, 1);
         g_active_bear_top = iLow(_Symbol, PERIOD_H1, 3);
         g_active_bear_size = gap_bear / pv;
         g_active_bear_ratio = (h1_atr > 0.0) ? (gap_bear / h1_atr) : 0.0;
      }

      g_h1_rsi = CalculatePythonRSI(_Symbol, PERIOD_H1, 14, 1);
      double h1_ema = CalculatePythonEMA(_Symbol, PERIOD_H1, 50, 1);
      g_h1_dist = (h1_ema > 0.0) ? (h1_c1 - h1_ema) / pv : 0.0;

      double pdh = iHigh(_Symbol, PERIOD_D1, 1);
      double pdl = iLow(_Symbol, PERIOD_D1, 1);
      g_h1_dist_pdh = (h1_h1 - pdh) / pv;
      g_h1_dist_pdl = (h1_l1 - pdl) / pv;

      double hi5 = h1_h1, lo5 = h1_l1;
      for(int i=1; i<=5; i++)
      {
         hi5 = MathMax(hi5, iHigh(_Symbol, PERIOD_H1, i));
         lo5 = MathMin(lo5, iLow(_Symbol, PERIOD_H1, i));
      }
      g_h1_swept_pdh = (hi5 >= pdh) ? 1.0 : 0.0;
      g_h1_swept_pdl = (lo5 <= pdl) ? 1.0 : 0.0;
   }

   // --- M15 eval ---
   if(m15_time != last_m15_bar)
   {
      last_m15_bar = m15_time;

      double m15_o1 = iOpen(_Symbol, PERIOD_M15, 1);
      double m15_c1 = iClose(_Symbol, PERIOD_M15, 1);
      double m15_h1 = iHigh(_Symbol, PERIOD_M15, 1);
      double m15_l1 = iLow(_Symbol, PERIOD_M15, 1);

      bool trade_bull = (g_active_bull_top > 0 && m15_l1 <= g_active_bull_top);
      bool trade_bear = (!trade_bull && g_active_bear_bottom > 0 && m15_h1 >= g_active_bear_bottom);

      if(!(trade_bull || trade_bear)) return;
      if(!IsNewsTradingAllowed()) return;
      if(CountMyPositions() > 0) return;

      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

      double fvg_top = trade_bull ? g_active_bull_top : g_active_bear_top;
      double fvg_bot = trade_bull ? g_active_bull_bottom : g_active_bear_bottom;
      double total_gap = fvg_top - fvg_bot;

      double wick_pen = 0.0;
      if(total_gap > 0.0)
         wick_pen = trade_bull ? ((fvg_top - m15_l1) / total_gap) : ((m15_h1 - fvg_bot) / total_gap);

      if(wick_pen < InpMinFillPct || wick_pen > InpMaxFillPct)
      {
         if(trade_bull) { g_active_bull_top=0; g_active_bull_bottom=0; }
         if(trade_bear) { g_active_bear_top=0; g_active_bear_bottom=0; }
         return;
      }

      double entry_dist_pips = trade_bull ? ((ask - fvg_top) / pv) : ((fvg_bot - bid) / pv);
      if(entry_dist_pips > InpMaxEntryDistance)
      {
         if(trade_bull) { g_active_bull_top=0; g_active_bull_bottom=0; }
         if(trade_bear) { g_active_bear_top=0; g_active_bear_bottom=0; }
         return;
      }

      double c_rng = m15_h1 - m15_l1;
      double body_pct=0, u_wick=0, d_wick=0;
      if(c_rng > 0.0)
      {
         body_pct = MathAbs(m15_o1 - m15_c1)/c_rng;
         u_wick   = (m15_h1 - MathMax(m15_o1, m15_c1))/c_rng;
         d_wick   = (MathMin(m15_o1, m15_c1) - m15_l1)/c_rng;
      }

      double m15_atr = CalculatePythonATR(_Symbol, PERIOD_M15, 14, 1);
      if(m15_atr <= 0.0) return;

      double h4_ema = CalculatePythonEMA(_Symbol, PERIOD_H4, 50, 1);
      float h4_dist = (h4_ema > 0.0) ? (float)((iClose(_Symbol, PERIOD_H4, 1) - h4_ema)/pv) : 0.0f;
      float h4_atr  = (float)(CalculatePythonATR(_Symbol, PERIOD_H4, 14, 1)/pv);

      float dxy_rsi = 50.0f, dxy_dist = 0.0f;
      if(g_dxy_ok && SymbolInfoDouble(InpDxySymbol, SYMBOL_BID) > 0.0)
      {
         double dxy_ema = CalculatePythonEMA(InpDxySymbol, PERIOD_M15, 50, 1);
         if(dxy_ema > 0.0)
         {
            dxy_rsi  = (float)CalculatePythonRSI(InpDxySymbol, PERIOD_M15, 14, 1);
            dxy_dist = (float)(iClose(InpDxySymbol, PERIOD_M15, 1) - dxy_ema);
         }
      }

      float is_asia, is_london, is_ny;
      int hour_now=0;
      GetSessionFlags(m15_time, is_asia, is_london, is_ny, hour_now);

      double session_thr = GetSessionThreshold(is_asia, is_london, is_ny);

      // 21D features
      float X[1][21];
      X[0][0]  = (float)iTickVolume(_Symbol, PERIOD_M15, 1);
      X[0][1]  = (float)CalculatePythonRSI(_Symbol, PERIOD_M15, 14, 1);
      X[0][2]  = (float)(m15_atr / pv);
      X[0][3]  = trade_bull ? (float)g_active_bull_ratio : 0.0f;
      X[0][4]  = trade_bear ? (float)g_active_bear_ratio : 0.0f;
      X[0][5]  = (float)body_pct;
      X[0][6]  = (float)u_wick;
      X[0][7]  = (float)d_wick;
      X[0][8]  = (float)g_h1_rsi;
      X[0][9]  = (float)g_h1_dist;
      X[0][10] = (float)g_h1_dist_pdh;
      X[0][11] = (float)g_h1_dist_pdl;
      X[0][12] = (float)g_h1_swept_pdh;
      X[0][13] = (float)g_h1_swept_pdl;
      X[0][14] = is_asia;
      X[0][15] = is_london;
      X[0][16] = is_ny;
      X[0][17] = h4_dist;
      X[0][18] = h4_atr;
      X[0][19] = dxy_rsi;
      X[0][20] = dxy_dist;

      // classifier
      float probs[1][2];
      ArrayInitialize(probs, 0.0f);

      bool ok=false;
      if(OnnxGetOutputCount(classifier_handle) >= 2)
      {
         long label[1];
         ok = OnnxRun(classifier_handle, ONNX_NO_CONVERSION, X, label, probs);
      }
      else ok = OnnxRun(classifier_handle, ONNX_NO_CONVERSION, X, probs);
      if(!ok) return;

      double raw_p = (double)probs[0][1];
      double p = CalibrateProb(raw_p);

      int tier = TierFromProb(p, session_thr);
      if(tier == 0) return;

      // regressor
      float mfe[1][1];
      if(!OnnxRun(regressor_handle, ONNX_NO_CONVERSION, X, mfe)) return;

      double atr_pips = m15_atr / pv;
      double h1_fvg_size = trade_bull ? g_active_bull_size : g_active_bear_size;

      double sl_pips = h1_fvg_size + (atr_pips * InpATRMultiplier);
      if(sl_pips < InpMinSLPips) sl_pips = InpMinSLPips;
      if(sl_pips > InpMaxSLPips) sl_pips = InpMaxSLPips;

      double tp_pips = (double)mfe[0][0] * InpMFEDampener;
      if(tp_pips > sl_pips * InpMaxRR) tp_pips = sl_pips * InpMaxRR;
      if(tp_pips < 10.0) tp_pips = 10.0;

      double sl_dist_price = sl_pips * pv;
      double tp_dist_price = tp_pips * pv;
      double min_stop = ((int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL)) * _Point;

      if(sl_dist_price < min_stop || tp_dist_price < min_stop) return;

      // lot sizing: probability + session + drawdown recovery
      double lot = CalculateDynamicLot(sl_pips, tier, hour_now, p, session_thr);

      // encode magic for manager (tier + risk_pts)
      int risk_pts = (int)MathRound(sl_dist_price / _Point);
      if(risk_pts < 1) risk_pts = 1;
      long magic = InpBaseMagic + (tier * 100000) + risk_pts;
      trade.SetExpertMagicNumber(magic);

      PrintFormat("V16.2 EXEC: raw=%.3f cal=%.3f thr=%.3f tier=%d lot=%.2f SL=%.1f TP=%.1f DDmult=%.2f",
                  raw_p, p, session_thr, tier, lot, sl_pips, tp_pips, GetSmoothDrawdownMult());

      if(trade_bull)
         trade.Buy(lot, _Symbol, ask,
                   NormalizeDouble(ask - sl_dist_price, _Digits),
                   NormalizeDouble(ask + tp_dist_price, _Digits));
      else
         trade.Sell(lot, _Symbol, bid,
                    NormalizeDouble(bid + sl_dist_price, _Digits),
                    NormalizeDouble(bid - tp_dist_price, _Digits));

      // reset FVG after placing
      if(trade_bull) { g_active_bull_top=0; g_active_bull_bottom=0; }
      else          { g_active_bear_top=0; g_active_bear_bottom=0; }
   }
}

void OnDeinit(const int reason)
{
   if(classifier_handle != INVALID_HANDLE) OnnxRelease(classifier_handle);
   if(regressor_handle  != INVALID_HANDLE) OnnxRelease(regressor_handle);

   // optional: do not delete global stage vars; they expire only if you delete manually
}