//+------------------------------------------------------------------+
//|                                                FVG_Portfolio.mq5 |
//|                             Production FVG Portfolio Engine V1.0 |
//+------------------------------------------------------------------+
#property copyright "Senior Quant Engineering"
#property version   "1.00"

#include <FVG_Portfolio/Core/FVG_Detector.mqh>

#include <FVG_Portfolio/Risk/RiskEngine.mqh>

#include <FVG_Portfolio/Data/Logger.mqh>
#include <FVG_Portfolio/Data/RegimeReader.mqh>

#include <FVG_Portfolio/Strategies/TrendFVGStrategy.mqh>
#include <FVG_Portfolio/Strategies/MeanReversionFVGStrategy.mqh>
#include <FVG_Portfolio/Strategies/BreakoutMicroFVGStrategy.mqh>
#include <FVG_Portfolio/Strategies/SessionGate.mqh>

#include <FVG_Portfolio/Core/IndicatorCache.mqh>
#include <FVG_Portfolio/Core/NewBarGate.mqh>

#include <FVG_Portfolio/Execution/Execution.mqh>
#include <FVG_Portfolio/Execution/OrderGate.mqh>
#include <FVG_Portfolio/Execution/TradeTracker.mqh>

input double InpBaseRiskPercent = 0.5;      // Base Risk per trade (%)
input double InpMaxSymbolRisk   = 1.0;      // Max Risk per Symbol (%)
input double InpMaxPortfolioRisk = 5.0;     // Max Open Risk (%)
input string InpSymbols         = "EURUSD,GBPJPY,XAUUSD,NAS100";
input string InpTimeframes      = "M15,H1";
input int    InpBaseMagic       = 80000;    // Base Magic Number
input int    InpMaxSpreadPoints = 30;       // Max Spread Limit
input double InpSLBufferATRTrend    = 1.5;  
input double InpSLBufferATRBreakout = 2.0;  
input double InpSLBufferATRMeanRev  = 0.5;  
input int    InpPendingExpiryMins   = 60;   
input int    InpMinReplacePoints    = 50;   
input int    InpHistoryLookbackDays = 30;   // <-- NEW: History Lookback Days

string          ActiveSymbols[];
ENUM_TIMEFRAMES ActiveTimeframes[];


CRiskEngine    Risk;

CLogger        Logger;
CRegimeReader  Regime;

CTrendFVGStrategy          StrategyTrend;
CMeanReversionFVGStrategy  StrategyMeanRev;
CBreakoutMicroFVGStrategy  StrategyBreakout;

CIndicatorCache Indicators;
CNewBarGate     BarGate;

CExecutionEngine Execution;
COrderGate OrderGate;
CTradeTracker   Tracker;


// Helper to convert string to TF
ENUM_TIMEFRAMES GetTimeframeFromString(string tf_str) {
    StringTrimLeft(tf_str); StringTrimRight(tf_str);
    if(tf_str == "M1") return PERIOD_M1;
    if(tf_str == "M5") return PERIOD_M5;
    if(tf_str == "M15") return PERIOD_M15;
    if(tf_str == "M30") return PERIOD_M30;
    if(tf_str == "H1") return PERIOD_H1;
    if(tf_str == "H4") return PERIOD_H4;
    if(tf_str == "D1") return PERIOD_D1;
    return PERIOD_H1; // Default
}


// Simple deterministic hash for Magic Number
long GenerateMagic(string symbol, ENUM_TIMEFRAMES tf, string strat_id) {
    long hash = 0;
    string combined = symbol + IntegerToString(tf) + strat_id;
    for(int i = 0; i < StringLen(combined); i++) {
        hash += StringGetCharacter(combined, i);
    }
    return InpBaseMagic + hash;
}


int OnInit() {
    Print("Initializing FVG Portfolio Engine...");
    
    // 1) Fix Input Parsing (Comma delimiter, trim, remove empty)
    string raw_symbols[];
    int count_sym = StringSplit(InpSymbols, ',', raw_symbols);
    int valid_s = 0;
    ArrayResize(ActiveSymbols, count_sym);
    for(int i = 0; i < count_sym; i++) {
        StringTrimLeft(raw_symbols[i]); StringTrimRight(raw_symbols[i]);
        if(StringLen(raw_symbols[i]) > 0) ActiveSymbols[valid_s++] = raw_symbols[i];
    }
    ArrayResize(ActiveSymbols, valid_s);
    
    string raw_tfs[];
    int count_tf = StringSplit(InpTimeframes, ',', raw_tfs);
    int valid_t = 0;
    ArrayResize(ActiveTimeframes, count_tf);
    for(int i = 0; i < count_tf; i++) {
        StringTrimLeft(raw_tfs[i]); StringTrimRight(raw_tfs[i]);
        if(StringLen(raw_tfs[i]) > 0) ActiveTimeframes[valid_t++] = GetTimeframeFromString(raw_tfs[i]);
    }
    ArrayResize(ActiveTimeframes, valid_t);

    // Init Strategies with ATR multipliers
    StrategyTrend.Init(InpSLBufferATRTrend);
    StrategyMeanRev.Init(InpSLBufferATRMeanRev);
    StrategyBreakout.Init(InpSLBufferATRBreakout);

    Indicators.Init(ActiveSymbols, ActiveTimeframes);
    Execution.Init(InpMaxSpreadPoints, InpPendingExpiryMins, InpMinReplacePoints);
    Risk.Init(InpBaseRiskPercent, InpMaxSymbolRisk, InpMaxPortfolioRisk);
    
    if(!Logger.Init("FVG_Logs.csv")) return INIT_FAILED;
    
    // Inject logger and lookback days into tracker
    Tracker.Init(&Logger, InpHistoryLookbackDays); 
    
    Regime.Init("Regime_Signals.csv");
    EventSetTimer(60); 
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
    EventKillTimer();
    Indicators.Deinit();
    Logger.Close();
    Print("FVG Portfolio Engine Shutdown.");
}

// --- New Event Handler ---
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result) {
                        
    // Pass transaction stream to the tracker for lifecycle logging
    Tracker.UpdateFromTradeTransaction(trans, request, result);
}

void OnTick() {
    // Update MAE/MFE tracking on every tick. 
    // Assuming magic numbers fall within base_magic and base_magic + 999999
    Tracker.UpdateOpenPositions(InpBaseMagic, InpBaseMagic + 999999);

    string current_regime = Regime.GetLatestRegime(); 
    ActiveSessions active_sessions = CSessionGate::GetActiveStrategies();

    for(int s = 0; s < ArraySize(ActiveSymbols); s++) {
        for(int t = 0; t < ArraySize(ActiveTimeframes); t++) {
            
            string sym = ActiveSymbols[s];
            ENUM_TIMEFRAMES tf = ActiveTimeframes[t];
            
            MqlRates rates[];
            ArraySetAsSeries(rates, true);
            if(CopyRates(sym, tf, 0, 100, rates) < 4) continue; 
            
            // New Bar Check to prevent overtrading
            if(!BarGate.IsNewBar(sym, tf, rates[1].time)) continue;
            
            FVG latest_fvg;
            if(!CFVGDetector::Detect(rates, 1, latest_fvg)) continue;

            double ema_50 = 0.0, atr_14 = 0.0;
            if(!Indicators.GetValues(sym, tf, 1, ema_50, atr_14)) continue;
            
            Signal best_signal;
            best_signal.direction = 0;
            best_signal.confidence = 0.0;

            if(active_sessions.allow_trend) {
                Signal sig = StrategyTrend.Evaluate(sym, tf, rates, latest_fvg, current_regime, ema_50, atr_14);
                if(sig.direction != 0 && sig.confidence > best_signal.confidence) best_signal = sig;
            }
            if(active_sessions.allow_mean_rev) {
                // Assuming MeanReversionFVGStrategy Evaluate signature was updated similarly
                Signal sig = StrategyMeanRev.Evaluate(sym, tf, rates, latest_fvg, current_regime, ema_50, atr_14);
                if(sig.direction != 0 && sig.confidence > best_signal.confidence) best_signal = sig;
            }
            if(active_sessions.allow_breakout) {
                Signal sig = StrategyBreakout.Evaluate(sym, tf, rates, latest_fvg, current_regime, ema_50, atr_14);
                if(sig.direction != 0 && sig.confidence > best_signal.confidence) best_signal = sig;
            }

            if(best_signal.direction != 0) {
                long magic = GenerateMagic(sym, tf, best_signal.strategy_id);
                ulong cancel_ticket = 0;
                
                ENUM_ORDER_ACTION action = OrderGate.CheckAction(sym, magic, best_signal.entry, InpMinReplacePoints, cancel_ticket);
                
                int spread = (int)SymbolInfoInteger(sym, SYMBOL_SPREAD);
                int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);

                if(action == ACTION_SKIP) {
                    Logger.LogDecision(sym, EnumToString(tf), best_signal.strategy_id, magic, current_regime,
                                       latest_fvg.top, latest_fvg.bottom, best_signal.direction, 
                                       best_signal.entry, best_signal.sl, best_signal.tp, 0.0, 
                                       spread, atr_14, "ORDER_SKIPPED", "Duplicate Position/Order", digits);
                    continue;
                }

                double lot_size = Risk.CalculateLotSize(sym, best_signal.sl, best_signal.entry, best_signal.direction, atr_14);
                
                if(lot_size > 0.0) {
                    string event_type = (action == ACTION_REPLACE) ? "ORDER_REPLACED" : "ORDER_PLACED";
                    ExecResult exec_res = Execution.ExecuteSignal(sym, best_signal, lot_size, magic, cancel_ticket);
                    
                    Logger.LogDecision(sym, EnumToString(tf), best_signal.strategy_id, magic, current_regime,
                                       latest_fvg.top, latest_fvg.bottom, best_signal.direction, 
                                       best_signal.entry, best_signal.sl, best_signal.tp, lot_size, 
                                       spread, atr_14, event_type, exec_res.comment, digits);
                } else {
                    Logger.LogDecision(sym, EnumToString(tf), best_signal.strategy_id, magic, current_regime,
                                       latest_fvg.top, latest_fvg.bottom, best_signal.direction, 
                                       best_signal.entry, best_signal.sl, best_signal.tp, 0.0, 
                                       spread, atr_14, "DECISION", "Risk Blocked/Zero Lot", digits);
                }
            }
        }
    }
}


void OnTimer() {
    Regime.Update();
}