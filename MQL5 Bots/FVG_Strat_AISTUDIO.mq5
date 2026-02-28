//+------------------------------------------------------------------+
//|                                    ICT_FVG_LiquiditySweep_EA.mq5 |
//|                                       Copyright 2024, Expert Dev |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Expert Dev"
#property link      "https://www.mql5.com"
#property version   "17.00"

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\PositionInfo.mqh>

//--- Enums
enum ENUM_VOLUME_MODE {
   VOLUME_FIXED,
   VOLUME_MANAGED,
   VOLUME_PERCENT,
   VOLUME_MONEY
};

enum ENUM_CALC_MODE {
   CALC_MODE_OFF,
   CALC_MODE_FACTOR,
   CALC_MODE_POINTS
};

enum ENUM_SL_MODE {
   SL_MODE_SWING,
   SL_MODE_ATR,
   SL_MODE_POINTS
};

//--- Inputs
input group "<Multi-Timeframe Settings>"
input ENUM_TIMEFRAMES    InpHTF                  = PERIOD_M15;
input ENUM_TIMEFRAMES    InpLTF                  = PERIOD_M5;

input group "<Trading Volume Modes>"
input ENUM_VOLUME_MODE   InpVolumeMode           = VOLUME_PERCENT;
input double             InpFixedLots            = 0.1;
input double             InpFixedLotsPerXMoney   = 1000;
input double             InpRiskPercent          = 3.0;     
input double             InpRiskMoney            = 100.0;
input int                InpOrderBufferPoints    = 0;

input group "<Dynamic Risk Management>"
input bool               InpEnableDynamicRisk    = true;
input double             InpDrawdownThresholdPct = 5.0;
input double             InpRecoveryTargetPct    = 75.0;

input group "<Stop Loss Settings>"
input ENUM_SL_MODE       InpStopLossMode         = SL_MODE_SWING;
input double             InpStopValuePoints      = 150;
input int                InpAtrPeriod            = 14;
input double             InpAtrMultiplier        = 1.5;
input double             InpSwingBufferPoints    = 10;

input group "<Take Profit Settings>"
input ENUM_CALC_MODE     InpTargetCalcMode       = CALC_MODE_FACTOR;
input double             InpTargetValue          = 0.2;

input group "<Risk Management: Spread Filter>"
input int                InpMaxSpreadPoints      = 30;

input group "<Risk Management: Break Even (RR Based)>"
input bool               InpEnableBE             = true;
input double             InpBETriggerRR          = 0.5;
input double             InpBEBufferPoints       = 10;

input group "<Risk Management: Partial TP (RR Based)>"
input bool               InpEnablePartialTP      = true;
input double             InpPartialTPTriggerRR   = 1.0;
input double             InpPartialTPVolumePct   = 50.0;

input group "<Risk Management: Trailing Stop (RR Based)>"
input bool               InpEnableTSL            = true;
input double             InpTslTriggerRR         = 1.0;
input double             InpTslDistanceRR        = 0.5;
input double             InpTslStepRR            = 0.1;

input group "<Time Settings>"
input int                InpTradeStartHour       = 0;
input int                InpTradeStartMinute     = 0;
input int                InpTradeEndHour         = 23;
input int                InpTradeEndMinute       = 59;

input group "<Trading Frequency Settings>"
input int                InpMaxTotalTrades       = 10;

input group "<ICT Strategy Settings>"
input int                InpSwingStrength          = 10;
input double             InpMinSwingSizePoints     = 50;
input double             InpMinSweepBreakPoints    = 10;
input bool               InpRequireCloseBack       = true;
input int                InpMaxCandlesOutsideSweep = 2;
input double             InpMinFVGSizePoints       = 5;
input int                InpMaxBarsAfterSweep      = 8;
input double             InpFVGEntryPercent        = 50.0; // 0% = Edge, 50% = Mid, 100% = Far Edge
input bool               InpCloseOnOppositeSweep   = true;

input group "<Hammer Criteria Settings>"
input double             InpMinWickMultiplier      = 1.5; // Main wick must be X times the body 
input double             InpMaxOppositeWickRatio   = 1.5; // Opposite wick can be up to X times the body 

input group "<More Settings>"
input color              InpBullishColor         = clrLimeGreen;
input color              InpBearishColor         = clrCrimson;
input color              InpSweepColor           = clrDodgerBlue;
input string             InpOrderComment         = "ICT_Breakout";
input ulong              InpMagicNumber          = 777777;
input bool               InpDebugMode            = true;

//--- Global Variables
CTrade         trade;
CSymbolInfo    symb;
CPositionInfo  posi;

int            atrHandle;

datetime       lastBarTimeHTF = 0;
datetime       lastBarTimeLTF = 0;

// Risk Management Tracking
ulong          processedPartials[];
double         highestBalance = 0;
double         referenceBalance = 0;
double         recoveryTarget = 0;
double         currentRiskMultiplier = 1.0;

// Initial risk cache (posId -> initialRiskPts)
ulong          riskPosIds[];
double         riskPts[];

// Swing Point Tracking (HTF)
double         activeSwingHigh = 0;
datetime       activeSwingHighTime = 0;
bool           activeSwingHighSwept = true;

double         activeSwingLow = 0;
datetime       activeSwingLowTime = 0;
bool           activeSwingLowSwept = true;

// Sweep State
int            barsSinceSweep = -1;
int            sweepDirection = 0;
double         sweepExtremePrice = 0;
datetime       sweepTime = 0;

struct FVG_Data {
   datetime time;
   int direction;
   double high;
   double low;
   bool traded;
};
FVG_Data currentFVG;

int tradesTodayTotal = 0;
datetime currentDay = 0;

//+------------------------------------------------------------------+
int OnInit() {
   symb.Name(_Symbol);
   symb.Refresh();
   trade.SetExpertMagicNumber(InpMagicNumber);

   uint filling = (uint)SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((filling & SYMBOL_FILLING_FOK) != 0) trade.SetTypeFilling(ORDER_FILLING_FOK);
   else if((filling & SYMBOL_FILLING_IOC) != 0) trade.SetTypeFilling(ORDER_FILLING_IOC);
   else trade.SetTypeFilling(ORDER_FILLING_RETURN);

   atrHandle = iATR(_Symbol, InpLTF, InpAtrPeriod);
   if(atrHandle == INVALID_HANDLE) Print("Failed to initialize ATR indicator.");

   highestBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   referenceBalance = highestBalance;
   currentRiskMultiplier = 1.0;

   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
   ObjectsDeleteAll(0, "ICT_");
   if(atrHandle != INVALID_HANDLE) IndicatorRelease(atrHandle);
}

void OnTick() {
   if(!symb.RefreshRates()) return;

   UpdateDrawdownState();

   datetime today = iTime(_Symbol, PERIOD_D1, 0);
   if(today != currentDay) {
      currentDay = today;
      tradesTodayTotal = 0;
      ArrayFree(processedPartials);
   }

   ManageBreakEven();
   ManagePartialTP();
   ManageTrailingStop();

   if(!IsTradingTime()) return;

   datetime currentBarTimeHTF = iTime(_Symbol, InpHTF, 0);
   if(currentBarTimeHTF != lastBarTimeHTF) {
      lastBarTimeHTF = currentBarTimeHTF;
      UpdateSwingPoints();
      if(sweepDirection == 0) DetectLiquiditySweep();
   }

   datetime currentBarTimeLTF = iTime(_Symbol, InpLTF, 0);
   if(currentBarTimeLTF != lastBarTimeLTF) {
      lastBarTimeLTF = currentBarTimeLTF;

      if(sweepDirection != 0) {
         barsSinceSweep++;
         if(barsSinceSweep > InpMaxBarsAfterSweep) ResetSweepState();
         else DetectFVG(); // Fallback if displacement FVG wasn't found instantly
      }

      // Check for Market Entry condition (Candle close + Hammer + Wick Touch)
      if(currentFVG.time != 0 && !currentFVG.traded) {
         CheckMarketEntry();
      }
   }
}

//+------------------------------------------------------------------+
//| Initial Risk Cache                                               |
//+------------------------------------------------------------------+
int FindRiskIndexByPosId(ulong posId) {
   int n = ArraySize(riskPosIds);
   for(int i=0;i<n;i++) if(riskPosIds[i] == posId) return i;
   return -1;
}

double GetInitialRiskPoints(ulong posId, double openPrice, double initialSL) {
   int idx = FindRiskIndexByPosId(posId);
   if(idx >= 0) return riskPts[idx];

   double pts = 0;
   if(initialSL > 0 && openPrice > 0) pts = MathAbs(openPrice - initialSL) / _Point;
   if(pts <= 0) pts = 100.0;

   int n = ArraySize(riskPosIds);
   ArrayResize(riskPosIds, n+1);
   ArrayResize(riskPts, n+1);
   riskPosIds[n] = posId;
   riskPts[n] = pts;
   return pts;
}

//+------------------------------------------------------------------+
//| Risk money at SL using OrderCalcProfit                           |
//+------------------------------------------------------------------+
double LossMoneyAtSLPerLot(int direction, double entryPrice, double slPrice) {
   double profit = 0.0;
   ENUM_ORDER_TYPE type = (direction == 1) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   if(!OrderCalcProfit(type, _Symbol, 1.0, entryPrice, slPrice, profit)) {
      return -1.0;
   }
   return MathAbs(profit);
}

//+------------------------------------------------------------------+
//| Lot sizing with HARD CAP: expected loss at SL <= riskAmount      |
//+------------------------------------------------------------------+
double CalculateLotSize(double entryPrice, double sl, int direction) {
   double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   double lot = InpFixedLots;
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double riskAmount = 0.0;

   if(InpVolumeMode == VOLUME_PERCENT) {
      riskAmount = equity * (InpRiskPercent / 100.0);
   } else if(InpVolumeMode == VOLUME_MONEY) {
      riskAmount = InpRiskMoney;
   } else if(InpVolumeMode == VOLUME_MANAGED) {
      lot = InpFixedLots * MathFloor(equity / InpFixedLotsPerXMoney);
      lot = lot * GetDynamicRiskMultiplier();
      return NormalizeLot(lot);
   } else {
      lot = InpFixedLots * GetDynamicRiskMultiplier();
      return NormalizeLot(lot);
   }

   riskAmount *= GetDynamicRiskMultiplier();

   if(entryPrice <= 0 || sl <= 0 || riskAmount <= 0) {
      return NormalizeLot(InpFixedLots * GetDynamicRiskMultiplier());
   }

   double lossPerLot = LossMoneyAtSLPerLot(direction, entryPrice, sl);
   if(lossPerLot <= 0) {
      return NormalizeLot(InpFixedLots * GetDynamicRiskMultiplier());
   }

   lot = riskAmount / lossPerLot;
   lot = MathMax(minLot, MathMin(maxLot, lot));
   lot = MathFloor(lot / stepLot) * stepLot;

   double expectedLoss = lot * lossPerLot;

   int guard = 0;
   while(expectedLoss > riskAmount && lot > minLot && guard < 1000) {
      lot -= stepLot;
      lot = MathMax(minLot, lot);
      expectedLoss = lot * lossPerLot;
      guard++;
   }

   double marginRequired = 0;
   ENUM_ORDER_TYPE orderType = (direction == 1) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   if(OrderCalcMargin(orderType, _Symbol, lot, entryPrice, marginRequired)) {
      double maxUsableMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE) * 0.95;
      if(marginRequired > maxUsableMargin && lot > minLot) {
         double ratio = maxUsableMargin / marginRequired;
         lot = lot * ratio;
         lot = MathMax(minLot, MathMin(maxLot, lot));
         lot = MathFloor(lot / stepLot) * stepLot;

         expectedLoss = lot * lossPerLot;
         guard = 0;
         while(expectedLoss > riskAmount && lot > minLot && guard < 1000) {
            lot -= stepLot;
            lot = MathMax(minLot, lot);
            expectedLoss = lot * lossPerLot;
            guard++;
         }
      }
   }

   if(InpDebugMode) {
      PrintFormat(">>> RISK DEBUG | Equity=%.2f RiskAmt=%.2f LossPerLot=%.2f Lot=%.2f ExpLoss=%.2f (dir=%d)",
                  equity, riskAmount, lossPerLot, lot, expectedLoss, direction);
   }

   return lot;
}

double NormalizeLot(double lot) {
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   lot = MathMax(minLot, MathMin(maxLot, lot));
   return MathFloor(lot / stepLot) * stepLot;
}

//+------------------------------------------------------------------+
//| Risk Management: Break Even (RR Based)                           |
//+------------------------------------------------------------------+
void ManageBreakEven() {
   if(!InpEnableBE) return;

   for(int i=PositionsTotal()-1; i>=0; i--) {
      if(posi.SelectByIndex(i) && posi.Symbol() == _Symbol && posi.Magic() == InpMagicNumber) {

         double initialRiskPts = GetInitialRiskPoints(posi.Identifier(), posi.PriceOpen(), posi.StopLoss());
         if(initialRiskPts <= 0) continue;

         double currentPrice = (posi.PositionType() == POSITION_TYPE_BUY) ? symb.Bid() : symb.Ask();
         double openPrice = posi.PriceOpen();
         double currentSL = posi.StopLoss();
         double profitPoints = MathAbs(currentPrice - openPrice) / _Point;
         double currentRR = profitPoints / initialRiskPts;

         if(posi.PositionType() == POSITION_TYPE_BUY && currentPrice > openPrice) {
            if(currentRR >= InpBETriggerRR) {
               double bePrice = openPrice + (InpBEBufferPoints * _Point);
               if(currentSL < bePrice) {
                  trade.PositionModify(posi.Ticket(), NormalizeDouble(bePrice, _Digits), posi.TakeProfit());
               }
            }
         } else if(posi.PositionType() == POSITION_TYPE_SELL && currentPrice < openPrice) {
            if(currentRR >= InpBETriggerRR) {
               double bePrice = openPrice - (InpBEBufferPoints * _Point);
               if(currentSL > bePrice || currentSL == 0) {
                  trade.PositionModify(posi.Ticket(), NormalizeDouble(bePrice, _Digits), posi.TakeProfit());
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Risk Management: Partial Take Profit (RR Based)                  |
//+------------------------------------------------------------------+
bool IsPartialTaken(ulong posId) {
   int size = ArraySize(processedPartials);
   for(int i=0; i<size; i++) if(processedPartials[i] == posId) return true;
   return false;
}

void MarkPartialTaken(ulong posId) {
   int size = ArraySize(processedPartials);
   ArrayResize(processedPartials, size + 1);
   processedPartials[size] = posId;
}

void ManagePartialTP() {
   if(!InpEnablePartialTP) return;

   for(int i=PositionsTotal()-1; i>=0; i--) {
      if(posi.SelectByIndex(i) && posi.Symbol() == _Symbol && posi.Magic() == InpMagicNumber) {
         ulong posId = posi.Identifier();
         if(IsPartialTaken(posId)) continue;

         double initialRiskPts = GetInitialRiskPoints(posId, posi.PriceOpen(), posi.StopLoss());
         if(initialRiskPts <= 0) continue;

         double currentPrice = (posi.PositionType() == POSITION_TYPE_BUY) ? symb.Bid() : symb.Ask();
         double openPrice = posi.PriceOpen();

         double profitPoints = 0;
         if(posi.PositionType() == POSITION_TYPE_BUY && currentPrice > openPrice) profitPoints = (currentPrice - openPrice) / _Point;
         if(posi.PositionType() == POSITION_TYPE_SELL && currentPrice < openPrice) profitPoints = (openPrice - currentPrice) / _Point;

         double currentRR = profitPoints / initialRiskPts;

         if(currentRR >= InpPartialTPTriggerRR) {
            double currentVol = posi.Volume();
            double closeVol = NormalizeLot(currentVol * (InpPartialTPVolumePct / 100.0));

            if(closeVol >= SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN)) {
               if(trade.PositionClosePartial(posi.Ticket(), closeVol)) {
                  MarkPartialTaken(posId);
               }
            } else {
               MarkPartialTaken(posId);
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Risk Management: Trailing Stop (RR Based)                        |
//+------------------------------------------------------------------+
void ManageTrailingStop() {
   if(!InpEnableTSL) return;

   for(int i=PositionsTotal()-1; i>=0; i--) {
      if(posi.SelectByIndex(i) && posi.Symbol() == _Symbol && posi.Magic() == InpMagicNumber) {

         double initialRiskPts = GetInitialRiskPoints(posi.Identifier(), posi.PriceOpen(), posi.StopLoss());
         if(initialRiskPts <= 0) continue;

         double currentPrice = (posi.PositionType() == POSITION_TYPE_BUY) ? symb.Bid() : symb.Ask();
         double openPrice = posi.PriceOpen();
         double currentSL = posi.StopLoss();

         double profitPoints = 0;
         if(posi.PositionType() == POSITION_TYPE_BUY && currentPrice > openPrice) profitPoints = (currentPrice - openPrice) / _Point;
         if(posi.PositionType() == POSITION_TYPE_SELL && currentPrice < openPrice) profitPoints = (openPrice - currentPrice) / _Point;

         double currentRR = profitPoints / initialRiskPts;

         if(currentRR >= InpTslTriggerRR) {
            double trailDistPts = initialRiskPts * InpTslDistanceRR;
            double trailStepPts = initialRiskPts * InpTslStepRR;

            if(posi.PositionType() == POSITION_TYPE_BUY) {
               double newSL = currentPrice - (trailDistPts * _Point);
               if(currentSL == 0 || newSL >= currentSL + (trailStepPts * _Point))
                  trade.PositionModify(posi.Ticket(), NormalizeDouble(newSL, _Digits), posi.TakeProfit());
            } else {
               double newSL = currentPrice + (trailDistPts * _Point);
               if(currentSL == 0 || newSL <= currentSL - (trailStepPts * _Point))
                  trade.PositionModify(posi.Ticket(), NormalizeDouble(newSL, _Digits), posi.TakeProfit());
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Update Swing Points (HTF)                                        |
//+------------------------------------------------------------------+
void UpdateSwingPoints() {
   int checkIdx = InpSwingStrength + 1;

   bool isSwingHigh = true;
   double highVal = iHigh(_Symbol, InpHTF, checkIdx);
   double lowestInWindowH = iLow(_Symbol, InpHTF, checkIdx);

   for(int i=1; i<=InpSwingStrength; i++) {
      if(iHigh(_Symbol, InpHTF, checkIdx - i) >= highVal || iHigh(_Symbol, InpHTF, checkIdx + i) >= highVal) { isSwingHigh = false; break; }
      lowestInWindowH = MathMin(lowestInWindowH, iLow(_Symbol, InpHTF, checkIdx - i));
      lowestInWindowH = MathMin(lowestInWindowH, iLow(_Symbol, InpHTF, checkIdx + i));
   }
   if(isSwingHigh && (highVal - lowestInWindowH) < InpMinSwingSizePoints * _Point) isSwingHigh = false;

   if(isSwingHigh && activeSwingHighTime != iTime(_Symbol, InpHTF, checkIdx)) {
      activeSwingHigh = highVal;
      activeSwingHighTime = iTime(_Symbol, InpHTF, checkIdx);
      activeSwingHighSwept = false;
   }

   bool isSwingLow = true;
   double lowVal = iLow(_Symbol, InpHTF, checkIdx);
   double highestInWindowL = iHigh(_Symbol, InpHTF, checkIdx);

   for(int i=1; i<=InpSwingStrength; i++) {
      if(iLow(_Symbol, InpHTF, checkIdx - i) <= lowVal || iLow(_Symbol, InpHTF, checkIdx + i) <= lowVal) { isSwingLow = false; break; }
      highestInWindowL = MathMax(highestInWindowL, iHigh(_Symbol, InpHTF, checkIdx - i));
      highestInWindowL = MathMax(highestInWindowL, iHigh(_Symbol, InpHTF, checkIdx + i));
   }
   if(isSwingLow && (highestInWindowL - lowVal) < InpMinSwingSizePoints * _Point) isSwingLow = false;

   if(isSwingLow && activeSwingLowTime != iTime(_Symbol, InpHTF, checkIdx)) {
      activeSwingLow = lowVal;
      activeSwingLowTime = iTime(_Symbol, InpHTF, checkIdx);
      activeSwingLowSwept = false;
   }
}

//+------------------------------------------------------------------+
//| Detect Liquidity Sweep (HTF) - FIXED BOS FILTER                  |
//+------------------------------------------------------------------+
void DetectLiquiditySweep() {
   double close1 = NormalizeDouble(iClose(_Symbol, InpHTF, 1), _Digits);
   double minBreak = MathMax(InpMinSweepBreakPoints * _Point, _Point);
   int maxLookback = 50;

   // 1. Check for Sell Side Liquidity (SSL) Sweep - Bullish Setup
   if(!activeSwingLowSwept && activeSwingLow > 0) {
      double swingLowNorm = NormalizeDouble(activeSwingLow, _Digits);

      bool breakFound=false, reenterFound=false;
      int candlesOutside=0;
      double extPrice=0; datetime extTime=0;

      int startIdx=0;
      for(int i=maxLookback;i>=1;i--) if(iTime(_Symbol, InpHTF, i) > activeSwingLowTime) { startIdx=i; break; }

      for(int i=startIdx;i>=1;i--) {
         datetime t=iTime(_Symbol, InpHTF, i);
         if(t<=activeSwingLowTime) break;

         double L=NormalizeDouble(iLow(_Symbol, InpHTF, i), _Digits);
         double C=NormalizeDouble(iClose(_Symbol, InpHTF, i), _Digits);

         if(!breakFound) {
            if(L <= swingLowNorm - minBreak) breakFound=true;
            else continue;
         }

         bool isOutside=(L < swingLowNorm || C < swingLowNorm);
         bool isInside=(L >= swingLowNorm && C >= swingLowNorm);

         if(isOutside) {
            candlesOutside++;
            if(extPrice==0 || L<extPrice){ extPrice=L; extTime=t; }
            if(candlesOutside > InpMaxCandlesOutsideSweep){ breakFound=false; break; }
         } else if(isInside) { reenterFound=true; break; }
         else {
            candlesOutside++;
            if(extPrice==0 || L<extPrice){ extPrice=L; extTime=t; }
            if(candlesOutside > InpMaxCandlesOutsideSweep){ breakFound=false; break; }
         }
      }

      bool isSweep=false;
      if(breakFound && candlesOutside>0 && candlesOutside<=InpMaxCandlesOutsideSweep) {
         if(InpRequireCloseBack) { if(reenterFound && close1 > swingLowNorm) isSweep=true; }
         else isSweep=true;
      }

      if(isSweep) {
         sweepDirection=1; sweepExtremePrice=extPrice; sweepTime=extTime; barsSinceSweep=0;
         activeSwingLowSwept=true;
         DrawSweepLine("ICT_SweepLow_" + TimeToString(sweepTime), activeSwingLowTime, activeSwingLow, sweepTime, InpSweepColor);
         if(InpCloseOnOppositeSweep) CheckOppositeSweepClose(1);
         
         ScanForDisplacementFVG(); // Instantly scan LTF to find the exact displacement FVG
      }
   }

   // 2. Check for Buy Side Liquidity (BSL) Sweep - Bearish Setup
   if(!activeSwingHighSwept && activeSwingHigh > 0) {
      double swingHighNorm = NormalizeDouble(activeSwingHigh, _Digits);

      bool breakFound=false, reenterFound=false;
      int candlesOutside=0;
      double extPrice=0; datetime extTime=0;

      int startIdx=0;
      for(int i=maxLookback;i>=1;i--) if(iTime(_Symbol, InpHTF, i) > activeSwingHighTime) { startIdx=i; break; }

      for(int i=startIdx;i>=1;i--) {
         datetime t=iTime(_Symbol, InpHTF, i);
         if(t<=activeSwingHighTime) break;

         double H=NormalizeDouble(iHigh(_Symbol, InpHTF, i), _Digits);
         double C=NormalizeDouble(iClose(_Symbol, InpHTF, i), _Digits);

         if(!breakFound) {
            if(H >= swingHighNorm + minBreak) breakFound=true;
            else continue;
         }

         bool isOutside=(H > swingHighNorm || C > swingHighNorm);
         bool isInside=(H <= swingHighNorm && C <= swingHighNorm);

         if(isOutside) {
            candlesOutside++;
            if(extPrice==0 || H>extPrice){ extPrice=H; extTime=t; }
            if(candlesOutside > InpMaxCandlesOutsideSweep){ breakFound=false; break; }
         } else if(isInside) { reenterFound=true; break; }
         else {
            candlesOutside++;
            if(extPrice==0 || H>extPrice){ extPrice=H; extTime=t; }
            if(candlesOutside > InpMaxCandlesOutsideSweep){ breakFound=false; break; }
         }
      }

      bool isSweep=false;
      if(breakFound && candlesOutside>0 && candlesOutside<=InpMaxCandlesOutsideSweep) {
         if(InpRequireCloseBack) { if(reenterFound && close1 < swingHighNorm) isSweep=true; }
         else isSweep=true;
      }

      if(isSweep) {
         sweepDirection=-1; sweepExtremePrice=extPrice; sweepTime=extTime; barsSinceSweep=0;
         activeSwingHighSwept=true;
         DrawSweepLine("ICT_SweepHigh_" + TimeToString(sweepTime), activeSwingHighTime, activeSwingHigh, sweepTime, InpSweepColor);
         if(InpCloseOnOppositeSweep) CheckOppositeSweepClose(-1);
         
         ScanForDisplacementFVG(); // Instantly scan LTF to find the exact displacement FVG
      }
   }
}

void CheckOppositeSweepClose(int newSweepDir) {
   for(int i=PositionsTotal()-1; i>=0; i--) {
      if(posi.SelectByIndex(i) && posi.Symbol() == _Symbol && posi.Magic() == InpMagicNumber) {
         double netProfit = posi.Profit() + posi.Swap() + posi.Commission();
         if(netProfit > 0) {
            if(posi.PositionType() == POSITION_TYPE_BUY && newSweepDir == -1) trade.PositionClose(posi.Ticket());
            else if(posi.PositionType() == POSITION_TYPE_SELL && newSweepDir == 1) trade.PositionClose(posi.Ticket());
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Scan for Historical Displacement FVG                             |
//+------------------------------------------------------------------+
void ScanForDisplacementFVG() {
   // Sweep time is the exact HTF candle that caused the sweep.
   // We rewind and scan the LTF candles that formed during that sweep.
   int startIdx = iBarShift(_Symbol, InpLTF, sweepTime);
   if(startIdx < 3) startIdx = 3;
   if(startIdx > 50) startIdx = 50; 
   
   bool found = false;
   
   for(int i = startIdx; i >= 1; i--) {
       // Ensure the FVG formation happened ON or AFTER the HTF sweep candle opened
       if(iTime(_Symbol, InpLTF, i+1) < sweepTime) continue; 
       
       double high3 = iHigh(_Symbol, InpLTF, i+2);
       double low3  = iLow(_Symbol, InpLTF, i+2);
       double high1 = iHigh(_Symbol, InpLTF, i);
       double low1  = iLow(_Symbol, InpLTF, i);
       
       double gapSize = 0;
       
       // Sell Side Liquidity (SSL) swept -> Price reversal UP -> Bullish FVG
       if(sweepDirection == 1 && low1 > high3) {
           gapSize = low1 - high3;
           if(gapSize >= InpMinFVGSizePoints * _Point) {
               currentFVG.time = iTime(_Symbol, InpLTF, i+1);
               currentFVG.direction = 1;
               currentFVG.high = low1;
               currentFVG.low = high3;
               currentFVG.traded = false;
               DrawFVG(currentFVG);
               found = true;
               break;
           }
       }
       // Buy Side Liquidity (BSL) swept -> Price reversal DOWN -> Bearish FVG
       else if(sweepDirection == -1 && high1 < low3) {
           gapSize = low3 - high1;
           if(gapSize >= InpMinFVGSizePoints * _Point) {
               currentFVG.time = iTime(_Symbol, InpLTF, i+1);
               currentFVG.direction = -1;
               currentFVG.high = low3;
               currentFVG.low = high1;
               currentFVG.traded = false;
               DrawFVG(currentFVG);
               found = true;
               break;
           }
       }
   }
   
   if(found) {
       ResetSweepState(); // FVG was found, no need to keep hunting in standard DetectFVG
   }
}


//+------------------------------------------------------------------+
//| Detect Fair Value Gap (LTF) - Forward Looking Fallback           |
//+------------------------------------------------------------------+
void DetectFVG() {
   double high3 = iHigh(_Symbol, InpLTF, 3);
   double low3  = iLow(_Symbol, InpLTF, 3);
   double high1 = iHigh(_Symbol, InpLTF, 1);
   double low1  = iLow(_Symbol, InpLTF, 1);

   double gapSize = 0;

   // Sell Side Liquidity swept -> Mark Bullish FVG
   if(sweepDirection == 1 && low1 > high3) {
      gapSize = low1 - high3;
      if(gapSize >= InpMinFVGSizePoints * _Point) {
         currentFVG.time = iTime(_Symbol, InpLTF, 2);
         currentFVG.direction = 1;
         currentFVG.high = low1;
         currentFVG.low = high3;
         currentFVG.traded = false;

         DrawFVG(currentFVG);
         ResetSweepState();
      }
   }
   // Buy Side Liquidity swept -> Mark Bearish FVG
   else if(sweepDirection == -1 && high1 < low3) {
      gapSize = low3 - high1;
      if(gapSize >= InpMinFVGSizePoints * _Point) {
         currentFVG.time = iTime(_Symbol, InpLTF, 2);
         currentFVG.direction = -1;
         currentFVG.high = low3;
         currentFVG.low = high1;
         currentFVG.traded = false;

         DrawFVG(currentFVG);
         ResetSweepState();
      }
   }
}

//+------------------------------------------------------------------+
//| Wick Touch + Candle Close + Hammer Execution                     |
//+------------------------------------------------------------------+
void CheckMarketEntry() {
   if(CountOpenPositions() > 0) return;
   if(tradesTodayTotal >= InpMaxTotalTrades) return;
   if(!IsSpreadValid()) return;

   // Get previous closed candle properties
   double O = iOpen(_Symbol, InpLTF, 1);
   double H = iHigh(_Symbol, InpLTF, 1);
   double L = iLow(_Symbol, InpLTF, 1);
   double C = iClose(_Symbol, InpLTF, 1);

   // Calculate candle shape
   double body = MathAbs(O - C);
   double range = H - L;
   double lowerWick = MathMin(O, C) - L;
   double upperWick = H - MathMax(O, C);

   // Adjustable Hammer Definitions
   bool isBullishHammer = (range > 0 && lowerWick >= InpMinWickMultiplier * body && upperWick <= InpMaxOppositeWickRatio * body);
   bool isBearishHammer = (range > 0 && upperWick >= InpMinWickMultiplier * body && lowerWick <= InpMaxOppositeWickRatio * body);

   double entryLevel = 0;
   double gapSize = currentFVG.high - currentFVG.low;
   
   bool touched = false;
   bool triggerBuy = false;
   bool triggerSell = false;

   if(currentFVG.direction == 1) { // Bullish FVG
      // Calculate % Level (0% = High Edge, 100% = Low Edge)
      entryLevel = currentFVG.high - (gapSize * (InpFVGEntryPercent / 100.0));
      entryLevel += InpOrderBufferPoints * _Point;

      // Check if the low of the candle touched or went below the calculated % entry level
      if(L <= entryLevel) {
         touched = true;
         // Now check if it satisfied the hammer and close conditions
         if(C > entryLevel && isBullishHammer) {
            triggerBuy = true;
         }
      }
   } else if(currentFVG.direction == -1) { // Bearish FVG
      // Calculate % Level (0% = Low Edge, 100% = High Edge)
      entryLevel = currentFVG.low + (gapSize * (InpFVGEntryPercent / 100.0));
      entryLevel -= InpOrderBufferPoints * _Point;

      // Check if the high of the candle touched or went above the calculated % entry level
      if(H >= entryLevel) {
         touched = true;
         // Now check if it satisfied the hammer and close conditions
         if(C < entryLevel && isBearishHammer) {
            triggerSell = true;
         }
      }
   }

   // If the FVG zone was touched, we mark it as traded regardless of the outcome
   if(touched) {
      currentFVG.traded = true; // This invalidates the FVG for future checks
      
      if(triggerBuy || triggerSell) {
         double ask = symb.Ask();
         double bid = symb.Bid();
         double execPrice = triggerBuy ? ask : bid;

         double sl = CalculateSL(execPrice, currentFVG.direction);
         double tp = CalculateTP(execPrice, sl, currentFVG.direction);

         CheckAndFixStops(currentFVG.direction, execPrice, sl, tp);

         double lot = CalculateLotSize(execPrice, sl, currentFVG.direction);
         if(lot <= 0) return;

         if(triggerBuy) {
            if(trade.Buy(lot, _Symbol, execPrice, sl, tp, InpOrderComment)) tradesTodayTotal++;
         } else {
            if(trade.Sell(lot, _Symbol, execPrice, sl, tp, InpOrderComment)) tradesTodayTotal++;
         }
      } else {
         if(InpDebugMode) Print("FVG % Level touched but Hammer conditions not met. FVG Invalidated.");
      }
   }
}

//+------------------------------------------------------------------+
//| Spread / Stops                                                   |
//+------------------------------------------------------------------+
bool IsSpreadValid() {
   if(InpMaxSpreadPoints <= 0) return true;
   double ask = symb.Ask();
   double bid = symb.Bid();
   return ((ask - bid) / _Point) <= InpMaxSpreadPoints;
}

void CheckAndFixStops(int direction, double price, double &sl, double &tp) {
   double minStop = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
   if(minStop == 0) minStop = 10 * _Point;

   if(direction == 1) {
      if(sl > 0 && price - sl < minStop) sl = price - minStop;
      if(tp > 0 && tp - price < minStop) tp = price + minStop;
   } else {
      if(sl > 0 && sl - price < minStop) sl = price + minStop;
      if(tp > 0 && price - tp < minStop) tp = price - minStop;
   }
   sl = NormalizeDouble(sl, _Digits);
   tp = NormalizeDouble(tp, _Digits);
}

double CalculateSL(double entryPrice, int direction) {
   double sl = 0;

   if(InpStopLossMode == SL_MODE_POINTS) {
      sl = (direction == 1) ? entryPrice - InpStopValuePoints * _Point : entryPrice + InpStopValuePoints * _Point;
   }
   else if(InpStopLossMode == SL_MODE_ATR) {
      double atrArray[1];
      if(CopyBuffer(atrHandle, 0, 1, 1, atrArray) <= 0) return 0;
      double atrDist = atrArray[0] * InpAtrMultiplier;
      sl = (direction == 1) ? entryPrice - atrDist : entryPrice + atrDist;
   }
   else if(InpStopLossMode == SL_MODE_SWING) {
      if(direction == 1) {
         double refLow = (sweepExtremePrice > 0 && sweepExtremePrice < entryPrice) ? sweepExtremePrice : activeSwingLow;
         sl = refLow - (InpSwingBufferPoints * _Point);
      } else {
         double refHigh = (sweepExtremePrice > 0 && sweepExtremePrice > entryPrice) ? sweepExtremePrice : activeSwingHigh;
         sl = refHigh + (InpSwingBufferPoints * _Point);
      }
   }

   return NormalizeDouble(sl, _Digits);
}

double CalculateTP(double entryPrice, double sl, int direction) {
   if(InpTargetCalcMode == CALC_MODE_OFF) return 0;

   double tp = 0;
   double risk = MathAbs(entryPrice - sl);
   if(risk <= 0) risk = 100 * _Point;

   if(InpTargetCalcMode == CALC_MODE_FACTOR) {
      tp = (direction == 1) ? entryPrice + risk * InpTargetValue : entryPrice - risk * InpTargetValue;
   } else if(InpTargetCalcMode == CALC_MODE_POINTS) {
      tp = (direction == 1) ? entryPrice + InpTargetValue * _Point : entryPrice - InpTargetValue * _Point;
   }
   return NormalizeDouble(tp, _Digits);
}

//+------------------------------------------------------------------+
//| Dynamic Risk                                                     |
//+------------------------------------------------------------------+
void UpdateDrawdownState() {
   if(!InpEnableDynamicRisk) return;

   double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);

   if(currentRiskMultiplier == 1.0 && currentBalance > highestBalance) {
      highestBalance = currentBalance;
      referenceBalance = highestBalance;
   }

   if(currentRiskMultiplier < 1.0 && currentBalance >= recoveryTarget) {
      currentRiskMultiplier = 1.0;
      referenceBalance = currentBalance;
      highestBalance = currentBalance;
   }

   double dropPercentage = 0;
   if(referenceBalance > 0) dropPercentage = ((referenceBalance - currentBalance) / referenceBalance) * 100.0;

   if(dropPercentage >= InpDrawdownThresholdPct) {
      currentRiskMultiplier *= 0.5;
      double lostAmount = referenceBalance - currentBalance;
      recoveryTarget = currentBalance + (lostAmount * (InpRecoveryTargetPct / 100.0));
      referenceBalance = currentBalance;
   }
}

double GetDynamicRiskMultiplier() {
   if(!InpEnableDynamicRisk) return 1.0;

   HistorySelect(0, TimeCurrent());
   int dealsTotal = HistoryDealsTotal();
   int consecutiveWins = 0;
   ulong lastPosId = 0;

   for(int i = dealsTotal - 1; i >= 0; i--) {
      ulong ticket = HistoryDealGetTicket(i);
      long entry = HistoryDealGetInteger(ticket, DEAL_ENTRY);

      if(entry == DEAL_ENTRY_OUT || entry == DEAL_ENTRY_INOUT) {
         ulong posId = (ulong)HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
         if(posId == lastPosId) continue;
         lastPosId = posId;

         double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
         double swap = HistoryDealGetDouble(ticket, DEAL_SWAP);
         double commission = HistoryDealGetDouble(ticket, DEAL_COMMISSION);
         double netProfit = profit + swap + commission;

         if(netProfit > 0) consecutiveWins++;
         else if(netProfit < 0) break;
      }
   }

   double finalMultiplier = currentRiskMultiplier;
   if(consecutiveWins >= 3) finalMultiplier = MathMin(currentRiskMultiplier, 0.5);
   return finalMultiplier;
}

//+------------------------------------------------------------------+
//| Helpers                                                          |
//+------------------------------------------------------------------+
int CountOpenPositions() {
   int count = 0;
   for(int i=PositionsTotal()-1; i>=0; i--) {
      if(posi.SelectByIndex(i) && posi.Symbol() == _Symbol && posi.Magic() == InpMagicNumber) count++;
   }
   return count;
}

bool IsTradingTime() {
   MqlDateTime dt; TimeCurrent(dt);
   int currentMins = dt.hour * 60 + dt.min;
   int startMins = InpTradeStartHour * 60 + InpTradeStartMinute;
   int endMins = InpTradeEndHour * 60 + InpTradeEndMinute;

   if(startMins < endMins) return (currentMins >= startMins && currentMins <= endMins);
   else if(startMins > endMins) return (currentMins >= startMins || currentMins <= endMins);
   return true;
}

void ResetSweepState() {
   sweepDirection = 0;
   barsSinceSweep = -1;
   sweepExtremePrice = 0;
   sweepTime = 0;
}

void DrawFVG(FVG_Data &fvg) {
   string name = "ICT_FVG_" + TimeToString(fvg.time);
   datetime timeEnd = fvg.time + PeriodSeconds(InpLTF) * 15;
   ObjectCreate(0, name, OBJ_RECTANGLE, 0, fvg.time, fvg.high, timeEnd, fvg.low);
   ObjectSetInteger(0, name, OBJPROP_COLOR, (fvg.direction == 1) ? InpBullishColor : InpBearishColor);
   ObjectSetInteger(0, name, OBJPROP_FILL, true);
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
}

void DrawSweepLine(string name, datetime t1, double price, datetime t2, color clr) {
   ObjectCreate(0, name, OBJ_TREND, 0, t1, price, t2, price);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DASHDOT);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, false);
}
//+------------------------------------------------------------------+