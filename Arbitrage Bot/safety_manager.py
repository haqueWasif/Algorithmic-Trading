"""
Safety Manager - Comprehensive safety features to prevent losses in the arbitrage bot.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

from config import RISK_CONFIG, TRADING_CONFIG
from trading_engine import ArbitrageTrade, TradeStatus
from exchange_manager import ArbitrageOpportunity
from error_handler import ErrorHandler, ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    """Safety alert levels."""
    GREEN = 1    # Normal operation
    YELLOW = 2   # Caution - increased monitoring
    ORANGE = 3   # Warning - restricted operation
    RED = 4      # Critical - emergency stop

class SafetyRule(Enum):
    """Types of safety rules."""
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    SINGLE_TRADE_LOSS = "single_trade_loss"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    BALANCE_THRESHOLD = "balance_threshold"
    PRICE_DEVIATION = "price_deviation"
    VOLUME_ANOMALY = "volume_anomaly"
    API_ERROR_RATE = "api_error_rate"
    EXECUTION_TIME = "execution_time"
    SPREAD_ANOMALY = "spread_anomaly"
    MARKET_VOLATILITY = "market_volatility"

@dataclass
class SafetyViolation:
    """Represents a safety rule violation."""
    rule: SafetyRule
    severity: SafetyLevel
    message: str
    current_value: float
    threshold_value: float
    timestamp: float
    component: str
    context: Dict[str, Any]

@dataclass
class SafetyMetrics:
    """Current safety metrics."""
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    last_trade_loss: float = 0.0
    api_error_rate: float = 0.0
    avg_execution_time: float = 0.0
    max_price_deviation: float = 0.0
    total_trades_today: int = 0
    emergency_stops_triggered: int = 0
    last_balance_check: float = 0.0
    balance_warnings: int = 0

class SafetyManager:
    def __init__(self, monitoring_system: Any, risk_config: Dict[str, Any]): # <--- MODIFIED LINE
        self.monitoring_system = monitoring_system
        self.risk_config = risk_config
        self.daily_profit_loss = 0.0
        self.consecutive_losses = 0
        self.circuit_breaker_active = False
        self.last_trade_time = None
        self.initial_balances: Dict[str, Dict[str, float]] = {}
        self.granular_circuit_breakers: Dict[str, Dict[str, Any]] = {} # {entity_id: {reason: str, cooldown_until: float}}
        self.last_daily_profit_loss_reset = time.time()

    async def initialize_balances(self, exchange_manager: Any):
        logger.info("Initializing safety manager with current balances...")
        self.initial_balances = await exchange_manager.get_all_balances()
        logger.info(f"Initial balances: {self.initial_balances}")

    def record_profit(self, profit_usd: float):
        self.daily_profit_loss += profit_usd
        self.consecutive_losses = 0
        self.last_trade_time = time.time()
        logger.info(f"Recorded profit: ${profit_usd:.2f}. Daily P/L: ${self.daily_profit_loss:.2f}")
        self.monitoring_system.alert_manager.create_alert(
            "Profit Recorded", f"Trade resulted in profit: ${profit_usd:.2f}", "info", "SafetyManager"
        )

    def record_loss(self, loss_usd: float):
        self.daily_profit_loss -= abs(loss_usd) # Ensure loss is negative
        self.consecutive_losses += 1
        self.last_trade_time = time.time()
        logger.warning(f"Recorded loss: ${loss_usd:.2f}. Daily P/L: ${self.daily_profit_loss:.2f}. Consecutive losses: {self.consecutive_losses}")
        self.monitoring_system.alert_manager.create_alert(
            "Loss Recorded", f"Trade resulted in loss: ${loss_usd:.2f}", "warning", "SafetyManager"
        )
        self._check_risk_limits()

    def _check_risk_limits(self):
        # Dynamic daily loss threshold
        adjusted_max_daily_loss = self.risk_config["max_daily_loss_usd"]
        if self.daily_profit_loss > 0: # If bot is profitable, allow slightly larger loss before stopping
            adjusted_max_daily_loss += self.daily_profit_loss * self.risk_config["dynamic_loss_threshold_factor"]

        if self.risk_config["circuit_breaker_enabled"]:
            if self.daily_profit_loss < -adjusted_max_daily_loss:
                self._activate_circuit_breaker("Max daily loss exceeded")
            elif self.consecutive_losses >= self.risk_config["max_consecutive_losses"]:
                self._activate_circuit_breaker("Max consecutive losses reached")

    def _activate_circuit_breaker(self, reason: str, entity_id: Optional[str] = None, entity_type: Optional[str] = None):
        if entity_id and entity_type:
            # Granular circuit breaker
            cooldown_minutes = self.risk_config["granular_circuit_breaker_cooldown_minutes"]
            cooldown_until = time.time() + cooldown_minutes * 60
            self.granular_circuit_breakers[f"{entity_type}:{entity_id}"] = {
                "reason": reason,
                "cooldown_until": cooldown_until,
                "activated_at": time.time()
            }
            from datetime import datetime
            logger.critical(f"GRANULAR CIRCUIT BREAKER ACTIVATED for {entity_type} {entity_id}: {reason}. Cooldown until {datetime.fromtimestamp(cooldown_until).strftime('%Y-%m-%d %H:%M:%S')}")
            self.monitoring_system.alert_manager.create_alert(
                "Granular Circuit Breaker Activated", f"{reason} for {entity_type} {entity_id}", "critical", "SafetyManager"
            )
        elif not self.circuit_breaker_active:
            # Global circuit breaker
            self.circuit_breaker_active = True
            logger.critical(f"GLOBAL CIRCUIT BREAKER ACTIVATED: {reason}")
            self.monitoring_system.alert_manager.create_alert(
                "Global Circuit Breaker Activated", reason, "critical", "SafetyManager"
            )
    def deactivate_circuit_breaker(self):
        if self.circuit_breaker_active:
            self.circuit_breaker_active = False
            self.consecutive_losses = 0 # Reset consecutive losses on deactivation
            logger.info("Circuit breaker deactivated.")
            self.monitoring_system.alert_manager.create_alert(
                "Circuit Breaker Deactivated", "Manual deactivation", "info", "SafetyManager"
            )

    def is_circuit_breaker_active(self, entity_id: Optional[str] = None, entity_type: Optional[str] = None) -> bool:
        # Check global circuit breaker
        if self.circuit_breaker_active:
            return True
        
        # Check granular circuit breaker
        if entity_id and entity_type:
            key = f"{entity_type}:{entity_id}"
            if key in self.granular_circuit_breakers:
                cooldown_until = self.granular_circuit_breakers[key]["cooldown_until"]
                if time.time() < cooldown_until:
                    return True
                else:
                    # Cooldown period ended, deactivate granular circuit breaker
                    del self.granular_circuit_breakers[key]
        return False

    def get_safety_status(self) -> Dict[str, Any]:
        return {
            "daily_profit_loss": self.daily_profit_loss,
            "consecutive_losses": self.consecutive_losses,
            "circuit_breaker_active": self.circuit_breaker_active,
            "last_trade_time": self.last_trade_time,
            "max_daily_loss_usd": self.risk_config["max_daily_loss_usd"],
            "max_consecutive_losses": self.risk_config["max_consecutive_losses"]
        }
        
    def _initialize_safety_rules(self) -> Dict[SafetyRule, Dict[str, Any]]:
        """Initialize safety rules with thresholds and actions."""
        return {
            SafetyRule.DAILY_LOSS_LIMIT: {
                "threshold": RISK_CONFIG["max_daily_loss_usd"],
                "action": "emergency_stop",
                "severity": SafetyLevel.RED
            },
            SafetyRule.SINGLE_TRADE_LOSS: {
                "threshold": RISK_CONFIG["max_single_trade_loss_usd"],
                "action": "pause_trading",
                "severity": SafetyLevel.ORANGE
            },
            SafetyRule.CONSECUTIVE_LOSSES: {
                "threshold": 5,  # 5 consecutive losses
                "action": "pause_trading",
                "severity": SafetyLevel.ORANGE
            },
            SafetyRule.BALANCE_THRESHOLD: {
                "threshold": 0.1,  # 10% of initial balance
                "action": "restrict_trading",
                "severity": SafetyLevel.YELLOW
            },
            SafetyRule.PRICE_DEVIATION: {
                "threshold": 0.05,  # 5% price deviation
                "action": "restrict_pair",
                "severity": SafetyLevel.YELLOW
            },
            SafetyRule.API_ERROR_RATE: {
                "threshold": 0.1,  # 10% error rate
                "action": "restrict_exchange",
                "severity": SafetyLevel.ORANGE
            },
            SafetyRule.EXECUTION_TIME: {
                "threshold": 5000,  # 5 seconds
                "action": "warning",
                "severity": SafetyLevel.YELLOW
            },
            SafetyRule.SPREAD_ANOMALY: {
                "threshold": 0.02,  # 2% spread
                "action": "restrict_pair",
                "severity": SafetyLevel.YELLOW
            },
            SafetyRule.MARKET_VOLATILITY: {
                "threshold": 0.1,  # 10% volatility
                "action": "pause_trading",
                "severity": SafetyLevel.ORANGE
            }
        }
    
    async def check_safety_before_trade(self, opportunity: ArbitrageOpportunity) -> Tuple[bool, Optional[str]]:
        """Check if it's safe to execute a trade."""
        
        # Check emergency stop
        if self.emergency_stop_active:
            return False, f"Emergency stop active: {self.emergency_stop_reason}"
        
        # Check if trading is paused
        if self.trading_paused:
            return False, "Trading is currently paused due to safety concerns"
        
        # Check restricted pairs
        if opportunity.symbol in self.restricted_pairs:
            return False, f"Trading pair {opportunity.symbol} is restricted"
        
        # Check restricted exchanges
        if (opportunity.buy_exchange in self.restricted_exchanges or 
            opportunity.sell_exchange in self.restricted_exchanges):
            return False, "One or both exchanges are restricted"
        
        # Check daily trade limit
        if self.metrics.total_trades_today >= TRADING_CONFIG["max_daily_trades"]:
            return False, "Daily trade limit reached"
        
        # Check price deviation
        if self._check_price_deviation(opportunity):
            return False, "Price deviation exceeds safety threshold"
        
        # Check spread anomaly
        if self._check_spread_anomaly(opportunity):
            return False, "Spread anomaly detected"
        
        # Check minimum profit threshold with safety margin
        safety_margin = 1.5  # 50% safety margin
        min_profit = TRADING_CONFIG["min_profit_threshold"] * safety_margin
        if opportunity.potential_profit_pct < min_profit:
            return False, f"Profit below safety threshold ({min_profit:.4f}%)"
        
        return True, None
    
    def _check_price_deviation(self, opportunity: ArbitrageOpportunity) -> bool:
        """Check if price deviation is within safe limits."""
        # Calculate price deviation from average
        avg_price = (opportunity.buy_price + opportunity.sell_price) / 2
        deviation = abs(opportunity.sell_price - opportunity.buy_price) / avg_price
        
        threshold = self.safety_rules[SafetyRule.PRICE_DEVIATION]["threshold"]
        
        if deviation > threshold:
            self._record_violation(
                SafetyRule.PRICE_DEVIATION,
                f"Price deviation {deviation:.4f} exceeds threshold {threshold:.4f}",
                deviation,
                threshold,
                "price_monitor",
                {"symbol": opportunity.symbol, "buy_price": opportunity.buy_price, "sell_price": opportunity.sell_price}
            )
            return True
        
        return False
    
    def _check_spread_anomaly(self, opportunity: ArbitrageOpportunity) -> bool:
        """Check for unusual spread that might indicate data issues."""
        # This is a simplified check - in practice, you'd compare against historical spreads
        spread_pct = opportunity.potential_profit_pct
        threshold = self.safety_rules[SafetyRule.SPREAD_ANOMALY]["threshold"]
        
        if spread_pct > threshold:
            self._record_violation(
                SafetyRule.SPREAD_ANOMALY,
                f"Spread {spread_pct:.4f} exceeds normal threshold {threshold:.4f}",
                spread_pct,
                threshold,
                "price_monitor",
                {"symbol": opportunity.symbol, "spread": spread_pct}
            )
            return True
        
        return False
    
    def record_trade_result(self, trade: ArbitrageTrade):
        """Record trade result for safety monitoring."""
        
        # Update metrics
        self.metrics.total_trades_today += 1
        self.metrics.daily_pnl += trade.actual_profit_usd
        
        # Track consecutive losses
        if trade.actual_profit_usd < 0:
            self.metrics.consecutive_losses += 1
            self.metrics.last_trade_loss = abs(trade.actual_profit_usd)
        else:
            self.metrics.consecutive_losses = 0
        
        # Update execution time
        if trade.execution_time_ms:
            self.metrics.avg_execution_time = (
                (self.metrics.avg_execution_time * (self.metrics.total_trades_today - 1) + 
                 trade.execution_time_ms) / self.metrics.total_trades_today
            )
        
        # Store trade history
        trade_record = {
            "timestamp": trade.timestamp,
            "symbol": trade.opportunity.symbol,
            "profit_usd": trade.actual_profit_usd,
            "status": trade.status.value,
            "execution_time_ms": trade.execution_time_ms
        }
        self.trade_history.append(trade_record)
        
        # Keep only recent history
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-500:]
        
        # Check safety rules after trade
        asyncio.create_task(self._check_post_trade_safety(trade))
    
    async def _check_post_trade_safety(self, trade: ArbitrageTrade):
        """Check safety rules after a trade is completed."""
        
        # Check daily loss limit
        if abs(self.metrics.daily_pnl) >= self.safety_rules[SafetyRule.DAILY_LOSS_LIMIT]["threshold"]:
            await self._trigger_emergency_stop(
                f"Daily loss limit exceeded: ${abs(self.metrics.daily_pnl):.2f}"
            )
            return
        
        # Check single trade loss
        if (trade.actual_profit_usd < 0 and 
            abs(trade.actual_profit_usd) >= self.safety_rules[SafetyRule.SINGLE_TRADE_LOSS]["threshold"]):
            self._record_violation(
                SafetyRule.SINGLE_TRADE_LOSS,
                f"Single trade loss ${abs(trade.actual_profit_usd):.2f} exceeds threshold",
                abs(trade.actual_profit_usd),
                self.safety_rules[SafetyRule.SINGLE_TRADE_LOSS]["threshold"],
                "trading_engine",
                {"trade_id": trade.id}
            )
            await self._pause_trading("Large single trade loss detected")
        
        # Check consecutive losses
        if self.metrics.consecutive_losses >= self.safety_rules[SafetyRule.CONSECUTIVE_LOSSES]["threshold"]:
            self._record_violation(
                SafetyRule.CONSECUTIVE_LOSSES,
                f"{self.metrics.consecutive_losses} consecutive losses detected",
                self.metrics.consecutive_losses,
                self.safety_rules[SafetyRule.CONSECUTIVE_LOSSES]["threshold"],
                "trading_engine",
                {}
            )
            await self._pause_trading("Too many consecutive losses")
        
        # Check execution time
        if (trade.execution_time_ms and 
            trade.execution_time_ms >= self.safety_rules[SafetyRule.EXECUTION_TIME]["threshold"]):
            self._record_violation(
                SafetyRule.EXECUTION_TIME,
                f"Execution time {trade.execution_time_ms}ms exceeds threshold",
                trade.execution_time_ms,
                self.safety_rules[SafetyRule.EXECUTION_TIME]["threshold"],
                "trading_engine",
                {"trade_id": trade.id}
            )

    def get_dynamic_trade_size(self, symbol: str, opportunity_profit_pct: float, market_volatility: float) -> float:
        """Calculates a dynamic trade size based on risk parameters."""
        base_trade_amount_usd = TRADING_CONFIG.get("max_trade_amount_usd", 100.0)

        # Adjust for profit potential (higher profit, larger size)
        profit_factor = 1.0 + (opportunity_profit_pct / 100.0) # Simple linear scaling

        # Adjust for market volatility (higher volatility, smaller size)
        volatility_factor = 1.0 / (1.0 + market_volatility * 10) # Inverse scaling

        # Combine factors
        dynamic_trade_amount_usd = base_trade_amount_usd * profit_factor * volatility_factor

        # Ensure the trade size is within reasonable bounds
        min_trade_amount = 10.0 # Minimum trade size in USD
        max_trade_amount = base_trade_amount_usd * 2 # Max trade size capped at 2x base
        dynamic_trade_amount_usd = max(min_trade_amount, min(dynamic_trade_amount_usd, max_trade_amount))

        logger.info(f"Dynamic trade size for {symbol}: ${dynamic_trade_amount_usd:.2f} (Profit: {opportunity_profit_pct:.4f}%, Volatility: {market_volatility:.4f})")
        return dynamic_trade_amount_usd

    async def rebalance_funds(self, exchange_manager: Any):
        """Monitors and rebalances funds across exchanges to maintain liquidity."""
        logger.info("Checking for fund rebalancing needs...")
        balances = await exchange_manager.get_all_balances()
        # This is a placeholder for a more complex rebalancing logic.
        # A real implementation would:
        # 1. Define target balance ratios for each asset on each exchange.
        # 2. Calculate the current balance distribution.
        # 3. If the deviation from the target exceeds a threshold (e.g., REBALANCING_THRESHOLD_PCT),
        #    trigger a rebalancing trade (e.g., withdraw from one exchange and deposit to another).
        # 4. This requires careful handling of withdrawal/deposit fees and times.
        logger.info("Fund rebalancing check complete. (Placeholder logic)")



