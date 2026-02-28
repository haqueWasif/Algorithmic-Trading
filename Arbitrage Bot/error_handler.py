"""
Comprehensive error handling and recovery mechanisms for the arbitrage bot.
"""

from collections import deque

import asyncio
import logging
import time
import traceback
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import functools



logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ErrorCategory(Enum):
    """Error categories for classification."""
    NETWORK = "network"
    API = "api"
    EXCHANGE = "exchange"
    TRADING = "trading"
    SYSTEM = "system"
    DATA = "data"
    CONFIGURATION = "configuration"

@dataclass
class ErrorEvent:
    """Represents an error event."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    component: str
    message: str
    exception: Optional[Exception]
    context: Dict[str, Any]
    timestamp: float
    retry_count: int = 0
    resolved: bool = False

class RetryStrategy:
    """Defines retry strategies for different error types."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 exponential_backoff: bool = True, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.exponential_backoff = exponential_backoff
        self.max_delay = max_delay
    
    def get_delay(self, retry_count: int) -> float:
        """Calculate delay for the given retry attempt."""
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** retry_count)
        else:
            delay = self.base_delay
        
        return min(delay, self.max_delay)



class ErrorHandler:
    def __init__(self, monitoring_system: Any): # <--- MODIFIED LINE
        self.monitoring_system = monitoring_system
        self.error_count = 0
        self.last_error_time = None
        self.recent_errors = deque(maxlen=100)

    def get_error_stats(self) -> Dict[str, Any]:
        return {
            "total_errors": self.error_count,
            "last_error_time": self.last_error_time,
            "recent_errors": list(self.recent_errors)
        }

    def reset_errors(self):
        self.error_count = 0
        self.last_error_time = None
        self.recent_errors.clear()
        logger.info("Error stats reset.")


    
    def _setup_default_retry_strategies(self):
        """Set up default retry strategies for different error categories."""
        self.retry_strategies = {
            ErrorCategory.NETWORK: RetryStrategy(max_retries=5, base_delay=1.0),
            ErrorCategory.API: RetryStrategy(max_retries=3, base_delay=2.0),
            ErrorCategory.EXCHANGE: RetryStrategy(max_retries=3, base_delay=5.0),
            ErrorCategory.TRADING: RetryStrategy(max_retries=2, base_delay=1.0),
            ErrorCategory.SYSTEM: RetryStrategy(max_retries=1, base_delay=10.0),
            ErrorCategory.DATA: RetryStrategy(max_retries=3, base_delay=1.0),
            ErrorCategory.CONFIGURATION: RetryStrategy(max_retries=0, base_delay=0.0)
        }
    
    def register_error_callback(self, category: ErrorCategory, callback: Callable):
        """Register a callback for specific error categories."""
        if category not in self.error_callbacks:
            self.error_callbacks[category] = []
        self.error_callbacks[category].append(callback)
    
    def handle_error(self, error: Exception, category: ErrorCategory, severity: ErrorSeverity, component: str, message: str, context: Dict[str, Any] = None):
        """Handle an error event."""
        
        self.error_count += 1
        self.last_error_time = time.time()
        error_details = {
            "timestamp": self.last_error_time,
            "component": component,
            "message": message,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "context": context or {}
        }
        self.recent_errors.append(error_details)
        logger.error(f"Error in {component} - {message}: {error}", exc_info=True)
        
        # Notify monitoring system
        if self.monitoring_system:
            self.monitoring_system.alert_manager.create_alert(
                "Application Error", 
                f"Error in {component} - {message}: {str(error)}", 
                "error", 
                component
            )

    
    def _update_error_stats(self, error_event: ErrorEvent):
        """Update error statistics."""
        self.error_stats['total_errors'] += 1
        
        # Update category stats
        category_name = error_event.category.value
        if category_name not in self.error_stats['errors_by_category']:
            self.error_stats['errors_by_category'][category_name] = 0
        self.error_stats['errors_by_category'][category_name] += 1
        
        # Update severity stats
        severity_name = error_event.severity.name
        if severity_name not in self.error_stats['errors_by_severity']:
            self.error_stats['errors_by_severity'][severity_name] = 0
        self.error_stats['errors_by_severity'][severity_name] += 1
    
    def _log_error(self, error_event: ErrorEvent):
        """Log the error event."""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[error_event.severity]
        
        log_message = f"[{error_event.category.value.upper()}] {error_event.component}: {error_event.message}"
        
        if error_event.exception:
            log_message += f"\nException: {str(error_event.exception)}"
            log_message += f"\nTraceback: {traceback.format_exc()}"
        
        if error_event.context:
            log_message += f"\nContext: {error_event.context}"
        
        logger.log(log_level, log_message)
    
    def _execute_error_callbacks(self, error_event: ErrorEvent):
        """Execute registered callbacks for the error category."""
        callbacks = self.error_callbacks.get(error_event.category, [])
        
        for callback in callbacks:
            try:
                callback(error_event)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    def _handle_critical_error(self, error_event: ErrorEvent):
        """Handle critical errors that require immediate attention."""
        logger.critical(f"CRITICAL ERROR DETECTED: {error_event.message}")
        
        # Trigger emergency procedures based on component
        if error_event.component in ['trading_engine', 'exchange_manager']:
            # Stop trading immediately
            logger.critical("Triggering emergency trading halt due to critical error")
            # This would trigger the circuit breaker in the trading engine
    
    async def retry_with_strategy(self, func: Callable, category: ErrorCategory,
                                component: str, *args, **kwargs) -> Any:
        """Retry a function call with the appropriate strategy for the error category."""
        
        strategy = self.retry_strategies.get(category, RetryStrategy())
        last_exception = None
        
        for attempt in range(strategy.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                if attempt < strategy.max_retries:
                    delay = strategy.get_delay(attempt)
                    
                    # Log retry attempt
                    logger.warning(f"Retry attempt {attempt + 1}/{strategy.max_retries} "
                                 f"for {component} after {delay}s delay. Error: {str(e)}")
                    
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed, handle the error
                    self.handle_error(
                        category=category,
                        severity=ErrorSeverity.HIGH,
                        component=component,
                        message=f"Function failed after {strategy.max_retries} retries",
                        exception=e,
                        context={'function': func.__name__, 'args': str(args)[:200]}
                    )
                    raise e
        
        # This should never be reached, but just in case
        if last_exception:
            raise last_exception

class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func):
        """Decorator to apply circuit breaker to a function."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper
    
    async def call(self, func, *args, **kwargs):
        """Call a function through the circuit breaker."""
        
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success - reset failure count
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")




def error_handler_decorator(category: ErrorCategory, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                          component: str = "unknown"):
    """Decorator to automatically handle errors in functions."""
    
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Get error handler from the first argument if it's available
                error_handler = None
                if args and hasattr(args[0], 'error_handler'):
                    error_handler = args[0].error_handler
                
                if error_handler:
                    error_handler.handle_error(
                        category=category,
                        severity=severity,
                        component=component,
                        message=f"Error in {func.__name__}: {str(e)}",
                        exception=e,
                        context={'function': func.__name__, 'args': str(args)[:200]}
                    )
                else:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                
                raise e
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get error handler from the first argument if it's available
                error_handler = None
                if args and hasattr(args[0], 'error_handler'):
                    error_handler = args[0].error_handler
                
                if error_handler:
                    error_handler.handle_error(
                        category=category,
                        severity=severity,
                        component=component,
                        message=f"Error in {func.__name__}: {str(e)}",
                        exception=e,
                        context={'function': func.__name__, 'args': str(args)[:200]}
                    )
                else:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                
                raise e
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

class HealthChecker:
    """System health checker that monitors various components."""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.health_checks: Dict[str, Callable] = {}
        self.last_check_results: Dict[str, bool] = {}
        self.check_intervals: Dict[str, float] = {}
        self.last_check_times: Dict[str, float] = {}
    
    def register_health_check(self, name: str, check_func: Callable, 
                            interval: float = 60.0):
        """Register a health check function."""
        self.health_checks[name] = check_func
        self.check_intervals[name] = interval
        self.last_check_times[name] = 0
        self.last_check_results[name] = True
    
    async def run_health_checks(self):
        """Run all registered health checks."""
        current_time = time.time()
        
        for name, check_func in self.health_checks.items():
            # Check if it's time to run this health check
            if current_time - self.last_check_times[name] >= self.check_intervals[name]:
                try:
                    if asyncio.iscoroutinefunction(check_func):
                        result = await check_func()
                    else:
                        result = check_func()
                    
                    self.last_check_results[name] = result
                    self.last_check_times[name] = current_time
                    
                    if not result:
                        self.error_handler.handle_error(
                            category=ErrorCategory.SYSTEM,
                            severity=ErrorSeverity.HIGH,
                            component="health_checker",
                            message=f"Health check failed: {name}",
                            context={'check_name': name}
                        )
                
                except Exception as e:
                    self.last_check_results[name] = False
                    self.last_check_times[name] = current_time
                    
                    self.error_handler.handle_error(
                        category=ErrorCategory.SYSTEM,
                        severity=ErrorSeverity.HIGH,
                        component="health_checker",
                        message=f"Health check error: {name}",
                        exception=e,
                        context={'check_name': name}
                    )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get the current health status of all components."""
        return {
            'overall_healthy': all(self.last_check_results.values()),
            'checks': self.last_check_results.copy(),
            'last_check_times': self.last_check_times.copy()
        }

# Example health check functions
async def check_exchange_connectivity(exchange_manager) -> bool:
    """Check if exchange connections are healthy."""
    try:
        # This would check if exchanges are responding
        for exchange_name, exchange in exchange_manager.exchanges.items():
            # Simple ping or market data request
            await exchange.fetch_ticker('BTC/USDT')
        return True
    except Exception:
        return False

def check_system_resources() -> bool:
    """Check if system resources are within acceptable limits."""
    try:
        import psutil
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 95:
            return False
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 95:
            return False
        
        # Check disk usage
        disk = psutil.disk_usage('/')
        if disk.percent > 95:
            return False
        
        return True
    except Exception:
        return False
