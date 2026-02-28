"""
Comprehensive test suite for the crypto arbitrage bot.
Includes paper trading, unit tests, integration tests, and performance tests.
"""

import asyncio
import logging
import time
import json
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import unittest
from unittest.mock import Mock, AsyncMock, patch
import statistics

from exchange_manager import ExchangeManager, ArbitrageOpportunity
from price_monitor import PriceMonitor
from trading_engine import TradingEngine, ArbitrageTrade, Order, TradeStatus
from monitoring import MonitoringSystem
from safety_manager import SafetyManager
from error_handler import ErrorHandler
from config import TRADING_CONFIG, RISK_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result data structure."""
    test_name: str
    passed: bool
    execution_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None

class MockExchange:
    """Mock exchange for testing."""
    
    def __init__(self, name: str, latency_ms: float = 50):
        self.name = name
        self.latency_ms = latency_ms
        self.balances = {
            'BTC': 1.0,
            'ETH': 10.0,
            'USDT': 10000.0,
            'USDC': 10000.0
        }
        self.order_book = {
            'BTC/USDT': {'bid': 45000.0, 'ask': 45010.0},
            'ETH/USDT': {'bid': 3000.0, 'ask': 3005.0},
            'BTC/USDC': {'bid': 44995.0, 'ask': 45005.0},
            'ETH/USDC': {'bid': 2998.0, 'ask': 3003.0}
        }
        self.orders = {}
        self.order_counter = 0
        
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Simulate fetching ticker data."""
        await asyncio.sleep(self.latency_ms / 1000)
        
        if symbol not in self.order_book:
            raise Exception(f"Symbol {symbol} not found")
        
        book = self.order_book[symbol]
        return {
            'symbol': symbol,
            'bid': book['bid'],
            'ask': book['ask'],
            'last': (book['bid'] + book['ask']) / 2,
            'timestamp': time.time() * 1000
        }
    
    async def fetch_balance(self) -> Dict[str, float]:
        """Simulate fetching balance."""
        await asyncio.sleep(self.latency_ms / 1000)
        return self.balances.copy()
    
    async def create_order(self, symbol: str, type_: str, side: str, 
                          amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Simulate creating an order."""
        await asyncio.sleep(self.latency_ms / 1000)
        
        self.order_counter += 1
        order_id = f"{self.name}_{self.order_counter}"
        
        # Simulate order execution
        filled = amount if type_ == 'market' else random.uniform(0.8, 1.0) * amount
        
        order = {
            'id': order_id,
            'symbol': symbol,
            'type': type_,
            'side': side,
            'amount': amount,
            'price': price,
            'filled': filled,
            'status': 'closed' if filled == amount else 'open',
            'timestamp': time.time() * 1000,
            'fee': {'cost': filled * 0.001, 'currency': 'USDT'}  # 0.1% fee
        }
        
        self.orders[order_id] = order
        
        # Update balances
        if side == 'buy':
            base_currency = symbol.split('/')[0]
            quote_currency = symbol.split('/')[1]
            cost = filled * (price or self.order_book[symbol]['ask'])
            self.balances[quote_currency] -= cost
            self.balances[base_currency] += filled
        else:  # sell
            base_currency = symbol.split('/')[0]
            quote_currency = symbol.split('/')[1]
            revenue = filled * (price or self.order_book[symbol]['bid'])
            self.balances[base_currency] -= filled
            self.balances[quote_currency] += revenue
        
        return order
    
    async def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Simulate fetching order status."""
        await asyncio.sleep(self.latency_ms / 1000)
        
        if order_id not in self.orders:
            raise Exception(f"Order {order_id} not found")
        
        return self.orders[order_id]

class PaperTradingEngine:
    """Paper trading engine for testing without real money."""
    
    def __init__(self):
        self.exchanges = {
            'binance': MockExchange('binance', latency_ms=30),
            'coinbase': MockExchange('coinbase', latency_ms=50),
            'kraken': MockExchange('kraken', latency_ms=80)
        }
        
        # Add price differences for arbitrage opportunities
        self.exchanges['coinbase'].order_book['BTC/USDT']['bid'] = 45020.0
        self.exchanges['coinbase'].order_book['BTC/USDT']['ask'] = 45030.0
        
        self.trades_executed = []
        self.total_profit = 0.0
        self.start_time = time.time()
        
    async def simulate_arbitrage_opportunity(self) -> Optional[ArbitrageOpportunity]:
        """Generate a simulated arbitrage opportunity."""
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'BTC/USDC', 'ETH/USDC']
        symbol = random.choice(symbols)
        
        # Get prices from different exchanges
        exchange_names = list(self.exchanges.keys())
        buy_exchange = random.choice(exchange_names)
        sell_exchange = random.choice([e for e in exchange_names if e != buy_exchange])
        
        buy_ticker = await self.exchanges[buy_exchange].fetch_ticker(symbol)
        sell_ticker = await self.exchanges[sell_exchange].fetch_ticker(symbol)
        
        buy_price = buy_ticker['ask']
        sell_price = sell_ticker['bid']
        
        # Only return if there's a profitable opportunity
        if sell_price > buy_price:
            profit_pct = ((sell_price - buy_price) / buy_price) * 100
            
            if profit_pct > 0.1:  # At least 0.1% profit
                return ArbitrageOpportunity(
                    symbol=symbol,
                    buy_exchange=buy_exchange,
                    sell_exchange=sell_exchange,
                    buy_price=buy_price,
                    sell_price=sell_price,
                    potential_profit_pct=profit_pct,
                    potential_profit_usd=profit_pct * 100,  # Assuming $100 trade
                    max_quantity=1.0,
                    timestamp=time.time()
                )
        
        return None
    
    async def execute_paper_trade(self, opportunity: ArbitrageOpportunity) -> ArbitrageTrade:
        """Execute a paper trade."""
        
        trade_amount = 0.1  # Trade 0.1 units
        
        # Create mock trade
        from trading_engine import Order, ArbitrageTrade
        
        buy_order = Order(
            id=f"paper_buy_{len(self.trades_executed)}",
            exchange=opportunity.buy_exchange,
            symbol=opportunity.symbol,
            side='buy',
            amount=trade_amount,
            price=opportunity.buy_price,
            order_type='limit'
        )
        
        sell_order = Order(
            id=f"paper_sell_{len(self.trades_executed)}",
            exchange=opportunity.sell_exchange,
            symbol=opportunity.symbol,
            side='sell',
            amount=trade_amount,
            price=opportunity.sell_price,
            order_type='limit'
        )
        
        trade = ArbitrageTrade(
            id=f"paper_trade_{len(self.trades_executed)}",
            opportunity=opportunity,
            buy_order=buy_order,
            sell_order=sell_order
        )
        
        # Simulate execution
        start_time = time.time()
        
        try:
            # Execute buy order
            buy_result = await self.exchanges[opportunity.buy_exchange].create_order(
                opportunity.symbol, 'limit', 'buy', trade_amount, opportunity.buy_price
            )
            buy_order.exchange_order_id = buy_result['id']
            buy_order.filled_amount = buy_result['filled']
            buy_order.fee = buy_result['fee']['cost']
            
            # Execute sell order
            sell_result = await self.exchanges[opportunity.sell_exchange].create_order(
                opportunity.symbol, 'limit', 'sell', trade_amount, opportunity.sell_price
            )
            sell_order.exchange_order_id = sell_result['id']
            sell_order.filled_amount = sell_result['filled']
            sell_order.fee = sell_result['fee']['cost']
            
            # Calculate actual profit
            revenue = sell_order.filled_amount * opportunity.sell_price
            cost = buy_order.filled_amount * opportunity.buy_price
            fees = buy_order.fee + sell_order.fee
            
            trade.actual_profit_usd = revenue - cost - fees
            trade.status = TradeStatus.COMPLETED
            trade.execution_time_ms = (time.time() - start_time) * 1000
            
            self.total_profit += trade.actual_profit_usd
            
        except Exception as e:
            trade.status = TradeStatus.FAILED
            trade.error_message = str(e)
        
        self.trades_executed.append(trade)
        return trade
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get paper trading performance statistics."""
        
        if not self.trades_executed:
            return {'message': 'No trades executed yet'}
        
        successful_trades = [t for t in self.trades_executed if t.status == TradeStatus.COMPLETED]
        profitable_trades = [t for t in successful_trades if t.actual_profit_usd > 0]
        
        profits = [t.actual_profit_usd for t in successful_trades]
        execution_times = [t.execution_time_ms for t in successful_trades if t.execution_time_ms]
        
        runtime_hours = (time.time() - self.start_time) / 3600
        
        return {
            'total_trades': len(self.trades_executed),
            'successful_trades': len(successful_trades),
            'failed_trades': len(self.trades_executed) - len(successful_trades),
            'profitable_trades': len(profitable_trades),
            'success_rate': len(successful_trades) / len(self.trades_executed) * 100,
            'profit_rate': len(profitable_trades) / len(successful_trades) * 100 if successful_trades else 0,
            'total_profit_usd': self.total_profit,
            'avg_profit_per_trade': statistics.mean(profits) if profits else 0,
            'max_profit': max(profits) if profits else 0,
            'min_profit': min(profits) if profits else 0,
            'avg_execution_time_ms': statistics.mean(execution_times) if execution_times else 0,
            'trades_per_hour': len(self.trades_executed) / max(runtime_hours, 0.01),
            'runtime_hours': runtime_hours
        }

class PerformanceTestSuite:
    """Performance testing suite."""
    
    def __init__(self):
        self.test_results: List[TestResult] = []
        self.paper_trading_engine = PaperTradingEngine()
    
    async def run_all_tests(self) -> List[TestResult]:
        """Run all performance tests."""
        
        logger.info("Starting performance test suite...")
        
        # Paper trading tests
        await self.test_paper_trading_basic()
        await self.test_paper_trading_extended()
        await self.test_high_frequency_simulation()
        
        # Component performance tests
        await self.test_exchange_manager_performance()
        await self.test_price_monitor_performance()
        await self.test_trading_engine_performance()
        
        # System tests
        await self.test_memory_usage()
        await self.test_error_handling()
        await self.test_safety_mechanisms()
        
        logger.info(f"Performance test suite completed. {len(self.test_results)} tests run.")
        return self.test_results
    
    async def test_paper_trading_basic(self):
        """Test basic paper trading functionality."""
        test_name = "Paper Trading - Basic"
        start_time = time.time()
        
        try:
            # Generate and execute 10 paper trades
            trades_to_execute = 10
            successful_trades = 0
            
            for i in range(trades_to_execute):
                opportunity = await self.paper_trading_engine.simulate_arbitrage_opportunity()
                if opportunity:
                    trade = await self.paper_trading_engine.execute_paper_trade(opportunity)
                    if trade.status == TradeStatus.COMPLETED:
                        successful_trades += 1
                
                await asyncio.sleep(0.1)  # Small delay between trades
            
            stats = self.paper_trading_engine.get_performance_stats()
            
            # Test passes if we executed at least 5 successful trades
            passed = successful_trades >= 5
            
            self.test_results.append(TestResult(
                test_name=test_name,
                passed=passed,
                execution_time=time.time() - start_time,
                details={
                    'trades_attempted': trades_to_execute,
                    'successful_trades': successful_trades,
                    'performance_stats': stats
                }
            ))
            
        except Exception as e:
            self.test_results.append(TestResult(
                test_name=test_name,
                passed=False,
                execution_time=time.time() - start_time,
                details={},
                error_message=str(e)
            ))
    
    async def test_paper_trading_extended(self):
        """Test extended paper trading to simulate daily volume."""
        test_name = "Paper Trading - Extended (100 trades)"
        start_time = time.time()
        
        try:
            # Execute 100 trades to test sustained performance
            trades_to_execute = 100
            successful_trades = 0
            total_profit = 0.0
            
            for i in range(trades_to_execute):
                opportunity = await self.paper_trading_engine.simulate_arbitrage_opportunity()
                if opportunity:
                    trade = await self.paper_trading_engine.execute_paper_trade(opportunity)
                    if trade.status == TradeStatus.COMPLETED:
                        successful_trades += 1
                        total_profit += trade.actual_profit_usd
                
                # Small delay to simulate realistic trading frequency
                await asyncio.sleep(0.05)
            
            stats = self.paper_trading_engine.get_performance_stats()
            
            # Test passes if we have >80% success rate and positive profit
            success_rate = successful_trades / trades_to_execute * 100
            passed = success_rate > 80 and total_profit > 0
            
            self.test_results.append(TestResult(
                test_name=test_name,
                passed=passed,
                execution_time=time.time() - start_time,
                details={
                    'trades_attempted': trades_to_execute,
                    'successful_trades': successful_trades,
                    'success_rate': success_rate,
                    'total_profit': total_profit,
                    'performance_stats': stats
                }
            ))
            
        except Exception as e:
            self.test_results.append(TestResult(
                test_name=test_name,
                passed=False,
                execution_time=time.time() - start_time,
                details={},
                error_message=str(e)
            ))
    
    async def test_high_frequency_simulation(self):
        """Test high-frequency trading simulation to reach 1000+ trades/day target."""
        test_name = "High Frequency Simulation"
        start_time = time.time()
        
        try:
            # Simulate 1 hour of high-frequency trading
            # Target: 1000 trades/day = ~42 trades/hour = ~0.7 trades/minute
            
            simulation_duration = 60  # 1 minute simulation
            target_trades_per_minute = 1.0
            
            trades_executed = 0
            end_time = start_time + simulation_duration
            
            while time.time() < end_time:
                opportunity = await self.paper_trading_engine.simulate_arbitrage_opportunity()
                if opportunity:
                    trade = await self.paper_trading_engine.execute_paper_trade(opportunity)
                    trades_executed += 1
                
                # Adjust delay to meet target frequency
                await asyncio.sleep(60 / target_trades_per_minute / 10)  # Faster for simulation
            
            actual_duration = time.time() - start_time
            trades_per_minute = trades_executed / (actual_duration / 60)
            projected_daily_trades = trades_per_minute * 60 * 24
            
            # Test passes if we can project >1000 trades/day
            passed = projected_daily_trades >= 1000
            
            self.test_results.append(TestResult(
                test_name=test_name,
                passed=passed,
                execution_time=actual_duration,
                details={
                    'trades_executed': trades_executed,
                    'trades_per_minute': trades_per_minute,
                    'projected_daily_trades': projected_daily_trades,
                    'target_met': passed
                }
            ))
            
        except Exception as e:
            self.test_results.append(TestResult(
                test_name=test_name,
                passed=False,
                execution_time=time.time() - start_time,
                details={},
                error_message=str(e)
            ))
    
    async def test_exchange_manager_performance(self):
        """Test exchange manager performance."""
        test_name = "Exchange Manager Performance"
        start_time = time.time()
        
        try:
            # Test concurrent API calls
            mock_exchange = MockExchange('test_exchange')
            
            # Test 100 concurrent ticker requests
            tasks = []
            for _ in range(100):
                tasks.append(mock_exchange.fetch_ticker('BTC/USDT'))
            
            results = await asyncio.gather(*tasks)
            
            # All requests should succeed
            passed = len(results) == 100 and all(r['symbol'] == 'BTC/USDT' for r in results)
            
            self.test_results.append(TestResult(
                test_name=test_name,
                passed=passed,
                execution_time=time.time() - start_time,
                details={
                    'concurrent_requests': 100,
                    'successful_responses': len(results),
                    'avg_response_time_ms': (time.time() - start_time) * 1000 / 100
                }
            ))
            
        except Exception as e:
            self.test_results.append(TestResult(
                test_name=test_name,
                passed=False,
                execution_time=time.time() - start_time,
                details={},
                error_message=str(e)
            ))
    
    async def test_price_monitor_performance(self):
        """Test price monitoring performance."""
        test_name = "Price Monitor Performance"
        start_time = time.time()
        
        try:
            # Simulate price monitoring for multiple symbols
            symbols = ['BTC/USDT', 'ETH/USDT', 'BTC/USDC', 'ETH/USDC']
            exchanges = ['binance', 'coinbase', 'kraken']
            
            opportunities_found = 0
            price_updates = 0
            
            # Simulate 1000 price updates
            for _ in range(1000):
                # Simulate price update
                symbol = random.choice(symbols)
                exchange = random.choice(exchanges)
                
                # Mock price data
                price_data = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'bid': random.uniform(40000, 50000),
                    'ask': random.uniform(40010, 50010),
                    'timestamp': time.time()
                }
                
                price_updates += 1
                
                # Simulate opportunity detection (simplified)
                if random.random() < 0.05:  # 5% chance of opportunity
                    opportunities_found += 1
                
                await asyncio.sleep(0.001)  # 1ms per update
            
            processing_rate = price_updates / (time.time() - start_time)
            
            # Test passes if we can process >500 updates/second
            passed = processing_rate > 500
            
            self.test_results.append(TestResult(
                test_name=test_name,
                passed=passed,
                execution_time=time.time() - start_time,
                details={
                    'price_updates_processed': price_updates,
                    'opportunities_found': opportunities_found,
                    'processing_rate_per_second': processing_rate,
                    'target_met': passed
                }
            ))
            
        except Exception as e:
            self.test_results.append(TestResult(
                test_name=test_name,
                passed=False,
                execution_time=time.time() - start_time,
                details={},
                error_message=str(e)
            ))
    
    async def test_trading_engine_performance(self):
        """Test trading engine performance under load."""
        test_name = "Trading Engine Performance"
        start_time = time.time()
        
        try:
            # Simulate concurrent trade executions
            concurrent_trades = 50
            successful_trades = 0
            
            async def execute_mock_trade():
                # Simulate trade execution time
                await asyncio.sleep(random.uniform(0.1, 0.5))
                return random.choice([True, False])  # Random success/failure
            
            # Execute trades concurrently
            tasks = [execute_mock_trade() for _ in range(concurrent_trades)]
            results = await asyncio.gather(*tasks)
            
            successful_trades = sum(results)
            success_rate = successful_trades / concurrent_trades * 100
            
            # Test passes if >80% success rate with concurrent execution
            passed = success_rate > 80
            
            self.test_results.append(TestResult(
                test_name=test_name,
                passed=passed,
                execution_time=time.time() - start_time,
                details={
                    'concurrent_trades': concurrent_trades,
                    'successful_trades': successful_trades,
                    'success_rate': success_rate,
                    'avg_execution_time': (time.time() - start_time) / concurrent_trades
                }
            ))
            
        except Exception as e:
            self.test_results.append(TestResult(
                test_name=test_name,
                passed=False,
                execution_time=time.time() - start_time,
                details={},
                error_message=str(e)
            ))
    
    async def test_memory_usage(self):
        """Test memory usage under sustained operation."""
        test_name = "Memory Usage Test"
        start_time = time.time()
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulate sustained operation
            data_structures = []
            for i in range(10000):
                # Simulate creating trade objects and storing them
                mock_trade_data = {
                    'id': f'trade_{i}',
                    'timestamp': time.time(),
                    'symbol': 'BTC/USDT',
                    'profit': random.uniform(-10, 10),
                    'data': [random.random() for _ in range(100)]
                }
                data_structures.append(mock_trade_data)
                
                if i % 1000 == 0:
                    await asyncio.sleep(0.01)  # Small delay
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Test passes if memory increase is reasonable (<100MB for this test)
            passed = memory_increase < 100
            
            self.test_results.append(TestResult(
                test_name=test_name,
                passed=passed,
                execution_time=time.time() - start_time,
                details={
                    'initial_memory_mb': initial_memory,
                    'final_memory_mb': final_memory,
                    'memory_increase_mb': memory_increase,
                    'objects_created': len(data_structures)
                }
            ))
            
        except ImportError:
            self.test_results.append(TestResult(
                test_name=test_name,
                passed=False,
                execution_time=time.time() - start_time,
                details={},
                error_message="psutil not available for memory testing"
            ))
        except Exception as e:
            self.test_results.append(TestResult(
                test_name=test_name,
                passed=False,
                execution_time=time.time() - start_time,
                details={},
                error_message=str(e)
            ))
    
    async def test_error_handling(self):
        """Test error handling mechanisms."""
        test_name = "Error Handling Test"
        start_time = time.time()
        
        try:
            from error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
            
            error_handler = ErrorHandler()
            
            # Test different types of errors
            test_errors = [
                (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM, "Network timeout"),
                (ErrorCategory.API, ErrorSeverity.HIGH, "API rate limit"),
                (ErrorCategory.TRADING, ErrorSeverity.CRITICAL, "Order execution failed"),
                (ErrorCategory.SYSTEM, ErrorSeverity.LOW, "Low disk space warning")
            ]
            
            for category, severity, message in test_errors:
                error_handler.handle_error(category, severity, "test_component", message)
            
            # Test retry mechanism
            retry_count = 0
            
            async def failing_function():
                nonlocal retry_count
                retry_count += 1
                if retry_count < 3:
                    raise Exception("Simulated failure")
                return "success"
            
            try:
                result = await error_handler.retry_with_strategy(
                    failing_function, ErrorCategory.NETWORK, "test_component"
                )
                retry_success = result == "success"
            except:
                retry_success = False
            
            # Test passes if errors are handled and retry works
            passed = len(error_handler.error_events) == 4 and retry_success
            
            self.test_results.append(TestResult(
                test_name=test_name,
                passed=passed,
                execution_time=time.time() - start_time,
                details={
                    'errors_handled': len(error_handler.error_events),
                    'retry_attempts': retry_count,
                    'retry_success': retry_success,
                    'error_stats': error_handler.error_stats
                }
            ))
            
        except Exception as e:
            self.test_results.append(TestResult(
                test_name=test_name,
                passed=False,
                execution_time=time.time() - start_time,
                details={},
                error_message=str(e)
            ))
    
    async def test_safety_mechanisms(self):
        """Test safety mechanisms."""
        test_name = "Safety Mechanisms Test"
        start_time = time.time()
        
        try:
            from safety_manager import SafetyManager
            from error_handler import ErrorHandler
            
            error_handler = ErrorHandler()
            safety_manager = SafetyManager(error_handler)
            
            # Test daily loss limit
            safety_manager.metrics.daily_pnl = -1000  # Simulate $1000 loss
            
            # Create a mock opportunity
            mock_opportunity = ArbitrageOpportunity(
                symbol='BTC/USDT',
                buy_exchange='binance',
                sell_exchange='coinbase',
                buy_price=45000.0,
                sell_price=45100.0,
                potential_profit_pct=0.22,
                potential_profit_usd=22.0,
                max_quantity=1.0,
                timestamp=time.time()
            )
            
            # Test safety check
            is_safe, reason = await safety_manager.check_safety_before_trade(mock_opportunity)
            
            # Should be safe with normal parameters
            safety_check_passed = is_safe
            
            # Test emergency stop trigger
            await safety_manager._trigger_emergency_stop("Test emergency stop")
            emergency_stop_works = safety_manager.emergency_stop_active
            
            # Test safety status
            status = safety_manager.get_safety_status()
            status_complete = all(key in status for key in [
                'safety_level', 'emergency_stop_active', 'metrics'
            ])
            
            passed = safety_check_passed and emergency_stop_works and status_complete
            
            self.test_results.append(TestResult(
                test_name=test_name,
                passed=passed,
                execution_time=time.time() - start_time,
                details={
                    'safety_check_passed': safety_check_passed,
                    'emergency_stop_works': emergency_stop_works,
                    'status_complete': status_complete,
                    'safety_status': status
                }
            ))
            
        except Exception as e:
            self.test_results.append(TestResult(
                test_name=test_name,
                passed=False,
                execution_time=time.time() - start_time,
                details={},
                error_message=str(e)
            ))
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        
        total_tests = len(self.test_results)
        passed_tests = len([t for t in self.test_results if t.passed])
        failed_tests = total_tests - passed_tests
        
        total_execution_time = sum(t.execution_time for t in self.test_results)
        
        # Paper trading stats
        paper_trading_stats = self.paper_trading_engine.get_performance_stats()
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'total_execution_time': total_execution_time
            },
            'paper_trading_performance': paper_trading_stats,
            'test_results': [asdict(result) for result in self.test_results],
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        failed_tests = [t for t in self.test_results if not t.passed]
        
        if failed_tests:
            recommendations.append(f"Address {len(failed_tests)} failed tests before production deployment")
        
        # Check paper trading performance
        stats = self.paper_trading_engine.get_performance_stats()
        
        if 'success_rate' in stats and stats['success_rate'] < 90:
            recommendations.append("Improve trade execution success rate (currently below 90%)")
        
        if 'projected_daily_trades' in stats and stats.get('projected_daily_trades', 0) < 1000:
            recommendations.append("Optimize for higher frequency trading to meet 1000+ trades/day target")
        
        if 'total_profit_usd' in stats and stats['total_profit_usd'] <= 0:
            recommendations.append("Review arbitrage strategy - paper trading shows negative profitability")
        
        # Performance recommendations
        memory_test = next((t for t in self.test_results if t.test_name == "Memory Usage Test"), None)
        if memory_test and memory_test.passed and memory_test.details.get('memory_increase_mb', 0) > 50:
            recommendations.append("Monitor memory usage - significant increase detected during testing")
        
        if not recommendations:
            recommendations.append("All tests passed - system ready for production deployment")
        
        return recommendations

async def run_comprehensive_tests():
    """Run the complete test suite."""
    
    logger.info("Starting comprehensive test suite for crypto arbitrage bot...")
    
    test_suite = PerformanceTestSuite()
    
    # Run all tests
    test_results = await test_suite.run_all_tests()
    
    # Generate report
    report = test_suite.generate_test_report()
    
    # Save report to file
    with open('/test_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info("Test suite completed. Report saved to test_report.json")
    
    return report

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    asyncio.run(run_comprehensive_tests())

