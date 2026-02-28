import logging
import asyncio
import psutil
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Any, List, Optional
import uuid
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    alert_id: str
    alert_type: str
    component: str
    message: str
    severity: str # e.g., info, warning, error, critical
    timestamp: float
    resolved: bool = False

class AlertManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alerts: deque[Alert] = deque(maxlen=1000) # Store last 1000 alerts
        self.telegram_enabled = bool(config.get("telegram_bot_token") and config.get("telegram_chat_id"))
        self.email_enabled = bool(config.get("email_smtp_server") and config.get("email_username") and config.get("email_password"))

    def create_alert(self, alert_type: str, message: str, severity: str, component: str):
        alert_id = str(uuid.uuid4())
        new_alert = Alert(alert_id, alert_type, component, message, severity, time.time())
        self.alerts.append(new_alert)
        logger.log(getattr(logging, severity.upper(), logging.INFO), f"ALERT ({severity.upper()}) - {component}: {message}")
        
        if severity in ["error", "critical", "warning"]:
            if self.telegram_enabled:
                asyncio.create_task(self._send_telegram_alert(new_alert))
            if self.email_enabled:
                asyncio.create_task(self._send_email_alert(new_alert))

    async def _send_telegram_alert(self, alert: Alert ):
        try:
            import httpx # Import httpx inside the function or at the top of the file
            
            bot_token = self.config.get("telegram_bot_token" )
            chat_id = self.config.get("telegram_chat_id")

            if not bot_token or not chat_id:
                logger.warning("Telegram bot token or chat ID not configured. Cannot send alert.")
                return

            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            message_text = (
                f"*ALERT ({alert.severity.upper()})*\n" +
                f"*Component:* `{alert.component}`\n" +
                f"*Message:* `{alert.message}`\n" +
                f"*Timestamp:* `{alert.timestamp}`"
            )
            payload = {
                "chat_id": chat_id,
                "text": message_text,
               # "parse_mode": "MarkdownV2" # Use MarkdownV2 for formatting
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=10)
                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            logger.info(f"Successfully sent Telegram alert: {alert.message}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to send Telegram alert (HTTP Error ): {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Failed to send Telegram alert (Request Error ): {e}")
        except Exception as e:
            logger.error(f"Failed to send Telegram alert (General Error): {e}")

    async def _send_email_alert(self, alert: Alert):
        try:
            # Create the simplest possible email
            msg = MIMEText(f"Alert: {alert.message}", "plain")
            msg["From"] = self.config["email_username"]
            msg["To"] = self.config.get("email_recipients", [""])[0]  # Use only the first recipient
            msg["Subject"] = f"Bot Alert - {alert.severity.upper()}"
            
            # Use SMTP_SSL on port 465 for Gmail (more reliable than STARTTLS)
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(self.config["email_username"], self.config["email_password"])
                server.send_message(msg)
            logger.info(f"Sent email alert: {alert.message}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")


    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in list(self.alerts) if alert.timestamp >= cutoff_time]

class PerformanceMonitor:
    def __init__(self):
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.avg_execution_time_ms = 0.0
        self.opportunities_per_minute = 0
        self.active_trades_count = 0
        self.last_update_time = time.time()

    def update_metrics(self, active_trades_count: int, opportunities_found: int, trade_execution_times: List[float]):
        self.cpu_usage = psutil.cpu_percent(interval=None) # Non-blocking
        self.memory_usage = psutil.virtual_memory().percent
        
        if trade_execution_times:
            self.avg_execution_time_ms = sum(trade_execution_times) / len(trade_execution_times)
        else:
            self.avg_execution_time_ms = 0.0

        current_time = time.time()
        time_diff = current_time - self.last_update_time
        if time_diff > 0:
            self.opportunities_per_minute = (opportunities_found / time_diff) * 60
        self.last_update_time = current_time
        self.active_trades_count = active_trades_count

    def get_current_metrics(self) -> Dict[str, Any]:
        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "opportunities_per_minute": self.opportunities_per_minute,
            "active_trades": self.active_trades_count
        }

class MonitoringSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_manager = AlertManager(config["MONITORING_CONFIG"])
        self.performance_monitor = PerformanceMonitor()
        self.is_running = False
        self.health_check_interval = config["PERFORMANCE_CONFIG"].get("main_loop_interval", 10) # Default 10 seconds

    async def start(self):
        if self.is_running:
            logger.info("Monitoring system is already running.")
            return
        logger.info("Starting monitoring system...")
        self.is_running = True
        asyncio.create_task(self._health_check_loop())

    async def stop(self):
        if not self.is_running:
            logger.info("Monitoring system is not running.")
            return
        logger.info("Stopping monitoring system...")
        self.is_running = False

    async def _health_check_loop(self):
        while self.is_running:
            await self.perform_health_check()
            await asyncio.sleep(self.health_check_interval)

    async def perform_health_check(self) -> Dict[str, Any]:
        # This is a simplified health check. In a real system, you\"d check:
        # - Exchange API connectivity
        # - Database connectivity
        # - Redis connectivity
        # - Internal queue sizes
        
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_usage = psutil.virtual_memory().percent
        
        # Simulate daily loss and consecutive losses for dashboard display
        # These should ideally come from the SafetyManager
        daily_loss = 0.0 # Placeholder
        consecutive_losses = 0 # Placeholder
        risk_level = "Low"

        # Example: Trigger warning if CPU or memory is high
        if cpu_usage > self.config["MONITORING_CONFIG"]["alert_thresholds"]["cpu_usage_percent"]:
            self.alert_manager.create_alert("High CPU Usage", f"CPU usage is {cpu_usage}%", "warning", "System")
            risk_level = "Medium"
        if memory_usage > self.config["MONITORING_CONFIG"]["alert_thresholds"]["memory_usage_percent"]:
            self.alert_manager.create_alert("High Memory Usage", f"Memory usage is {memory_usage}%", "warning", "System")
            risk_level = "Medium"

        return {
            "system_status": "Operational",
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "last_checked": datetime.now().isoformat(),
            "circuit_breaker_active": False, # This should come from SafetyManager
            "daily_loss": daily_loss,
            "consecutive_losses": consecutive_losses,
            "risk_level": risk_level
        }

    def update_performance_metrics(self, active_trades_count: int, opportunities_found: int, trade_execution_times: List[float]):
        self.performance_monitor.update_metrics(active_trades_count, opportunities_found, trade_execution_times)

    def get_current_performance_metrics(self) -> Dict[str, Any]:
        return self.performance_monitor.get_current_metrics()