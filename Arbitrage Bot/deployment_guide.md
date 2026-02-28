# Crypto Arbitrage Bot - Deployment Guide

## 🚀 Production Deployment

This guide covers deploying the crypto arbitrage bot to production environments.

## 📋 Pre-Deployment Checklist

### ✅ System Requirements

- **Operating System**: Ubuntu 20.04+ or CentOS 8+
- **Python**: 3.11 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 100GB SSD minimum
- **Network**: Stable internet with low latency to exchanges
- **CPU**: 4+ cores recommended for high-frequency trading

### ✅ Exchange Setup

1. **Create Exchange Accounts**
   - Binance, Coinbase Pro, Kraken (minimum)
   - Complete KYC verification
   - Enable API access

2. **Generate API Keys**
   - Create read-only keys for monitoring
   - Create trading keys with appropriate permissions
   - Whitelist server IP addresses
   - Store keys securely

3. **Fund Accounts**
   - Distribute capital across exchanges
   - Maintain minimum balances for trading
   - Consider withdrawal limits and fees

### ✅ Security Preparation

- [ ] Secure server with firewall rules
- [ ] Set up SSL certificates
- [ ] Configure VPN access if needed
- [ ] Prepare backup and recovery procedures
- [ ] Set up monitoring and alerting

## 🛠️ Installation Steps

### 1. Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev -y

# Install system dependencies
sudo apt install build-essential curl git -y

# Create application user
sudo useradd -m -s /bin/bash arbitrage
sudo usermod -aG sudo arbitrage
```

### 2. Application Deployment

```bash
# Switch to application user
sudo su - arbitrage

# Clone repository
git clone <repository-url> crypto_arbitrage_bot
cd crypto_arbitrage_bot

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create configuration
cp .env.example .env
# Edit .env with your settings
```

### 3. Configuration

#### Environment Variables (.env)

```env
# Production Environment
ENVIRONMENT=production
DEBUG=false

# Exchange API Credentials
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET=your_binance_secret
BINANCE_SANDBOX=false

COINBASE_API_KEY=your_coinbase_api_key
COINBASE_SECRET=your_coinbase_secret
COINBASE_PASSPHRASE=your_coinbase_passphrase
COINBASE_SANDBOX=false

KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_SECRET=your_kraken_secret

# Trading Configuration
MAX_DAILY_TRADES=5000
MAX_TRADE_AMOUNT_USD=1000
MIN_PROFIT_THRESHOLD=0.15
MAX_DAILY_LOSS_USD=1000
MAX_SINGLE_TRADE_LOSS_USD=100

# Risk Management
ENABLE_CIRCUIT_BREAKER=true
MAX_CONSECUTIVE_LOSSES=5
BALANCE_CHECK_INTERVAL=300

# Monitoring
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/arbitrage/main.log
```

### 4. System Service Setup

Create systemd service file:

```bash
sudo nano /etc/systemd/system/arbitrage-bot.service
```

```ini
[Unit]
Description=Crypto Arbitrage Bot
After=network.target

[Service]
Type=simple
User=arbitrage
Group=arbitrage
WorkingDirectory=/home/arbitrage/crypto_arbitrage_bot
Environment=PATH=/home/arbitrage/crypto_arbitrage_bot/venv/bin
ExecStart=/home/arbitrage/crypto_arbitrage_bot/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Create dashboard service:

```bash
sudo nano /etc/systemd/system/arbitrage-dashboard.service
```

```ini
[Unit]
Description=Crypto Arbitrage Dashboard
After=network.target

[Service]
Type=simple
User=arbitrage
Group=arbitrage
WorkingDirectory=/home/arbitrage/crypto_arbitrage_bot/dashboard
Environment=PATH=/home/arbitrage/crypto_arbitrage_bot/dashboard/venv/bin
ExecStart=/home/arbitrage/crypto_arbitrage_bot/dashboard/venv/bin/python src/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 5. Enable and Start Services

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable arbitrage-bot
sudo systemctl enable arbitrage-dashboard

# Start services
sudo systemctl start arbitrage-bot
sudo systemctl start arbitrage-dashboard

# Check status
sudo systemctl status arbitrage-bot
sudo systemctl status arbitrage-dashboard
```

## 🔧 Configuration Tuning

### Production Settings

#### config.py Adjustments

```python
# Production configuration
TRADING_CONFIG = {
    'max_daily_trades': 5000,
    'max_trade_amount_usd': 1000,
    'min_profit_threshold': 0.15,  # 0.15% minimum profit
    'order_timeout_seconds': 30,
    'max_open_positions': 10
}

RISK_CONFIG = {
    'max_daily_loss_usd': 1000,
    'max_single_trade_loss_usd': 100,
    'max_consecutive_losses': 5,
    'circuit_breaker_enabled': True
}

PERFORMANCE_CONFIG = {
    'price_update_interval': 0.1,  # 100ms
    'opportunity_scan_interval': 0.05,  # 50ms
    'max_concurrent_trades': 5
}
```

### Exchange-Specific Settings

```python
EXCHANGES = {
    'binance': {
        'api_key': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET'),
        'sandbox': False,
        'rate_limit': 1200,  # requests per minute
        'trading_fee': 0.001  # 0.1%
    },
    'coinbase': {
        'api_key': os.getenv('COINBASE_API_KEY'),
        'secret': os.getenv('COINBASE_SECRET'),
        'passphrase': os.getenv('COINBASE_PASSPHRASE'),
        'sandbox': False,
        'rate_limit': 10,  # requests per second
        'trading_fee': 0.005  # 0.5%
    }
}
```

## 📊 Monitoring Setup

### 1. Log Management

```bash
# Create log directory
sudo mkdir -p /var/log/arbitrage
sudo chown arbitrage:arbitrage /var/log/arbitrage

# Set up log rotation
sudo nano /etc/logrotate.d/arbitrage
```

```
/var/log/arbitrage/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 arbitrage arbitrage
    postrotate
        systemctl reload arbitrage-bot
    endscript
}
```

### 2. Monitoring Dashboard

Access the dashboard at: `http://your-server-ip:5000`

### 3. Alerting Setup

Configure Telegram alerts:

1. Create a Telegram bot via @BotFather
2. Get your chat ID
3. Add credentials to `.env`

Configure email alerts:

1. Set up app password for Gmail
2. Add SMTP settings to `.env`

## 🔒 Security Hardening

### 1. Firewall Configuration

```bash
# Enable UFW
sudo ufw enable

# Allow SSH
sudo ufw allow ssh

# Allow dashboard (restrict to specific IPs)
sudo ufw allow from YOUR_IP to any port 5000

# Allow outbound connections
sudo ufw default allow outgoing
```

### 2. SSL Certificate (Optional)

```bash
# Install Certbot
sudo apt install certbot -y

# Generate certificate
sudo certbot certonly --standalone -d your-domain.com

# Configure nginx proxy (optional)
sudo apt install nginx -y
```

### 3. API Key Security

- Store API keys in environment variables only
- Use read-only keys where possible
- Rotate keys regularly
- Monitor API usage

## 🚨 Monitoring and Alerts

### Key Metrics to Monitor

1. **Trading Performance**
   - Trades per hour
   - Success rate
   - Profit/loss
   - Execution times

2. **System Health**
   - CPU usage
   - Memory usage
   - Network latency
   - Disk space

3. **Exchange Status**
   - API response times
   - Error rates
   - Balance levels
   - Order book depth

### Alert Thresholds

```python
ALERT_THRESHOLDS = {
    'daily_loss_usd': 500,
    'consecutive_losses': 3,
    'cpu_usage_percent': 80,
    'memory_usage_percent': 85,
    'api_error_rate': 0.05,
    'execution_time_ms': 1000
}
```

## 🔄 Backup and Recovery

### 1. Database Backup

```bash
# Create backup script
nano /home/arbitrage/backup.sh
```

```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/home/arbitrage/backups"

mkdir -p $BACKUP_DIR

# Backup configuration
cp /home/arbitrage/crypto_arbitrage_bot/.env $BACKUP_DIR/env_$DATE

# Backup logs
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz /var/log/arbitrage/

# Backup trade data
cp /home/arbitrage/crypto_arbitrage_bot/dashboard/src/database/app.db $BACKUP_DIR/trades_$DATE.db

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
find $BACKUP_DIR -name "*.db" -mtime +30 -delete
```

### 2. Automated Backups

```bash
# Add to crontab
crontab -e

# Backup every 6 hours
0 */6 * * * /home/arbitrage/backup.sh
```

## 🔧 Maintenance

### Regular Tasks

1. **Daily**
   - Check system status
   - Review trading performance
   - Monitor alerts

2. **Weekly**
   - Update dependencies
   - Review logs
   - Check disk space

3. **Monthly**
   - Rotate API keys
   - Update system packages
   - Review and optimize configuration

### Update Procedure

```bash
# Stop services
sudo systemctl stop arbitrage-bot arbitrage-dashboard

# Backup current version
cp -r /home/arbitrage/crypto_arbitrage_bot /home/arbitrage/crypto_arbitrage_bot_backup

# Update code
cd /home/arbitrage/crypto_arbitrage_bot
git pull origin main

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt

# Run tests
python quick_test.py

# Start services
sudo systemctl start arbitrage-bot arbitrage-dashboard
```

## 🚨 Troubleshooting

### Common Issues

1. **Service Won't Start**
   ```bash
   # Check logs
   sudo journalctl -u arbitrage-bot -f
   
   # Check configuration
   python -c "import config; print('Config OK')"
   ```

2. **API Connection Issues**
   ```bash
   # Test API connectivity
   python -c "
   import ccxt
   exchange = ccxt.binance({'apiKey': 'key', 'secret': 'secret'})
   print(exchange.fetch_ticker('BTC/USDT'))
   "
   ```

3. **High Memory Usage**
   ```bash
   # Monitor memory
   htop
   
   # Check for memory leaks
   python -m memory_profiler main.py
   ```

### Emergency Procedures

1. **Emergency Stop**
   ```bash
   sudo systemctl stop arbitrage-bot
   ```

2. **Cancel All Orders**
   ```bash
   python emergency_cancel.py
   ```

3. **Restore from Backup**
   ```bash
   sudo systemctl stop arbitrage-bot arbitrage-dashboard
   cp /home/arbitrage/backups/latest/* /home/arbitrage/crypto_arbitrage_bot/
   sudo systemctl start arbitrage-bot arbitrage-dashboard
   ```

## 📈 Performance Optimization

### Production Optimizations

1. **System Tuning**
   ```bash
   # Increase file descriptor limits
   echo "arbitrage soft nofile 65536" | sudo tee -a /etc/security/limits.conf
   echo "arbitrage hard nofile 65536" | sudo tee -a /etc/security/limits.conf
   
   # Optimize network settings
   echo "net.core.rmem_max = 16777216" | sudo tee -a /etc/sysctl.conf
   echo "net.core.wmem_max = 16777216" | sudo tee -a /etc/sysctl.conf
   ```

2. **Python Optimizations**
   ```bash
   # Use PyPy for better performance (optional)
   sudo apt install pypy3 pypy3-dev
   ```

### Scaling Considerations

- **Horizontal Scaling**: Run multiple instances with different trading pairs
- **Load Balancing**: Distribute load across multiple servers
- **Database Optimization**: Use PostgreSQL for better performance
- **Caching**: Implement Redis for price data caching

## ✅ Go-Live Checklist

### Final Verification

- [ ] All tests pass
- [ ] Configuration validated
- [ ] API keys tested
- [ ] Monitoring configured
- [ ] Alerts working
- [ ] Backups scheduled
- [ ] Emergency procedures documented
- [ ] Team trained on operations

### Launch Sequence

1. **Start with Paper Trading**
   - Enable paper trading mode
   - Run for 24 hours
   - Verify all systems working

2. **Gradual Rollout**
   - Start with small trade amounts
   - Monitor closely for first week
   - Gradually increase limits

3. **Full Production**
   - Enable full trading limits
   - 24/7 monitoring
   - Regular performance reviews

---

**Deployment Status**: ✅ **READY FOR PRODUCTION**

**Support**: Create GitHub issues for deployment questions

**Last Updated**: July 2025

