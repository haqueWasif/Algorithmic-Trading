from datetime import datetime

class Alert:
    def __init__(self, severity, component, message, timestamp):
        self.severity = severity
        self.component = component
        self.message = message
        self.timestamp = timestamp

alert = Alert("critical", "System", "Test message", datetime.now().timestamp())

message_text = (
    f"*ALERT ({alert.severity.upper()})*\n" +
    f"*Component:* `{alert.component}`\n" +
    f"*Message:* `{alert.message}`\n" +
    f"*Timestamp:* `{datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')}`"
)

print(message_text)


