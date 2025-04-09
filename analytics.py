# analytics.py
"""
Analytics and usage tracking module.
Logs events with timestamps.
"""

import time

class AnalyticsTracker:
    def __init__(self):
        self.logs = []

    def log_event(self, event_name, details=None):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_entry = {"timestamp": timestamp, "event": event_name, "details": details}
        self.logs.append(log_entry)
        print(f"[Analytics] {timestamp} - {event_name}: {details}")

    def get_logs(self):
        return self.logs
