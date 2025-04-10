# analytics.py
"""
Analytics and usage‑tracking module.
Logs events with timestamps.
"""

import time

class AnalyticsTracker:
    def __init__(self, logs=None):
        """
        Parameters
        ----------
        logs : list | None
            Pass an existing list (e.g. st.session_state.analytics_logs)
            so the data survives Streamlit reruns.
        """
        self.logs = logs if logs is not None else []

    def log_event(self, event_name, details=None):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        entry = {"timestamp": timestamp, "event": event_name, "details": details}
        self.logs.append(entry)
        # still print to server log for debugging
        print(f"[Analytics] {timestamp} – {event_name}: {details}")

    def get_logs(self):
        return self.logs
