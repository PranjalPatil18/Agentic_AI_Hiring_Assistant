# memory.py
"""
Module for session-based memory.
This simple in-memory store is used by our agent to track conversation state.
"""

class SessionMemory:
    def __init__(self):
        self.memory = {}

    def get(self, key, default=None):
        return self.memory.get(key, default)

    def set(self, key, value):
        self.memory[key] = value

    def clear(self):
        self.memory.clear()
