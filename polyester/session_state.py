import enum

# This is needed by object.py, but session.py imports object.py
# To break the circular import, this is its own tiny module


class SessionState(enum.Enum):
    NONE = "none"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
