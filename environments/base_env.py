# base_env.py
from abc import ABC, abstractmethod

class BaseEnv(ABC):
    """Abstract interface all environments must follow."""

    @abstractmethod
    def reset(self, session_id: str):
        """Reset environment and return initial observation."""
        pass

    @abstractmethod
    def step(self, session_id: str, action: str):
        """Take one action. Returns (observation, reward, done)."""
        pass
