from abc import ABC, abstractmethod
from typing import Dict, Any


class DataProvider(ABC):
    """
    Parent interface for all data providers.

    This defines the contract that every provider
    in the system MUST follow.

    Key design principles:
    - No implementation logic
    - No assumptions about transport (HTTP, DB, cache)
    - Strong typing for predictability
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable identifier for the provider.
        Used for logging, caching, and metrics.
        """
        pass

    @abstractmethod
    def fetch(self) -> Dict[str, Any]:
        """
        Fetch data from the provider.

        Responsibilities:
        - Return structured data
        - Raise exceptions on failure
        - Be side-effect free

        NOTE:
        This is intentionally synchronous for now.
        We will evolve this in later steps.
        """
        pass
