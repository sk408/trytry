"""Notification models and constants."""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class NotificationCategory(str, Enum):
    INJURY = "injury"
    MATCHUP = "matchup"
    INSIGHT = "insight"


class NotificationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Notification:
    id: Optional[int] = None
    category: str = "info"
    severity: str = "info"
    title: str = ""
    message: str = ""
    created_at: str = ""
    read: bool = False
    data: str = ""  # JSON extra data
