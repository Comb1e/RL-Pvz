"""Read-only pickle compatibility for retired search checkpoints.

No search implementation remains. Autonomous resume rejects these payloads;
the enum permits inspection and compatible weights-only initialization.
"""

from enum import StrEnum


class SearchPhase(StrEnum):
    QUEUED = "queued"
    PILOT = "pilot"
    REFERENCE = "reference_frozen"
    DISCOVERY = "discovery"
    CONFIRMATION = "confirmation"
    READY = "ready"
    COMMITTED = "action_committed"
