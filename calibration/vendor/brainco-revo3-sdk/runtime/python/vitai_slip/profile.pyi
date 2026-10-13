from enum import IntEnum

__all__ = ['SlipState']

class SlipState(IntEnum):
    CONTACT = 1
    INCIPIENT_SLIP = 2
    PARTIAL_SLIP = 3
    COMPLETE_SLIP = 4
    NO_OBJ = 5
    STEADY_HOLD = 6
