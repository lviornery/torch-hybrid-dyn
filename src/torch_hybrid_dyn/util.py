import enum


class NetType(enum.Enum):
    FORCE = enum.auto()
    EVENT = enum.auto()
    RESET = enum.auto()
    DYNAMICS = enum.auto()

class SeriesType(enum.Enum):
    FULL = enum.auto()
    FLOW = enum.auto()
    RESET = enum.auto()  