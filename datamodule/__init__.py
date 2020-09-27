from enum import Enum, auto


class DatasetSplit(Enum):
    """Enum for train, validation and test split.
    """
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()
