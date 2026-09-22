from .base import Task as Task
from .counting import Counting as Counting
from .existence import Existence as Existence

TASKS: dict[str, type[Task]] = {"counting": Counting, "existence": Existence}
