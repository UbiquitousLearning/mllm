import platform
from . import gguf, spinquant

if platform.machine().lower().startswith("arm"):
    from . import kai
