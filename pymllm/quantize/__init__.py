# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import platform
from . import gguf, spinquant
from . import quantize_pass
from . import solver

if platform.machine().lower().startswith("arm"):
    from . import kai
