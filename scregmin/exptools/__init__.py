#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
from .static import run_all_static_retailer

from typing import Any, Dict
import numpy as np

def get_best_final_value(values: Dict[Any, np.ndarray]) -> Any:

    best_idx = np.argmax([values[key][-1] for key in values])
    best_key = list(values.keys())[best_idx]

    return best_key
