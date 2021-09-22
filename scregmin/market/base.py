#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
from typing import Optional, Callable
import numpy as np
from abc import ABC, abstractmethod


class BaseMarket(ABC):

    def __init__(self, seed: Optional[int] = None):

        if seed is None:
            seed = sum([ord(s) for s in "market"])

        self.seed = seed
        self.reset_rng()

    def reset_rng(self, seed: Optional[int] = None):
        if seed is None:
            self.rng = np.random.RandomState(self.seed)
        else:
            self.rng = np.random.RandomState(seed)
        
    @abstractmethod
    def act(self, retail_price: float) -> float:
        raise NotImplementedError


class RandomMarket(BaseMarket):

    def act(self, retail_price: float) -> float:
        return self.rng.random()


class ConstantMarket(BaseMarket):

    def __init__(self, demand: float):
        self.demand = demand
        super().__init__()

    def act(self, retail_price: float) -> float:
        return self.demand


class DeterministicMarket(BaseMarket):

    def __init__(self, demand_function: Callable[[float], float]):
        self.demand_function = demand_function
        super().__init__()

    def act(self, retail_price: float) -> float:
        return self.demand_function(retail_price)
