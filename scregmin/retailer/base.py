#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
from typing import Tuple, Optional, Dict, Union
import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod


class BaseRetailer(ABC):

    def __init__(self, seed: Optional[int] = None):

        if seed is None:
            seed = sum([ord(s) for s in "retailer"])

        self.seed = seed
        self.reset_rng()

    def reset_rng(self, seed: Optional[int] = None):
        if seed is None:
            self.rng = np.random.RandomState(self.seed)
        else:
            self.rng = np.random.RandomState(seed)

    @abstractmethod
    def act(self, wholesale_price: float) -> Tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def learn(self, demand: float):
        raise NotImplementedError


class RandomRetailer(BaseRetailer):

    def act(self, wholesale_price: float) -> Tuple[float, float]:
        retail_price, quantity = self.rng.random(2)
        return retail_price, quantity

    def learn(self, demand: float):
        pass


class ConstantRetailer(BaseRetailer):

    def __init__(self,
                 retail_price: float,
                 quantity: float,
                 seed: Optional[int] = None):

        self.retail_price = retail_price
        self.quantity = quantity

    def act(self, wholesale_price: float) -> Tuple[float, float]:
        return self.retail_price, self.quantity

    def learn(self, demand: float):
        pass
