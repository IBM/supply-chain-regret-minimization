#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
from typing import Optional
import numpy as np
from abc import ABC, abstractmethod


class BaseWholeSeller(ABC):

    def __init__(self, seed: Optional[int] = None):

        if seed is None:
            seed = sum([ord(s) for s in "wholeseller"])

        self.seed = seed
        self.reset_rng()

    def reset_rng(self, seed: Optional[int] = None):
        if seed is None:
            self.rng = np.random.RandomState(self.seed)
        else:
            self.rng = np.random.RandomState(seed)

    @abstractmethod
    def act(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def learn(self, quantity: float):
        raise NotImplementedError


class RandomWholeSeller(BaseWholeSeller):

    def act(self) -> float:
        return self.rng.random()

    def learn(self, quantity: float):
        pass


class ConstantWholeSeller(BaseWholeSeller):

    def __init__(self, wholesale_price: float):
        self.wholesale_price = wholesale_price
        super().__init__()

    def act(self) -> float:
        return self.wholesale_price

    def learn(self, quantity: float):
        pass
