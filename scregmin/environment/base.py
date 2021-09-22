#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
from typing import Tuple, Optional
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from ..market import BaseMarket
from ..retailer import BaseRetailer
from ..wholeseller import BaseWholeSeller


class BaseSupplyChainEnv(ABC):

    def __init__(
            self,
            horizon: int,
            market: BaseMarket,
            retailer: BaseRetailer,
            wholeseller: BaseWholeSeller,
            seed: Optional[int] = None
    ):

        self.horizon = horizon
        self.market = market
        self.retailer = retailer
        self.wholeseller = wholeseller
    
        if seed is None:
            seed = sum([ord(s) for s in "environment"])
        self.seed = seed
        self.reset_rng()

        self.reset()

    def reset_rng(self, seed: Optional[int] = None):
        if seed is None:
            self.rng = np.random.RandomState(self.seed)
        else:
            self.rng = np.random.RandomState(seed)
        
    def reset(self):
        self.history = defaultdict(list)

    def set_retailer(self, retailer: BaseRetailer):
        self.retailer = retailer

    def set_wholeseller(self, wholeseller: BaseWholeSeller):
        self.wholeseller = wholeseller

    def set_market(self, market: BaseMarket):
        self.market = market

    @abstractmethod
    def step(self):
        raise NotImplementedError

    def _add_history(self,
                     wholesale_price: float,
                     retail_price: float,
                     quantity: float,
                     demand: float):
        self.history["wholesale_price"].append(wholesale_price)
        self.history["retail_price"].append(retail_price)
        self.history["quantity"].append(quantity)
        self.history["demand"].append(demand)

    def get_history(self):
        return self.history
        
    def get_retailer_total_profit(self) -> np.ndarray:

        T = len(self.history["demand"])

        for key in self.history:
            self.history[key] = np.array(self.history[key])
            assert len(self.history[key]) == T

        sold = np.minimum(self.history["quantity"], self.history["demand"])
        assert sold.shape == (T,)

        revenue = self.history["retail_price"] * sold
        assert revenue.shape == (T,)

        cost = self.history["wholesale_price"] * self.history["quantity"]
        assert cost.shape == (T,)

        profit = revenue - cost
        assert profit.shape == (T,)

        return profit.cumsum()  # shape = (T,)

    def get_best_static_retailer(
            self,
            eps: float
    ) -> Tuple[np.ndarray, float, float]:

        assert eps > 0 and eps < 1

        price_vector = np.arange(0, 1, eps)
        quantity_vector = np.arange(0, 1+eps, eps)

        T = len(self.history["demand"])
        P = len(price_vector)
        Q = len(quantity_vector)

        for key in self.history:
            self.history[key] = np.array(self.history[key])
            assert len(self.history[key]) == T

        # Compute counterfactual retailer profit with fixed p, q for all p, q
        sold = np.minimum(
            quantity_vector.reshape((-1, 1)),
            self.history["demand"]
        )
        assert sold.shape == (Q, T)

        revenue = np.outer(
            price_vector,
            sold
        ).reshape(price_vector.shape + sold.shape)
        assert revenue.shape == (P, Q, T)

        cost = np.outer(
            quantity_vector,
            self.history["wholesale_price"]
        )
        assert cost.shape == (Q, T)

        profit = revenue - cost.reshape((1,) + cost.shape)
        assert profit.shape == (P, Q, T)

        total_profit = profit.cumsum(axis=2)
        assert total_profit.shape == (P, Q, T)

        # Find best static p, q
        best_price_idx, best_quantity_idx = np.unravel_index(
            total_profit[:, :, -1].argmax(),
            total_profit[:, :, -1].shape
        )
        best_total_profit = total_profit[best_price_idx, best_quantity_idx, :]
        best_price = price_vector[best_price_idx]

        best_quantity = quantity_vector[best_quantity_idx]

        # return best_total_profit, best_price, best_quantity
        return {
            "prices": price_vector,
            "quantities": quantity_vector,
            "total profit": total_profit,
            "best price index": best_price_idx,
            "best quantity index": best_quantity_idx
        }


class SupplyChainEnv(BaseSupplyChainEnv):

    def __init__(
            self,
            horizon: int,
            market: BaseMarket,
            retailer: BaseRetailer,
            wholeseller: BaseWholeSeller,
            seed: Optional[int] = None
    ):
        super().__init__(horizon, market, retailer, wholeseller, seed)
        self.reset()

    def reset(self):
        self.t = 0
        super().reset()

    def step(self) -> bool:

        wholesale_price = self.wholeseller.act()
        retail_price, quantity = self.retailer.act(wholesale_price)
        demand = self.market.act(retail_price)

        self.wholeseller.learn(quantity)
        self.retailer.learn(demand)

        self.t += 1

        self._add_history(wholesale_price, retail_price, quantity, demand)

        return (self.t >= self.horizon)
