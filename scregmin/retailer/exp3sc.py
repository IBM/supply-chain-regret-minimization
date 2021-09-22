#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
from typing import Tuple, Optional, Dict, Union
import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod
from .base import BaseRetailer


class Exp3SCRetailer(BaseRetailer):

    def __init__(self,
                 learning_rate: float,
                 exploration_param: float,
                 n_bins: int,
                 seed: Optional[int] = None):

        self.eta = learning_rate
        self.gamma = exploration_param
        self.K = n_bins

        self.reset()
        super().__init__(seed)

    def reset(self):

        # initialization
        self.pi = np.full((self.K, self.K + 1), 1 / (self.K * (self.K + 1)))
        self.cum_loss = np.zeros((self.K, self.K + 1))

        self._update_mu()

    def _update_mu(self):
        # Compute distribution with exploration
        self.mu = (1 - self.gamma) * self.pi
        self.mu[:, -1] += self.gamma / self.K  # exploration term

    def act(self, wholesale_price: float) -> Tuple[float, float]:

        # Sample from the distribution
        idx = self.rng.choice(np.prod(self.mu.shape), p=self.mu.reshape(-1))
        price_idx, quantity_idx = np.unravel_index(idx, self.mu.shape)

        price = price_idx * self.gamma  # gamma -> 1/K?
        quantity = quantity_idx * self.gamma  # gamma -> 1/K?

        self._last_wholesale_price = wholesale_price
        self._last_price_idx = price_idx
        self._last_quantity_idx = quantity_idx

        return price, quantity

    def learn(self, demand: float):

        # Retrieve last price and quantity for learning
        price = self._last_price_idx * self.gamma
        quantity = self._last_quantity_idx * self.gamma

        # vector of quantities below quantity
        quantity_vector = (
            np.arange(self._last_quantity_idx + 1) * self.gamma
        ). reshape((1, -1))
        
        # vector of quanitities sold to the market (when the quantities are below quantity)
        sold_vector = np.clip(quantity_vector, a_min=-np.inf, a_max=demand)        

        # vector of profits given the quantities below quantity
        profit_vector = price * sold_vector - quantity_vector * self._last_wholesale_price

        # profit -> loss
        loss_vector = (1 - profit_vector) / 2

        # probability of observing the losses (j=0,...,quantity_idx)
        # = sum_{k=j,...,K+1} mu(price_idx,k)
        obs_prob_vector = np.flip(
            np.cumsum(np.flip(self.mu[self._last_price_idx, :]))
        )[:(self._last_quantity_idx + 1)]

        # Compute the estimated loss
        estimated_loss = np.zeros((self.K, self.K + 1))
        estimated_loss[
            self._last_price_idx,
            :(self._last_quantity_idx + 1)
        ] = loss_vector / obs_prob_vector

        # Update distribution
        self.cum_loss += estimated_loss
        
        # subtract baseline for numerical stability
        self.cum_loss -= self.cum_loss.min()

        # Update the distribution without exploration
        self.pi = np.exp(-self.eta * self.cum_loss)
        self.pi /= self.pi.sum()

        # Update the distribution with exploration
        self._update_mu()
