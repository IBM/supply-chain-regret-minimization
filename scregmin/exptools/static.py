#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
from typing import Dict, List, Tuple
import numpy as np
from ..retailer import ConstantRetailer
from ..environment import SupplyChainEnv


def run_all_static_retailer(
        env: SupplyChainEnv,
        price_vector: List[float],
        quantity_vector: List[float]
) -> Dict[Tuple[float, float], np.ndarray]:

    # Store states (TODO)
    env_rng_state = env.rng.get_state()
    market_rng_state = env.market.rng.get_state()
    wholeseller_rng_state = env.wholeseller.rng.get_state()
    org_retailer = env.retailer

    total_profit = dict()
    for price in price_vector:
        for quantity in quantity_vector:
            static_retailer = ConstantRetailer(price, quantity)
            env.set_retailer(static_retailer)
            env.reset_rng()

            env.reset()
            done = False
            while not done:
                done = env.step()

            total_profit[(price, quantity)] = env.get_retailer_total_profit()

    # Restore states (TODO)
    env.set_retailer(org_retailer)
    env.rng.set_state(env_rng_state)
    env.market.rng.set_state(market_rng_state)
    env.wholeseller.rng.set_state(wholeseller_rng_state)

    return total_profit
