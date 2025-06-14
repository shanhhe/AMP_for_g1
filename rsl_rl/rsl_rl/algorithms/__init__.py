#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .ppo import PPO
from .amp_ppo import AMPPPO
from .distillation import Distillation

__all__ = ["PPO", "AMPPPO", "Distillation"]