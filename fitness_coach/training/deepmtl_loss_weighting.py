"""Lightweight multitask loss helpers aligned with amazon-science/DeepMTL2R.

Implementations here deliberately avoid importing the full DeepMTL2R / allrank
stack (cvxpy, Hessian-heavy solvers, …). This module only carries **Dynamic Weight
Average** logic; phased or fixed linear weights are implemented alongside the
trainer in ``train_xlstm_egoexo_multitask.py``.
References
----------
- Liu et al., "End-to-End Multi-Task Learning with Attention"; Dynamic Loss
  Weighting / Dynamic Weight Average (``dwa`` in DeepMTL2R README), ported from
  ``allrank.methods.weight_methods.DynamicWeightAverage``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class DynamicWeightAverage:
    """Dynamic Weight Average (DWA): DeepMTL2R ``DynamicWeightAverage`` semantics.

    Each training step observes a vector of per-task scalar losses ``L``.
    Task weights adapt from the ratio of recent average loss trends.
    Objective per step: ``(w * L).mean()`` matching DeepMTL2R upstream.
    """

    n_tasks: int
    iteration_window: int = 25
    temp: float = 2.0

    def __post_init__(self) -> None:
        w = max(2, int(self.iteration_window))
        self.iteration_window = w
        self.running_iterations = 0
        fifo = max(2, w * 2)
        self._costs = np.ones((fifo, self.n_tasks), dtype=np.float32)
        self._weights = np.ones(self.n_tasks, dtype=np.float32)

    def weighted_loss_mean(self, stacked_losses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Combine per-task scalar losses stacked as shape ``(n_tasks,)``.

        Returns
        -------
        loss_scalar
            Mean-weighted multitask objective (DeepMTL2R convention).
        task_weights_tensor
            Weights shaped ``(n_tasks,)`` on ``stacked_losses.device``.
        """
        if stacked_losses.ndim != 1 or stacked_losses.numel() != self.n_tasks:
            raise ValueError(
                f"stacked_losses must be shape ({self.n_tasks},), got {tuple(stacked_losses.shape)}"
            )
        cost = stacked_losses.detach().cpu().numpy().astype(np.float32)
        fifo = self._costs.shape[0]
        self._costs[:-1, :] = self._costs[1:, :]
        self._costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            w = self.iteration_window
            ws = self._costs[w:, :].mean(0) / self._costs[:w, :].mean(0)
            ew = np.exp(ws / float(self.temp))
            self._weights = (self.n_tasks * ew) / float(np.sum(ew))

        tw = torch.from_numpy(self._weights.astype(np.float32)).to(stacked_losses.device)
        loss = (tw * stacked_losses).mean()
        self.running_iterations += 1
        return loss, tw
