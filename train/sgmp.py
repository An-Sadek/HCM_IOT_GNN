"""Spectral Graph Markov Process (SGMP) imputation."""

import torch
import torch.nn as nn


class SGMPImputer(nn.Module):
    """Complete graph signals recursively with a Chebyshev spectral filter."""

    def __init__(self, order: int = 2):
        super().__init__()
        if order < 1:
            raise ValueError("SGMP Chebyshev order must be at least 1")
        self.order = order
        self.theta = nn.Parameter(torch.zeros(order + 1))
        with torch.no_grad():
            self.theta[0] = 1.0
        self.bias = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _propagate(x, source, destination, num_nodes):
        loops = torch.arange(num_nodes, device=x.device)
        original_source = source
        original_destination = destination
        source = torch.cat((original_source, original_destination, loops))
        destination = torch.cat((original_destination, original_source, loops))
        degree = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        degree.index_add_(0, destination, torch.ones_like(destination, dtype=x.dtype))
        norm = degree.clamp_min(1).rsqrt()
        message = x[source] * norm[source] * norm[destination]
        output = torch.zeros_like(x)
        output.index_add_(0, destination, message)
        return output

    def _spectral_transition(self, x, source, destination):
        terms = [x, self._propagate(x, source, destination, x.shape[0])]
        for _ in range(2, self.order + 1):
            terms.append(
                2.0 * self._propagate(terms[-1], source, destination, x.shape[0])
                - terms[-2]
            )
        return sum(weight * term for weight, term in zip(self.theta, terms)) + self.bias

    def forward(self, values, observed_mask, source, destination):
        """Complete ``values`` shaped ``(nodes, time)`` while preserving observations."""
        completed = []
        previous = torch.zeros_like(values[:, 0])
        for timestamp in range(values.shape[1]):
            estimate = self._spectral_transition(previous, source, destination)
            mask = observed_mask[:, timestamp].clamp(0.0, 1.0)
            current = mask * values[:, timestamp] + (1.0 - mask) * estimate
            completed.append(current)
            previous = current
        return torch.stack(completed, dim=1)
