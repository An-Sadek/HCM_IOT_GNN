"""Spectral Graph Markov Process (SGMP) imputation."""

import torch
import torch.nn as nn


class SGMPImputer(nn.Module):
    """
    Điền giá trị bị thiếu sử dụng SGMP

    Parameters:
        gamma: Hệ số giảm dần theo thời gian (Gamma)
        order: Bậc truyền thông tin
    """

    def __init__(self, order: int = 2, gamma: float = 0.9):
        """
        Parameters:
            order: k-hop thông tin của SGMP
        """
        super().__init__()
        if order < 1:
            raise ValueError("k phải lớn hơn hoặc bằng 1")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.order = order
        self.gamma = float(gamma)
        self.theta = nn.Parameter(torch.zeros(order + 1)) # Tham số học được
        with torch.no_grad():
            self.theta[0] = 1.0
        self.bias = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _propagate(x, source, destination, num_nodes):
        """
        Truyền thông tin cho loại nút.

        """
        # A(G) + A(G)^T + I
        loops = torch.arange(num_nodes, device=x.device)
        original_source = source
        original_destination = destination
        source = torch.cat((original_source, original_destination, loops))
        destination = torch.cat((original_destination, original_source, loops))

        # D(G)
        degree = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        degree.index_add_(0, destination, torch.ones_like(destination, dtype=x.dtype))

        # Chuẩn hoá ma trận Laplace \tilde{L}(G).
        # m[u][w] <- x[u] * sqrt(d_u * d_w)
        norm = degree.clamp_min(1).rsqrt() # Tránh chia cho 0
        message = x[source] * norm[source] * norm[destination]
        output = torch.zeros_like(x)
        output.index_add_(0, destination, message)
        return output

    def _spectral_transition(self, x, source, destination):
        """
        Điền giá trị bị thiếu bằng ước lượng.
        Công thức (20) trong paper.
        """
        terms = [x, self._propagate(x, source, destination, x.shape[0])]
        for _ in range(2, self.order + 1):
            terms.append(
                2.0 * self._propagate(terms[-1], source, destination, x.shape[0])
                - terms[-2]
            )

        return sum(weight * term for weight, term in zip(self.theta, terms)) + self.bias

    def forward(
        self,
        values,
        observed_mask,
        source,
        destination,
        *,
        batch_size=1,
    ):
        """
        Điền giá trị bị thiếu vào.

        Parameters:
            values: Giá trị thật input.
            observed_mask: Mask thể hiện giá trị thật và giá trị giả được điền.
            source: Tập các nút nguồn (edge_list[0]).
            destination: Tập các nút đích(edges_list[1]).

        Returns:
            completed: Giá trị thực và giá trị được điền.
        """
        if values.ndim != 2 or observed_mask.shape != values.shape:
            raise ValueError("values and observed_mask must both have shape (nodes, time)")
        if batch_size < 1 or values.shape[0] % batch_size:
            raise ValueError(
                f"Cannot split {values.shape[0]} nodes into batch_size={batch_size}"
            )

        # DGL flattens a batch to (batch * nodes, time).  Keep that memory-efficient
        # representation for sparse graph propagation, but expose the batch axis
        # while combining temporal lags.  Consequently SGMP works on exactly the
        # mini-batch selected by run_train instead of materialising the full dataset.
        nodes_per_graph = values.shape[0] // batch_size
        values_batched = values.reshape(batch_size, nodes_per_graph, -1)
        mask_batched = observed_mask.reshape_as(values_batched).clamp(0.0, 1.0)
        completed = []
        for timestamp in range(values.shape[1]):
            estimate = torch.zeros_like(values_batched[:, :, timestamp])
            normalizer = 0.0
            available_lags = min(self.order, timestamp)
            for lag in range(1, available_lags + 1):
                weight = self.gamma ** lag
                previous = completed[timestamp - lag].reshape(-1)
                transitioned = self._spectral_transition(
                    previous, source, destination
                ).reshape(batch_size, nodes_per_graph)
                estimate = estimate + weight * transitioned
                normalizer += weight

            # There is no history at t=0.  Zero is the neutral value because the
            # velocity channel has already been standardized with train-only stats.
            if normalizer:
                estimate = estimate / normalizer
            mask = mask_batched[:, :, timestamp]
            current = mask * values_batched[:, :, timestamp] + (1.0 - mask) * estimate
            completed.append(current)

        return torch.stack(completed, dim=2).reshape_as(values)
