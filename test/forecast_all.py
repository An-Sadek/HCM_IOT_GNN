"""Forecast every segment and timestamp with ``result/htgnn``.

The resulting CSV has one row per segment (|V|) and one column per timestamp
(|T|). Predictions from overlapping horizons are averaged. Missing history
before timestamp 0 is padded as unobserved zero input, so every column receives
a model prediction.

Run from the repository root::

    python test/forecast_all.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from data import SEHTGNNDataset, _require_dgl  # noqa: E402
from HTGNN import HTGNN, NodePredictor as HTGNNNodePredictor  # noqa: E402
from model import NodePredictor as SENodePredictor, SEHTGNN  # noqa: E402
from run_train import chronological_split  # noqa: E402


DEFAULT_CHECKPOINT = ROOT / "result" / "htgnn" / "htgnn_best.pt"
DEFAULT_OUTPUT = ROOT / "result" / "htgnn" / "forecast_all.csv"


class _ForecastWindows:
    """Inference-only windows, including virtual negative starting indices."""

    def __init__(self, dataset, sample_ids):
        self.dataset = dataset
        self.sample_ids = range(sample_ids.start, sample_ids.stop)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        return self.dataset.build_graph(self.sample_ids[index])


def _collate_forecast(graphs):
    return _require_dgl().batch(graphs)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_runtime(checkpoint_path: str | Path = DEFAULT_CHECKPOINT, device="cpu"):
    """Recreate the dataset and model exactly as recorded in the checkpoint."""
    checkpoint_path = _resolve(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture")
    if architecture not in {"htgnn", "sehtgnn"}:
        raise ValueError(f"Unsupported architecture {architecture!r}: {checkpoint_path}")
    if architecture == "htgnn" and checkpoint.get("htgnn_variant") != "yeslab_reference":
        raise ValueError("Checkpoint is not the supported YesLab-reference HTGNN")

    args = checkpoint["args"]
    dataset = SEHTGNNDataset(
        preprocess_root=_resolve(args["preprocess_root"]),
        dynamic_path=_resolve(args["dynamic_path"]),
        target_path=_resolve(args["target_path"]),
        target_mask_path=_resolve(args["target_mask_path"]),
        window_size=args["window_size"],
        horizon=args["horizon"],
        separate_dynamic=False,
    )
    gap = args.get("split_gap")
    gap = args["window_size"] + args["horizon"] - 1 if gap is None else gap
    train_ds, val_ds, test_ds = chronological_split(
        dataset, args["train_ratio"], args["val_ratio"], gap
    )
    train_end = train_ds.indices[-1] + args["window_size"] + args["horizon"]
    dataset.fit_input_velocity_scaler(
        train_ds.indices[0],
        train_end,
        velocity_channel=args.get("velocity_channel", 0),
        velocity_mask_channel=args.get("velocity_mask_channel", 1),
    )

    device = torch.device(device)
    sample_graph, _, _ = dataset[0]
    encoder_class = HTGNN if architecture == "htgnn" else SEHTGNN
    predictor_class = HTGNNNodePredictor if architecture == "htgnn" else SENodePredictor
    encoder_kwargs = dict(
        graph=sample_graph,
        n_inp=args["hidden_dim"],
        n_hid=args["hidden_dim"],
        n_layers=args["layers"],
        n_heads=args.get("heads", 1),
        time_window=args["window_size"],
        norm=True,
        device=device,
        dropout=args["dropout"],
        inp_list=checkpoint.get("inp_list", dataset.inp_list),
    )
    if architecture == "htgnn":
        encoder_kwargs["dynamic_input_dim"] = dataset.dynamic_dim
    else:
        llm_feature = checkpoint.get("llm_feature")
        if llm_feature is None:
            raise ValueError(f"SEHTGNN checkpoint has no llm_feature: {checkpoint_path}")
        encoder_kwargs["LLM_feature"] = llm_feature
    encoder = encoder_class(**encoder_kwargs).to(device)
    predictor = predictor_class(args["hidden_dim"], args["horizon"]).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    predictor.load_state_dict(checkpoint["predictor"])
    encoder.eval()
    predictor.eval()
    return dataset, encoder, predictor, args, (train_ds, val_ds, test_ds), device


def forecast_range(
    dataset,
    encoder,
    predictor,
    device,
    start_time: int,
    end_time: int,
    segment_ids: list[int] | np.ndarray | None = None,
    batch_size: int = 16,
    progress_callback=None,
):
    """Forecast inclusive timestamps ``start_time..end_time``.

    Returns an array shaped ``(selected_segments, end_time-start_time+1)``.
    Only windows whose forecast horizon intersects the requested range are run.
    """
    start_time, end_time = int(start_time), int(end_time)
    if start_time < 0 or end_time >= dataset.total_timesteps:
        raise ValueError(
            f"Valid forecast interval is [0, {dataset.total_timesteps - 1}]"
        )
    if start_time > end_time:
        raise ValueError("start_time must be <= end_time")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    if segment_ids is None:
        segment_ids = np.arange(dataset.num_segments, dtype=np.int64)
    else:
        segment_ids = np.asarray(segment_ids, dtype=np.int64)
    if segment_ids.ndim != 1 or len(segment_ids) == 0:
        raise ValueError("At least one segment must be selected")
    if segment_ids.min() < 0 or segment_ids.max() >= dataset.num_segments:
        raise ValueError(f"Segment IDs must be in [0, {dataset.num_segments - 1}]")

    # sample_id=-window_size is the virtual window immediately preceding t=0.
    first_sample = max(
        -dataset.window_size,
        start_time - dataset.window_size - dataset.horizon + 1,
    )
    last_sample = min(len(dataset) - 1, end_time - dataset.window_size)
    sample_ids = range(first_sample, last_sample + 1)
    loader = DataLoader(
        _ForecastWindows(dataset, sample_ids),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_forecast,
    )
    width = end_time - start_time + 1
    # float32 keeps a full 10k x 14k forecast practical in memory (~0.56 GB).
    pred_sum = np.zeros((len(segment_ids), width), dtype=np.float32)
    pred_count = np.zeros(width, dtype=np.int16)
    completed = 0

    with torch.inference_mode():
        for graph in loader:
            current_batch = graph.batch_size
            graph = graph.to(device)
            output = predictor(encoder(graph, predict_type="segment"))
            output = output.view(current_batch, dataset.num_segments, dataset.horizon)
            selected = output[:, segment_ids, :].detach().cpu().numpy()
            for row in range(current_batch):
                sample_id = first_sample + completed + row
                forecast_start = sample_id + dataset.window_size
                left = max(forecast_start, start_time)
                right = min(forecast_start + dataset.horizon - 1, end_time)
                if left <= right:
                    src = slice(left - forecast_start, right - forecast_start + 1)
                    dst = slice(left - start_time, right - start_time + 1)
                    pred_sum[:, dst] += selected[row, :, src]
                    pred_count[dst] += 1
            completed += current_batch
            if progress_callback:
                progress_callback(completed, len(sample_ids))

    result = np.full((len(segment_ids), width), np.nan, dtype=np.float32)
    valid = pred_count > 0
    result[:, valid] = (pred_sum[:, valid] / pred_count[valid]).astype(np.float32)
    return result


def write_forecast_csv(path, predictions, segment_ids, timestamps):
    """Write a |V| x |T| matrix; row labels are stored as the CSV index."""
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["segment_id", *timestamps])
        for segment_id, row in zip(segment_ids, predictions):
            writer.writerow([int(segment_id), *("" if np.isnan(x) else float(x) for x in row)])
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    cli = parse_args()
    dataset, encoder, predictor, _, _, device = load_runtime(cli.checkpoint, cli.device)

    def report(done, total):
        print(f"\rForecasting windows: {done:,}/{total:,}", end="", flush=True)

    forecast = forecast_range(
        dataset, encoder, predictor, device,
        0, dataset.total_timesteps - 1,
        batch_size=cli.batch_size, progress_callback=report,
    )
    output = write_forecast_csv(
        cli.output, forecast, np.arange(dataset.num_segments),
        range(dataset.total_timesteps),
    )
    print(
        f"\nSaved {dataset.num_segments} x {dataset.total_timesteps} "
        f"forecasts to {output}"
    )


if __name__ == "__main__":
    main()
