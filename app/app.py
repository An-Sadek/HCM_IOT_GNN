"""Interactive, end-to-end forecast viewer for result/model18."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from data import SEHTGNNDataset, collate_sehtgnn  # noqa: E402
from htgnn_model import HTGNN, NodePredictor  # noqa: E402
from run_train import chronological_split  # noqa: E402


CHECKPOINT = ROOT / "result" / "model18" / "htgnn_best.pt"
CACHE_DIR = ROOT / "app" / ".cache"


st.set_page_config(page_title="Model18 traffic forecast", layout="wide")


@st.cache_resource(show_spinner="Đang nạp model18 và dữ liệu...")
def load_runtime(device_name: str):
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    args = checkpoint["args"]
    dataset = SEHTGNNDataset(
        preprocess_root=ROOT / args["preprocess_root"],
        dynamic_path=ROOT / args["dynamic_path"],
        target_path=ROOT / args["target_path"],
        window_size=args["window_size"],
        horizon=args["horizon"],
    )
    gap = args.get("split_gap")
    gap = args["window_size"] + args["horizon"] - 1 if gap is None else gap
    train_ds, val_ds, test_ds = chronological_split(
        dataset, args["train_ratio"], args["val_ratio"], gap
    )
    train_end = train_ds.indices[-1] + args["window_size"] + args["horizon"]
    dataset.fit_input_velocity_scaler(
        0, train_end, velocity_channel=args.get("velocity_channel", 0)
    )

    device = torch.device(device_name)
    graph, _ = dataset[0]
    encoder = HTGNN(
        graph=graph,
        n_hid=args["hidden_dim"],
        n_layers=args["layers"],
        time_window=args["window_size"],
        dropout=args["dropout"],
        inp_list=checkpoint.get("inp_list", dataset.inp_list),
    ).to(device)
    predictor = NodePredictor(args["hidden_dim"], args["horizon"]).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    predictor.load_state_dict(checkpoint["predictor"])
    encoder.eval()
    predictor.eval()
    return dataset, encoder, predictor, args, (train_ds, val_ds, test_ds), device


@st.cache_data(show_spinner=False)
def segment_catalog():
    frame = pd.read_csv(ROOT / "data" / "preprocess" / "segments.csv")
    names = frame["name"].fillna("Không rõ").astype(str)
    return [f"{idx} — {name}" for idx, name in zip(frame["id"], names)]


@st.cache_data(show_spinner=False)
def timeline(total_timesteps: int):
    status_path = ROOT / "data" / "raw" / "segment_status.csv"
    try:
        dates = pd.read_csv(status_path, usecols=["updated_at"])["updated_at"]
        start = pd.to_datetime(dates, utc=True).min().floor("30min").tz_localize(None)
        return pd.date_range(start=start, periods=total_timesteps, freq="30min")
    except Exception:
        return pd.RangeIndex(total_timesteps, name="time_index")


def cache_path(segment_id: int):
    return CACHE_DIR / f"model18_segment_{segment_id}.npz"


def run_forecast(dataset, encoder, predictor, device, segment_id, batch_size):
    """Predict every sliding window and average overlapping horizons."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sehtgnn
    )
    pred_sum = np.zeros(dataset.total_timesteps, dtype=np.float64)
    pred_count = np.zeros(dataset.total_timesteps, dtype=np.int32)
    progress = st.progress(0, text="Bắt đầu dự báo...")
    sample_start = 0
    with torch.inference_mode():
        for batch_no, (graph, _) in enumerate(loader, start=1):
            current_batch = graph.batch_size
            graph = graph.to(device)
            output = predictor(encoder(graph, predict_type="segment"))
            output = output.view(current_batch, dataset.num_segments, dataset.horizon)
            selected = output[:, segment_id, :].detach().cpu().numpy()
            for row in range(current_batch):
                begin = sample_start + row + dataset.window_size
                end = begin + dataset.horizon
                pred_sum[begin:end] += selected[row]
                pred_count[begin:end] += 1
            sample_start += current_batch
            progress.progress(
                min(sample_start / len(dataset), 1.0),
                text=f"Đã xử lý {sample_start:,}/{len(dataset):,} cửa sổ",
            )
    progress.empty()
    prediction = np.full(dataset.total_timesteps, np.nan, dtype=np.float32)
    valid = pred_count > 0
    prediction[valid] = (pred_sum[valid] / pred_count[valid]).astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path(segment_id), prediction=prediction)
    return prediction


def split_regions(dataset, splits, args):
    regions = []
    for label, subset, color in zip(
        ("Train", "Validation", "Test"),
        splits,
        ("rgba(46, 204, 113, .12)", "rgba(241, 196, 15, .14)", "rgba(231, 76, 60, .12)"),
    ):
        start = subset.indices[0] + args["window_size"]
        end = subset.indices[-1] + args["window_size"] + args["horizon"] - 1
        regions.append((label, start, min(end, dataset.total_timesteps - 1), color))
    return regions


def make_chart(x, actual, prediction, regions):
    fig = go.Figure()
    for label, start, end, color in regions:
        fig.add_vrect(
            x0=x[start], x1=x[end], fillcolor=color, line_width=0,
            annotation_text=label, annotation_position="top left",
        )
    fig.add_trace(go.Scatter(x=x, y=actual, name="Thực tế", line=dict(width=1.4)))
    fig.add_trace(go.Scatter(x=x, y=prediction, name="Model18", line=dict(width=1.4)))
    fig.update_layout(
        height=650, hovermode="x unified", margin=dict(l=30, r=20, t=45, b=30),
        xaxis=dict(title="Thời gian", rangeslider=dict(visible=True), type="date" if isinstance(x, pd.DatetimeIndex) else "linear"),
        yaxis_title="Vận tốc",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


st.title("Dự báo vận tốc toàn chuỗi — Model18")
st.caption("Kéo thanh ngang bên dưới biểu đồ để phóng to hoặc di chuyển trên toàn bộ dữ liệu.")

available_device = "cuda" if torch.cuda.is_available() else "cpu"
with st.sidebar:
    st.header("Thiết lập")
    labels = segment_catalog()
    selected = st.selectbox("Đoạn đường", labels, index=0)
    segment_id = int(selected.split(" — ", 1)[0])
    batch_size = st.number_input("Batch size suy luận", 1, 256, 16, 1)
    st.info(f"Thiết bị: {available_device.upper()}")

dataset, encoder, predictor, args, splits, device = load_runtime(available_device)
cached = cache_path(segment_id)
prediction = None
if cached.exists():
    prediction = np.load(cached)["prediction"]

button_text = "Chạy dự báo toàn bộ dữ liệu" if prediction is None else "Chạy lại dự báo"
if st.button(button_text, type="primary"):
    prediction = run_forecast(
        dataset, encoder, predictor, device, segment_id, int(batch_size)
    )

if prediction is None:
    st.warning("Chọn đoạn đường rồi bấm **Chạy dự báo toàn bộ dữ liệu**. Kết quả sẽ được cache cho lần mở sau.")
    st.stop()

actual = np.asarray(dataset.targets[:, segment_id], dtype=np.float32)
x = timeline(dataset.total_timesteps)
valid = np.isfinite(prediction)
rmse = float(np.sqrt(np.mean((prediction[valid] - actual[valid]) ** 2)))
mae = float(np.mean(np.abs(prediction[valid] - actual[valid])))
c1, c2, c3 = st.columns(3)
c1.metric("RMSE toàn chuỗi", f"{rmse:.3f}")
c2.metric("MAE toàn chuỗi", f"{mae:.3f}")
c3.metric("Số timestamp dự báo", f"{valid.sum():,}/{len(actual):,}")
st.plotly_chart(
    make_chart(x, actual, prediction, split_regions(dataset, splits, args)),
    use_container_width=True,
)

