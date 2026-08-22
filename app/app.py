"""Streamlit viewer for partial and full-dataset HTGNN forecasts."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from forecast_all import load_runtime  # noqa: E402


st.set_page_config(page_title="Dự báo giao thông HTGNN", layout="wide")


MODEL_FILES = {
    f"_htgnn{i}": ROOT / "result" / f"_htgnn{i}" / "htgnn_best.pt"
    for i in range(1, 5)
} | {
    f"_sehtgnn{i}": ROOT / "result" / f"_sehtgnn{i}" / "sehtgnn_best.pt"
    for i in range(1, 3)
}


@st.cache_resource(show_spinner="Đang nạp model và dữ liệu...")
def cached_runtime(checkpoint_path, device_name):
    return load_runtime(checkpoint_path, device_name)


@st.cache_data(show_spinner=False)
def segment_catalog():
    frame = pd.read_csv(ROOT / "data" / "preprocess" / "segments.csv")
    names = frame["name"].fillna("Không rõ").astype(str)
    return {f"{idx} — {name}": int(idx) for idx, name in zip(frame["id"], names)}


@st.cache_data(show_spinner=False)
def timeline(total_timesteps):
    try:
        dates = pd.read_csv(
            ROOT / "data" / "raw" / "segment_status.csv", usecols=["updated_at"]
        )["updated_at"]
        start = pd.to_datetime(dates, utc=True).min().floor("30min").tz_localize(None)
        return pd.date_range(start=start, periods=total_timesteps, freq="30min")
    except Exception:
        return pd.RangeIndex(total_timesteps, name="time_index")


@st.cache_data(show_spinner="Đang đọc forecast_all.csv...")
def read_segment_forecast(path_string, modified_time, segment_id):
    """Scan the CSV for one segment without loading the multi-GB file into RAM."""
    del modified_time  # cache key invalidates automatically when the file changes
    with open(path_string, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if not header or header[0] != "segment_id":
            raise ValueError("Cột đầu tiên của CSV phải là segment_id")
        for row in reader:
            if row and int(row[0]) == segment_id:
                values = np.asarray(
                    [np.nan if value == "" else float(value) for value in row[1:]],
                    dtype=np.float32,
                )
                return values, len(header) - 1
    raise KeyError(segment_id)


def make_chart(x, actual, prediction, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=actual, name="Thực tế", line=dict(width=1.4)))
    fig.add_trace(go.Scatter(x=x, y=prediction, name="HTGNN", line=dict(width=1.4)))
    fig.update_layout(
        title=title, height=620, hovermode="x unified",
        margin=dict(l=30, r=100, t=55, b=30),
        xaxis=dict(title="Thời gian", rangeslider=dict(visible=True)),
        yaxis_title="Vận tốc",
        legend=dict(
            orientation="v",
            x=1.01,
            xanchor="left",
            y=1,
            yanchor="top",
        ),
    )
    return fig


def forecast_future(dataset, encoder, predictor, device, observed_time, segment_id):
    """Predict the complete model horizon after the last observed timestamp."""
    window_start = int(observed_time) - dataset.window_size + 1
    graph = dataset.build_graph(window_start).to(device)
    with torch.inference_mode():
        output = predictor(encoder(graph, predict_type="segment"))
        output = output.view(dataset.num_segments, dataset.horizon)
    return output[int(segment_id)].detach().cpu().numpy().astype(np.float32)


def future_timeline(all_times, observed_time, horizon):
    """Build labels for t+1..t+horizon, including times beyond the dataset."""
    observed_time = int(observed_time)
    if isinstance(all_times, pd.DatetimeIndex):
        step = all_times.freq or pd.Timedelta(minutes=30)
        start = all_times[observed_time] + step
        return pd.date_range(start=start, periods=horizon, freq=step)
    return pd.RangeIndex(observed_time + 1, observed_time + horizon + 1)


st.title("Dự báo vận tốc giao thông — HTGNN")
st.caption("Chọn mốc quan sát cuối cùng để dự báo nhiều bước trong tương lai.")

device_name = "cuda" if torch.cuda.is_available() else "cpu"
with st.sidebar:
    st.header("Thiết lập")
    model_name = st.selectbox("Model", list(MODEL_FILES))

checkpoint_path = MODEL_FILES[model_name]
dataset, encoder, predictor, args, _, device = cached_runtime(str(checkpoint_path), device_name)
catalog = segment_catalog()
all_times = timeline(dataset.total_timesteps)
forecast_csv = checkpoint_path.parent / "forecast_all.csv"

with st.sidebar:
    selected_label = st.selectbox("Đoạn đường", list(catalog))
    segment_id = catalog[selected_label]
    st.caption(f"Thiết bị suy luận: {device_name.upper()}")
    st.subheader("Mốc quan sát")
    if isinstance(all_times, pd.DatetimeIndex):
        time_index = st.selectbox(
            "Thời điểm cuối đã quan sát",
            range(dataset.total_timesteps),
            format_func=lambda index: all_times[index].strftime("%d/%m/%Y %H:%M"),
        )
        st.caption(
            f"Hợp lệ: {all_times[0]:%d/%m/%Y %H:%M} – "
            f"{all_times[-1]:%d/%m/%Y %H:%M} (mỗi 30 phút)"
        )
    else:
        time_index = st.selectbox(
            "Chỉ số cuối đã quan sát", range(dataset.total_timesteps)
        )
        st.caption(f"Khoảng hợp lệ: 0–{dataset.total_timesteps - 1}")
    st.caption(
        f"Model sẽ dự báo {dataset.horizon} bước kế tiếp "
        f"({dataset.horizon * 30} phút nếu mỗi bước là 30 phút)."
    )

partial_button, full_button = st.columns(2)
run_partial = partial_button.button("Dự báo tương lai", type="primary", use_container_width=True)
run_full = full_button.button("Dự báo toàn bộ", use_container_width=True)

if run_partial:
    with st.spinner("Đang dự báo các bước tương lai..."):
        prediction = forecast_future(
            dataset, encoder, predictor, device, time_index, segment_id
        )
    future_start = int(time_index) + 1
    future_end = future_start + dataset.horizon
    available_end = min(future_end, dataset.total_timesteps)
    actual = np.full(dataset.horizon, np.nan, dtype=np.float32)
    if future_start < dataset.total_timesteps:
        actual[:available_end - future_start] = np.asarray(
            dataset.targets[future_start:available_end, segment_id], dtype=np.float32
        )
    x = future_timeline(all_times, time_index, dataset.horizon)
    st.session_state["forecast_result"] = (
        x,
        actual,
        prediction,
        f"{model_name} — {dataset.horizon} bước tương lai",
    )

if run_full:
    if not forecast_csv.exists():
        st.error(
            f"Chưa có {forecast_csv}. Hãy chạy `forecast_all.py` cho model này trước."
        )
    else:
        try:
            prediction, csv_width = read_segment_forecast(
                str(forecast_csv), forecast_csv.stat().st_mtime_ns, segment_id
            )
        except KeyError:
            st.error(f"CSV không chứa segment_id={segment_id}.")
        except (OSError, ValueError) as exc:
            st.error(f"Không thể đọc CSV: {exc}")
        else:
            if csv_width != dataset.total_timesteps:
                st.error(
                    f"CSV có {csv_width} timestamp, dữ liệu cần {dataset.total_timesteps}."
                )
                st.stop()
            actual = np.asarray(dataset.targets[:, segment_id], dtype=np.float32)
            st.session_state["forecast_result"] = (
                all_times, actual, prediction, f"{model_name} — toàn bộ dữ liệu"
            )

if "forecast_result" not in st.session_state:
    st.info("Chọn chế độ dự báo để hiển thị kết quả.")
else:
    x, actual, prediction, title = st.session_state["forecast_result"]
    valid = np.isfinite(prediction) & np.isfinite(actual)
    if valid.any():
        rmse = float(np.sqrt(np.mean((prediction[valid] - actual[valid]) ** 2)))
        mae = float(np.mean(np.abs(prediction[valid] - actual[valid])))
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{rmse:.3f}")
        c2.metric("MAE", f"{mae:.3f}")
        c3.metric("Số timestamp dự báo", f"{valid.sum():,}/{len(prediction):,}")
    st.plotly_chart(make_chart(x, actual, prediction, title), use_container_width=True)
