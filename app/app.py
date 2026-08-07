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

from forecast_all import DEFAULT_CHECKPOINT, DEFAULT_OUTPUT, forecast_range, load_runtime  # noqa: E402


st.set_page_config(page_title="Dự báo giao thông HTGNN", layout="wide")


@st.cache_resource(show_spinner="Đang nạp model HTGNN và dữ liệu...")
def cached_runtime(device_name):
    return load_runtime(DEFAULT_CHECKPOINT, device_name)


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
        margin=dict(l=30, r=20, t=55, b=30),
        xaxis=dict(title="Thời gian", rangeslider=dict(visible=True)),
        yaxis_title="Vận tốc", legend=dict(orientation="h", y=1.08),
    )
    return fig


st.title("Dự báo vận tốc giao thông — HTGNN")
st.caption("Chọn đoạn đường, sau đó dự báo một khoảng hoặc đọc dự báo toàn bộ từ CSV.")

device_name = "cuda" if torch.cuda.is_available() else "cpu"
dataset, encoder, predictor, args, _, device = cached_runtime(device_name)
catalog = segment_catalog()
all_times = timeline(dataset.total_timesteps)

with st.sidebar:
    st.header("Thiết lập")
    selected_label = st.selectbox("Đoạn đường", list(catalog))
    segment_id = catalog[selected_label]
    st.caption(f"Thiết bị suy luận: {device_name.upper()}")
    batch_size = st.number_input("Batch size", min_value=1, max_value=256, value=16)
    st.subheader("Khoảng dự báo một phần")
    minimum = 0
    start_time = st.number_input(
        "Chỉ số bắt đầu", min_value=minimum,
        max_value=dataset.total_timesteps - 1, value=minimum,
    )
    end_default = min(47, dataset.total_timesteps - 1)
    end_time = st.number_input(
        "Chỉ số kết thúc", min_value=minimum,
        max_value=dataset.total_timesteps - 1, value=end_default,
    )
    st.caption(f"Khoảng hợp lệ: {minimum}–{dataset.total_timesteps - 1} (bao gồm hai đầu)")

partial_button, full_button = st.columns(2)
run_partial = partial_button.button("Dự báo một phần", type="primary", use_container_width=True)
run_full = full_button.button("Dự báo toàn bộ", use_container_width=True)

if run_partial:
    if start_time > end_time:
        st.error("Chỉ số bắt đầu phải nhỏ hơn hoặc bằng chỉ số kết thúc.")
    else:
        bar = st.progress(0, text="Đang dự báo...")

        def update(done, total):
            bar.progress(min(done / total, 1.0), text=f"Đã xử lý {done:,}/{total:,} cửa sổ")

        prediction = forecast_range(
            dataset, encoder, predictor, device, int(start_time), int(end_time),
            [segment_id], int(batch_size), update,
        )[0]
        bar.empty()
        actual = np.asarray(dataset.targets[int(start_time):int(end_time) + 1, segment_id])
        x = all_times[int(start_time):int(end_time) + 1]
        st.session_state["forecast_result"] = (x, actual, prediction, "Dự báo một phần")

if run_full:
    if not DEFAULT_OUTPUT.exists():
        st.error(
            f"Chưa có {DEFAULT_OUTPUT}. Hãy chạy `python test/forecast_all.py` trước."
        )
    else:
        try:
            prediction, csv_width = read_segment_forecast(
                str(DEFAULT_OUTPUT), DEFAULT_OUTPUT.stat().st_mtime_ns, segment_id
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
                all_times, actual, prediction, "Dự báo toàn bộ dữ liệu"
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
