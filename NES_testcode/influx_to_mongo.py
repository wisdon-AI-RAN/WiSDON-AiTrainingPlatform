#!/usr/bin/env python3
import os
import time
import base64
import gzip
import datetime
from io import BytesIO

import numpy as np
from influxdb_client import InfluxDBClient
from pymongo import MongoClient

def to_onehot_1x256(action_idx: int, dim: int = 256):
    v = [0] * dim
    if 0 <= action_idx < dim:
        v[action_idx] = 1
    return [v]  # 1 x 256

# ----------------------------
# Env (InfluxDB v2)
# ----------------------------
INFLUX_URL = os.environ.get("INFLUX_URL", "http://127.0.0.1:8086")
INFLUX_TOKEN = os.environ.get("INFLUX_TOKEN", "")
INFLUX_ORG = os.environ.get("INFLUX_ORG", "my-org")
INFLUX_BUCKET = os.environ.get("INFLUX_BUCKET", "api_bucket")

RUN_ID = os.environ.get("RUN_ID", "demo_run_001")
EPISODE = os.environ.get("EPISODE", "0")  # keep string for tag match

# ----------------------------
# Env (MongoDB)
# ----------------------------
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://140.113.144.121:27019")
APP_NAME = os.environ.get("APP_NAME", "NES")       # db name in your example :contentReference[oaicite:1]{index=1}
PROJECT_ID = os.environ.get("PROJECT_ID", "ED8F")            # collection name in your example :contentReference[oaicite:2]{index=2}
MODEL_VERSION = os.environ.get("MODEL_VERSION", "0.0.1")

# ----------------------------
# Transfer control
# ----------------------------
BATCH_STEPS = int(os.environ.get("BATCH_STEPS", "100"))
POLL_SEC = float(os.environ.get("POLL_SEC", "1.0"))
RANGE_START = os.environ.get("RANGE_START", "-2h")  # Influx range start for queries
DATA_INTERVAL_SEC = float(os.environ.get("DATA_INTERVAL_SEC", "1.0"))

# If set, start transferring from this step (inclusive)
START_STEP = int(os.environ.get("START_STEP", "0"))


def utc_now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def b64_gzip_npy_to_list(state_b64: str):
    """
    Decode base64(gzip(npy_bytes)) -> np.ndarray -> python list
    """
    raw = base64.b64decode(state_b64.encode("utf-8"))
    with gzip.GzipFile(fileobj=BytesIO(raw), mode="rb") as gz:
        npy_bytes = gz.read()
    arr = np.load(BytesIO(npy_bytes), allow_pickle=False)
    return arr.tolist()


def influx_client():
    return InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)


def mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client[APP_NAME]
    return db[PROJECT_ID]


def query_state_map(client: InfluxDBClient, step_start: int, step_end: int):
    """
    Return dict: step(int) -> state_b64(str) for steps in [step_start, step_end]
    We query nes_state and pivot to get step + state_b64 aligned on same _time.
    """
    flux = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {RANGE_START})
  |> filter(fn: (r) => r._measurement == "nes_state")
  |> filter(fn: (r) => r.run_id == "{RUN_ID}" and r.episode == "{EPISODE}")
  |> filter(fn: (r) => r._field == "step" or r._field == "state_b64")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> filter(fn: (r) => r.step >= {step_start} and r.step <= {step_end})
  |> keep(columns: ["step","state_b64"])
'''
    tables = client.query_api().query(flux)
    out = {}
    for tb in tables:
        for rec in tb.records:
            step = rec.values.get("step")
            sb64 = rec.values.get("state_b64")
            if step is None or sb64 is None:
                continue
            out[int(step)] = str(sb64)
    return out


def query_transition_rows(client: InfluxDBClient, step_start: int, step_end: int):
    """
    Return list of dict rows for nes_transition steps in [step_start, step_end]
    pivot to get (step, reward, done, action_idx, capacity, total_energy ...)
    """
    flux = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {RANGE_START})
  |> filter(fn: (r) => r._measurement == "nes_transition")
  |> filter(fn: (r) => r.run_id == "{RUN_ID}" and r.episode == "{EPISODE}")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> filter(fn: (r) => r.step >= {step_start} and r.step <= {step_end})
  |> sort(columns: ["step"], desc: false)
'''
    tables = client.query_api().query(flux)
    rows = []
    for tb in tables:
        for rec in tb.records:
            rows.append(dict(rec.values))
    # ensure sorted by step
    rows.sort(key=lambda x: int(x.get("step", 0)))
    return rows


def query_latest_transition_step(client: InfluxDBClient):
    flux = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {RANGE_START})
  |> filter(fn: (r) => r._measurement == "nes_transition")
  |> filter(fn: (r) => r.run_id == "{RUN_ID}" and r.episode == "{EPISODE}")
  |> filter(fn: (r) => r._field == "step")
  |> last()
'''
    tables = client.query_api().query(flux)
    for tb in tables:
        for rec in tb.records:
            v = rec.get_value()
            if v is None:
                continue
            return int(v)
    return None


def make_doc(step: int, state_list, action, reward: float, next_state_list, done: bool):
    # Follow the style of your push_data.py :contentReference[oaicite:3]{index=3}
    return {
        "project_id": PROJECT_ID,
        "app_name": APP_NAME,
        "model_version": MODEL_VERSION,
        "input_format": ["state", "next_state"],
        "input": {"state": state_list, "next_state": next_state_list},
        "output_format": ["RU_OnOff"],
        "output": action,
        "KPI_format": ["reward"],
        "KPI": reward,
        "data_interval": DATA_INTERVAL_SEC,
        "timestamp": utc_now_iso(),

        # extra metadata (helpful for tracing)
        "run_id": RUN_ID,
        "episode": str(EPISODE),
        "step": int(step),
        "done": bool(done),
    }


def main():
    if not INFLUX_TOKEN:
        raise RuntimeError("INFLUX_TOKEN is empty. Please export INFLUX_TOKEN.")

    ic = influx_client()
    col = mongo_collection()

    print(f"[XFER] influx={INFLUX_URL} org={INFLUX_ORG} bucket={INFLUX_BUCKET}", flush=True)
    print(f"[XFER] run_id={RUN_ID} episode={EPISODE} range_start={RANGE_START}", flush=True)
    print(f"[XFER] mongo={MONGO_URI} db={APP_NAME} collection={PROJECT_ID}", flush=True)
    print(f"[XFER] batch_steps={BATCH_STEPS} poll_sec={POLL_SEC} start_step={START_STEP}", flush=True)

    next_batch_start = START_STEP

    while True:
        latest = query_latest_transition_step(ic)
        if latest is None:
            print("[XFER] no nes_transition yet, waiting...", flush=True)
            time.sleep(POLL_SEC)
            continue

        # We can only build next_state for step s if state(s+1) exists.
        # So require at least end_step+1 state available; we will check when assembling.
        while next_batch_start + BATCH_STEPS - 1 <= latest:
            step_start = next_batch_start
            step_end = next_batch_start + BATCH_STEPS - 1

            # Fetch states for [start, end+1] so we can construct next_state
            state_map = query_state_map(ic, step_start, step_end + 1)
            trans_rows = query_transition_rows(ic, step_start, step_end)

            docs = []
            missing = 0

            for row in trans_rows:
                s = int(row.get("step"))
                sb64 = state_map.get(s)
                nsb64 = state_map.get(s + 1)  # next_state from next step
                if sb64 is None or nsb64 is None:
                    missing += 1
                    continue

                try:
                    state_list = b64_gzip_npy_to_list(sb64)
                    next_state_list = b64_gzip_npy_to_list(nsb64)
                except Exception as e:
                    print(f"[XFER][WARN] step={s} decode failed: {e}", flush=True)
                    missing += 1
                    continue

                # action: prefer action_idx stored in transition; if absent, store None
                #action_idx = row.get("action_idx", None)
                #action = {"action_idx": int(action_idx)} if action_idx is not None else {}
                action_idx = row.get("action_idx", None)
                if action_idx is None:
                    # 若缺 action_idx，你可以選擇：
                    # A) 跳過這筆（推薦，避免存不完整資料）
                    # continue
                    # B) 仍存，但 output 為 None
                    action = None
                else:
                    action = to_onehot_1x256(int(action_idx), dim=256)

                reward = float(row.get("reward", 0.0))
                done = bool(row.get("done", False))

                docs.append(make_doc(s, state_list, action, reward, next_state_list, done))

            if docs:
                col.insert_many(docs, ordered=False)

            print(
                f"[XFER] batch steps [{step_start}..{step_end}] "
                f"latest_transition_step={latest} inserted={len(docs)} missing={missing}",
                flush=True
            )

            next_batch_start += BATCH_STEPS

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
