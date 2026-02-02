import json
import os
import time
from pathlib import Path
import numpy as np
import io
import base64
import gzip



from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timezone

import ddpg_nes_allscen_lstm_8f_v22 as v22  # 依你檔名改


def load_local_env():
    """
    Load shared env vars from franz_v011/.env.local so simulator and app use the same settings.
    Keep OS environment overrides intact by only filling missing keys.
    """
    env_path = Path(__file__).resolve().parent.parent / "franz_v011" / ".env.local"
    if not env_path.exists():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, value.strip())

    print(f"[SIM-LOCAL] Loaded env from {env_path}")


load_local_env()

# ---- Patch v22.decode_one_hot to CPU-only (avoid torch/cuda) ----

def decode_one_hot_cpu(one_hot_vector):
    """
    one_hot_vector: length 256 one-hot (numpy array)
    return: list[int] length TOTAL_BS=8 bits after role enforcement
    """
    vec = np.asarray(one_hot_vector).astype(np.float32).reshape(-1)
    idx = int(np.argmax(vec))  # 0..255

    # 8-bit binary string
    bits_str = format(idx, f"0{v22.TOTAL_BS}b")  # e.g. "01010101"

    # Assume MSB->RU0. If mapping is reversed in your original v22, flip it with [::-1]
    bits = [int(c) for c in bits_str]

    # Enforce roles if these lists exist in v22; otherwise treat as empty
    inactive = getattr(v22, "INACTIVE_RUS", [])
    coverage = getattr(v22, "COVERAGE_RUS", [])

    for ru in inactive:
        bits[ru] = 0
    for ru in coverage:
        bits[ru] = 1

    return bits

v22.decode_one_hot = decode_one_hot_cpu
print("[SIM-LOCAL] Patched v22.decode_one_hot -> CPU-only")


INFLUX_URL    = os.environ.get("INFLUX_URL", "http://127.0.0.1:8086")
INFLUX_TOKEN  = os.environ.get("INFLUX_TOKEN", "")
INFLUX_ORG    = os.environ.get("INFLUX_ORG", "my-org")
INFLUX_BUCKET = os.environ.get("INFLUX_BUCKET", "api_bucket")

PROJECT_ID = os.environ.get("PROJECT_ID", "ED8F")
APP_NAME = os.environ.get("APP_NAME", "NES")
MEAS_RADIO_RAW = os.environ.get("MEAS_RADIO_RAW", f"{APP_NAME.lower()}_radio_raw")
MEAS_ACTION = os.environ.get("MEAS_ACTION", f"{APP_NAME.lower()}_action")
MEAS_STATE = os.environ.get("MEAS_STATE", f"{APP_NAME.lower()}_state")
MEAS_TRANSITION = os.environ.get("MEAS_TRANSITION", f"{APP_NAME.lower()}_transition")

RUN_ID  = os.environ.get("RUN_ID", "demo_run_001")
EPISODE = int(os.environ.get("EPISODE", "0"))

STEP_PERIOD_SEC = float(os.environ.get("STEP_PERIOD_SEC", "1.0"))
FALLBACK_ACTION_IDX = int(os.environ.get("FALLBACK_ACTION_IDX", "0"))
BEST_SEARCH_FLAG = int(os.environ.get("BEST_SEARCH_FLAG", "0"))
WAIT_FOR_ACTION_SEC = float(os.environ.get("WAIT_FOR_ACTION_SEC", "2.0"))
UE_ID = os.environ.get("UE_ID", "0001")
# Prefer explicit UE_IDS env; otherwise derive sequential IDs based on v22.numberOfUE
NUMBER_OF_UE = int(os.environ.get("NUMBER_OF_UE", getattr(v22, "numberOfUE", 1)))
ue_ids_env = [u.strip() for u in os.environ.get("UE_IDS", "").split(",") if u.strip()]
if ue_ids_env:
    UE_IDS = ue_ids_env
else:
    UE_IDS = [f"{i+1:04d}" for i in range(NUMBER_OF_UE)]

def influx_client():
    return InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)

def write_points(client, points):
    client.write_api(write_options=SYNCHRONOUS).write(bucket=INFLUX_BUCKET, record=points)

def query_one_field_last(client, flux: str):
    tables = client.query_api().query(flux)
    for tb in tables:
        for rec in tb.records:
            return rec.get_value()
    return None


def build_rsrp_snapshot(state: np.ndarray, ue_id: str, step: int, episode: int) -> dict:
    """
    Turn the environment state into an UE-centric radio snapshot:
    rsrps is a list of {pci, rsrp}. Uses mean per RU to keep shape-independent.
    """
    flat = state.reshape(-1, state.shape[-1])
    rsrps = flat.mean(axis=0)  # length = TOTAL_BS
    payload = {
        "ue_id": ue_id,
        "rsrps": [{"pci": int(100 + i), "rsrp": float(r)} for i, r in enumerate(rsrps)],
        "ts": time.time(),
        "step": int(step),
        "episode": int(episode),
    }
    return payload


def build_user_snapshots_from_trace(step_idx: int, episode: int) -> list[dict]:
    """
    Use v22 trace (ueRouteFromFile) to emit per-UE RSRP vectors.
    Valid UE: pos_x and pos_y > -100.
    """
    snapshots = []
    trace = getattr(v22, "ueRouteFromFile", None)
    if trace is None:
        return snapshots

    num_users, route_len, cols = trace.shape
    t = step_idx % route_len

    for i in range(num_users):
        row = trace[i][t]
        pos_x = float(row[1])
        pos_y = float(row[2])
        if pos_x <= -100 or pos_y <= -100:
            continue  # invalid user

        rsrps = []
        for bs in range(v22.TOTAL_BS):
            col = 3 + bs
            val = float(row[col]) if col < cols else -255.0
            rsrps.append({"pci": int(100 + bs), "rsrp": val})

        snapshots.append({
            "ue_id": f"{i+1:04d}",
            "pos_x": pos_x,
            "pos_y": pos_y,
            "rsrps": rsrps,
            "ts": time.time(),
            "step": int(step_idx),
            "episode": int(episode),
        })

    return snapshots

def flux_get_action_idx(run_id: str, episode: int, step: int) -> str:
    return f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: -30m)
  |> filter(fn: (r) => r._measurement == "{MEAS_ACTION}")
  |> filter(fn: (r) => r.project_id == "{PROJECT_ID}" and r.app_name == "{APP_NAME}")
  |> filter(fn: (r) => r.run_id == "{run_id}" and r.episode == "{episode}" and r.step == "{step}")
  |> filter(fn: (r) => r._field == "action_idx")
  |> last()
'''




def make_onehot(action_idx: int, action_dim: int) -> np.ndarray:
    a = np.zeros((action_dim,), dtype=np.float32)
    a[action_idx] = 1.0
    return a

def tensor_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32), allow_pickle=False)
    gz = gzip.compress(buf.getvalue())
    return base64.b64encode(gz).decode("ascii")


def clear_influx_for_run(client) -> None:
    """
    Clear prior nes_radio_raw and nes_action for this RUN_ID/EPISODE to avoid stale data.
    """
    start = "1970-01-01T00:00:00Z"
    stop = datetime.now(timezone.utc).isoformat()
    for measurement in (MEAS_RADIO_RAW, MEAS_ACTION):
        predicate = f'_measurement="{measurement}" AND run_id="{RUN_ID}" AND project_id="{PROJECT_ID}" AND app_name="{APP_NAME}"'
        try:
            client.delete_api().delete(
                start=start,
                stop=stop,
                predicate=predicate,
                bucket=INFLUX_BUCKET,
                org=INFLUX_ORG,
            )
            print(f"[SIM-LOCAL] Cleared {measurement} for run_id={RUN_ID}", flush=True)
        except Exception as e:
            print(f"[SIM-LOCAL][WARN] failed to clear {measurement}: {e}", flush=True)

def main():
    client = influx_client()

    env = v22.Environment()
    state = env.reset()
    step = 0
    episode = EPISODE
    clear_influx_for_run(client)

    print(f"[SIM-LOCAL] INFLUX_URL={INFLUX_URL} ORG={INFLUX_ORG} BUCKET={INFLUX_BUCKET}")
    print(f"[SIM-LOCAL] RUN_ID={RUN_ID} EPISODE={episode} PROJECT_ID={PROJECT_ID} APP_NAME={APP_NAME}")
    print(f"[SIM-LOCAL] ACTION_DIM={v22.ACTION_DIM} TOTAL_BS={v22.TOTAL_BS} UE_IDS={UE_IDS} STEP_PERIOD_SEC={STEP_PERIOD_SEC}")

    while True:
        t0 = time.time()

        # 1) write ue-centric radio snapshots (pci/rsrp pairs) keyed by ue_id
        radio_points = []
        snapshots = build_user_snapshots_from_trace(step, episode)
        if snapshots:
            for snap in snapshots:
                p_radio = (
                    Point(MEAS_RADIO_RAW)
                    .tag("run_id", RUN_ID)
                    .tag("episode", str(episode))
                    .tag("ue_id", snap["ue_id"])
                    .tag("project_id", PROJECT_ID)
                    .tag("app_name", APP_NAME)
                    .field("step", int(step))
                    .field("rsrps_json", json.dumps(snap["rsrps"]))
                    .field("pos_x", float(snap["pos_x"]))
                    .field("pos_y", float(snap["pos_y"]))
                    .field("ts", float(snap["ts"]))
                )
                radio_points.append(p_radio)
        else:
            # Write an empty snapshot to allow downstream pipeline to build state/action
            p_radio = (
                Point(MEAS_RADIO_RAW)
                .tag("run_id", RUN_ID)
                .tag("episode", str(episode))
                .tag("ue_id", "none")
                .tag("project_id", PROJECT_ID)
                .tag("app_name", APP_NAME)
                .field("step", int(step))
                .field("rsrps_json", "[]")
                .field("pos_x", float(-255.0))
                .field("pos_y", float(-255.0))
                .field("ts", float(time.time()))
            )
            radio_points.append(p_radio)

        write_points(client, radio_points)

        # 1b) write state snapshot (current state before action)
        try:
            state_b64 = tensor_to_b64(np.asarray(state))
            p_state = (
                Point(MEAS_STATE)
                .tag("run_id", RUN_ID)
                .tag("episode", str(episode))
                .tag("project_id", PROJECT_ID)
                .tag("app_name", APP_NAME)
                .field("step", int(step))
                .field("state_b64", state_b64)
            )
            write_points(client, [p_state])
        except Exception as e:
            print(f"[SIM-LOCAL][WARN] failed to write state: {e}", flush=True)

        # 2) read action_idx for this step (if inference already wrote it)
        action_idx = None
        deadline = time.time() + WAIT_FOR_ACTION_SEC  # allow worker some time to read & respond
        while time.time() < deadline:
            action_idx = query_one_field_last(client, flux_get_action_idx(RUN_ID, episode, step))
            if action_idx is not None:
                break
            time.sleep(0.05)

        if action_idx is None:
            action_idx = FALLBACK_ACTION_IDX
        else:
            action_idx = int(action_idx)


        action_onehot = make_onehot(action_idx, v22.ACTION_DIM)

        # 3) step env
        next_state, reward, capacity, totalEnergy, bestReward, bestAction, bestCapacity, bestEnergy, done = env.step(
            state, action_onehot, BEST_SEARCH_FLAG
        )

        # 4) write transition (current -> next)
        try:
            next_state_b64 = tensor_to_b64(np.asarray(next_state))
            p_tr = (
                Point(MEAS_TRANSITION)
                .tag("run_id", RUN_ID)
                .tag("episode", str(episode))
                .tag("project_id", PROJECT_ID)
                .tag("app_name", APP_NAME)
                .field("step", int(step))
                .field("action_idx", int(action_idx))
                .field("reward", float(reward))
                .field("capacity", float(capacity))
                .field("energy", float(totalEnergy))
                .field("done", int(done))
                .field("state_b64", state_b64)
                .field("next_state_b64", next_state_b64)
            )
            write_points(client, [p_tr])
        except Exception as e:
            print(f"[SIM-LOCAL][WARN] failed to write transition: {e}", flush=True)

        # advance
        state = next_state
        step += 1
        
        if step % 10 == 0:
            print(f"[SIM-LOCAL] step={step} wrote ue snapshot, action_idx={action_idx}, reward={reward:.3f}, energy={totalEnergy:.3f}")


        if done:
            episode += 1
            step = 0
            state = env.reset()
            print(f"[SIM-LOCAL] episode -> {episode}")

        # pacing
        dt = time.time() - t0
        time.sleep(max(0.0, STEP_PERIOD_SEC - dt))

if __name__ == "__main__":
    main()
