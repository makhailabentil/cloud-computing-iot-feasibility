from rle_mpy import RunLengthCompressor
from delta_mpy import DeltaEncodingCompressor
from quant_mpy import QuantizationCompressor
import network, urequests, time, gc, ujson

# --- Configuration ---
SSID = "network"
PASSWORD = "greentea01"
SERVER_IP = "10.0.0.134"
SERVER_PORT = 5001
BASE_URL = f"http://{SERVER_IP}:{SERVER_PORT}"

PARTICIPANT_ID = "P001"
AXES = ["x", "y", "z"]           # now we request all 3 axes
WINDOW_SIZE = 200
MAX_SEGMENTS = 50
DELAY_BETWEEN_SEGMENTS = 0.5   # seconds between loops

# --- Wi-Fi ---
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, PASSWORD)
while not wlan.isconnected():
    time.sleep(0.5)
print("✅ Connected:", wlan.ifconfig())

# --- Compressors ---
ALGORITHMS = {
    "RLE": RunLengthCompressor(),
    "Delta": DeltaEncodingCompressor(),
    "Quant": QuantizationCompressor()
}

# --- Helpers ---
def fetch_segment(idx):
    """Download one segment (all axes) from Flask server"""
    url = f"{BASE_URL}/participant/{PARTICIPANT_ID}/segment/{idx}?axes=" + ",".join(AXES) + f"&window_size={WINDOW_SIZE}"
    print(f"\n[GET] {url}")
    try:
        t0 = time.ticks_ms()
        resp = urequests.get(url)
        t1 = time.ticks_diff(time.ticks_ms(), t0)
        if resp.status_code != 200:
            print("❌ GET failed:", resp.status_code)
            return None
        payload = resp.json()
        resp.close()
        data = payload.get("data", {})
        print(f"✅ Segment {idx}: {len(data[AXES[0]])} samples/axis, {t1} ms transfer")
        return data
    except Exception as e:
        print("❌ Fetch failed:", e)
        return None


def upload_payload(algorithm, idx, compressed_bundle):
    url = f"{BASE_URL}/upload"
    headers = {"Content-Type": "application/json"}
    try:
        # Prepare payload first
        t0 = time.ticks_ms()
        payload = {
            "algorithm": algorithm,
            "participant_id": PARTICIPANT_ID,
            "segment_index": idx,
            "axes": AXES,
            "compressed_len": len(ujson.dumps(compressed_bundle)),
            "compressed": compressed_bundle,
        }

        # Measure upload + include duration
        data_json = ujson.dumps(payload)
        resp = urequests.post(url, headers=headers, data=data_json)
        elapsed = time.ticks_diff(time.ticks_ms(), t0)

        print(f"⬆️ {algorithm} upload ({len(data_json)} B) → {elapsed} ms → {resp.status_code}")
        resp.close()
        return elapsed

    except Exception as e:
        print("❌ Upload failed:", e)
        return None



# --- Continuous Loop ---
for seg_idx in range(MAX_SEGMENTS):
    gc.collect()
    print(f"\n================ SEGMENT {seg_idx}/{MAX_SEGMENTS} ================")
    segment_axes = fetch_segment(seg_idx)
    try:
        raw_json = ujson.dumps(segment_axes)
        original_size = len(raw_json)
        print(f"📦 Original segment size: {original_size} bytes ({len(AXES)} axes)")
    except Exception as e:
        print("⚠️ Could not measure original size:", e)
        original_size = 0

    if not segment_axes:
        print("No more segments or fetch error, stopping.")
        break

    # Compress + upload (each algorithm applied to all axes)
    for name, compressor in ALGORITHMS.items():
        try:
            t0 = time.ticks_ms()
            compressed_bundle = {}
            for axis in AXES:
                axis_data = segment_axes[axis]
                if name == "Delta":
                    deltas, first, ratio = compressor.compress(axis_data)
                    compressed_bundle[axis] = {"deltas": deltas, "first": first, "ratio": ratio}
                else:
                    compressed_bundle[axis] = compressor.compress(axis_data)
            comp_time = time.ticks_diff(time.ticks_ms(), t0)
            print(f"{name}: total comp {comp_time} ms | {len(AXES)} axes")
            upload_payload(name, seg_idx, compressed_bundle)
            gc.collect()
            time.sleep_ms(50)
        except Exception as e:
            print(f"❌ {name} compression error:", e)
            gc.collect()

    time.sleep(DELAY_BETWEEN_SEGMENTS)

print("\n✅ Continuous multi-axis streaming complete.")
wlan.disconnect()
