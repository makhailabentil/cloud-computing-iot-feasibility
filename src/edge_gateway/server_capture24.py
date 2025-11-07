import ujson
from flask import Flask, jsonify, request
from pathlib import Path
import time
import numpy as np
import pandas as pd
import threading
import matplotlib
matplotlib.use("TkAgg")  # ensures a real macOS GUI window
import matplotlib.pyplot as plt

from src.data_processing import Capture24Loader
from src.evaluation import SystematicEvaluator


# ------------------- Flask setup ------------------- #
app = Flask(__name__)
loader = Capture24Loader(data_dir="../../data/capture24")
evaluator = SystematicEvaluator()

results_file = Path("stream_compression_results.csv")
if not results_file.exists():
    pd.DataFrame(columns=[
        "timestamp", "algorithm", "participant", "axis",
        "segment_index", "upload_bytes", "upload_time_ms"
    ]).to_csv(results_file, index=False)


# ------------------- Upload endpoint ------------------- #
@app.route("/upload", methods=["POST"])
def upload():
    """Receive compressed uploads and log their size and server-side upload time."""
    t0 = time.time()  # start timing as soon as request begins
    try:
        # Parse JSON payload
        data = request.get_json(force=True)
        t1 = time.time()
        upload_time_ms = (t1 - t0) * 1000  # milliseconds from request arrival → JSON parsed

        algorithm = data["algorithm"]
        participant = data["participant_id"]
        seg = data["segment_index"]
        axes = data.get("axes", ["x"])
        compressed = data["compressed"]

        total_bytes = len(request.data)
        now = time.strftime("%H:%M:%S")

        print(
            f"[{now}] Upload {algorithm} | "
            f"P={participant} | Seg={seg} | {len(axes)} axes | "
            f"{total_bytes} bytes | {upload_time_ms:.2f} ms"
        )

        rows = []
        for ax in axes:
            comp_str = ujson.dumps(compressed.get(ax, {}))
            rows.append({
                "timestamp": now,
                "participant": participant,
                "segment_index": seg,
                "axis": ax,
                "algorithm": algorithm,
                "upload_bytes": len(comp_str),
                "upload_time_ms": round(upload_time_ms, 3),
            })

        # consistent column order
        columns = [
            "timestamp",
            "participant",
            "segment_index",
            "axis",
            "algorithm",
            "upload_bytes",
            "upload_time_ms",
        ]
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(results_file, mode="a", header=not results_file.exists(), index=False)

        return jsonify({"status": "ok", "received_bytes": total_bytes,
                        "upload_time_ms": upload_time_ms})

    except Exception as e:
        print("❌ Upload error:", e)
        return jsonify({"error": str(e)}), 500



# ------------------- Participant endpoints ------------------- #
@app.route("/participants", methods=["GET"])
def list_participants():
    participants = loader.list_participants()
    return jsonify({"participants": participants})


@app.route("/participant/<participant_id>/segments", methods=["GET"])
def list_segments(participant_id):
    """List how many segments are available for a participant."""
    window_size = int(request.args.get("window_size", 10000))
    max_segments = int(request.args.get("max_segments", 10))
    axes = request.args.get("axes", "x,y,z").split(",")

    try:
        data = loader.load_participant_data(participant_id, axes=axes,
                                            max_samples=window_size * max_segments)
        segments = loader.segment_data(data['x'], window_size)
        return jsonify({
            "participant_id": participant_id,
            "segment_count": len(segments),
            "window_size": window_size
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/participant/<participant_id>/segment/<int:idx>", methods=["GET"])
def get_segment(participant_id, idx):
    """Return a single data segment."""
    window_size = int(request.args.get("window_size", 10000))
    axes = request.args.get("axes", "x,y,z").split(",")

    try:
        data = loader.load_participant_data(participant_id, axes=axes)
        segments_per_axis = {axis: loader.segment_data(data[axis], window_size) for axis in axes}
        if idx < 0 or idx >= len(segments_per_axis[axes[0]]):
            return jsonify({"error": "Segment index out of range"}), 404

        segs = {axis: segments_per_axis[axis][idx].tolist() for axis in axes}
        return jsonify({
            "participant_id": participant_id,
            "segment_index": idx,
            "axes": axes,
            "data": segs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/participant/<participant_id>/compress", methods=["POST"])
def compress_segment(participant_id):
    """Evaluate and return compression results for a single segment."""
    axis = request.args.get("axis", "x")
    window_size = int(request.args.get("window_size", 10000))
    idx = int(request.args.get("idx", 0))
    data = loader.load_participant_data(participant_id, axes=[axis])
    segments = loader.segment_data(data[axis], window_size)
    if idx >= len(segments):
        return jsonify({"error": "Segment index out of range"}), 404

    segment = segments[idx]
    results_df = evaluator.evaluate_segments([segment], axis_name=axis, participant_id=participant_id)
    results = results_df.to_dict(orient="records")
    return jsonify({"participant_id": participant_id, "axis": axis, "results": results})

# ------------------- Run Flask + Plot ------------------- #
if __name__ == "__main__":
    # Run Flask in background thread
    app.run(host="0.0.0.0", port=5001, debug=True)