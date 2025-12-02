import ujson
from flask import Flask, jsonify, request, render_template_string
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


# ------------------- Visualization endpoints ------------------- #
@app.route("/visualize", methods=["GET"])
def visualize():
    """Simple web-based visualization dashboard."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ESP32 Streaming Visualization</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
            h1 { color: #667eea; }
            .stats { display: flex; gap: 20px; margin: 20px 0; }
            .stat-box { flex: 1; padding: 15px; background: #f8f9fa; border-radius: 5px; text-align: center; }
            .stat-value { font-size: 32px; font-weight: bold; color: #667eea; }
            .stat-label { color: #666; margin-top: 5px; }
            .chart { margin: 20px 0; }
            button { padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #5568d3; }
            .auto-refresh { margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📡 ESP32 Streaming Data Visualization</h1>
            <div class="auto-refresh">
                <button onclick="toggleAutoRefresh()" id="refreshBtn">Enable Auto-Refresh</button>
                <span id="status"></span>
            </div>
            <div class="stats" id="stats"></div>
            <div class="chart" id="sizeChart"></div>
            <div class="chart" id="timeChart"></div>
            <div class="chart" id="comparisonChart"></div>
        </div>
        <script>
            let autoRefresh = false;
            let refreshInterval = null;
            
            function toggleAutoRefresh() {
                autoRefresh = !autoRefresh;
                const btn = document.getElementById('refreshBtn');
                const status = document.getElementById('status');
                if (autoRefresh) {
                    btn.textContent = 'Disable Auto-Refresh';
                    status.textContent = ' (Refreshing every 2 seconds)';
                    refreshInterval = setInterval(loadData, 2000);
                } else {
                    btn.textContent = 'Enable Auto-Refresh';
                    status.textContent = '';
                    clearInterval(refreshInterval);
                }
            }
            
            async function loadData() {
                try {
                    const response = await fetch('/api/visualize');
                    const data = await response.json();
                    updateStats(data.stats);
                    updateCharts(data);
                } catch (error) {
                    console.error('Error loading data:', error);
                }
            }
            
            function updateStats(stats) {
                const statsDiv = document.getElementById('stats');
                statsDiv.innerHTML = `
                    <div class="stat-box">
                        <div class="stat-value">${stats.total_uploads}</div>
                        <div class="stat-label">Total Uploads</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">${stats.avg_size}</div>
                        <div class="stat-label">Avg Size (bytes)</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">${stats.avg_time.toFixed(2)}</div>
                        <div class="stat-label">Avg Time (ms)</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">${stats.segments}</div>
                        <div class="stat-label">Segments</div>
                    </div>
                `;
            }
            
            function updateCharts(data) {
                // Size over time
                const sizeTrace = {
                    x: data.segments,
                    y: data.avg_sizes,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Avg Upload Size',
                    line: { color: '#667eea' }
                };
                Plotly.newPlot('sizeChart', [sizeTrace], {
                    title: 'Average Upload Size Over Time',
                    xaxis: { title: 'Segment Index' },
                    yaxis: { title: 'Size (bytes)' }
                });
                
                // Time over time
                const timeTrace = {
                    x: data.segments,
                    y: data.avg_times,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Avg Upload Time',
                    line: { color: '#764ba2' }
                };
                Plotly.newPlot('timeChart', [timeTrace], {
                    title: 'Average Upload Time Over Time',
                    xaxis: { title: 'Segment Index' },
                    yaxis: { title: 'Time (ms)' }
                });
                
                // Algorithm comparison
                const algorithms = Object.keys(data.by_algorithm);
                const sizes = algorithms.map(a => data.by_algorithm[a].avg_size);
                const times = algorithms.map(a => data.by_algorithm[a].avg_time);
                
                const comparisonTrace1 = {
                    x: algorithms,
                    y: sizes,
                    type: 'bar',
                    name: 'Avg Size (bytes)',
                    marker: { color: '#667eea' }
                };
                const comparisonTrace2 = {
                    x: algorithms,
                    y: times,
                    type: 'bar',
                    name: 'Avg Time (ms)',
                    marker: { color: '#764ba2' },
                    yaxis: 'y2'
                };
                
                Plotly.newPlot('comparisonChart', [comparisonTrace1, comparisonTrace2], {
                    title: 'Algorithm Comparison',
                    xaxis: { title: 'Algorithm' },
                    yaxis: { title: 'Size (bytes)', side: 'left' },
                    yaxis2: { title: 'Time (ms)', side: 'right', overlaying: 'y' },
                    barmode: 'group'
                });
            }
            
            // Load data on page load
            loadData();
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)


@app.route("/api/visualize", methods=["GET"])
def api_visualize():
    """API endpoint for visualization data."""
    try:
        if not results_file.exists():
            return jsonify({
                "stats": {
                    "total_uploads": 0,
                    "avg_size": 0,
                    "avg_time": 0,
                    "segments": 0
                },
                "segments": [],
                "avg_sizes": [],
                "avg_times": [],
                "by_algorithm": {}
            })
        
        df = pd.read_csv(results_file)
        
        if df.empty:
            return jsonify({
                "stats": {
                    "total_uploads": 0,
                    "avg_size": 0,
                    "avg_time": 0,
                    "segments": 0
                },
                "segments": [],
                "avg_sizes": [],
                "avg_times": [],
                "by_algorithm": {}
            })
        
        # Calculate statistics
        stats = {
            "total_uploads": len(df),
            "avg_size": int(df['upload_bytes'].mean()) if 'upload_bytes' in df.columns else 0,
            "avg_time": float(df['upload_time_ms'].mean()) if 'upload_time_ms' in df.columns else 0,
            "segments": int(df['segment_index'].nunique()) if 'segment_index' in df.columns else 0
        }
        
        # Time series data
        segments = []
        avg_sizes = []
        avg_times = []
        if 'segment_index' in df.columns:
            segment_stats = df.groupby('segment_index').agg({
                'upload_bytes': 'mean',
                'upload_time_ms': 'mean'
            }).reset_index()
            segments = segment_stats['segment_index'].tolist()
            avg_sizes = segment_stats['upload_bytes'].tolist()
            avg_times = segment_stats['upload_time_ms'].tolist()
        
        # Algorithm comparison
        by_algorithm = {}
        if 'algorithm' in df.columns:
            for algo in df['algorithm'].unique():
                algo_df = df[df['algorithm'] == algo]
                by_algorithm[algo] = {
                    "count": len(algo_df),
                    "avg_size": int(algo_df['upload_bytes'].mean()) if 'upload_bytes' in algo_df.columns else 0,
                    "avg_time": float(algo_df['upload_time_ms'].mean()) if 'upload_time_ms' in algo_df.columns else 0
                }
        
        return jsonify({
            "stats": stats,
            "segments": segments,
            "avg_sizes": avg_sizes,
            "avg_times": avg_times,
            "by_algorithm": by_algorithm
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------- Run Flask + Plot ------------------- #
if __name__ == "__main__":
    # Run Flask in background thread
    app.run(host="0.0.0.0", port=5001, debug=True)