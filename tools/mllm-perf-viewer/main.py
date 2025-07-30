import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from enum import Enum
import numpy as np
import heapq


class TimeUnit(Enum):
    NS = "ns"
    US = "us"
    MS = "ms"
    S = "s"


def convert_time(time_ns, base_time, unit):
    """Convert nanosecond time to specified unit"""
    if base_time is None or base_time == 0:
        relative_ns = time_ns
    else:
        relative_ns = time_ns - base_time

    if unit == TimeUnit.NS.value:
        return relative_ns
    elif unit == TimeUnit.US.value:
        return relative_ns / 1000.0
    elif unit == TimeUnit.MS.value:
        return relative_ns / 1e6
    elif unit == TimeUnit.S.value:
        return relative_ns / 1e9
    return relative_ns


def assign_tracks(events):
    """Assign tracks to events using an efficient algorithm to handle overlaps"""
    if not events:
        return []

    # Sort events by start time
    events = sorted(events, key=lambda e: e["start"])

    # Initialize tracks: each track stores the end time of the last event
    tracks = []  # list of end times for each track
    assignments = []  # list of (event, track_index) tuples

    for event in events:
        start = event["start"]
        end = event["end"]

        # Check for an available track
        allocated = False
        for track_idx, track_end in enumerate(tracks):
            if track_end <= start:
                # This track is available
                tracks[track_idx] = end
                assignments.append((event, track_idx))
                allocated = True
                break

        if not allocated:
            # Need a new track
            track_idx = len(tracks)
            tracks.append(end)
            assignments.append((event, track_idx))

    return assignments


def create_timeline_figure(events, base_time, unit, title, color):
    """Create a timeline figure for events"""
    if not events:
        return go.Figure()

    # Prepare event data with converted times
    event_list = []
    for i, event in enumerate(events):
        start = convert_time(event["start_time"], base_time, unit)
        end = convert_time(event["end_time"], base_time, unit)

        # Create event name
        if "op_name" in event:
            event_name = event["op_name"]
        else:
            uuid_val = event.get("uuid", i)
            event_name = f"Memory {uuid_val}"

        event_list.append(
            {
                "name": event_name,
                "start": start,
                "end": end,
                "duration": end - start,
                "uuid": event.get("uuid", i),
                "device_type": event.get("device_type", "CPU"),
                "memory_usage": event.get("memory_usage", 0),
            }
        )

    # Assign tracks to events to handle overlaps
    track_assignments = assign_tracks(
        [{"start": e["start"], "end": e["end"], "data": e} for e in event_list]
    )

    # Prepare data for plotting
    plot_data = []
    for assignment in track_assignments:
        event_info, track_idx = assignment
        event = event_info["data"]

        # Create hover text
        hover_text = (
            f"<b>{event['name']}</b><br>"
            f"Start: {event['start']:.3f}{unit}<br>"
            f"End: {event['end']:.3f}{unit}<br>"
            f"Duration: {event['duration']:.3f}{unit}<br>"
            f"UUID: {event['uuid']}<br>"
            f"Device: {event['device_type']}"
        )
        if "memory_usage" in event:
            hover_text += f"<br>Memory: {event['memory_usage']} bytes"

        plot_data.append(
            {
                "Event": event["name"],
                "Start": event["start"],
                "End": event["end"],
                "Duration": event["duration"],
                "Track": track_idx,
                "HoverText": hover_text,
            }
        )

    if not plot_data:
        return go.Figure()

    df = pd.DataFrame(plot_data)

    # Find min and max times for axis scaling
    min_time = df["Start"].min()
    max_time = df["End"].max()
    time_range = max_time - min_time
    padding = time_range * 0.05  # 5% padding

    # Create figure using Plotly Graph Objects for more control
    fig = go.Figure()

    # Add a bar for each event
    for _, row in df.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row["Duration"]],  # width of the bar
                y=[row["Track"]],  # y position (track)
                base=[row["Start"]],  # starting point on x-axis
                orientation="h",
                marker_color=color,
                name=row["Event"],
                hoverinfo="text",
                hovertext=row["HoverText"],
                width=0.8,  # bar height
            )
        )

    # Set layout
    fig.update_layout(
        title=title,
        xaxis_title=f"Time ({unit})",
        yaxis_title="Track",
        barmode="stack",
        showlegend=False,
        height=300 + len(df) * 20,
        hovermode="closest",
        xaxis=dict(
            type="linear",
            range=[min_time - padding, max_time + padding],
            tickformat=".3f",
        ),
        yaxis=dict(
            type="linear",
            tickmode="array",
            tickvals=df["Track"].unique(),
            ticktext=[f"Track {i}" for i in df["Track"].unique()],
            autorange=True,
        ),
    )

    return fig


def create_memory_plot(memory_blobs, base_time, unit):
    """Create memory usage plot"""
    if not memory_blobs:
        return go.Figure()

    # Create list of memory events
    events = []
    for event in memory_blobs:
        start = convert_time(event["start_time"], base_time, unit)
        end = convert_time(event["end_time"], base_time, unit)
        usage = event["memory_usage"]
        events.append(
            {"start": start, "end": end, "usage": usage, "uuid": event.get("uuid", "")}
        )

    # Sort by start time
    events.sort(key=lambda e: e["start"])

    # Create time points for memory changes
    time_points = []
    for event in events:
        time_points.append((event["start"], event["usage"]))
        time_points.append((event["end"], -event["usage"]))

    # Sort by time
    time_points.sort(key=lambda x: x[0])

    # Calculate cumulative memory usage
    times = []
    values = []
    current_memory = 0

    # Start from the first event
    if time_points:
        first_time = time_points[0][0] - 0.001  # slightly before first event
        times.append(first_time)
        values.append(0)

    for time, delta in time_points:
        if times and time == times[-1]:
            # Update existing point if same time
            current_memory += delta
            values[-1] = current_memory
        else:
            if times:
                # Add intermediate point to create step effect
                times.append(time)
                values.append(current_memory)
            current_memory += delta
            times.append(time)
            values.append(current_memory)

    # Add a final point for better visualization
    if times:
        times.append(times[-1] + 0.001)
        values.append(values[-1])

    # Create figure
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=values,
            fill="tozeroy",
            mode="lines",
            name="Memory Usage",
            line=dict(color="royalblue"),
            hovertemplate="<b>Memory:</b> %{y} bytes<br><b>Time:</b> %{x:.3f}"
            + unit
            + "<extra></extra>",
        )
    )

    # Set layout
    min_time = min(times) if times else 0
    max_time = max(times) if times else 1
    time_range = max_time - min_time
    padding = time_range * 0.05

    fig.update_layout(
        title="Memory Usage Over Time",
        xaxis_title=f"Time ({unit})",
        yaxis_title="Memory (bytes)",
        hovermode="x unified",
        height=400,
        xaxis=dict(
            type="linear",
            range=[min_time - padding, max_time + padding],
            tickformat=".3f",
        ),
    )

    return fig


def visualize_perf(file_path, time_unit):
    """Visualize performance data"""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        return f"Error loading file: {str(e)}", None, None, None

    # Get base data
    base_time = data.get("init_time", 0)
    memory_blobs = data.get("memory_blobs", [])
    op_blobs = data.get("op_blobs", [])

    # Calculate summary statistics
    mem_duration = 0
    op_duration = 0
    mem_peak = 0
    op_count = len(op_blobs)

    if memory_blobs:
        mem_start = min(e["start_time"] for e in memory_blobs)
        mem_end = max(e["end_time"] for e in memory_blobs)
        mem_duration = convert_time(mem_end - mem_start, 0, time_unit)

        # Calculate peak memory
        points = []
        for event in memory_blobs:
            points.append((event["start_time"], event["memory_usage"]))
            points.append((event["end_time"], -event["memory_usage"]))

        points.sort(key=lambda x: x[0])
        current = 0
        peak = 0
        for _, delta in points:
            current += delta
            if current > peak:
                peak = current
        mem_peak = peak

    if op_blobs:
        op_start = min(e["start_time"] for e in op_blobs)
        op_end = max(e["end_time"] for e in op_blobs)
        op_duration = convert_time(op_end - op_start, 0, time_unit)

    # Create summary
    summary = f"""
    ## 🚀 MLLM Performance Summary
    
    ### 📊 Memory Metrics
    - **Events**: {len(memory_blobs)}
    - **Duration**: {mem_duration:.3f}{time_unit}
    - **Peak Usage**: {mem_peak:,} bytes
    
    ### ⚙️ Operation Metrics
    - **Events**: {op_count}
    - **Duration**: {op_duration:.3f}{time_unit}
    
    ### ⏱️ Time Reference
    - **Base Time**: {base_time}
    - **Time Unit**: {time_unit}
    """

    # Create plots
    memory_plot = create_memory_plot(memory_blobs, base_time, time_unit)
    memory_timeline = create_timeline_figure(
        memory_blobs,
        base_time,
        time_unit,
        "Memory Events Timeline",
        "rgb(255, 153, 51)",
    )
    op_timeline = create_timeline_figure(
        op_blobs, base_time, time_unit, "Operation Events Timeline", "rgb(51, 153, 255)"
    )

    return summary, memory_plot, memory_timeline, op_timeline


# Create Gradio interface
with gr.Blocks(title="MLLM Performance Viewer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 MLLM Performance Viewer")
    gr.Markdown("Visualize memory and operation events from perf.json files")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload perf.json", type="filepath")
            time_unit = gr.Radio(
                choices=[t.value for t in TimeUnit],
                value=TimeUnit.US.value,
                label="Time Unit",
            )
            plot_btn = gr.Button("Visualize", variant="primary")

        with gr.Column(scale=2):
            summary = gr.Markdown("## Summary will appear here after visualization")

    with gr.Row():
        memory_plot = gr.Plot(label="📊 Memory Usage Over Time")

    with gr.Row():
        memory_timeline = gr.Plot(label="🧠 Memory Events Timeline")

    with gr.Row():
        op_timeline = gr.Plot(label="⚙️ Operation Events Timeline")

    plot_btn.click(
        visualize_perf,
        inputs=[file_input, time_unit],
        outputs=[summary, memory_plot, memory_timeline, op_timeline],
    )

if __name__ == "__main__":
    demo.launch()
