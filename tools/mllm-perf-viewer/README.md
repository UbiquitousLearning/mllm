# MLLM Performance Viewer

This tool visualizes performance data from `perf.json` files, which contain memory and operation events collected during program execution. The visualization includes memory usage charts and timeline views, allowing you to analyze memory allocation patterns and operation execution times.

## Features

- **Memory Usage Plot**: Shows total memory consumption over time
- **Memory Events Timeline**: Displays memory allocation/deallocation events
- **Operation Events Timeline**: Shows execution timelines for operations
- **Time Unit Conversion**: Switch between ns, us, ms, and s
- **Overlap Handling**: Automatically manages overlapping events
- **Interactive Exploration**: Zoom, pan, and hover for detailed information
- **Performance Summary**: Key metrics at a glance

## Installation

```shell
pip install -r requirements.txt
```

## Usage

### Running the Visualizer

```bash
python main.py
```

The application will start and provide a local URL (usually `http://127.0.0.1:7860`). Open this URL in your browser.

### Using the Interface

1. **Upload your perf.json file**:
   - Click the "Upload perf.json" button
   - Select your performance data file

2. **Select time unit**:
   - Choose between nanoseconds (ns), microseconds (us), milliseconds (ms), or seconds (s)

3. **Click "Visualize"**:
   - The tool will process your data and display three visualizations:
     1. Memory Usage Over Time
     2. Memory Events Timeline
     3. Operation Events Timeline

4. **Interact with the visualizations**:
   - Hover over events to see detailed information
   - Use the zoom and pan tools to explore specific time ranges
   - Click and drag to select areas for closer inspection

### Understanding the Visualizations

1. **Memory Usage Plot**:
   - Shows total memory consumption over time
   - The area chart fills to the x-axis, showing active memory allocations
   - Hover to see exact memory usage at specific times

2. **Memory Events Timeline**:
   - Each horizontal bar represents a memory allocation
   - The length of the bar shows the duration of the allocation
   - Different tracks (vertical levels) show concurrent allocations
   - Orange color indicates memory events

3. **Operation Events Timeline**:
   - Each horizontal bar represents an operation execution
   - The length of the bar shows the operation duration
   - Different tracks show concurrent operations
   - Blue color indicates operation events

### Keyboard Shortcuts

- **Z**: Zoom to selection
- **P**: Pan the view
- **Double-click**: Reset view
- **Arrow keys**: Pan while zoomed in

## Input File Format

The visualizer expects a JSON file with the following structure:

```json
{
  "init_time": 1753865973438836,
  "memory_blobs": [
    {
      "device_type": "CPU",
      "end_time": 1753865973439167,
      "memory_usage": 8388608,
      "start_time": 1753865973438862,
      "uuid": 0
    },
    // ... more memory events
  ],
  "op_blobs": [
    {
      "device_type": "CPU",
      "end_time": 1753865973439167,
      "op_name": "convolution",
      "start_time": 1753865973438862
    },
    // ... more operation events
  ]
}
```
