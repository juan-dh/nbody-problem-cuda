# N-Body Problem with CUDA

## Overview

![N-Body Simulation GIF](nbody_visualization.gif)

This project implements a CUDA-based N-body simulation and a Python visualization pipeline that can export the simulation as either a GIF or an MP4 video.

The repository includes:

- A CUDA simulator in `src/nbodysimulation.cu`
- A `Makefile` to build and run the project
- A shared Python visualization script in `nbody-visualization.py`
- Generated outputs such as `nbody_data.bin`, `nbody_proof_of_execution.csv`, `nbody_visualization.gif`, and `nbody_visualization.mp4`


## How it works

### Installation

Installation requirements are described in [`INSTALL`](/INSTALL).

### Build

To compile the CUDA simulator:

```bash
make build
```

This produces the executable:

```text
bin/nbodysimulation
```

### Run the Simulation

To build and run in one step:

```bash
make run
```

You can also run the binary directly:

```bash
./bin/nbodysimulation
```

The program accepts positional arguments in this order:

```text
./bin/nbodysimulation [n_bodies] [n_steps] [init_method] [step_size]
```

Example:

```bash
./bin/nbodysimulation 100000 1500 "multiple gaussians" 0.001
```

### Simulation Output

After execution, the simulator writes:

- `nbody_data.bin`
  This is the full binary output used by the visualization script.
- `nbody_proof_of_execution.csv`
  This is a smaller CSV export used as a lightweight proof of execution.

### Create Visualizations

The main visualization entry point is:

```text
nbody-visualization.py
```

It reads `nbody_data.bin` and exports an animation based on the file extension you choose for `output_path`.

If `output_path` ends with:

- `.gif`, the script uses `PillowWriter`
- `.mp4`, the script uses `FFMpegWriter`

#### Visualization Parameters

The shared script supports several configurable parameters, including:

- `input_file`
- `output_path`
- `max_particles`
- `frame_stride`
- `fps`
- `marker_size`
- `alpha`
- `elev`
- `azim`
- `dpi`
- `xlim`, `ylim`, `zlim`
- `seed`
- `zoom`

These parameters let you control sampling density, camera angle, opacity, render bounds, and output quality.

### Clean Generated Files

To clean the build artifacts defined in the `Makefile`:

```bash
make clean
```

## Notes

- The CUDA build currently expects CUDA headers under `/usr/local/cuda/include`.
- The simulator currently writes `nbody_data.bin`, which is the input file expected by the visualization scripts.
