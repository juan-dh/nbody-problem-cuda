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

## The science behind the simulation

The math and physics will be explained in the following notes.

![N-Body notes](notes/notes.png)

## Shared Memory Optimization

The current implementation in `src/nbodysimulation.cu` improves the force computation by using a tiled shared-memory strategy inside the `computeForceTiled` kernel. Instead of having each thread read every body directly from global memory for every interaction, the kernel loads a block of positions and masses into shared memory and then lets all threads in the block reuse that tile. In practice, this means that once a tile is loaded, the data can be accessed many times from fast on-chip memory before moving on to the next tile. This is especially useful in the N-body problem because the same particle data is needed repeatedly when computing all pairwise interactions.

This is a major difference from `src/nbodysimulationold1.cu`, where the algorithm first builds full pairwise arrays of distances and position differences for all bodies and stores them in global memory. That older version explicitly materializes `N x N` interaction data, writes it out to device memory, and later reads it again inside the Leapfrog update kernels. While that approach is conceptually straightforward, it creates a much heavier memory footprint and significantly more traffic to global memory.

The shared-memory version avoids that overhead by computing force contributions directly while streaming tiles of particle data through the multiprocessor. As a result, it no longer needs to allocate and revisit large global arrays containing every pairwise distance and displacement. The computation stays closer to the arithmetic work that actually matters, and the GPU can spend less time waiting on memory. Compared with `nbodysimulationold1.cu`, the current approach is therefore more scalable, uses memory more efficiently, and follows a more standard CUDA optimization pattern for dense all-to-all interaction problems.
