# Adaptive LiDAR Resolution

Adaptive LiDAR resolution using real RELLIS-3D LiDAR data.

## Workflow

REAL LiDAR
-> Ground / Non-Ground
-> Terrain Understanding
-> Traversability
-> Adaptive Resolution
-> 2.5D Map
-> Drivable Path
-> Vehicle Replay
-> Measured Comparison

## Features

- Ground and non-ground separation
- Terrain cell generation
- Local terrain geometry analysis
- Terrain complexity estimation
- Traversability classification
- Distance-based adaptive LiDAR resolution
- Terrain-priority-based refinement
- 2.5D terrain mapping
- Drivable-space and path visualization
- Vehicle replay using real LiDAR frames
- Benchmark comparison of fixed and adaptive resolution

## Dataset

This project uses the RELLIS-3D dataset.

The raw LiDAR dataset is not included in this repository.

Place the dataset locally at:

data\RELLIS3D

## Project Structure

adaptive-lidar-resolution/
|-- demo/
|-- docs/
|-- results/
|-- scripts/
|-- src/
|-- tests/
|-- .gitignore
|-- README.md
|-- requirements.txt

## Requirements

Install the Python dependencies with:

pip install -r requirements.txt

## Running the Final Demo

Run:

python .\scripts\final_adaptive_lidar_demo.py

The demo supports replay of the available RELLIS-3D sequences.

## Important Notes

The current implementation is an offline replay of real LiDAR scans with terrain-derived drivable-path simulation.

Terrain refinement priority is used to allocate finer spatial resolution to areas requiring more information.

The current system does not use a trained machine-learning pothole or hazard detector.

## Results

The results/ directory contains selected visualizations and benchmark outputs used to evaluate the adaptive-resolution pipeline.

The benchmark files contain measured comparisons between fixed, distance-adaptive, and distance-plus-terrain approaches.

## Dataset and Large Files

The original LiDAR dataset is excluded from Git because of its size.
