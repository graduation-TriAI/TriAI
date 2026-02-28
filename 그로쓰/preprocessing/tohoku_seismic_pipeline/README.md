# Tohoku Seismic Pipeline

## Overview

This project preprocesses seismic station metadata associated with the 2011 Tohoku earthquake.

The pipeline includes:
- Extraction of station coordinates
- Distance computation from the Tohoku epicenter

## Data Source

Seismic station metadata and waveform data were obtained from:

National Research Institute for Earth Science and Disaster Resilience (NIED).  
Hi-net (High Sensitivity Seismograph Network Japan),  
2011 Tohoku earthquake records.  
https://www.hinet.bosai.go.jp/

Accessed: 2026-02-24.

## Project Structure
```
data/
  raw/        # original downloaded data (not tracked in Git)
  csv/        # processed summary CSV files (not tracked in Git)

scripts/      # main preprocessing scripts
```

## How to Run

Example:
```
python -m scripts.extract_station_latlon
python -m scripts.compute_distance_to_tohoku
```