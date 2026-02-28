# Tohoku Seismic Pipeline

## Overview

This project preprocesses seismic station metadata associated with the 2011 Tohoku earthquake.

The pipeline includes:
- Extraction of station coordinates
- Distance computation from the Tohoku epicenter

## Data Source



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