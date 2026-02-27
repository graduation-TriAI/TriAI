# Tohoku GNSS Pipeline

## Overview

This project preprocesses high-rate GNSS displacement data from the 2011 Tohoku earthquake.

The pipeline includes:
- Extraction of station coordinates
- Distance computation from the Tohoku epicenter
- Filtering stations within 300 km
- Slicing GNSS time series into 600s windows with 300s overlap
- Exporting processed data as compressed NPZ files

## Data Source

GNSS displacement data were obtained from:

Shu, Y., & Xu, P. (2020).  
*1-Hz PPP displacements at GEONET stations during the 2011 Tohoku Mw9.0 earthquake*  
[dataset publication series]. PANGAEA.  
https://doi.org/10.1594/PANGAEA.914110  

Accessed on: 2026-02-19.

## Project Structure
```
data/
  raw/        # original downloaded data (not tracked in Git)
  npz/        # generated windowed GNSS data (not tracked in Git)
  csv/        # processed summary CSV files
  sample/     # small example outputs

scripts/      # main preprocessing scripts
tools/        # utility / helper scripts
```

## How to Run

Example:
```
python -m scripts.extract_station_lat_lon
python -m scripts.compute_distance_to_tohoku
python -m scripts.filter_within_300km
python -m scripts.slice_gnss_600_300
```