# WebGIS environmental study

Streamlit study application for intersecting an uploaded polygon and raster
with bundled environmental layers: IBAMA embargoes, MapBiomas alerts and
Indigenous lands. It also renders class colors and estimates areas.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run main.py
```

The application expects one polygon and one GeoTIFF upload. The tracked Parquet
files are read from `dados/` and are not downloaded by the app.

## Current status and limits

This is an untested study project, not a production WebGIS. No automated tests
or CI are present, and the interactive flow was not run during this audit.
Area values for raster classes assume 900 m² per pixel without checking raster
resolution or NoData. Chart column positions and input schemas are also assumed.

The repository contains environmental datasets whose provenance, update dates,
redistribution terms and privacy review are not documented. The embargo layer's
schema contains identity/contact-related fields; the application drops selected
columns only after reading the file. Do not publish, redistribute or demonstrate
the bundled data until ownership, legal basis and minimization are confirmed.

The existing `LICENSE` applies only as stated there and does not establish rights
to third-party datasets.
