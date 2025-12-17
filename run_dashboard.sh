#!/bin/bash

# Ensure required directories exist
mkdir -p input_videos output_videos stubs models memdump logs

# Run the dashboard
streamlit run dashboard/app.py 