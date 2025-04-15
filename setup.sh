#!/bin/bash

# Create the .streamlit directory in user's home if it doesn't exist
mkdir -p ~/.streamlit/

# Create a Streamlit configuration file with server settings
echo "[server]
headless = true
enableCORS = false
enableXsrfProtection = false
" > ~/.streamlit/config.toml

# Use environment variable PORT if it exists, otherwise use 8501 as default
PORT=${PORT:-8501}

# Start Streamlit with the port specified explicitly via command line
echo "Starting Streamlit app on port $PORT"
exec streamlit run app.py --server.port=$PORT