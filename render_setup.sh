#!/bin/bash

# This script runs when the app is deployed to Render

# Install the required packages
pip install -r requirements.txt

# If alternative model doesn't exist, create it
if [ ! -f pipe_no_xgb.pkl ]; then
    echo "Alternative model not found. Creating it now..."
    python create_alternative_model.py
fi

# Update the app to use the alternative model
python update_app_model.py

# Rename the updated app file to app.py
mv app_updated.py app.py

echo "Setup complete. The app is ready to run."
