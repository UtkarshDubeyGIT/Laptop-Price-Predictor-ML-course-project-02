# Create the .streamlit directory in user's home if it doesn't exist
mkdir -p ~/.streamlit/

# Create a Streamlit configuration file with server settings
# This configures the server to run in headless mode and use the port provided by Render
# We use a slightly different approach to avoid TOML parsing issues with variables
echo "[server]
headless = true
enableCORS = false
enableXsrfProtection = false
" > ~/.streamlit/config.toml

# Use environment variable PORT if it exists, otherwise use 8501 as default
PORT=${PORT:-8501}

# Start Streamlit with the port specified explicitly via command line
# This avoids the TOML parsing issue with $PORT variable
exec streamlit run app.py --server.port=$PORT