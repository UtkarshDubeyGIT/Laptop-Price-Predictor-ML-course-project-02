# Create the .streamlit directory in user's home if it doesn't exist
mkdir -p ~/.streamlit/

# Create a Streamlit configuration file with server settings
# This configures the server to run in headless mode and use the port provided by Render
# Also disables CORS and XSRF protection for API compatibility
echo "[server]
headless = true
port = \$PORT
enableCORS = false
enableXsrfProtection = false
" > ~/.streamlit/config.toml