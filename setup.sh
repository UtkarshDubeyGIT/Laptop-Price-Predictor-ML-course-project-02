#!/bin/bash

# Create the .streamlit directory in user's home if it doesn't exist
mkdir -p ~/.streamlit/

# Create a Streamlit configuration file with server settings
cat > ~/.streamlit/config.toml << EOT
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
EOT

# Use environment variable PORT if it exists, otherwise use 8501 as default
PORT=${PORT:-8501}

# Fix NumPy version compatibility issues
echo "Fixing potential NumPy compatibility issues..."
pip install --upgrade numpy==1.23.5
pip install --upgrade scikit-learn==1.2.2

# Create fallback model if missing
if [ ! -f "pipe_no_xgb.pkl" ]
then
    echo "Main model file (pipe.pkl) may have compatibility issues."
    echo "Creating a simple fallback model..."
    
    # Create a simple python script to generate a basic model
    cat > create_fallback_model.py << EOT
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

try:
    print("Loading dataframe...")
    df = pickle.load(open('df.pkl', 'rb'))
    print("Creating a simple model pipeline...")
    
    # Create a basic model
    X = df.drop(columns=['Price'])
    y = np.log(df['Price'])
    
    # Create a simpler model that's less likely to have issues
    step1 = ColumnTransformer(transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
    ], remainder='passthrough')
    
    # Use a simple GradientBoostingRegressor with minimal parameters
    step2 = GradientBoostingRegressor(n_estimators=100, max_depth=5)
    
    pipe = Pipeline([
        ('step1', step1),
        ('step2', step2)
    ])
    
    print("Training fallback model...")
    pipe.fit(X, y)
    
    print("Saving fallback model...")
    pickle.dump(pipe, open('pipe_fallback.pkl', 'wb'))
    print("Fallback model created successfully!")
    
except Exception as e:
    print(f"Error creating fallback model: {e}")
EOT

    # Run the script to create the fallback model
    python create_fallback_model.py
    
    # Create a wrapper to update the app to use the fallback model
    cat > update_app.py << EOT
import os

def modify_app_for_fallback():
    """Update app.py to use fallback model"""
    if not os.path.exists('app.py'):
        print("app.py not found!")
        return
        
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Update the model loading code
    updated_content = content.replace(
        "pipe = pickle.load(open('pipe.pkl', 'rb'))",
        """try:
        pipe = pickle.load(open('pipe.pkl', 'rb'))
    except Exception as e:
        print(f"Error loading main model: {e}")
        print("Falling back to simplified model")
        pipe = pickle.load(open('pipe_fallback.pkl', 'rb'))"""
    )
    
    with open('app.py', 'w') as f:
        f.write(updated_content)
    print("Updated app.py to use fallback model when needed")

if __name__ == "__main__":
    modify_app_for_fallback()
EOT
    
    # Update the app
    python update_app.py
fi

# Fix the text encoding in app.py if it exists
if [ -f "app.py" ]
then
    echo "Checking app.py for encoding issues..."
    # Use tr instead of sed for better compatibility
    cat app.py | tr -cd '\11\12\15\40-\176' > app_clean.py
    mv app_clean.py app.py
fi

echo "Starting Streamlit app on port $PORT"
exec streamlit run app.py --server.port=$PORT