#!/bin/bash
mkdir -p ~/.streamlit/
cat > ~/.streamlit/config.toml << EOT
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
EOT
PORT=${PORT:-8501}
pip install --upgrade numpy==1.23.5
pip install --upgrade scikit-learn==1.2.2
if [ ! -f "pipe_no_xgb.pkl" ]
then
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
    X = df.drop(columns=['Price'])
    y = np.log(df['Price'])
    step1 = ColumnTransformer(transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
    ], remainder='passthrough')
    step2 = GradientBoostingRegressor(n_estimators=100, max_depth=5)
    pipe = Pipeline([
        ('step1', step1),
        ('step2', step2)
    ])
    pipe.fit(X, y)
    pickle.dump(pipe, open('pipe_fallback.pkl', 'wb'))
except Exception as e:
    print(f"Error creating fallback model: {e}")
EOT
    python create_fallback_model.py
    cat > update_app.py << EOT
import os
def modify_app_for_fallback():
    if not os.path.exists('app.py'):
        return
    with open('app.py', 'r') as f:
        content = f.read()
    updated_content = content.replace(
        "pipe = pickle.load(open('pipe.pkl', 'rb'))",
        """try:
        pipe = pickle.load(open('pipe.pkl', 'rb'))
    except Exception as e:
        print("Falling back to simplified model")
        pipe = pickle.load(open('pipe_fallback.pkl', 'rb'))"""
    )
    with open('app.py', 'w') as f:
        f.write(updated_content)
if __name__ == "__main__":
    modify_app_for_fallback()
EOT
    python update_app.py
fi
if [ -f "app.py" ]
then
    cat app.py | tr -cd '\11\12\15\40-\176' > app_clean.py
    mv app_clean.py app.py
fi
exec streamlit run app.py --server.port=$PORT