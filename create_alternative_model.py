import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os
import sys

# Make sure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Loading dataset...")
try:
    # Try to load the prepared dataframe
    df = pickle.load(open('df.pkl', 'rb'))
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please make sure df.pkl exists in the current directory")
    sys.exit(1)

# Prepare the data
print("Preparing data for training...")
X = df.drop(columns=['Price'])
y = np.log(df['Price'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Create a model that doesn't use XGBoost
print("Building model pipeline...")
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

# We'll use Gradient Boosting as it's a good balance of performance and simplicity
gb_model = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.1)
print(f"Selected model: GradientBoostingRegressor")

# Build the pipeline
pipe = Pipeline([
    ('step1', step1),
    ('step2', gb_model)
])

# Train the model
print("Training model... (this may take a few minutes)")
try:
    pipe.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Model performance - R2 score: {r2:.4f}, MSE: {mse:.4f}')
    
    # Save the alternative model
    print("Saving model...")
    pickle.dump(pipe, open('pipe_no_xgb.pkl', 'wb'))
    print("Alternative model without XGBoost saved as 'pipe_no_xgb.pkl'")
except Exception as e:
    print(f"Error training model: {e}")
    sys.exit(1)
