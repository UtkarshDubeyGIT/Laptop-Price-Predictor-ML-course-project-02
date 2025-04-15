import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

print("Loading dataset...")
try:
    # Try to load the prepared dataframe
    df = pickle.load(open('df.pkl', 'rb'))
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please make sure df.pkl exists in the current directory")
    exit(1)

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

# Option 1: Random Forest - shown to have good performance
rf_model = RandomForestRegressor(
    n_estimators=350, 
    random_state=3,
    max_samples=0.5,
    max_features=0.75,
    max_depth=15
)

# Option 2: Gradient Boosting - also good and no XGBoost dependency
gb_model = GradientBoostingRegressor(n_estimators=500)

# Choose which model to use (Gradient Boosting might be better for deployment)
selected_model = gb_model
print(f"Selected model: {selected_model.__class__.__name__}")

# Build the pipeline
pipe = Pipeline([
    ('step1', step1),
    ('step2', selected_model)
])

# Train the model
print("Training model... (this may take a few minutes)")
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
print("\nTo use this model, run update_app_model.py to update your app.py file")
