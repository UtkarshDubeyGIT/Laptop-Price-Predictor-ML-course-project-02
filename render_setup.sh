#!/bin/bash

echo "===== Starting Render Deployment Setup ====="

# Install pip dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create the model if it doesn't exist
if [ ! -f "pipe_no_xgb.pkl" ]; then
    echo "Creating alternative model without XGBoost..."
    python create_alternative_model.py
fi

# Create preprocessing code patch
echo "Creating preprocessing code patch..."
cat > preprocess_patch.py << 'EOF'
import os

def patch_app_with_preprocessing():
    """Add preprocessing function to app.py to handle Unicode issues"""
    if not os.path.exists('app.py'):
        print("app.py not found!")
        return False
        
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Check if preprocessing function already exists
    if "def preprocess_text_input(text):" in content:
        print("Preprocessing function already exists in app.py")
        return True
        
    # Add preprocessing function after load_data
    preprocess_function = '''
# Add a preprocessing function to sanitize input data
def preprocess_text_input(text):
    """Handle potential unicode issues by normalizing text input"""
    if isinstance(text, str):
        # Remove non-ASCII characters and normalize text
        return ''.join(char for char in text if ord(char) < 128)
    return text
'''
    
    # Insert the preprocessing function
    if "pipe, df = load_data()" in content:
        modified_content = content.replace(
            "pipe, df = load_data()",
            "pipe, df = load_data()" + preprocess_function
        )
        
        # Update the prediction code to preprocess inputs
        if "predicted_price = np.exp(pipe.predict(query)[0])" in modified_content:
            old_prediction = "predicted_price = np.exp(pipe.predict(query)[0])"
            new_prediction = """try:
                # Preprocess string inputs
                for col in query.columns:
                    if query[col].dtype == 'object':
                        query[col] = query[col].apply(preprocess_text_input)
                
                predicted_price = np.exp(pipe.predict(query)[0])
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("Try adjusting your selections or use simpler text values to avoid encoding issues.")
                return"""
                
            modified_content = modified_content.replace(old_prediction, new_prediction)
            
            # Write the modified content
            with open('app.py', 'w') as f:
                f.write(modified_content)
            print("Successfully added preprocessing to app.py")
            return True
    
    print("Could not patch app.py, pattern not found")
    return False

if __name__ == "__main__":
    patch_app_with_preprocessing()
EOF

# Update app.py to use the alternative model and add preprocessing
echo "Patching app.py..."
python preprocess_patch.py

echo "===== Setup Complete ====="
