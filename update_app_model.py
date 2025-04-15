import pickle
import os
import shutil

# You can run this script to create a backup of the original app.py and create a new one
if os.path.exists('app.py'):
    shutil.copy('app.py', 'app_backup.py')
    print("Created backup of original app.py as app_backup.py")

# Read the original file
with open('app.py', 'r') as f:
    content = f.read()

# Replace the pipe loading line
updated_content = content.replace(
    "pipe = pickle.load(open('pipe.pkl', 'rb'))", 
    "pipe = pickle.load(open('pipe_no_xgb.pkl', 'rb'))"
)

# Add preprocessing function if it doesn't exist
if "def preprocess_text_input(text):" not in updated_content:
    preprocessing_function = '''
# Add a preprocessing function to sanitize input data
def preprocess_text_input(text):
    """Handle potential unicode issues by normalizing text input"""
    if isinstance(text, str):
        # Remove non-ASCII characters and normalize text
        return ''.join(char for char in text if ord(char) < 128)
    return text
'''
    # Insert after the load_data function
    updated_content = updated_content.replace(
        "pipe, df = load_data()",
        "pipe, df = load_data()\n" + preprocessing_function
    )

# Modify the prediction code to handle errors
if "predicted_price = np.exp(pipe.predict(query)[0])" in updated_content:
    old_prediction_code = '''    predicted_price = np.exp(pipe.predict(query)[0])
                st.markdown(f"""
                <div class="prediction-result">
                    Predicted Price: ₹{predicted_price:,.2f}
                </div>
                """, unsafe_allow_html=True)'''
                
    new_prediction_code = '''    try:
                # Preprocess string inputs
                for col in query.columns:
                    if query[col].dtype == 'object':
                        query[col] = query[col].apply(preprocess_text_input)
                
                # Make prediction with error handling
                predicted_price = np.exp(pipe.predict(query)[0])
                st.markdown(f"""
                <div class="prediction-result">
                    Predicted Price: ₹{predicted_price:,.2f}
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("Try adjusting your selections or use simpler text values to avoid encoding issues.")'''
    
    updated_content = updated_content.replace(old_prediction_code, new_prediction_code)

# Write updated content
with open('app_updated.py', 'w') as f:
    f.write(updated_content)
    
print("Created updated app file as app_updated.py")
print("To use it, rename it to app.py or run: mv app_updated.py app.py")
