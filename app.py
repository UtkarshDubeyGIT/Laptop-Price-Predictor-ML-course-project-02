import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
import time
import math

# Set page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #3498db;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3498db;
        margin-bottom: 1rem;
    }
    .highlight {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #3498db, #8e44ad);
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        margin: 20px 0;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 5px solid #3498db;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .stSlider > div > div > div {
        background-color: #3498db !important;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #2980b9;
        box-shadow: 0 3px 8px rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }
    .specs-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 20px;
    }
    .spec-item {
        background-color: #f1f1f1;
        padding: 10px;
        border-radius: 5px;
        min-width: 120px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    pipe = pickle.load(open('pipe.pkl', 'rb'))
    df = pickle.load(open('df.pkl', 'rb'))
    return pipe, df

pipe, df = load_data()

# Creating sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)
    st.image("https://img.freepik.com/free-vector/laptop-concept-illustration_114360-464.jpg", use_column_width=True)
    
    st.markdown("### Navigation")
    app_mode = st.radio("", ["Predict Price", "Explore Data", "About"])
    
    st.markdown("---")
    st.markdown("### Created By")
    st.markdown("ML Enthusiast")
    st.markdown("---")
    st.markdown("#### How it works")
    st.info("This app uses machine learning to predict laptop prices based on specifications. The model was trained on a dataset of laptop prices and features.")

# Main app
if app_mode == "Predict Price":
    st.markdown("<h1 class='main-header'>Laptop Price Predictor</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Enter the specifications of the laptop to get an estimated price.
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 class='sub-header'>Basic Specifications</h3>", unsafe_allow_html=True)
        
        # brand
        company = st.selectbox('Brand', df['Company'].unique())
        
        # type of laptop
        type = st.selectbox('Type', df['TypeName'].unique())
        
        # Ram
        ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
        
        # weight
        weight = st.number_input('Weight of the Laptop (kg)', min_value=0.5, max_value=5.0, value=1.5, step=0.1)
        
        # Touchscreen
        touchscreen = st.toggle('Touchscreen')
        
        # IPS
        ips = st.toggle('IPS Display')
    
    with col2:
        st.markdown("<h3 class='sub-header'>Display & Performance</h3>", unsafe_allow_html=True)
        
        # screen size
        screen_size = st.slider('Screen Size (inches)', min_value=10.0, max_value=18.0, value=15.0, step=0.1)
        
        # resolution
        resolution = st.selectbox('Screen Resolution', [
            '1920x1080', '1366x768', '1600x900', '3840x2160',
            '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
        ])
        
        # cpu
        cpu = st.selectbox('CPU', df['Cpu brand'].unique())
        
        # gpu
        gpu = st.selectbox('GPU', df['Gpu brand'].unique())
        
        # os
        os = st.selectbox('Operating System', df['os'].unique())
    
    # Storage options
    st.markdown("<h3 class='sub-header'>Storage Options</h3>", unsafe_allow_html=True)
    
    storage_col1, storage_col2 = st.columns(2)
    
    with storage_col1:
        # hdd
        hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
    
    with storage_col2:
        # ssd
        ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
    
    # Prediction section
    st.markdown("---")
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    
    with predict_col2:
        predict_button = st.button('Predict Price', use_container_width=True)
    
    if predict_button:
        with st.spinner('Calculating...'):
            time.sleep(1)  # Simulated delay for better UX
            
            # Process inputs
            touchscreen = 1 if touchscreen else 0
            ips = 1 if ips else 0
            
            X_res = int(resolution.split('x')[0])
            Y_res = int(resolution.split('x')[1])
            ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
            
            # Create query dataframe
            query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
            query = query.reshape(1, 12)
            
            # Predict
            predicted_price = np.exp(pipe.predict(query)[0])
            
            # Show result
            st.markdown(f"""
            <div class="prediction-result">
                Predicted Price: ${predicted_price:,.2f}
            </div>
            """, unsafe_allow_html=True)
            
            # Show laptop specs summary
            st.markdown("<h3 class='sub-header'>Laptop Specifications Summary</h3>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="highlight">
                <div class="specs-container">
                    <div class="spec-item">
                        <strong>Brand</strong><br>{company}
                    </div>
                    <div class="spec-item">
                        <strong>Type</strong><br>{type}
                    </div>
                    <div class="spec-item">
                        <strong>RAM</strong><br>{ram} GB
                    </div>
                    <div class="spec-item">
                        <strong>Weight</strong><br>{weight} kg
                    </div>
                    <div class="spec-item">
                        <strong>Screen</strong><br>{screen_size}"
                    </div>
                    <div class="spec-item">
                        <strong>Resolution</strong><br>{resolution}
                    </div>
                    <div class="spec-item">
                        <strong>CPU</strong><br>{cpu}
                    </div>
                    <div class="spec-item">
                        <strong>GPU</strong><br>{gpu}
                    </div>
                    <div class="spec-item">
                        <strong>Storage</strong><br>HDD: {hdd}GB<br>SSD: {ssd}GB
                    </div>
                    <div class="spec-item">
                        <strong>OS</strong><br>{os}
                    </div>
                </div>
            </div>
            """.format(
                company=company, type=type, ram=ram, weight=weight,
                screen_size=screen_size, resolution=resolution,
                cpu=cpu, gpu=gpu, hdd=hdd, ssd=ssd, os=os
            ), unsafe_allow_html=True)
            
            # Similar price range laptops
            similar_price_laptops = df[
                (df['Price'] >= predicted_price*0.9) & 
                (df['Price'] <= predicted_price*1.1)
            ].sample(min(5, len(df[(df['Price'] >= predicted_price*0.9) & (df['Price'] <= predicted_price*1.1)])))
            
            if not similar_price_laptops.empty:
                st.markdown("<h3 class='sub-header'>Similar Laptops in This Price Range</h3>", unsafe_allow_html=True)
                
                for i, row in similar_price_laptops.iterrows():
                    with st.expander(f"{row['Company']} - {row['TypeName']} (${row['Price']:,.2f})"):
                        spec_col1, spec_col2 = st.columns(2)
                        with spec_col1:
                            st.write(f"**CPU:** {row['Cpu brand']}")
                            st.write(f"**RAM:** {row['Ram']} GB")
                            st.write(f"**Storage:** HDD {row.get('HDD', 0)} GB, SSD {row.get('SSD', 0)} GB")
                        with spec_col2:
                            st.write(f"**GPU:** {row['Gpu brand']}")
                            st.write(f"**Screen:** {row['Inches']} inches")
                            st.write(f"**OS:** {row['os']}")

elif app_mode == "Explore Data":
    st.markdown("<h1 class='main-header'>Explore Laptop Data</h1>", unsafe_allow_html=True)
    
    # Analytics tabs
    tab1, tab2, tab3 = st.tabs(["Price Analysis", "Brand Comparison", "Feature Importance"])
    
    with tab1:
        st.markdown("<h3 class='sub-header'>Price Distribution</h3>", unsafe_allow_html=True)
        
        # Create price distribution plot with matplotlib instead of plotly
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Price'], bins=50, ax=ax)
        ax.set_title("Laptop Price Distribution")
        ax.set_xlabel("Price (USD)")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        
        # Price ranges
        st.markdown("<h3 class='sub-header'>Price Ranges</h3>", unsafe_allow_html=True)
        
        price_ranges = {
            "Budget (<$500)": len(df[df['Price'] < 500]),
            "Mid-range ($500-$1000)": len(df[(df['Price'] >= 500) & (df['Price'] < 1000)]),
            "High-end ($1000-$1500)": len(df[(df['Price'] >= 1000) & (df['Price'] < 1500)]),
            "Premium ($1500-$2000)": len(df[(df['Price'] >= 1500) & (df['Price'] < 2000)]),
            "Ultra Premium (>$2000)": len(df[df['Price'] >= 2000])
        }
        
        # Pie chart for price ranges
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(list(price_ranges.values()), labels=list(price_ranges.keys()), autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title("Laptop Price Ranges")
        st.pyplot(fig)
    
    with tab2:
        st.markdown("<h3 class='sub-header'>Brand Comparison</h3>", unsafe_allow_html=True)
        
        # Brand price comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(x='Company', y='Price', data=df, ax=ax)
        ax.set_title("Price Distribution by Brand")
        ax.set_xlabel("Brand")
        ax.set_ylabel("Price (USD)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Brand market share
        brand_counts = df['Company'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(brand_counts.values, labels=brand_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title("Brand Market Share")
        st.pyplot(fig)
        
        # Average price by brand and type
        avg_price = df.groupby(['Company', 'TypeName'])['Price'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(x='Company', y='Price', hue='TypeName', data=avg_price, ax=ax)
        ax.set_title("Average Price by Brand and Type")
        ax.set_xlabel("Brand")
        ax.set_ylabel("Average Price (USD)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        st.markdown("<h3 class='sub-header'>Feature Analysis</h3>", unsafe_allow_html=True)
        
        # Correlation between RAM and Price
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Ram', y='Price', data=df, ax=ax)
        ax.set_title("Correlation: RAM vs Price")
        ax.set_xlabel("RAM (GB)")
        ax.set_ylabel("Price (USD)")
        st.pyplot(fig)
        
        # Touchscreen vs Non-touchscreen
        touch_data = df.groupby('Touchscreen')['Price'].mean().reset_index()
        touch_data['Touchscreen'] = touch_data['Touchscreen'].map({0: 'No', 1: 'Yes'})
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Touchscreen', y='Price', data=touch_data, ax=ax)
        ax.set_title("Average Price: Touchscreen vs Non-touchscreen")
        ax.set_xlabel("Touchscreen")
        ax.set_ylabel("Average Price (USD)")
        st.pyplot(fig)
        
        # OS comparison
        os_data = df.groupby('os')['Price'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='os', y='Price', data=os_data, ax=ax)
        ax.set_title("Average Price by Operating System")
        ax.set_xlabel("Operating System")
        ax.set_ylabel("Average Price (USD)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

else:  # About section
    st.markdown("<h1 class='main-header'>About This App</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
        <h3>Laptop Price Predictor</h3>
        <p>This application uses machine learning to predict laptop prices based on their specifications. 
        The prediction model is trained on a dataset of various laptop configurations and their market prices.</p>
        
        <h3>How It Works</h3>
        <p>The app uses a trained regression model to predict laptop prices. 
        The model takes into account various features including:</p>
        <ul>
            <li>Brand and type of laptop</li>
            <li>RAM, storage capacity (HDD/SSD)</li>
            <li>Processor and graphics card</li>
            <li>Screen size, resolution, and display features</li>
            <li>Weight and other physical characteristics</li>
        </ul>
        
        <h3>Model Performance</h3>
        <p>The prediction model achieves an R² score of approximately 0.86, 
        which means it can explain about 86% of the variance in laptop prices.</p>
        
        <h3>Technologies Used</h3>
        <ul>
            <li>Streamlit: For the web application framework</li>
            <li>Scikit-learn: For machine learning models</li>
            <li>Pandas & NumPy: For data manipulation</li>
            <li>Matplotlib & Seaborn: For data visualization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    **Disclaimer**: The predictions provided by this app are based on historical data and should be used for reference only. 
    Actual laptop prices may vary based on market conditions, availability, and other factors.
    """)
