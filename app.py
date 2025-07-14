# Import all necessary libraries
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
warnings.filterwarnings('ignore')

# Set Streamlit page config
st.set_page_config(
    page_title="Water Pollutants Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #0077b6;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #caf0f8 0%, #90e0ef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #03045e;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1.2rem;
        color: #0077b6;
        font-weight: 600;
    }
    .info-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #0077b6;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background: #d1edff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'favorite_stations' not in st.session_state:
    st.session_state.favorite_stations = []

# Define pollutant information based on your dataset
pollutant_info = {
    'NH4': {'name': 'Ammonia', 'unit': 'mg/L', 'safe_range': (0, 0.5), 'description': 'Nitrogen compound, toxic to aquatic life'},
    'BSK5': {'name': 'BOD5', 'unit': 'mg/L', 'safe_range': (0, 3), 'description': 'Biochemical Oxygen Demand'},
    'Suspended': {'name': 'Suspended Solids', 'unit': 'mg/L', 'safe_range': (0, 25), 'description': 'Total suspended particles'},
    'O2': {'name': 'Dissolved Oxygen', 'unit': 'mg/L', 'safe_range': (5, 14), 'description': 'Essential for aquatic life'},
    'NO3': {'name': 'Nitrate', 'unit': 'mg/L', 'safe_range': (0, 10), 'description': 'Common agricultural pollutant'},
    'NO2': {'name': 'Nitrite', 'unit': 'mg/L', 'safe_range': (0, 1), 'description': 'Intermediate nitrogen compound'},
    'SO4': {'name': 'Sulfate', 'unit': 'mg/L', 'safe_range': (0, 250), 'description': 'Industrial pollutant'},
    'PO4': {'name': 'Phosphate', 'unit': 'mg/L', 'safe_range': (0, 0.1), 'description': 'Causes eutrophication'},
    'CL': {'name': 'Chloride', 'unit': 'mg/L', 'safe_range': (0, 250), 'description': 'Salt contamination indicator'}
}

# Data loading and preprocessing functions
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the water quality dataset"""
    try:
        # Try to load the dataset
        file_path = r"C:\Users\HP\OneDrive\Desktop\WaterPrediction\afa2e701598d20110228.xls"
        
        if os.path.exists(file_path):
            # Load the Excel file
            df = pd.read_excel(file_path)
        else:
            st.error(f"File not found: {file_path}")
            return None
        
        # Rename columns to match expected format
        column_mapping = {
            'id': 'station_id',
            'date': 'date',
            'NH4': 'NH4',
            'BSK5': 'BSK5',
            'Suspended': 'Suspended',
            'O2': 'O2',
            'NO3': 'NO3',
            'NO2': 'NO2',
            'SO4': 'SO4',
            'PO4': 'PO4',
            'CL': 'CL'
        }
        
        # Check if columns exist and rename
        existing_columns = df.columns.tolist()
        df.columns = [column_mapping.get(col, col) for col in existing_columns]
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
        
        # Extract year, month, day for modeling
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Handle missing values
        pollutant_columns = ['NH4', 'BSK5', 'Suspended', 'O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
        for col in pollutant_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].mean())
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['date'])
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load the actual dataset
data = load_and_preprocess_data()

# Model training and loading functions
@st.cache_resource
def train_or_load_model():
    """Train a new model or load existing one"""
    if data is None:
        return None, None, None
    
    try:
        # Try to load existing model
        model = joblib.load("pollution_model.pkl")
        model_cols = joblib.load("model_columns.pkl")
        scaler = joblib.load("scaler.pkl")
        st.success("✅ Loaded existing model successfully!")
        return model, model_cols, scaler
    except:
        st.info("🔄 Training new model...")
        return train_new_model()

def train_new_model():
    """Train a new Random Forest model"""
    if data is None:
        return None, None, None
    
    try:
        # Prepare features and targets
        feature_columns = ['station_id', 'year', 'month', 'day_of_year']
        target_columns = ['NH4', 'BSK5', 'Suspended', 'O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
        
        # Encode station_id
        le = LabelEncoder()
        df_model = data.copy()
        df_model['station_id_encoded'] = le.fit_transform(df_model['station_id'].astype(str))
        
        # Prepare final feature set
        X = df_model[['station_id_encoded', 'year', 'month', 'day_of_year']]
        y = df_model[target_columns]
        
        # Remove rows with missing target values
        mask = ~y.isnull().any(axis=1)
        X = X[mask]
        y = y[mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Save model and metadata
        model_info = {
            'model': model,
            'feature_columns': X.columns.tolist(),
            'target_columns': target_columns,
            'label_encoder': le,
            'train_score': model.score(X_train, y_train),
            'test_score': model.score(X_test, y_test)
        }
        
        # Save to files
        joblib.dump(model, "pollution_model.pkl")
        joblib.dump(X.columns.tolist(), "model_columns.pkl")
        joblib.dump(le, "label_encoder.pkl")
        
        st.success(f"✅ Model trained successfully! Test R² Score: {model_info['test_score']:.3f}")
        
        return model, X.columns.tolist(), le
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None

# Load model
model, model_cols, label_encoder = train_or_load_model()

# Sidebar Navigation
st.sidebar.markdown("# 🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Home", "🔍 Predictor", "📊 Data Analysis", "📈 Visualizations", "🎯 Model Insights", "📋 History", "ℹ️ About"]
)

# Function to assess water quality
def assess_water_quality(pollutant_values):
    """Assess overall water quality based on pollutant levels"""
    scores = []
    for pollutant, value in pollutant_values.items():
        if pollutant in pollutant_info:
            safe_min, safe_max = pollutant_info[pollutant]['safe_range']
            if safe_min <= value <= safe_max:
                scores.append(100)
            elif value < safe_min:
                scores.append(max(0, 100 - (safe_min - value) / safe_min * 100))
            else:
                scores.append(max(0, 100 - (value - safe_max) / safe_max * 100))
    
    overall_score = np.mean(scores)
    if overall_score >= 80:
        return "Excellent", "success"
    elif overall_score >= 60:
        return "Good", "success"
    elif overall_score >= 40:
        return "Fair", "warning"
    else:
        return "Poor", "error"

# PAGE: HOME
if page == "🏠 Home":
    st.markdown("<h1 class='main-header'>💧 Water Quality Monitoring System</h1>", unsafe_allow_html=True)
    
    if data is not None:
        # Welcome message
        st.markdown("""
        <div class='info-box'>
            <h3>🌊 Welcome to the Water Quality Monitoring System</h3>
            <p>This comprehensive platform analyzes your water quality dataset and provides AI-powered predictions 
            for various pollutant levels across different monitoring stations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Dataset overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Records", f"{len(data):,}")
        with col2:
            st.metric("🏭 Monitoring Stations", data['station_id'].nunique())
        with col3:
            st.metric("📅 Date Range", f"{data['year'].min()} - {data['year'].max()}")
        with col4:
            st.metric("🧪 Pollutants Tracked", len(pollutant_info))
        
        # Recent measurements
        st.markdown("## 📊 Recent Measurements")
        latest_data = data.nlargest(5, 'date')
        st.dataframe(latest_data[['station_id', 'date', 'NH4', 'O2', 'NO3', 'PO4']], use_container_width=True)
        
        # Quick statistics
        st.markdown("## 📈 Quick Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            # Average pollutant levels
            avg_pollutants = data[list(pollutant_info.keys())].mean()
            st.markdown("### Average Pollutant Levels")
            for pollutant, avg_val in avg_pollutants.items():
                safe_min, safe_max = pollutant_info[pollutant]['safe_range']
                status = "🟢" if safe_min <= avg_val <= safe_max else "🔴"
                st.write(f"{status} {pollutant}: {avg_val:.3f} {pollutant_info[pollutant]['unit']}")
        
        with col2:
            # Station with most measurements
            station_counts = data['station_id'].value_counts()
            st.markdown("### Most Active Stations")
            for station, count in station_counts.head(5).items():
                st.write(f"🏭 Station {station}: {count} measurements")
    
    else:
        st.error("❌ Unable to load dataset. Please check the file path and format.")

# PAGE: PREDICTOR
elif page == "🔍 Predictor":
    st.markdown("<h1 class='main-header'>🔍 Water Pollutants Predictor</h1>", unsafe_allow_html=True)
    
    if data is None or model is None:
        st.error("❌ Model or data not available. Please check your dataset and model files.")
        st.stop()
    
    # Input section
    st.markdown("## 📝 Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Get available stations from data
        available_stations = sorted(data['station_id'].unique())
        station_id = st.selectbox(
            "🏞️ Select Station ID",
            options=available_stations,
            help="Choose from available monitoring stations"
        )
    
    with col2:
        year_input = st.number_input(
            "📆 Enter Year",
            min_value=int(data['year'].min()),
            max_value=2030,
            value=int(data['year'].max()),
            help="Year for prediction"
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        month_input = st.slider(
            "📅 Month",
            min_value=1,
            max_value=12,
            value=6,
            help="Month of the year (1-12)"
        )
    
    with col4:
        day_of_year = st.slider(
            "📅 Day of Year",
            min_value=1,
            max_value=365,
            value=150,
            help="Day of the year (1-365)"
        )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        show_confidence = st.checkbox("Show prediction confidence", value=True)
        compare_historical = st.checkbox("Compare with historical data", value=True)
        save_to_favorites = st.checkbox("Save station to favorites", value=False)
    
    # Predict button
    if st.button('🔍 Predict Water Quality', type="primary"):
        try:
            # Prepare input data
            station_encoded = label_encoder.transform([str(station_id)])[0]
            input_data = np.array([[station_encoded, year_input, month_input, day_of_year]])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Create prediction dictionary
            target_columns = ['NH4', 'BSK5', 'Suspended', 'O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
            predicted_pollutants = dict(zip(target_columns, prediction))
            
            # Assess water quality
            quality_rating, quality_type = assess_water_quality(predicted_pollutants)
            
            # Display results
            st.markdown(f"<h3 style='color:#023e8a;'>Results for Station <code>{station_id}</code> on {year_input}-{month_input:02d}</h3>", unsafe_allow_html=True)
            
            # Overall quality score
            if quality_type == "success":
                st.success(f"🎉 Water Quality: **{quality_rating}**")
            elif quality_type == "warning":
                st.warning(f"⚠️ Water Quality: **{quality_rating}**")
            else:
                st.error(f"🚨 Water Quality: **{quality_rating}**")
            
            # Pollutant results
            st.markdown("### 🧪 Predicted Pollutant Levels")
            cols = st.columns(3)
            
            for i, (pollutant, value) in enumerate(predicted_pollutants.items()):
                with cols[i % 3]:
                    if pollutant in pollutant_info:
                        info = pollutant_info[pollutant]
                        safe_min, safe_max = info['safe_range']
                        
                        # Determine status
                        if safe_min <= value <= safe_max:
                            status = "✅ Safe"
                            color = "#28a745"
                        elif value < safe_min:
                            status = "⬇️ Low"
                            color = "#ffc107"
                        else:
                            status = "⚠️ High"
                            color = "#dc3545"
                        
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h4 style='color:{color};'>{pollutant}</h4>
                            <div class='metric-value'>{value:.3f}</div>
                            <div class='metric-label'>{info['unit']}</div>
                            <p style='margin-top:0.5rem;'>{status}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Save to history
            prediction_record = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'station': station_id,
                'year': year_input,
                'month': month_input,
                'quality': quality_rating,
                'pollutants': predicted_pollutants
            }
            st.session_state.prediction_history.append(prediction_record)
            
            # Save to favorites if requested
            if save_to_favorites and station_id not in st.session_state.favorite_stations:
                st.session_state.favorite_stations.append(station_id)
                st.success(f"Station {station_id} added to favorites!")
            
            # Compare with historical data if requested
            if compare_historical:
                st.markdown("### 📈 Historical Comparison")
                station_data = data[data['station_id'] == station_id].sort_values('date')
                
                if not station_data.empty:
                    # Create comparison chart
                    fig = go.Figure()
                    
                    # Add historical data
                    for pollutant in ['NH4', 'O2', 'NO3', 'PO4']:  # Show key pollutants
                        if pollutant in station_data.columns:
                            fig.add_trace(go.Scatter(
                                x=station_data['date'],
                                y=station_data[pollutant],
                                mode='lines+markers',
                                name=f'{pollutant} (Historical)',
                                line=dict(width=2)
                            ))
                    
                    # Add prediction point
                    pred_date = f"{year_input}-{month_input:02d}-15"  # Use 15th of month
                    for pollutant in ['NH4', 'O2', 'NO3', 'PO4']:
                        if pollutant in predicted_pollutants:
                            fig.add_trace(go.Scatter(
                                x=[pred_date],
                                y=[predicted_pollutants[pollutant]],
                                mode='markers',
                                name=f'{pollutant} (Predicted)',
                                marker=dict(size=12, symbol='star')
                            ))
                    
                    fig.update_layout(
                        title=f"Historical vs Predicted Values for Station {station_id}",
                        xaxis_title="Date",
                        yaxis_title="Concentration (mg/L)",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No historical data available for this station.")
            
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {str(e)}")

# PAGE: DATA ANALYSIS
elif page == "📊 Data Analysis":
    st.markdown("<h1 class='main-header'>📊 Data Analysis</h1>", unsafe_allow_html=True)
    
    if data is None:
        st.error("❌ Dataset not available.")
        st.stop()
    
    # Dataset overview
    st.markdown("## 📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Records", f"{len(data):,}")
    with col2:
        st.metric("🏭 Unique Stations", data['station_id'].nunique())
    with col3:
        st.metric("📅 Date Range", f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")
    with col4:
        st.metric("🧪 Pollutants", len(pollutant_info))
    
    # Data preview
    st.markdown("### 👁️ Data Preview")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Statistical summary
    st.markdown("### 📊 Statistical Summary")
    pollutant_cols = list(pollutant_info.keys())
    available_cols = [col for col in pollutant_cols if col in data.columns]
    
    if available_cols:
        summary_stats = data[available_cols].describe()
        st.dataframe(summary_stats, use_container_width=True)
    
    # Missing data analysis
    st.markdown("### 🔍 Missing Data Analysis")
    missing_data = data[available_cols].isnull().sum()
    if missing_data.sum() > 0:
        fig = px.bar(
            x=missing_data.index,
            y=missing_data.values,
            title="Missing Data Count by Pollutant",
            labels={'x': 'Pollutant', 'y': 'Missing Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing data found!")
    
    # Correlation analysis
    st.markdown("### 🔗 Pollutant Correlations")
    if len(available_cols) > 1:
        corr_matrix = data[available_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Correlation Matrix of Pollutants"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Temporal trends
    st.markdown("### 📈 Temporal Trends")
    yearly_avg = data.groupby('year')[available_cols].mean().reset_index()
    
    fig = px.line(
        yearly_avg,
        x='year',
        y=available_cols,
        title="Average Pollutant Levels Over Time",
        labels={'value': 'Average Concentration (mg/L)', 'year': 'Year'}
    )
    st.plotly_chart(fig, use_container_width=True)

# PAGE: VISUALIZATIONS
elif page == "📈 Visualizations":
    st.markdown("<h1 class='main-header'>📈 Data Visualizations</h1>", unsafe_allow_html=True)
    
    if data is None:
        st.error("❌ Dataset not available.")
        st.stop()
    
    # Visualization controls
    st.markdown("## 🎛️ Visualization Controls")
    
    col1, col2, col3 = st.columns(3)
    
    available_pollutants = [col for col in pollutant_info.keys() if col in data.columns]
    
    with col1:
        viz_type = st.selectbox(
            "Choose visualization type:",
            ["Time Series", "Distribution", "Station Comparison", "Seasonal Analysis", "Correlation Heatmap"]
        )
    
    with col2:
        selected_pollutant = st.selectbox(
            "Select pollutant:",
            options=available_pollutants
        )
    
    with col3:
        selected_stations = st.multiselect(
            "Select stations:",
            options=sorted(data['station_id'].unique()),
            default=sorted(data['station_id'].unique())[:5]
        )
    
    # Filter data
    if selected_stations:
        filtered_data = data[data['station_id'].isin(selected_stations)]
    else:
        filtered_data = data
    
    # Generate visualizations
    if viz_type == "Time Series":
        st.markdown("### 📈 Time Series Analysis")
        
        if not filtered_data.empty:
            # Group by date and calculate mean for multiple stations
            time_series = filtered_data.groupby('date')[selected_pollutant].mean().reset_index()
            
            fig = px.line(
                time_series,
                x='date',
                y=selected_pollutant,
                title=f"{selected_pollutant} Levels Over Time",
                labels={'date': 'Date', selected_pollutant: f'{selected_pollutant} ({pollutant_info[selected_pollutant]["unit"]})'}
            )
            
            # Add safe range lines
            if selected_pollutant in pollutant_info:
                safe_min, safe_max = pollutant_info[selected_pollutant]['safe_range']
                fig.add_hline(y=safe_min, line_dash="dash", line_color="green", annotation_text="Safe Min")
                fig.add_hline(y=safe_max, line_dash="dash", line_color="red", annotation_text="Safe Max")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for selected stations.")
    
    elif viz_type == "Distribution":
        st.markdown("### 📊 Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(
                filtered_data,
                x=selected_pollutant,
                nbins=30,
                title=f"Distribution of {selected_pollutant}",
                labels={selected_pollutant: f'{selected_pollutant} ({pollutant_info[selected_pollutant]["unit"]})'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(
                filtered_data,
                y=selected_pollutant,
                title=f"Box Plot of {selected_pollutant}",
                labels={selected_pollutant: f'{selected_pollutant} ({pollutant_info[selected_pollutant]["unit"]})'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Station Comparison":
        st.markdown("### 🏭 Station Comparison")
        
        if selected_stations:
            comparison_data = filtered_data[filtered_data['station_id'].isin(selected_stations)]
            
            fig = px.violin(
                comparison_data,
                x='station_id',
                y=selected_pollutant,
                title=f"{selected_pollutant} Levels by Station",
                labels={'station_id': 'Station ID', selected_pollutant: f'{selected_pollutant} ({pollutant_info[selected_pollutant]["unit"]})'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one station.")
    
    elif viz_type == "Seasonal Analysis":
        st.markdown("### 🌱 Seasonal Analysis")
        
        # Add month names
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                      7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        
        filtered_data['month_name'] = filtered_data['month'].map(month_names)
        
        seasonal_data = filtered_data.groupby('month_name')[selected_pollutant].mean().reset_index()
        
        fig = px.bar(
            seasonal_data,
            x='month_name',
            y=selected_pollutant,
            title=f"Monthly Average {selected_pollutant} Levels",
            labels={'month_name': 'Month', selected_pollutant: f'{selecte