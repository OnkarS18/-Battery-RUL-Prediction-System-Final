import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Page configuration
st.set_page_config(
    page_title="Battery RUL Prediction",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    .battery-animation {
        text-align: center;
        font-size: 5rem;
        animation: charge 3s infinite;
    }
    @keyframes charge {
        0% { opacity: 0.3; }
        50% { opacity: 1; }
        100% { opacity: 0.3; }
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🔋 Battery RUL Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Machine Learning for Battery Health Monitoring</p>', unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_models():
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure 'best_model.pkl' and 'scaler.pkl' are in the same directory.")
        return None, None

model, scaler = load_models()

# Load dataset for visualizations
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('Battery_dataset.csv')
        return df
    except FileNotFoundError:
        return None

df = load_dataset()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 Data Visualization", "📈 Model Performance", "ℹ️ About"])

# TAB 1: PREDICTION
with tab1:
    if model is not None and scaler is not None:
        st.markdown("### 🔌 Enter Battery Parameters")
        
        # Create two columns for input
        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown("#### 🔋 Battery Information")
            battery_id = st.selectbox(
                "Battery ID",
                options=["B5", "B6", "B7", "B18"],
                help="Select the battery identifier"
            )
            cycle = st.number_input(
                "Cycle Number",
                min_value=1,
                max_value=300,
                value=50,
                help="Current cycle number of the battery"
            )
            
            st.markdown("#### ⚡ Charging Parameters")
            chI = st.number_input(
                "Charging Current (chI)",
                min_value=0.0,
                max_value=3.0,
                value=1.4,
                step=0.01,
                format="%.2f",
                help="Charging current in Amperes"
            )
            chV = st.number_input(
                "Charging Voltage (chV)",
                min_value=3.0,
                max_value=5.0,
                value=4.2,
                step=0.01,
                format="%.2f",
                help="Charging voltage in Volts"
            )
            chT = st.number_input(
                "Charging Temperature (chT)",
                min_value=20.0,
                max_value=35.0,
                value=26.5,
                step=0.1,
                format="%.1f",
                help="Charging temperature in Celsius"
            )
        
        with col2:
            st.markdown("#### 🔻 Discharging Parameters")
            disI = st.number_input(
                "Discharging Current (disI)",
                min_value=0.0,
                max_value=3.0,
                value=2.0,
                step=0.01,
                format="%.2f",
                help="Discharging current in Amperes"
            )
            disV = st.number_input(
                "Discharging Voltage (disV)",
                min_value=2.0,
                max_value=5.0,
                value=3.5,
                step=0.01,
                format="%.2f",
                help="Discharging voltage in Volts"
            )
            disT = st.number_input(
                "Discharging Temperature (disT)",
                min_value=25.0,
                max_value=40.0,
                value=33.0,
                step=0.1,
                format="%.1f",
                help="Discharging temperature in Celsius"
            )
            
            st.markdown("#### 💚 Battery State")
            BCt = st.number_input(
                "Battery Capacity (BCt)",
                min_value=0.5,
                max_value=2.5,
                value=1.5,
                step=0.01,
                format="%.2f",
                help="Battery capacity"
            )
            SOH = st.number_input(
                "State of Health (SOH)",
                min_value=30.0,
                max_value=100.0,
                value=75.0,
                step=0.1,
                format="%.1f",
                help="State of Health percentage"
            )
        
        # Encode battery_id
        battery_id_mapping = {"B5": 0, "B6": 1, "B7": 2, "B18": 3}
        battery_id_encoded = battery_id_mapping.get(battery_id, 0)
        
        # Prediction button
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_button = st.button("🔮 Predict RUL", type="primary", use_container_width=True)
        
        if predict_button:
            # Charging animation
            with st.spinner("🔋 Analyzing battery parameters..."):
                time.sleep(1)
                
            # Prepare input data (order: battery_id, cycle, chI, chV, chT, disI, disV, disT, BCt, SOH)
            input_data = np.array([[battery_id_encoded, cycle, chI, chV, chT, disI, disV, disT, BCt, SOH]])
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display result with animation
            st.markdown("---")
            st.markdown('<div class="prediction-box">🎯 Predicted RUL: {:.0f} Cycles</div>'.format(prediction), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Additional information with better styling
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("""
                <div class="metric-card">
                    <h3>📊 Current Cycle</h3>
                    <h1>{}</h1>
                </div>
                """.format(cycle), unsafe_allow_html=True)
            with col_b:
                st.markdown("""
                <div class="metric-card">
                    <h3>⏱️ Predicted RUL</h3>
                    <h1>{:.0f}</h1>
                    <p>cycles remaining</p>
                </div>
                """.format(prediction), unsafe_allow_html=True)
            with col_c:
                estimated_total = cycle + prediction
                st.markdown("""
                <div class="metric-card">
                    <h3>🔋 Total Life</h3>
                    <h1>{:.0f}</h1>
                    <p>estimated cycles</p>
                </div>
                """.format(estimated_total), unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Battery health visualization
            health_percentage = (prediction / 250) * 100  # Assuming max RUL is ~250
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = health_percentage,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Battery Health Status", 'font': {'size': 24}},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 20], 'color': '#ff4444'},
                        {'range': [20, 40], 'color': '#ffaa44'},
                        {'range': [40, 60], 'color': '#ffff44'},
                        {'range': [60, 80], 'color': '#aaff44'},
                        {'range': [80, 100], 'color': '#44ff44'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Health status with better styling
            if prediction > 100:
                st.success("✅ **Excellent Condition** - Battery is in good condition with significant remaining life.")
            elif prediction > 50:
                st.warning("⚠️ **Moderate Condition** - Battery has moderate remaining life. Consider monitoring closely.")
            else:
                st.error("🔴 **Critical Condition** - Battery is approaching end of life. Replacement recommended soon.")

    else:
        st.warning("⚠️ Please ensure the model files are available to use this application.")
        st.info("""
        **Required files:**
        - `best_model.pkl` - Trained Random Forest model
        - `scaler.pkl` - Feature scaler
        
        These files should be generated from the Jupyter notebook.
        """)

# TAB 2: DATA VISUALIZATION
with tab2:
    st.markdown("### 📊 Battery Dataset Analysis")
    
    if df is not None:
        # Dataset overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Batteries", df['battery_id'].nunique())
        with col3:
            st.metric("Features", len(df.columns) - 1)
        with col4:
            st.metric("Max Cycles", df['cycle'].max())
        
        st.markdown("---")
        
        # Create sub-tabs for better organization
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "📈 Distribution Analysis", 
            "🔗 Correlation Analysis", 
            "🔋 Battery Degradation",
            "🌡️ Temperature & Voltage Analysis"
        ])
        
        # VIZ TAB 1: Distribution Analysis
        with viz_tab1:
            st.markdown("#### 📊 Feature Distributions")
            
            # RUL Distribution with KDE
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=df, x='RUL', kde=True, bins=50, color='#667eea', ax=ax)
                ax.set_title('RUL Distribution with KDE', fontsize=16, fontweight='bold')
                ax.set_xlabel('Remaining Useful Life (Cycles)', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.violinplot(data=df, x='battery_id', y='RUL', palette='Set2', ax=ax)
                ax.set_title('RUL Distribution by Battery (Violin Plot)', fontsize=16, fontweight='bold')
                ax.set_xlabel('Battery ID', fontsize=12)
                ax.set_ylabel('RUL (Cycles)', fontsize=12)
                st.pyplot(fig)
                plt.close()
            
            # SOH and BCt distributions
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=df, x='SOH', hue='battery_id', kde=True, palette='Set2', ax=ax)
                ax.set_title('State of Health Distribution', fontsize=16, fontweight='bold')
                ax.set_xlabel('SOH (%)', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=df, x='BCt', hue='battery_id', kde=True, palette='Set2', ax=ax)
                ax.set_title('Battery Capacity Distribution', fontsize=16, fontweight='bold')
                ax.set_xlabel('Battery Capacity', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                st.pyplot(fig)
                plt.close()
            
            # Box plots for all numeric features
            st.markdown("#### 📦 Box Plots - Outlier Detection")
            numeric_features = ['chI', 'chV', 'chT', 'disI', 'disV', 'disT', 'BCt', 'SOH']
            
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()
            
            for idx, feature in enumerate(numeric_features):
                sns.boxplot(data=df, y=feature, palette='Set2', ax=axes[idx])
                axes[idx].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
                axes[idx].set_ylabel(feature, fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # VIZ TAB 2: Correlation Analysis
        with viz_tab2:
            st.markdown("#### 🔗 Correlation Analysis")
            
            # Correlation heatmap with seaborn
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title('Feature Correlation Heatmap', fontsize=18, fontweight='bold', pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig)
            plt.close()
            
            # Pairplot for key features
            st.markdown("#### 🎯 Pairwise Relationships")
            key_features = ['RUL', 'SOH', 'BCt', 'cycle']
            
            with st.spinner("Generating pairplot... This may take a moment."):
                fig = sns.pairplot(df[key_features + ['battery_id']], 
                                  hue='battery_id', 
                                  palette='Set2',
                                  diag_kind='kde',
                                  plot_kws={'alpha': 0.6})
                fig.fig.suptitle('Pairwise Feature Relationships', y=1.02, fontsize=16, fontweight='bold')
                st.pyplot(fig)
                plt.close()
            
            # Correlation with RUL
            st.markdown("#### 🎯 Feature Correlation with RUL")
            rul_corr = corr_matrix['RUL'].sort_values(ascending=False)[1:]  # Exclude RUL itself
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#44ff44' if x > 0 else '#ff4444' for x in rul_corr.values]
            rul_corr.plot(kind='barh', color=colors, ax=ax)
            ax.set_title('Feature Correlation with RUL', fontsize=16, fontweight='bold')
            ax.set_xlabel('Correlation Coefficient', fontsize=12)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            st.pyplot(fig)
            plt.close()
        
        # VIZ TAB 3: Battery Degradation
        with viz_tab3:
            st.markdown("#### 🔋 Battery Degradation Analysis")
            
            # RUL vs Cycle with regression
            fig, ax = plt.subplots(figsize=(14, 6))
            sns.scatterplot(data=df, x='cycle', y='RUL', hue='battery_id', 
                          palette='Set2', s=100, alpha=0.6, ax=ax)
            sns.regplot(data=df, x='cycle', y='RUL', scatter=False, 
                       color='red', ax=ax, label='Overall Trend')
            ax.set_title('RUL Degradation Over Cycles', fontsize=18, fontweight='bold')
            ax.set_xlabel('Cycle Number', fontsize=14)
            ax.set_ylabel('Remaining Useful Life', fontsize=14)
            ax.legend(title='Battery ID', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            # SOH vs RUL
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df, x='SOH', y='RUL', hue='battery_id', 
                              palette='Set2', s=100, alpha=0.6, ax=ax)
                sns.regplot(data=df, x='SOH', y='RUL', scatter=False, 
                           color='darkblue', ax=ax)
                ax.set_title('State of Health vs RUL', fontsize=16, fontweight='bold')
                ax.set_xlabel('SOH (%)', fontsize=12)
                ax.set_ylabel('RUL (Cycles)', fontsize=12)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df, x='BCt', y='RUL', hue='battery_id', 
                              palette='Set2', s=100, alpha=0.6, ax=ax)
                sns.regplot(data=df, x='BCt', y='RUL', scatter=False, 
                           color='darkgreen', ax=ax)
                ax.set_title('Battery Capacity vs RUL', fontsize=16, fontweight='bold')
                ax.set_xlabel('Battery Capacity', fontsize=12)
                ax.set_ylabel('RUL (Cycles)', fontsize=12)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            # Line plot showing degradation
            st.markdown("#### 📉 Battery Health Degradation Timeline")
            fig, ax = plt.subplots(figsize=(14, 6))
            for battery in df['battery_id'].unique():
                battery_data = df[df['battery_id'] == battery].sort_values('cycle')
                ax.plot(battery_data['cycle'], battery_data['SOH'], 
                       marker='o', label=battery, linewidth=2, markersize=4)
            ax.set_title('SOH Degradation Over Cycles', fontsize=18, fontweight='bold')
            ax.set_xlabel('Cycle Number', fontsize=14)
            ax.set_ylabel('State of Health (%)', fontsize=14)
            ax.legend(title='Battery ID')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # VIZ TAB 4: Temperature & Voltage Analysis
        with viz_tab4:
            st.markdown("#### 🌡️ Temperature and Voltage Analysis")
            
            # Temperature analysis
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df, x='chT', y='RUL', hue='battery_id', 
                              palette='Set2', s=100, alpha=0.6, ax=ax)
                ax.set_title('Charging Temperature vs RUL', fontsize=16, fontweight='bold')
                ax.set_xlabel('Charging Temperature (°C)', fontsize=12)
                ax.set_ylabel('RUL (Cycles)', fontsize=12)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df, x='disT', y='RUL', hue='battery_id', 
                              palette='Set2', s=100, alpha=0.6, ax=ax)
                ax.set_title('Discharging Temperature vs RUL', fontsize=16, fontweight='bold')
                ax.set_xlabel('Discharging Temperature (°C)', fontsize=12)
                ax.set_ylabel('RUL (Cycles)', fontsize=12)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            # Voltage analysis
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df, x='chV', y='RUL', hue='battery_id', 
                              palette='Set2', s=100, alpha=0.6, ax=ax)
                ax.set_title('Charging Voltage vs RUL', fontsize=16, fontweight='bold')
                ax.set_xlabel('Charging Voltage (V)', fontsize=12)
                ax.set_ylabel('RUL (Cycles)', fontsize=12)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df, x='disV', y='RUL', hue='battery_id', 
                              palette='Set2', s=100, alpha=0.6, ax=ax)
                ax.set_title('Discharging Voltage vs RUL', fontsize=16, fontweight='bold')
                ax.set_xlabel('Discharging Voltage (V)', fontsize=12)
                ax.set_ylabel('RUL (Cycles)', fontsize=12)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            # Current analysis
            st.markdown("#### ⚡ Current Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.kdeplot(data=df, x='chI', hue='battery_id', 
                          palette='Set2', fill=True, alpha=0.5, ax=ax)
                ax.set_title('Charging Current Distribution', fontsize=16, fontweight='bold')
                ax.set_xlabel('Charging Current (A)', fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.kdeplot(data=df, x='disI', hue='battery_id', 
                          palette='Set2', fill=True, alpha=0.5, ax=ax)
                ax.set_title('Discharging Current Distribution', fontsize=16, fontweight='bold')
                ax.set_xlabel('Discharging Current (A)', fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                st.pyplot(fig)
                plt.close()
            
            # Joint plot for temperature relationship
            st.markdown("#### 🔥 Temperature Relationship Analysis")
            fig = sns.jointplot(data=df, x='chT', y='disT', hue='battery_id', 
                              palette='Set2', kind='scatter', alpha=0.6, height=8)
            fig.fig.suptitle('Charging vs Discharging Temperature', 
                           y=1.02, fontsize=16, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
    else:
        st.error("Dataset not found. Please ensure 'Battery_dataset.csv' is in the directory.")

# TAB 3: MODEL PERFORMANCE
with tab3:
    st.markdown("### 📈 Model Performance Metrics")
    
    if model is not None and df is not None:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Prepare data
        data = df.copy()
        le = LabelEncoder()
        data['battery_id'] = le.fit_transform(data['battery_id'])
        
        X = data.drop(columns=['RUL'])
        y = data['RUL']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📉 MAE</h3>
                <h1>{:.2f}</h1>
                <p>Mean Absolute Error</p>
            </div>
            """.format(mae), unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 RMSE</h3>
                <h1>{:.2f}</h1>
                <p>Root Mean Squared Error</p>
            </div>
            """.format(rmse), unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 R² Score</h3>
                <h1>{:.4f}</h1>
                <p>Coefficient of Determination</p>
            </div>
            """.format(r2), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Prediction vs Actual
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(x=y_test, y=y_pred, 
                           labels={'x': 'Actual RUL', 'y': 'Predicted RUL'},
                           title='Predicted vs Actual RUL',
                           color_discrete_sequence=['#667eea'])
            fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(),
                         x1=y_test.max(), y1=y_test.max(),
                         line=dict(color='red', dash='dash'))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            residuals = y_test - y_pred
            fig = px.histogram(residuals, nbins=50,
                             title='Residuals Distribution',
                             labels={'value': 'Residuals'},
                             color_discrete_sequence=['#764ba2'])
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            st.markdown("### 🎯 Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_importance, x='Importance', y='Feature',
                        orientation='h',
                        title='Feature Importance in Random Forest Model',
                        color='Importance',
                        color_continuous_scale='Viridis')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("Model or dataset not available for performance analysis.")

# TAB 4: ABOUT
with tab4:
    st.markdown("### ℹ️ About This Application")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        #### 🔋 Battery RUL Prediction System
        
        This application uses **Machine Learning** to predict the Remaining Useful Life (RUL) 
        of batteries based on their operational parameters and historical data.
        
        #### 🎯 Key Features:
        - **Real-time Prediction**: Get instant RUL predictions for your battery
        - **Interactive Visualizations**: Explore battery data through interactive charts
        - **Model Performance**: View detailed metrics about prediction accuracy
        - **User-Friendly Interface**: Easy-to-use interface with helpful tooltips
        
        #### 🤖 Model Information:
        - **Algorithm**: Random Forest Regressor with GridSearchCV
        - **Features**: 10 input parameters
        - **Target**: Remaining Useful Life (RUL) in cycles
        - **Optimization**: Hyperparameter tuning for best performance
        
        #### 📊 Input Features:
        1. **Battery ID**: Identifier for different batteries (B5, B6, B7, B18)
        2. **Cycle**: Current cycle number
        3. **Charging Parameters**: Current (chI), Voltage (chV), Temperature (chT)
        4. **Discharging Parameters**: Current (disI), Voltage (disV), Temperature (disT)
        5. **Battery State**: Capacity (BCt), State of Health (SOH)
        
        #### 🎓 Use Cases:
        - Battery health monitoring in electric vehicles
        - Predictive maintenance for battery systems
        - Research and development in battery technology
        - Quality control in battery manufacturing
        """)
    
    with col2:
        st.markdown('<div class="battery-animation">🔋</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.info("""
        **💡 Tips for Best Results:**
        
        - Ensure input values are within realistic ranges
        - Use actual measured values when available
        - Monitor battery health regularly
        - Consider environmental factors
        """)
        
        st.success("""
        **✅ Model Accuracy:**
        
        The Random Forest model has been trained on real battery data and optimized 
        for accurate predictions.
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit and Scikit-learn</p>
        <p>© 2025 Battery RUL Prediction System</p>
    </div>
    """, unsafe_allow_html=True)
