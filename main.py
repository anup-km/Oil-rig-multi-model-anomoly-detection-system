import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from sentence_transformers import SentenceTransformer
import warnings

import EDA_for_generated_data as EDA

warnings.filterwarnings('ignore')

class OilRigDataGenerator:
    """Generate synthetic oil rig sensor data and operator logs"""
    
    def __init__(self, start_date='2024-01-01', months=6):
        self.start_date = pd.to_datetime(start_date)
        self.months = months
        self.equipment_types = [
            'Drilling_Pump', 'Rotary_Table', 'Mud_Circulator', 
            'BOB_Preventer', 'Generator_Engine'
        ]
        
    def generate_sensor_data(self):
        """Generate time series sensor data with normal behavior and anomalies"""
        np.random.seed(42)
        
        # Generate timestamps (hourly data)
        end_date = self.start_date + timedelta(days=30 * self.months)
        timestamps = pd.date_range(self.start_date, end_date, freq='H')
        
        data = []
        anomaly_info = []
        
        for equipment in self.equipment_types:
            equipment_data = self._generate_equipment_data(equipment, timestamps, anomaly_info)
            data.extend(equipment_data)
            
        df = pd.DataFrame(data)
        anomaly_df = pd.DataFrame(anomaly_info)
        
        return df, anomaly_df
    
    def _generate_equipment_data(self, equipment, timestamps, anomaly_info):
        """Generate data for specific equipment with realistic patterns"""
        data = []
        
        # Equipment-specific normal ranges and patterns
        params = self._get_equipment_params(equipment)
        
        for i, ts in enumerate(timestamps):
            # Base values with daily/weekly patterns
            daily_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 24)  # Daily cycle
            weekly_factor = 1 + 0.05 * np.sin(2 * np.pi * i / (24 * 7))  # Weekly cycle
            
            # Generate normal readings
            pressure = params['pressure_base'] * daily_factor * weekly_factor + np.random.normal(0, params['pressure_noise'])
            temperature = params['temp_base'] * daily_factor + np.random.normal(0, params['temp_noise'])
            vibration = params['vib_base'] + np.random.normal(0, params['vib_noise'])
            flow_rate = params['flow_base'] * daily_factor + np.random.normal(0, params['flow_noise'])
            
            # Inject anomalies
            anomaly_type = None
            if random.random() < 0.003:  # 0.3% chance of anomaly
                anomaly_type = self._inject_anomaly(equipment, ts, pressure, temperature, 
                                                  vibration, flow_rate, params, anomaly_info)
            
            # Add missing values occasionally
            if random.random() < 0.001:  # 0.1% missing data
                pressure = np.nan
                
            data.append({
                'timestamp': ts,
                'equipment_id': equipment,
                'pressure_psi': max(0, pressure),
                'temperature_f': temperature,
                'vibration_mm_s': max(0, vibration),
                'flow_rate_bpm': max(0, flow_rate),
                'anomaly_type': anomaly_type
            })
            
        return data
    
    def _get_equipment_params(self, equipment):
        """Equipment-specific parameter ranges"""
        params_dict = {
            'Drilling_Pump': {
                'pressure_base': 3000, 'pressure_noise': 50,
                'temp_base': 180, 'temp_noise': 5,
                'vib_base': 2.5, 'vib_noise': 0.3,
                'flow_base': 500, 'flow_noise': 20
            },
            'Rotary_Table': {
                'pressure_base': 1500, 'pressure_noise': 30,
                'temp_base': 160, 'temp_noise': 8,
                'vib_base': 4.0, 'vib_noise': 0.5,
                'flow_base': 0, 'flow_noise': 0
            },
            'Mud_Circulator': {
                'pressure_base': 800, 'pressure_noise': 25,
                'temp_base': 120, 'temp_noise': 6,
                'vib_base': 1.8, 'vib_noise': 0.2,
                'flow_base': 800, 'flow_noise': 30
            },
            'BOB_Preventer': {
                'pressure_base': 5000, 'pressure_noise': 100,
                'temp_base': 100, 'temp_noise': 4,
                'vib_base': 0.5, 'vib_noise': 0.1,
                'flow_base': 0, 'flow_noise': 0
            },
            'Generator_Engine': {
                'pressure_base': 45, 'pressure_noise': 3,
                'temp_base': 200, 'temp_noise': 10,
                'vib_base': 3.2, 'vib_noise': 0.4,
                'flow_base': 150, 'flow_noise': 8
            }
        }
        return params_dict.get(equipment, params_dict['Drilling_Pump'])
    
    def _inject_anomaly(self, equipment, timestamp, pressure, temperature, 
                       vibration, flow_rate, params, anomaly_info):
        """Inject different types of anomalies"""
        anomaly_types = ['sudden_spike', 'gradual_drift', 'oscillation', 'dropout']
        anomaly_type = random.choice(anomaly_types)
        
        anomaly_info.append({
            'timestamp': timestamp,
            'equipment_id': equipment,
            'anomaly_type': anomaly_type,
            'severity': random.choice(['low', 'medium', 'high'])
        })
        
        return anomaly_type
    
    def generate_operator_logs(self, sensor_df, anomaly_df):
        """Generate realistic operator logs correlated with sensor data"""
        logs = []
        
        # Template messages for different situations
        templates = {
            'maintenance': [
                "Performed routine maintenance on {equipment}",
                "Scheduled service completed for {equipment} - all systems normal",
                "Replaced filters on {equipment}, pressure readings stable"
            ],
            'observation': [
                "Noticed unusual {metric} readings on {equipment}",
                "Monitoring {equipment} - {metric} showing irregular pattern",
                "{equipment} {metric} levels need attention"
            ],
            'anomaly_response': [
                "Investigating pressure spike on {equipment}",
                "Temperature alarm triggered on {equipment} - checking cooling system",
                "Vibration levels elevated on {equipment} - conducting inspection",
                "Flow rate anomaly detected on {equipment}"
            ],
            'normal_operation': [
                "All systems operating within normal parameters",
                "Shift handover - no issues reported on {equipment}",
                "Regular monitoring check - {equipment} performing well"
            ]
        }
        
        # Generate logs based on anomalies
        for _, anomaly in anomaly_df.iterrows():
            if random.random() < 0.7:  # 70% chance operator logs the anomaly
                template = random.choice(templates['anomaly_response'])
                log_time = anomaly['timestamp'] + timedelta(minutes=random.randint(5, 60))
                
                logs.append({
                    'timestamp': log_time,
                    'operator_id': f"OP_{random.randint(1, 10):02d}",
                    'equipment_id': anomaly['equipment_id'],
                    'log_text': template.format(equipment=anomaly['equipment_id']),
                    'log_type': 'anomaly_response'
                })
        
        # Generate routine logs
        routine_timestamps = pd.date_range(
            sensor_df['timestamp'].min(), 
            sensor_df['timestamp'].max(), 
            freq='4H'
        )
        
        for ts in routine_timestamps:
            if random.random() < 0.3:  # 30% chance of routine log
                equipment = random.choice(self.equipment_types)
                log_type = random.choice(['maintenance', 'observation', 'normal_operation'])
                template = random.choice(templates[log_type])
                
                metric = random.choice(['pressure', 'temperature', 'vibration', 'flow'])
                
                logs.append({
                    'timestamp': ts,
                    'operator_id': f"OP_{random.randint(1, 10):02d}",
                    'equipment_id': equipment,
                    'log_text': template.format(equipment=equipment, metric=metric),
                    'log_type': log_type
                })
        
        return pd.DataFrame(logs).sort_values('timestamp').reset_index(drop=True)

class MultiModalAnomalyDetector:
    """Multi-modal anomaly detection combining time series and text analysis"""
    
    def __init__(self):
        self.isolation_forest = None
        self.lstm_autoencoder = None
        self.scaler = StandardScaler()
        self.sentence_model = None
        self.is_trained = False
        
    def load_sentence_transformer(self):
        """Load sentence transformer model for text embedding"""
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            st.warning("Could not load sentence transformer. Using basic text analysis.")
            self.sentence_model = None
    
    def preprocess_sensor_data(self, df):
        """Preprocess sensor data for anomaly detection"""
        # Handle missing values
        df = df.copy()
        numeric_cols = ['pressure_psi', 'temperature_f', 'vibration_mm_s', 'flow_rate_bpm']
        
        # Forward fill then backward fill
        for col in numeric_cols:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Create features for each equipment
        features = []
        for equipment in df['equipment_id'].unique():
            eq_data = df[df['equipment_id'] == equipment][numeric_cols].values
            if len(eq_data) > 0:
                features.extend(eq_data)
        
        return np.array(features)
    
    def train_isolation_forest(self, features):
        """Train Isolation Forest for anomaly detection"""
        features_scaled = self.scaler.fit_transform(features)
        self.isolation_forest = IsolationForest(
            contamination=0.1, 
            random_state=42,
            n_estimators=100
        )
        self.isolation_forest.fit(features_scaled)
        
    def build_lstm_autoencoder(self, timesteps, n_features):
        """Build LSTM autoencoder for time series anomaly detection"""
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(timesteps, n_features), return_sequences=True),
            LSTM(32, activation='relu', return_sequences=False),
            RepeatVector(timesteps),
            LSTM(32, activation='relu', return_sequences=True),
            LSTM(64, activation='relu', return_sequences=True),
            TimeDistributed(Dense(n_features))
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def detect_sensor_anomalies(self, df):
        """Detect anomalies in sensor data using multiple methods"""
        features = self.preprocess_sensor_data(df)
        
        if not self.is_trained:
            self.train_isolation_forest(features)
            self.is_trained = True
        
        # Isolation Forest detection
        features_scaled = self.scaler.transform(features)
        anomaly_scores = self.isolation_forest.decision_function(features_scaled)
        anomalies = self.isolation_forest.predict(features_scaled)
        
        # Statistical anomaly detection (Z-score)
        z_scores = np.abs((features - np.mean(features, axis=0)) / np.std(features, axis=0))
        z_anomalies = np.any(z_scores > 3, axis=1)
        
        # Combine results
        combined_anomalies = (anomalies == -1) | z_anomalies
        
        return {
            'anomaly_flags': combined_anomalies,
            'anomaly_scores': anomaly_scores,
            'z_scores': z_scores
        }
    
    def embed_text_logs(self, logs):
        """Create embeddings for text logs"""
        if self.sentence_model is None:
            # Simple keyword-based approach if transformer not available
            return self._simple_text_features(logs)
        
        embeddings = self.sentence_model.encode(logs['log_text'].tolist())
        return embeddings
    
    def _simple_text_features(self, logs):
        """Simple text feature extraction using keywords"""
        keywords = [
            'pressure', 'temperature', 'vibration', 'flow', 'spike', 'high', 'low',
            'abnormal', 'unusual', 'alarm', 'fault', 'maintenance', 'repair'
        ]
        
        features = []
        for text in logs['log_text']:
            text_lower = text.lower()
            feature_vec = [1 if keyword in text_lower else 0 for keyword in keywords]
            features.append(feature_vec)
        
        return np.array(features)
    
    def correlate_anomalies_with_logs(self, sensor_anomalies, logs, sensor_df):
        """Correlate sensor anomalies with operator logs"""
        correlations = []
        
        # Get anomalous timestamps
        anomaly_indices = np.where(sensor_anomalies['anomaly_flags'])[0]
        
        if len(anomaly_indices) == 0:
            return pd.DataFrame(correlations)
        
        # Create embeddings for logs
        log_embeddings = self.embed_text_logs(logs)
        
        # for idx in anomaly_indices[:10]:  # Limit to first 10 anomalies for demo
        for idx in anomaly_indices:
            if idx < len(sensor_df):
                anomaly_time = sensor_df.iloc[idx]['timestamp']
                equipment = sensor_df.iloc[idx]['equipment_id']
                
                # Find logs within time window (±4 hours) and same equipment
                time_window = timedelta(hours=4)
                relevant_logs = logs[
                    (logs['timestamp'] >= anomaly_time - time_window) &
                    (logs['timestamp'] <= anomaly_time + time_window) &
                    (logs['equipment_id'] == equipment)
                ]
                
                if len(relevant_logs) > 0:
                    # Calculate similarity scores
                    for _, log_row in relevant_logs.iterrows():
                        correlation_score = self._calculate_correlation_score(
                            anomaly_time, log_row, sensor_anomalies['anomaly_scores'][idx]
                        )
                        
                        correlations.append({
                            'anomaly_timestamp': anomaly_time,
                            'equipment_id': equipment,
                            'log_timestamp': log_row['timestamp'],
                            'log_text': log_row['log_text'],
                            'correlation_score': correlation_score,
                            'anomaly_score': sensor_anomalies['anomaly_scores'][idx]
                        })
        
        return pd.DataFrame(correlations).sort_values('correlation_score', ascending=False)
    
    def _calculate_correlation_score(self, anomaly_time, log_row, anomaly_score):
        """Calculate correlation score between anomaly and log"""
        # Time proximity score (closer = higher score)
        time_diff = abs((log_row['timestamp'] - anomaly_time).total_seconds() / 3600)  # hours
        time_score = max(0, 1 - time_diff / 4)  # 4-hour window
        
        # Text relevance score (simple keyword matching)
        anomaly_keywords = ['spike', 'high', 'abnormal', 'unusual', 'alarm', 'fault']
        text_lower = log_row['log_text'].lower()
        text_score = sum(1 for keyword in anomaly_keywords if keyword in text_lower) / len(anomaly_keywords)
        
        # Combined score
        return 0.6 * time_score + 0.4 * text_score

class InsightGenerator:
    """Generate insights using rule-based approach (simulating GenAI)"""
    
    def __init__(self):
        self.insight_templates = {
            'pressure_spike': [
                "Sudden pressure increase detected. Possible causes: blockage in system, pump malfunction, or valve closure.",
                "Pressure spike observed. Recommend immediate inspection of downstream components and safety valves."
            ],
            'temperature_high': [
                "Elevated temperature readings suggest cooling system issues or increased load conditions.",
                "High temperature detected. Check cooling fluid levels and heat exchanger performance."
            ],
            'vibration_anomaly': [
                "Unusual vibration patterns may indicate mechanical wear, misalignment, or bearing issues.",
                "Vibration anomaly detected. Schedule maintenance inspection for rotating components."
            ],
            'flow_anomaly': [
                "Flow rate deviation from normal. Check for partial blockages or pump efficiency issues.",
                "Abnormal flow patterns detected. Inspect filters, pipes, and pump components."
            ]
        }
    
    def generate_anomaly_insights(self, correlations_df, equipment_id, anomaly_type=None):
        """Generate insights for detected anomalies"""
        insights = []
        
        if len(correlations_df) == 0:
            return ["No correlating operator logs found for this anomaly."]
        
        # Analyze correlation patterns
        top_correlations = correlations_df.head(3)
        
        for _, corr in top_correlations.iterrows():
            # Determine likely anomaly type from text
            text = corr['log_text'].lower()
            detected_type = self._classify_anomaly_type(text)
            
            # Generate insight
            if detected_type in self.insight_templates:
                insight = random.choice(self.insight_templates[detected_type])
            else:
                insight = f"Anomaly detected in {equipment_id}. Operator noted: '{corr['log_text']}'"
            
            insights.append({
                'insight': insight,
                'confidence': corr['correlation_score'],
                'supporting_log': corr['log_text'],
                'timestamp': corr['anomaly_timestamp']
            })
        
        return insights
    
    def _classify_anomaly_type(self, text):
        """Classify anomaly type from text"""
        if any(word in text for word in ['pressure', 'psi']):
            return 'pressure_spike'
        elif any(word in text for word in ['temperature', 'temp', 'hot', 'heat']):
            return 'temperature_high'
        elif any(word in text for word in ['vibration', 'shake', 'rattle']):
            return 'vibration_anomaly'
        elif any(word in text for word in ['flow', 'rate', 'pump']):
            return 'flow_anomaly'
        else:
            return 'general_anomaly'
    
    def generate_summary_report(self, anomalies_count, correlations_df, equipment_summary):
        """Generate overall summary report"""
        report = f"""
        ## Oil Rig Anomaly Detection Summary
        
        **Total Anomalies Detected:** {anomalies_count}
        **Time Period:** Last analyzed dataset
        
        ### Equipment Status Overview:
        """
        
        for equipment, count in equipment_summary.items():
            status = "⚠️ ATTENTION REQUIRED" if count > 0 else "✅ NORMAL"
            report += f"- **{equipment}:** {count} anomalies detected - {status}\n"
        
        if len(correlations_df) > 0:
            report += f"""
            ### Key Findings:
            - {len(correlations_df)} correlations found between sensor anomalies and operator logs
            - Highest correlation score: {correlations_df['correlation_score'].max():.2f}
            - Most frequent equipment in logs: {correlations_df['equipment_id'].mode().iloc[0] if not correlations_df.empty else 'N/A'}
            
            ### Recommendations:
            1. Schedule immediate inspection for equipment with high anomaly counts
            2. Review maintenance schedules for frequently flagged equipment
            3. Enhance monitoring for equipment showing gradual degradation patterns
            """
        
        return report

# Streamlit Application
def main():
    st.set_page_config(
        page_title="Oil Rig Anomaly Detection System",
        page_icon="🛢️",
        layout="wide"
    )
    
    st.title("🛢️ Oil Rig Multi-Modal Anomaly Detection System")
    st.markdown("---")
    
    # Initialize components
    if 'data_generator' not in st.session_state:
        st.session_state.data_generator = OilRigDataGenerator()
        st.session_state.detector = MultiModalAnomalyDetector()
        st.session_state.insight_generator = InsightGenerator()
    
    # Sidebar controls
    st.sidebar.header("System Controls")
    
    if st.sidebar.button("🔄 Generate New Data", type="primary"):
        with st.spinner("Generating synthetic oil rig data..."):
            sensor_df, anomaly_df = st.session_state.data_generator.generate_sensor_data()
            logs_df = st.session_state.data_generator.generate_operator_logs(sensor_df, anomaly_df)
            
            sensor_df.to_csv(r'G:\anup_prep\additonal_test\oil-rig-anomaly-detection\data\sensor_data.csv')
            anomaly_df.to_csv(r'G:\anup_prep\additonal_test\oil-rig-anomaly-detection\data\anamoly.csv')
            logs_df.to_csv(r'G:\anup_prep\additonal_test\oil-rig-anomaly-detection\data\logs_df.csv')


            st.session_state.sensor_df = sensor_df
            st.session_state.anomaly_df = anomaly_df
            st.session_state.logs_df = logs_df
        st.success("✅ New data generated successfully!")

    if st.sidebar.button("📊 Run Exploratory Data Analysis"):
        if 'sensor_df' in st.session_state and 'logs_df' in st.session_state:
            with st.spinner("Performing exploratory data analysis..."):
                # Create EDA instance
                eda = EDA.ExploratoryDataAnalysis(
                    st.session_state.sensor_df,
                    st.session_state.logs_df,
                    st.session_state.get('anomaly_df', pd.DataFrame())
                )
                
                # Display EDA dashboard
                eda.display_eda_dashboard()
                
                # Store EDA results in session state
                st.session_state.eda_complete = True
        else:
            st.error("Please generate data first before running EDA!")


    
    # Check if data exists
    if 'sensor_df' not in st.session_state:
        st.info("👆 Click 'Generate New Data' to start the analysis")
        return
    
    # Load sentence transformer
    if not hasattr(st.session_state.detector, 'sentence_model') or st.session_state.detector.sentence_model is None:
        with st.spinner("Loading NLP model..."):
            st.session_state.detector.load_sentence_transformer()
    
    # Main analysis
    if st.sidebar.button("🔍 Run Anomaly Detection"):
        run_anomaly_analysis()
    
    # Display current data overview
    display_data_overview()
    
    # Display results if available
    if 'anomaly_results' in st.session_state:
        display_anomaly_results()

def custom_main():
    
    data_generator = OilRigDataGenerator()
    detector = MultiModalAnomalyDetector()
    insight_generator = InsightGenerator()


    sensor_df, anomaly_df = data_generator.generate_sensor_data()
    logs_df = data_generator.generate_operator_logs(sensor_df, anomaly_df)
    
    # sensor_df.to_csv(r'G:\anup_prep\additonal_test\oil-rig-anomaly-detection\data\sensor_data.csv')
    # anomaly_df.to_csv(r'G:\anup_prep\additonal_test\oil-rig-anomaly-detection\data\anamoly.csv')
    # logs_df.to_csv(r'G:\anup_prep\additonal_test\oil-rig-anomaly-detection\data\logs_df.csv')
    
    run_anomaly_analysis_custom(detector,sensor_df,anomaly_df,logs_df,insight_generator)


def run_anomaly_analysis_custom(detector,sensor_df,anomaly_df,logs_df,insight_generator):


    anomaly_results = detector.detect_sensor_anomalies(sensor_df)
    
    correlations = detector.correlate_anomalies_with_logs(
        anomaly_results, 
        logs_df, 
  sensor_df
    )
    insights = []
    equipment_summary = {}
    
    for equipment in sensor_df['equipment_id'].unique():
        eq_correlations = correlations[correlations['equipment_id'] == equipment]
        eq_insights = insight_generator.generate_anomaly_insights(
            eq_correlations, equipment
        )
        insights.extend(eq_insights)
        equipment_summary[equipment] = len(eq_correlations)
    
    # Store results
    anomaly_results = anomaly_results
    correlations = correlations
    insights = insights
    equipment_summary = equipment_summary

    print("its done")
    



def run_anomaly_analysis():
    """Run the complete anomaly detection pipeline"""
    with st.spinner("Running anomaly detection pipeline..."):
        detector = st.session_state.detector
        
        # Step 1: Detect sensor anomalies
        st.write("🔍 **Step 1:** Detecting sensor anomalies...")
        anomaly_results = detector.detect_sensor_anomalies(st.session_state.sensor_df)
        
        # Step 2: Correlate with logs
        st.write("🔗 **Step 2:** Correlating anomalies with operator logs...")
        correlations = detector.correlate_anomalies_with_logs(
            anomaly_results, 
            st.session_state.logs_df, 
            st.session_state.sensor_df
        )
        
        # Step 3: Generate insights
        st.write("💡 **Step 3:** Generating insights...")
        insights = []
        equipment_summary = {}
        
        for equipment in st.session_state.sensor_df['equipment_id'].unique():
            eq_correlations = correlations[correlations['equipment_id'] == equipment]
            eq_insights = st.session_state.insight_generator.generate_anomaly_insights(
                eq_correlations, equipment
            )
            insights.extend(eq_insights)
            equipment_summary[equipment] = len(eq_correlations)
        
        # Store results
        st.session_state.anomaly_results = anomaly_results
        st.session_state.correlations = correlations
        st.session_state.insights = insights
        st.session_state.equipment_summary = equipment_summary
        
    st.success("✅ Anomaly detection complete!")

def display_data_overview():
    """Display overview of generated data"""
    st.header("📊 Data Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sensor Records", len(st.session_state.sensor_df))
    
    with col2:
        st.metric("Operator Logs", len(st.session_state.logs_df))
    
    with col3:
        st.metric("Equipment Types", len(st.session_state.sensor_df['equipment_id'].unique()))
    
    # Display sample data
    st.subheader("Sample Sensor Data")
    st.dataframe(st.session_state.sensor_df.head(10), use_container_width=True)
    
    st.subheader("Sample Operator Logs")
    st.dataframe(st.session_state.logs_df.head(5), use_container_width=True)

def display_anomaly_results():
    """Display anomaly detection results"""
    st.header("🚨 Anomaly Detection Results")
    
    anomaly_results = st.session_state.anomaly_results
    correlations = st.session_state.correlations
    
    # Summary metrics
    total_anomalies = np.sum(anomaly_results['anomaly_flags'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Anomalies", total_anomalies)
    with col2:
        st.metric("Correlations Found", len(correlations))
    with col3:
        avg_score = correlations['correlation_score'].mean() if len(correlations) > 0 else 0
        st.metric("Avg Correlation Score", f"{avg_score:.2f}")
    
    # Visualization
    st.subheader("📈 Sensor Data with Anomalies")
    
    # Select equipment for visualization
    equipment_options = st.session_state.sensor_df['equipment_id'].unique()
    selected_equipment = st.selectbox("Select Equipment for Visualization", equipment_options)
    
    # Plot sensor data with anomalies
    eq_data = st.session_state.sensor_df[st.session_state.sensor_df['equipment_id'] == selected_equipment]
    
    fig = go.Figure()
    
    # Add normal data
    fig.add_trace(go.Scatter(
        x=eq_data['timestamp'],
        y=eq_data['pressure_psi'],
        mode='lines',
        name='Pressure (PSI)',
        line=dict(color='blue')
    ))
    
    # Add anomaly points
    anomaly_indices = np.where(anomaly_results['anomaly_flags'])[0]
    if len(anomaly_indices) > 0:
        anomaly_data = st.session_state.sensor_df.iloc[anomaly_indices]
        anomaly_eq_data = anomaly_data[anomaly_data['equipment_id'] == selected_equipment]
        
        if len(anomaly_eq_data) > 0:
            fig.add_trace(go.Scatter(
                x=anomaly_eq_data['timestamp'],
                y=anomaly_eq_data['pressure_psi'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=8, symbol='x')
            ))
    
    fig.update_layout(
        title=f"Pressure Readings - {selected_equipment}",
        xaxis_title="Timestamp",
        yaxis_title="Pressure (PSI)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation results
    if len(correlations) > 0:
        st.subheader("🔗 Top Anomaly-Log Correlations")
        display_correlations = correlations[['anomaly_timestamp', 'equipment_id', 'log_text', 'correlation_score']].head(10)
        st.dataframe(display_correlations, use_container_width=True)
    
    # Insights
    if 'insights' in st.session_state and st.session_state.insights:
        st.subheader("💡 Generated Insights")
        for i, insight in enumerate(st.session_state.insights[:5]):  # Show top 5
            if isinstance(insight, dict):
                st.info(f"**Insight {i+1}:** {insight['insight']}\n\n*Supporting Log:* {insight['supporting_log']}")
    
    # Summary report
    if 'equipment_summary' in st.session_state:
        st.subheader("📋 Summary Report")
        summary_report = st.session_state.insight_generator.generate_summary_report(
            total_anomalies, correlations, st.session_state.equipment_summary
        )
        st.markdown(summary_report)

if __name__ == "__main__":
    # main()
    custom_main()