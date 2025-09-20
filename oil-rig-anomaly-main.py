import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sentence_transformers import SentenceTransformer
import warnings
import traceback
import EDA_for_generated_data as EDA


warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class OilRigDataGenerator:
    """Generate synthetic oil rig sensor data and operator logs"""
    
    def __init__(self, start_date='2024-01-01', months=6):
        self.start_date = pd.to_datetime(start_date)
        self.months = months
        self.equipment_types = [
            'Drilling_Pump', 'Rotary_Table', 'Mud_Circulator', 
            'BOP_Preventer', 'Generator_Engine'
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
        
        # Track ongoing anomalies for gradual drift
        drift_active = False
        drift_factor = 1.0
        drift_duration = 0
        
        for i, ts in enumerate(timestamps):
            # Base values with daily/weekly patterns
            daily_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 24)  # Daily cycle
            weekly_factor = 1 + 0.05 * np.sin(2 * np.pi * i / (24 * 7))  # Weekly cycle
            
            # Generate normal readings
            pressure = params['pressure_base'] * daily_factor * weekly_factor + np.random.normal(0, params['pressure_noise'])
            temperature = params['temp_base'] * daily_factor + np.random.normal(0, params['temp_noise'])
            vibration = params['vib_base'] + np.random.normal(0, params['vib_noise'])
            flow_rate = params['flow_base'] * daily_factor + np.random.normal(0, params['flow_noise'])
            
            # Handle gradual drift anomalies
            if drift_active:
                pressure *= drift_factor
                temperature *= (drift_factor * 0.8 + 0.2)  # Temperature changes less dramatically
                vibration *= drift_factor
                flow_rate *= drift_factor
                
                drift_duration -= 1
                if drift_duration <= 0:
                    drift_active = False
                    drift_factor = 1.0
            
            # Inject anomalies
            anomaly_type = None
            if random.random() < 0.003:  # 0.3% chance of anomaly
                anomaly_type, new_pressure, new_temp, new_vib, new_flow = self._inject_anomaly(
                    equipment, ts, pressure, temperature, vibration, flow_rate, params, anomaly_info
                )
                
                if anomaly_type == 'gradual_drift' and not drift_active:
                    drift_active = True
                    drift_factor = random.uniform(1.15, 1.4)  # 15-40% increase
                    drift_duration = random.randint(24, 72)  # 1-3 days
                elif anomaly_type in ['sudden_spike', 'oscillation', 'dropout']:
                    pressure, temperature, vibration, flow_rate = new_pressure, new_temp, new_vib, new_flow
            
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
            'BOP_Preventer': {
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
        
        severity = random.choice(['low', 'medium', 'high'])
        
        anomaly_info.append({
            'timestamp': timestamp,
            'equipment_id': equipment,
            'anomaly_type': anomaly_type,
            'severity': severity
        })
        
        # Modify values based on anomaly type
        if anomaly_type == 'sudden_spike':
            multiplier = {'low': 1.2, 'medium': 1.5, 'high': 2.0}[severity]
            pressure *= multiplier
            temperature *= (multiplier * 0.8 + 0.2)
            vibration *= multiplier
            
        elif anomaly_type == 'dropout':
            multiplier = {'low': 0.8, 'medium': 0.5, 'high': 0.1}[severity]
            pressure *= multiplier
            flow_rate *= multiplier
            
        elif anomaly_type == 'oscillation':
            osc_factor = {'low': 0.1, 'medium': 0.2, 'high': 0.4}[severity]
            pressure *= (1 + osc_factor * np.sin(np.random.random() * 10))
            vibration *= (1 + osc_factor * np.sin(np.random.random() * 15))
        
        return anomaly_type, pressure, temperature, vibration, flow_rate
    
    def generate_operator_logs(self, sensor_df, anomaly_df):
        """Generate realistic operator logs correlated with sensor data"""
        logs = []
        
        # Template messages for different situations
        templates = {
            'maintenance': [
                "Performed routine maintenance on {equipment}",
                "Scheduled service completed for {equipment} - all systems normal",
                "Replaced filters on {equipment}, pressure readings stable",
                "Lubrication service completed on {equipment}",
                "Calibrated sensors on {equipment}"
            ],
            'observation': [
                "Noticed unusual {metric} readings on {equipment}",
                "Monitoring {equipment} - {metric} showing irregular pattern",
                "{equipment} {metric} levels need attention",
                "Observed fluctuations in {equipment} {metric}",
                "Slight increase in {equipment} {metric} detected"
            ],
            'anomaly_response': [
                "Investigating pressure spike on {equipment}",
                "Temperature alarm triggered on {equipment} - checking cooling system",
                "Vibration levels elevated on {equipment} - conducting inspection",
                "Flow rate anomaly detected on {equipment}",
                "Emergency shutdown initiated on {equipment}",
                "Abnormal readings on {equipment} - maintenance team notified",
                "Critical alarm on {equipment} - immediate action required"
            ],
            'normal_operation': [
                "All systems operating within normal parameters",
                "Shift handover - no issues reported on {equipment}",
                "Regular monitoring check - {equipment} performing well",
                "{equipment} operating smoothly",
                "No anomalies detected on {equipment} during shift"
            ]
        }
        
        # Generate logs based on anomalies (higher correlation)
        for _, anomaly in anomaly_df.iterrows():
            if random.random() < 0.8:  # 80% chance operator logs the anomaly
                template = random.choice(templates['anomaly_response'])
                log_time = anomaly['timestamp'] + timedelta(minutes=random.randint(5, 60))
                
                logs.append({
                    'timestamp': log_time,
                    'operator_id': f"OP_{random.randint(1, 10):02d}",
                    'equipment_id': anomaly['equipment_id'],
                    'log_text': template.format(equipment=anomaly['equipment_id']),
                    'log_type': 'anomaly_response',
                    'severity': anomaly['severity']
                })
        
        # Generate routine logs
        routine_timestamps = pd.date_range(
            sensor_df['timestamp'].min(), 
            sensor_df['timestamp'].max(), 
            freq='6H'  # Every 6 hours
        )
        
        for ts in routine_timestamps:
            if random.random() < 0.4:  # 40% chance of routine log
                equipment = random.choice(self.equipment_types)
                log_type = random.choice(['maintenance', 'observation', 'normal_operation'])
                template = random.choice(templates[log_type])
                
                metric = random.choice(['pressure', 'temperature', 'vibration', 'flow'])
                
                logs.append({
                    'timestamp': ts,
                    'operator_id': f"OP_{random.randint(1, 10):02d}",
                    'equipment_id': equipment,
                    'log_text': template.format(equipment=equipment, metric=metric),
                    'log_type': log_type,
                    'severity': 'normal'
                })
        
        return pd.DataFrame(logs).sort_values('timestamp').reset_index(drop=True)


class MultiModalAnomalyDetector:
    """Multi-modal anomaly detection combining time series and text analysis"""
    
    def __init__(self):
        self.isolation_forest = None
        self.lstm_autoencoder = None
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.sentence_model = None
        self.is_trained = False
        self.sequence_length = 24  # 24 hours of data for LSTM
        
    def load_sentence_transformer(self):
        """Load sentence transformer model for text embedding"""
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            return True
        except Exception as e:
            st.warning(f"Could not load sentence transformer: {str(e)}. Using basic text analysis.")
            self.sentence_model = None
            return False
    
    def preprocess_sensor_data(self, df):
        """Preprocess sensor data for anomaly detection"""
        # Handle missing values
        df = df.copy()
        numeric_cols = ['pressure_psi', 'temperature_f', 'vibration_mm_s', 'flow_rate_bpm']
        
        # Forward fill then backward fill
        for col in numeric_cols:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Create features for each equipment
        features_dict = {}
        
        for equipment in df['equipment_id'].unique():
            eq_data = df[df['equipment_id'] == equipment][numeric_cols].values
            if len(eq_data) > 0:
                features_dict[equipment] = eq_data
        
        return features_dict
    
    def create_sequences(self, data, sequence_length):
        """Create sequences for LSTM autoencoder"""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:(i + sequence_length)])
        return np.array(sequences)
    
    def train_isolation_forest(self, features):
        """Train Isolation Forest for anomaly detection"""
        # Flatten all features for isolation forest
        all_features = []
        for equipment, eq_features in features.items():
            all_features.extend(eq_features)
        
        all_features = np.array(all_features)
        features_scaled = self.scaler.fit_transform(all_features)
        
        self.isolation_forest = IsolationForest(
            contamination=0.2, 
            random_state=42,
            n_estimators=100
        )
        self.isolation_forest.fit(features_scaled)
        
        return features_scaled
    
    def build_lstm_autoencoder(self, timesteps, n_features):
        """Build LSTM autoencoder for time series anomaly detection"""
        model = Sequential([
            # Encoder
            LSTM(64, activation='relu', input_shape=(timesteps, n_features), 
                 return_sequences=True),
            Dropout(0.1),
            LSTM(32, activation='relu', return_sequences=False),
            Dropout(0.1),
            
            # Repeat vector to match decoder input
            RepeatVector(timesteps),
            
            # Decoder
            LSTM(32, activation='relu', return_sequences=True),
            Dropout(0.1),
            LSTM(64, activation='relu', return_sequences=True),
            TimeDistributed(Dense(n_features))
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train_lstm_autoencoder(self, features_dict):
        """Train LSTM autoencoder for each equipment type"""
        lstm_models = {}
        reconstruction_errors = {}
        
        for equipment, features in features_dict.items():
            if len(features) < self.sequence_length * 2:  # Need enough data
                continue
                
            # Normalize features
            features_normalized = self.min_max_scaler.fit_transform(features)
            
            # Create sequences
            sequences = self.create_sequences(features_normalized, self.sequence_length)
            
            if len(sequences) < 10:  # Need minimum sequences
                continue
            
            # Split train/validation
            split_idx = int(0.8 * len(sequences))
            train_sequences = sequences[:split_idx]
            val_sequences = sequences[split_idx:]
            
            # Build and train model
            n_features = features.shape[1]
            model = self.build_lstm_autoencoder(self.sequence_length, n_features)
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                train_sequences, train_sequences,
                validation_data=(val_sequences, val_sequences),
                epochs=50,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Calculate reconstruction errors for threshold
            train_pred = model.predict(train_sequences, verbose=0)
            train_errors = np.mean(np.square(train_sequences - train_pred), axis=(1, 2))
            threshold = np.percentile(train_errors, 95)  # 95th percentile as threshold
            
            lstm_models[equipment] = model
            reconstruction_errors[equipment] = threshold
        
        return lstm_models, reconstruction_errors
    
    def detect_sensor_anomalies(self, df):
        """Detect anomalies in sensor data using multiple methods"""
        features_dict = self.preprocess_sensor_data(df)
        
        if not self.is_trained:
            # Train Isolation Forest
            features_scaled = self.train_isolation_forest(features_dict)
            
            # Train LSTM Autoencoders
            lstm_models, reconstruction_thresholds = self.train_lstm_autoencoder(features_dict)
            
            self.lstm_models = lstm_models
            self.reconstruction_thresholds = reconstruction_thresholds
            self.is_trained = True
        
        # Initialize results
        anomaly_flags = np.zeros(len(df), dtype=bool)
        anomaly_scores = np.zeros(len(df))
        anomaly_methods = [''] * len(df)
        
        # Process each equipment separately
        for equipment in df['equipment_id'].unique():
            eq_mask = df['equipment_id'] == equipment
            eq_indices = np.where(eq_mask)[0]
            eq_features = features_dict[equipment]
            
            # Isolation Forest detection
            if len(eq_features) > 0:
                eq_features_scaled = self.scaler.transform(eq_features)
                if_scores = self.isolation_forest.decision_function(eq_features_scaled)
                if_anomalies = self.isolation_forest.predict(eq_features_scaled) == -1
                
                # Statistical anomaly detection (Z-score)
                z_scores = np.abs((eq_features - np.mean(eq_features, axis=0)) / 
                                (np.std(eq_features, axis=0) + 1e-8))
                z_anomalies = np.any(z_scores > 3, axis=1)
                
                # LSTM autoencoder detection
                lstm_anomalies = np.zeros(len(eq_features), dtype=bool)
                if equipment in self.lstm_models:
                    model = self.lstm_models[equipment]
                    threshold = self.reconstruction_thresholds[equipment]
                    
                    # Normalize features
                    eq_features_norm = self.min_max_scaler.fit_transform(eq_features)
                    sequences = self.create_sequences(eq_features_norm, self.sequence_length)
                    
                    if len(sequences) > 0:
                        predictions = model.predict(sequences, verbose=0)
                        reconstruction_errors = np.mean(np.square(sequences - predictions), axis=(1, 2))
                        
                        # Map back to original indices
                        for i, error in enumerate(reconstruction_errors):
                            if error > threshold:
                                start_idx = i
                                end_idx = min(i + self.sequence_length, len(lstm_anomalies))
                                lstm_anomalies[start_idx:end_idx] = True
                
                # Combine all methods
                combined_anomalies = if_anomalies | z_anomalies | lstm_anomalies
                
                # Assign results to corresponding indices
                for i, eq_idx in enumerate(eq_indices):
                    if i < len(combined_anomalies):
                        anomaly_flags[eq_idx] = combined_anomalies[i]
                        anomaly_scores[eq_idx] = if_scores[i] if i < len(if_scores) else 0
                        
                        # Determine detection method
                        methods = []
                        if i < len(if_anomalies) and if_anomalies[i]:
                            methods.append('IF')
                        if i < len(z_anomalies) and z_anomalies[i]:
                            methods.append('Z-Score')
                        if i < len(lstm_anomalies) and lstm_anomalies[i]:
                            methods.append('LSTM')
                        
                        anomaly_methods[eq_idx] = ','.join(methods)
        
        return {
            'anomaly_flags': anomaly_flags,
            'anomaly_scores': anomaly_scores,
            'anomaly_methods': anomaly_methods
        }
    
    def embed_text_logs(self, logs):
        """Create embeddings for text logs"""
        if self.sentence_model is None:
            return self._simple_text_features(logs)
        
        embeddings = self.sentence_model.encode(logs['log_text'].tolist())
        return embeddings
    
    def _simple_text_features(self, logs):
        """Simple text feature extraction using keywords"""
        keywords = [
            'pressure', 'temperature', 'vibration', 'flow', 'spike', 'high', 'low',
            'abnormal', 'unusual', 'alarm', 'fault', 'maintenance', 'repair', 'critical',
            'emergency', 'shutdown', 'investigating', 'elevated', 'anomaly'
        ]
        
        features = []
        for text in logs['log_text']:
            text_lower = text.lower()
            feature_vec = [1 if keyword in text_lower else 0 for keyword in keywords]
            features.append(feature_vec)
        
        return np.array(features)
    
    def correlate_anomalies_with_logs(self, sensor_anomalies, logs, sensor_df):
        """Correlate sensor anomalies with operator logs using advanced NLP and semantic similarity"""
        correlations = []
        
        # Get anomalous timestamps
        anomaly_indices = np.where(sensor_anomalies['anomaly_flags'])[0]
        
        if len(anomaly_indices) == 0 or len(logs) == 0:
            return pd.DataFrame(correlations)
        
        # Create embeddings for logs
        log_embeddings = self.embed_text_logs(logs)
        
        # Create anomaly context embeddings (describe the anomaly type and severity)
        anomaly_contexts = self._create_anomaly_contexts(sensor_anomalies, sensor_df, anomaly_indices)
        anomaly_embeddings = self._embed_anomaly_contexts(anomaly_contexts)
        
        for i, idx in enumerate(anomaly_indices):
            if idx < len(sensor_df):
                anomaly_time = sensor_df.iloc[idx]['timestamp']
                equipment = sensor_df.iloc[idx]['equipment_id']
                detection_method = sensor_anomalies['anomaly_methods'][idx]
                
                # Find logs within time window (±6 hours) and same equipment
                time_window = timedelta(hours=6)
                relevant_logs_mask = (
                    (logs['timestamp'] >= anomaly_time - time_window) &
                    (logs['timestamp'] <= anomaly_time + time_window) &
                    (logs['equipment_id'] == equipment)
                )
                relevant_logs = logs[relevant_logs_mask]
                
                if len(relevant_logs) > 0:
                    # Get embeddings for relevant logs
                    relevant_log_indices = relevant_logs.index.tolist()
                    relevant_embeddings = log_embeddings[[logs.index.get_loc(idx) for idx in relevant_log_indices]]
                    
                    # Calculate semantic similarity between anomaly context and logs
                    if anomaly_embeddings is not None and len(anomaly_embeddings) > i:
                        anomaly_embedding = anomaly_embeddings[i].reshape(1, -1)
                        semantic_similarities = cosine_similarity(anomaly_embedding, relevant_embeddings)[0]
                    else:
                        # Fallback to basic similarity if embeddings not available
                        semantic_similarities = np.ones(len(relevant_logs)) * 0.5
                    
                    # Process each relevant log
                    for j, (_, log_row) in enumerate(relevant_logs.iterrows()):
                        # Enhanced correlation score combining multiple factors
                        correlation_score = self._calculate_enhanced_correlation_score(
                            anomaly_time, log_row, sensor_anomalies['anomaly_scores'][idx],
                            detection_method, semantic_similarities[j] if j < len(semantic_similarities) else 0.5
                        )
                        
                        correlations.append({
                            'anomaly_timestamp': anomaly_time,
                            'equipment_id': equipment,
                            'log_timestamp': log_row['timestamp'],
                            'log_text': log_row['log_text'],
                            'log_type': log_row.get('log_type', 'unknown'),
                            'correlation_score': correlation_score,
                            'anomaly_score': sensor_anomalies['anomaly_scores'][idx],
                            'detection_method': detection_method,
                            'operator_id': log_row.get('operator_id', 'unknown'),
                            'semantic_similarity': semantic_similarities[j] if j < len(semantic_similarities) else 0.5,
                            'time_proximity': self._calculate_time_proximity(anomaly_time, log_row['timestamp'])
                        })
        
        return pd.DataFrame(correlations).sort_values('correlation_score', ascending=False)
    
    def _create_anomaly_contexts(self, sensor_anomalies, sensor_df, anomaly_indices):
        """Create textual contexts for anomalies to enable semantic comparison with logs"""
        contexts = []
        
        for idx in anomaly_indices:
            if idx < len(sensor_df):
                row = sensor_df.iloc[idx]
                equipment = row['equipment_id']
                detection_method = sensor_anomalies['anomaly_methods'][idx]
                
                # Determine anomaly characteristics
                pressure = row['pressure_psi']
                temperature = row['temperature_f']
                vibration = row['vibration_mm_s']
                flow_rate = row['flow_rate_bpm']
                
                # Create descriptive context based on sensor values and detection method
                context_parts = []
                
                # Equipment context
                context_parts.append(f"{equipment} equipment")
                
                # Detection method context
                if 'LSTM' in detection_method:
                    context_parts.append("unusual pattern detected")
                if 'IF' in detection_method:
                    context_parts.append("outlier behavior identified")
                if 'Z-Score' in detection_method:
                    context_parts.append("statistical anomaly found")
                
                # Parameter-specific contexts
                if pressure > 0:
                    if pressure > 4000:  # High pressure threshold
                        context_parts.append("high pressure readings")
                    elif pressure < 100:  # Low pressure threshold
                        context_parts.append("low pressure readings")
                    else:
                        context_parts.append("pressure anomaly")
                
                if temperature > 250:  # High temperature threshold
                    context_parts.append("elevated temperature")
                elif temperature < 50:  # Low temperature threshold
                    context_parts.append("low temperature")
                
                if vibration > 5:  # High vibration threshold
                    context_parts.append("excessive vibration")
                elif vibration < 0.1:  # Low vibration threshold
                    context_parts.append("minimal vibration")
                
                if flow_rate > 1000:  # High flow threshold
                    context_parts.append("high flow rate")
                elif flow_rate < 10 and flow_rate > 0:  # Low flow threshold
                    context_parts.append("low flow rate")
                
                # Combine into coherent context
                context = " ".join(context_parts) + " anomaly detected requiring investigation"
                contexts.append(context)
        
        return contexts
    
    def _embed_anomaly_contexts(self, contexts):
        """Create embeddings for anomaly contexts"""
        if self.sentence_model is None or not contexts:
            return None
        
        try:
            embeddings = self.sentence_model.encode(contexts)
            return embeddings
        except Exception as e:
            st.warning(f"Failed to create anomaly embeddings: {str(e)}")
            return None
    
    def _calculate_time_proximity(self, anomaly_time, log_time):
        """Calculate time proximity score between anomaly and log"""
        time_diff = abs((log_time - anomaly_time).total_seconds() / 3600)  # hours
        return max(0, 1 - time_diff / 6)  # 6-hour window
    
    def _calculate_enhanced_correlation_score(self, anomaly_time, log_row, anomaly_score, 
                                           detection_method, semantic_similarity=0.5):
        """Calculate enhanced correlation score including semantic similarity"""
        # Time proximity score
        time_score = self._calculate_time_proximity(anomaly_time, log_row['timestamp'])
        
        # Semantic similarity score (from embeddings)
        semantic_score = semantic_similarity
        
        # Text relevance score with weighted keywords
        anomaly_keywords = {
            'critical': 3.0, 'emergency': 3.0, 'alarm': 2.5, 'fault': 2.0,
            'spike': 2.0, 'high': 1.5, 'abnormal': 2.0, 'unusual': 1.5,
            'investigating': 1.5, 'elevated': 1.5, 'anomaly': 2.0,
            'maintenance': 1.0, 'inspection': 1.5, 'malfunction': 2.5,
            'failure': 2.5, 'shutdown': 3.0, 'warning': 2.0
        }
        
        text_lower = log_row['log_text'].lower()
        keyword_score = 0
        total_weight = sum(anomaly_keywords.values())
        
        for keyword, weight in anomaly_keywords.items():
            if keyword in text_lower:
                keyword_score += weight / total_weight
        
        # Normalize keyword score
        keyword_score = min(1.0, keyword_score * 2)  # Multiply by 2 and cap at 1.0
        
        # Log type bonus
        log_type_bonus = {
            'anomaly_response': 0.3,
            'observation': 0.2,
            'maintenance': 0.1,
            'normal_operation': 0.0
        }
        type_bonus = log_type_bonus.get(log_row.get('log_type', 'unknown'), 0.05)
        
        # Detection method confidence
        method_confidence = {
            'LSTM': 0.9,
            'IF': 0.7,
            'Z-Score': 0.6
        }
        
        methods = detection_method.split(',') if detection_method else ['']
        max_confidence = max([method_confidence.get(method.strip(), 0.5) for method in methods])
        
        # Severity assessment from log text
        severity_multiplier = 1.0
        if any(word in text_lower for word in ['critical', 'emergency', 'severe']):
            severity_multiplier = 1.3
        elif any(word in text_lower for word in ['high', 'elevated', 'unusual']):
            severity_multiplier = 1.1
        
        # Combined score with semantic similarity as a key component
        # Weighted combination: semantic similarity (40%), time proximity (25%), 
        # keyword relevance (20%), detection confidence (10%), type bonus (5%)
        combined_score = (
            0.40 * semantic_score +
            0.25 * time_score + 
            0.20 * keyword_score + 
            0.10 * max_confidence + 
            0.05 * type_bonus
        ) * severity_multiplier
        
        return min(1.0, combined_score)


class EnhancedInsightGenerator:
    """Generate insights using rule-based approach with enhanced analysis"""
    
    def __init__(self):
        self.insight_templates = {
            'pressure_spike': {
                'low': "Minor pressure increase detected. Monitor for trends and check system load.",
                'medium': "Significant pressure spike observed. Inspect downstream components and safety valves immediately.",
                'high': "CRITICAL: Severe pressure spike detected. Emergency inspection required for potential blockage or pump malfunction."
            },
            'temperature_high': {
                'low': "Slight temperature elevation noted. Check cooling system performance.",
                'medium': "Elevated temperature readings suggest cooling system issues. Inspect heat exchangers and fluid levels.",
                'high': "CRITICAL: High temperature alarm. Immediate cooling system inspection required to prevent equipment damage."
            },
            'vibration_anomaly': {
                'low': "Minor vibration changes detected. Schedule routine bearing inspection.",
                'medium': "Unusual vibration patterns indicate potential mechanical wear or misalignment. Plan maintenance window.",
                'high': "CRITICAL: Severe vibration anomaly detected. Stop equipment immediately and inspect for bearing failure or structural issues."
            },
            'flow_anomaly': {
                'low': "Minor flow deviation observed. Check filter status and pump efficiency.",
                'medium': "Flow rate anomaly detected. Inspect for partial blockages and pump performance.",
                'high': "CRITICAL: Severe flow anomaly. Check for major blockages, pump failure, or system integrity issues."
            }
        }
        
        self.equipment_specific_insights = {
            'Drilling_Pump': "Critical equipment for drilling operations. Any anomaly requires immediate attention.",
            'BOP_Preventer': "Safety-critical equipment. Anomalies pose significant safety risks.",
            'Generator_Engine': "Power generation equipment. Failures can affect entire rig operations.",
            'Mud_Circulator': "Essential for drilling fluid circulation. Monitor for efficiency impacts.",
            'Rotary_Table': "Core drilling equipment. Anomalies can halt drilling operations."
        }
    
    def generate_anomaly_insights(self, correlations_df, equipment_id, anomaly_type=None):
        """Generate enhanced insights for detected anomalies"""
        insights = []
        
        if len(correlations_df) == 0:
            return [{
                'insight': f"Sensor anomaly detected on {equipment_id} with no correlating operator logs. Recommend immediate manual inspection.",
                'confidence': 0.5,
                'supporting_log': 'No operator logs found',
                'timestamp': None,
                'severity': 'medium',
                'recommendations': [
                    f"Conduct manual inspection of {equipment_id}",
                    "Review recent maintenance records",
                    "Check sensor calibration"
                ]
            }]
        
        # Analyze correlation patterns
        top_correlations = correlations_df.head(5)
        
        for _, corr in top_correlations.iterrows():
            # Determine anomaly type and severity
            text = corr['log_text'].lower()
            detected_type = self._classify_anomaly_type(text)
            severity = self._assess_severity(text, corr['correlation_score'])
            
            # Generate insight
            if detected_type in self.insight_templates:
                insight_text = self.insight_templates[detected_type][severity]
            else:
                insight_text = f"Anomaly detected in {equipment_id}. Operator noted: '{corr['log_text']}'"
            
            # Add equipment-specific context
            if equipment_id in self.equipment_specific_insights:
                equipment_context = self.equipment_specific_insights[equipment_id]
                insight_text += f" Note: {equipment_context}"
            
            # Generate recommendations
            recommendations = self._generate_recommendations(detected_type, severity, equipment_id, corr['detection_method'])
            
            insights.append({
                'insight': insight_text,
                'confidence': corr['correlation_score'],
                'supporting_log': corr['log_text'],
                'timestamp': corr['anomaly_timestamp'],
                'severity': severity,
                'anomaly_type': detected_type,
                'detection_method': corr['detection_method'],
                'recommendations': recommendations,
                'operator_id': corr.get('operator_id', 'unknown'),
                'equipment_id': equipment_id
            })
        
        return insights
    
    def _classify_anomaly_type(self, text):
        """Classify anomaly type from text with enhanced detection"""
        classification_keywords = {
            'pressure_spike': ['pressure', 'psi', 'spike', 'surge', 'buildup'],
            'temperature_high': ['temperature', 'temp', 'hot', 'heat', 'cooling', 'thermal'],
            'vibration_anomaly': ['vibration', 'shake', 'rattle', 'oscillation', 'bearing'],
            'flow_anomaly': ['flow', 'rate', 'pump', 'circulation', 'blockage']
        }
        
        scores = {}
        for anomaly_type, keywords in classification_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            scores[anomaly_type] = score
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'general_anomaly'
    
    def _assess_severity(self, text, correlation_score):
        """Assess severity based on text content and correlation score"""
        high_severity_keywords = ['critical', 'emergency', 'shutdown', 'alarm', 'severe', 'immediate']
        medium_severity_keywords = ['elevated', 'unusual', 'investigating', 'abnormal', 'attention']
        
        if any(keyword in text for keyword in high_severity_keywords) or correlation_score > 0.8:
            return 'high'
        elif any(keyword in text for keyword in medium_severity_keywords) or correlation_score > 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, anomaly_type, severity, equipment_id, detection_method):
        """Generate specific recommendations based on anomaly characteristics"""
        base_recommendations = {
            'pressure_spike': {
                'low': ["Monitor pressure trends", "Check system load"],
                'medium': ["Inspect safety valves", "Check downstream components", "Review pump settings"],
                'high': ["IMMEDIATE: Stop operations", "Emergency pressure relief", "Full system inspection"]
            },
            'temperature_high': {
                'low': ["Check cooling system", "Monitor temperature trends"],
                'medium': ["Inspect heat exchangers", "Check cooling fluid levels", "Review thermal loads"],
                'high': ["IMMEDIATE: Activate emergency cooling", "Stop equipment operation", "Thermal stress inspection"]
            },
            'vibration_anomaly': {
                'low': ["Schedule bearing inspection", "Monitor vibration patterns"],
                'medium': ["Check alignment", "Inspect rotating components", "Plan maintenance window"],
                'high': ["IMMEDIATE: Stop equipment", "Emergency bearing inspection", "Structural integrity check"]
            },
            'flow_anomaly': {
                'low': ["Check filters", "Monitor flow patterns"],
                'medium': ["Inspect pumps", "Check for blockages", "Review system efficiency"],
                'high': ["IMMEDIATE: Check system integrity", "Emergency pump inspection", "Full flow path analysis"]
            }
        }
        
        recommendations = base_recommendations.get(anomaly_type, {}).get(severity, ["Investigate anomaly", "Review equipment status"])
        
        # Add equipment-specific recommendations
        equipment_recommendations = {
            'Drilling_Pump': ["Check drilling fluid properties", "Inspect pump seals"],
            'BOP_Preventer': ["Test safety systems", "Verify hydraulic pressure"],
            'Generator_Engine': ["Check fuel system", "Inspect electrical connections"],
            'Mud_Circulator': ["Check mud properties", "Inspect circulation path"],
            'Rotary_Table': ["Check drive mechanism", "Inspect rotary seals"]
        }
        
        if equipment_id in equipment_recommendations:
            recommendations.extend(equipment_recommendations[equipment_id])
        
        # Add detection method specific recommendations
        if 'LSTM' in detection_method:
            recommendations.append("LSTM detected pattern anomaly - review historical data")
        if 'Z-Score' in detection_method:
            recommendations.append("Statistical anomaly - check sensor calibration")
        
        return recommendations
    
    def generate_summary_report(self, anomalies_count, correlations_df, equipment_summary, insights):
        """Generate comprehensive summary report"""
        # Calculate statistics
        total_correlations = len(correlations_df)
        avg_correlation = correlations_df['correlation_score'].mean() if total_correlations > 0 else 0
        
        # Severity analysis
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        for insight in insights:
            if isinstance(insight, dict) and 'severity' in insight:
                severity_counts[insight['severity']] += 1
        
        # Equipment risk assessment
        high_risk_equipment = [eq for eq, count in equipment_summary.items() if count >= 3]
        medium_risk_equipment = [eq for eq, count in equipment_summary.items() if 1 <= count < 3]
        
        report = f"""
# 🛢️ Oil Rig Anomaly Detection Summary Report

## 📊 Executive Summary
- **Total Anomalies Detected:** {anomalies_count}
- **Correlations with Operator Logs:** {total_correlations}
- **Average Correlation Score:** {avg_correlation:.2f}
- **Analysis Period:** {correlations_df['anomaly_timestamp'].min().strftime('%Y-%m-%d') if total_correlations > 0 else 'N/A'} to {correlations_df['anomaly_timestamp'].max().strftime('%Y-%m-%d') if total_correlations > 0 else 'N/A'}

## ⚠️ Severity Breakdown
- **🔴 High Severity:** {severity_counts['high']} anomalies
- **🟡 Medium Severity:** {severity_counts['medium']} anomalies  
- **🟢 Low Severity:** {severity_counts['low']} anomalies

## 🏭 Equipment Status Assessment
"""
        
        if high_risk_equipment:
            report += f"""
### 🔴 HIGH RISK EQUIPMENT (≥3 anomalies):
"""
            for equipment in high_risk_equipment:
                count = equipment_summary[equipment]
                report += f"- **{equipment}:** {count} anomalies - IMMEDIATE ATTENTION REQUIRED\n"
        
        if medium_risk_equipment:
            report += f"""
### 🟡 MEDIUM RISK EQUIPMENT (1-2 anomalies):
"""
            for equipment in medium_risk_equipment:
                count = equipment_summary[equipment]
                report += f"- **{equipment}:** {count} anomalies - SCHEDULED INSPECTION RECOMMENDED\n"
        
        normal_equipment = [eq for eq, count in equipment_summary.items() if count == 0]
        if normal_equipment:
            report += f"""
### ✅ NORMAL OPERATION:
"""
            for equipment in normal_equipment:
                report += f"- **{equipment}:** No anomalies detected\n"
        
        if total_correlations > 0:
            # Detection method analysis
            detection_methods = correlations_df['detection_method'].value_counts()
            report += f"""

## 🔍 Detection Method Performance
"""
            for method, count in detection_methods.head().items():
                report += f"- **{method}:** {count} detections\n"
            
            # Top insights
            high_confidence_insights = [i for i in insights if isinstance(i, dict) and i.get('confidence', 0) > 0.7]
            if high_confidence_insights:
                report += f"""

## 💡 Key Insights (High Confidence)
"""
                for i, insight in enumerate(high_confidence_insights[:3], 1):
                    report += f"""
**{i}. {insight['anomaly_type'].replace('_', ' ').title()} - {insight['equipment_id']}**
- {insight['insight']}
- Confidence: {insight['confidence']:.2f}
- Timestamp: {insight['timestamp'].strftime('%Y-%m-%d %H:%M') if insight['timestamp'] else 'N/A'}
"""
        
        report += f"""

## 📋 Immediate Action Items
1. **Critical Equipment Inspection:** Focus on high-risk equipment identified above
2. **Operator Training:** Review correlation patterns to improve anomaly reporting
3. **Preventive Maintenance:** Schedule maintenance for medium-risk equipment
4. **System Monitoring:** Enhance monitoring for equipment showing pattern anomalies

## 🔧 Technical Recommendations
1. **LSTM Model Performance:** {"LSTM autoencoder successfully detecting pattern anomalies" if any("LSTM" in method for method in correlations_df.get('detection_method', [])) else "Consider training LSTM models with more historical data"}
2. **Sensor Calibration:** Review sensors for equipment with statistical anomalies
3. **Data Quality:** {"Good correlation between sensor data and operator logs" if avg_correlation > 0.6 else "Improve operator log quality and timeliness"}

---
*Report generated by Multi-Modal Anomaly Detection System*
"""
        
        return report

# Streamlit Application
def main():
    st.set_page_config(
        page_title="Enhanced Oil Rig Anomaly Detection System",
        page_icon="🛢️",
        layout="wide"
    )
    
    st.title("🛢️ Enhanced Multi-Modal Oil Rig Anomaly Detection System")
    st.markdown("*Combining LSTM Autoencoders, Isolation Forest, and NLP for Comprehensive Anomaly Detection*")
    st.markdown("---")
    
    # Initialize components
    if 'data_generator' not in st.session_state:
        st.session_state.data_generator = OilRigDataGenerator()
        st.session_state.detector = MultiModalAnomalyDetector()
        st.session_state.insight_generator = EnhancedInsightGenerator()
        st.session_state.model_trained = False
    
    # Sidebar controls
    st.sidebar.header("🎛️ System Controls")
    
    # Model configuration
    st.sidebar.subheader("Model Configuration")
    sequence_length = st.sidebar.slider("LSTM Sequence Length (hours)", 12, 48, 24)
    contamination_rate = st.sidebar.slider("Isolation Forest Contamination", 0.05, 0.2, 0.1)
    
    st.session_state.detector.sequence_length = sequence_length
    
    if st.sidebar.button("🔄 Generate New Data", type="primary"):
        with st.spinner("Generating synthetic oil rig data..."):
            sensor_df, anomaly_df = st.session_state.data_generator.generate_sensor_data()
            logs_df = st.session_state.data_generator.generate_operator_logs(sensor_df, anomaly_df)
            
            st.session_state.sensor_df = sensor_df
            st.session_state.anomaly_df = anomaly_df
            st.session_state.logs_df = logs_df
            st.session_state.model_trained = False

            # sensor_df.to_csv(r'G:\anup_prep\additonal_test\oil-rig-anomaly-detection\data\sensor_data.csv')
            # anomaly_df.to_csv(r'G:\anup_prep\additonal_test\oil-rig-anomaly-detection\data\anamoly.csv')
            # logs_df.to_csv(r'G:\anup_prep\additonal_test\oil-rig-anomaly-detection\data\logs_df.csv')
        
        st.success("✅ New data generated successfully!")
        # st.rerun()

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
    if not hasattr(st.session_state.detector, 'sentence_model'):
        with st.spinner("Loading NLP model..."):
            success = st.session_state.detector.load_sentence_transformer()
            if success:
                st.success("✅ NLP model loaded successfully!")
            else:
                st.warning("⚠️ Using basic text analysis (Sentence Transformer not available)")
    
    # Main analysis
    if st.sidebar.button("🔍 Run Complete Analysis", type="secondary"):
        run_complete_analysis()
    
    # Display current data overview
    display_enhanced_data_overview()
    
    # Display results if available
    if 'enhanced_results' in st.session_state:
        display_enhanced_results()
    
    # Model performance metrics
    if st.session_state.model_trained:
        display_model_performance()

def run_complete_analysis():
    """Run the complete enhanced anomaly detection pipeline"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Load and prepare data
        status_text.text("🔍 Step 1/5: Preparing data and training models...")
        progress_bar.progress(20)
        
        detector = st.session_state.detector
        
        # Step 2: Detect sensor anomalies (includes training)
        status_text.text("🤖 Step 2/5: Training LSTM autoencoders and detecting anomalies...")
        anomaly_results = detector.detect_sensor_anomalies(st.session_state.sensor_df)
        progress_bar.progress(40)
        st.session_state.model_trained = True
        
        # Step 3: Correlate with logs
        status_text.text("🔗 Step 3/5: Correlating anomalies with operator logs...")
        correlations = detector.correlate_anomalies_with_logs(
            anomaly_results, 
            st.session_state.logs_df, 
            st.session_state.sensor_df
        )
        progress_bar.progress(60)
        
        # Step 4: Generate enhanced insights
        status_text.text("💡 Step 4/5: Generating AI-powered insights...")
        insights = []
        equipment_summary = {}
        
        for equipment in st.session_state.sensor_df['equipment_id'].unique():
            eq_correlations = correlations[correlations['equipment_id'] == equipment]
            eq_insights = st.session_state.insight_generator.generate_anomaly_insights(
                eq_correlations, equipment
            )
            insights.extend(eq_insights)
            equipment_summary[equipment] = len(eq_correlations)
        
        progress_bar.progress(80)
        
        # Step 5: Generate comprehensive report
        status_text.text("📊 Step 5/5: Generating comprehensive report...")
        total_anomalies = np.sum(anomaly_results['anomaly_flags'])
        # summary_report = st.session_state.insight_generator.generate_summary_report(
        #     total_anomalies, correlations, equipment_summary, insights
        # )
        
        progress_bar.progress(100)
        
        # Store enhanced results
        st.session_state.enhanced_results = {
            'anomaly_results': anomaly_results,
            'correlations': correlations,
            'insights': insights,
            'equipment_summary': equipment_summary,
            # 'summary_report': summary_report,
            'total_anomalies': total_anomalies
        }
        
        status_text.text("✅ Analysis complete!")
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        status_text.text("❌ Analysis failed")
        traceback.print_stack()
        traceback.print_exc()
        progress_bar.progress(0)

def display_enhanced_data_overview():
    """Display enhanced overview of generated data"""
    st.header("📊 Data Overview")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sensor Records", f"{len(st.session_state.sensor_df):,}")
    
    with col2:
        st.metric("Operator Logs", f"{len(st.session_state.logs_df):,}")
    
    with col3:
        st.metric("Equipment Types", len(st.session_state.sensor_df['equipment_id'].unique()))
    
    with col4:
        time_range = (st.session_state.sensor_df['timestamp'].max() - 
                     st.session_state.sensor_df['timestamp'].min()).days
        st.metric("Time Range (Days)", time_range)
    
    # Data quality metrics
    st.subheader("📈 Data Quality Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Missing data analysis
        numeric_cols = ['pressure_psi', 'temperature_f', 'vibration_mm_s', 'flow_rate_bpm']
        missing_data = st.session_state.sensor_df[numeric_cols].isnull().sum()
        
        if missing_data.sum() > 0:
            fig = px.bar(
                x=missing_data.index, 
                y=missing_data.values,
                title="Missing Data by Parameter",
                labels={'x': 'Parameter', 'y': 'Missing Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing data detected")
    
    with col2:
        # Log type distribution
        if 'log_type' in st.session_state.logs_df.columns:
            log_type_counts = st.session_state.logs_df['log_type'].value_counts()
            fig = px.pie(
                values=log_type_counts.values,
                names=log_type_counts.index,
                title="Operator Log Types Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Equipment activity overview
    st.subheader("🏭 Equipment Activity Overview")
    
    equipment_stats = []
    for equipment in st.session_state.sensor_df['equipment_id'].unique():
        eq_data = st.session_state.sensor_df[st.session_state.sensor_df['equipment_id'] == equipment]
        eq_logs = st.session_state.logs_df[st.session_state.logs_df['equipment_id'] == equipment]
        
        stats = {
            'Equipment': equipment,
            'Sensor Readings': len(eq_data),
            'Operator Logs': len(eq_logs),
            'Avg Pressure': eq_data['pressure_psi'].mean(),
            'Avg Temperature': eq_data['temperature_f'].mean(),
            'Data Coverage': f"{len(eq_data) / len(st.session_state.sensor_df) * 100:.1f}%"
        }
        equipment_stats.append(stats)
    
    equipment_df = pd.DataFrame(equipment_stats)
    st.dataframe(equipment_df, use_container_width=True)

def display_enhanced_results():
    """Display comprehensive anomaly detection results"""
    results = st.session_state.enhanced_results
    
    st.header("🚨 Enhanced Anomaly Detection Results")
    
    # Executive Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Anomalies", results['total_anomalies'])
    
    with col2:
        st.metric("Correlations Found", len(results['correlations']))
    
    with col3:
        avg_score = results['correlations']['correlation_score'].mean() if len(results['correlations']) > 0 else 0
        st.metric("Avg Correlation Score", f"{avg_score:.3f}")
    
    with col4:
        high_confidence = sum(1 for i in results['insights'] if isinstance(i, dict) and i.get('confidence', 0) > 0.7)
        st.metric("High Confidence Insights", high_confidence)
    
    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["📈 Time Series Analysis", "🔗 Correlations", "💡 Insights"])
    
    with tab1:
        display_time_series_analysis(results)
    
    with tab2:
        display_correlation_analysis(results)
    
    with tab3:
        display_insights_analysis(results)
    
    # with tab4:
    #     st.markdown(results['summary_report'])

def display_time_series_analysis(results):
    """Display time series analysis with anomalies"""
    st.subheader("📈 Time Series Analysis with Multi-Method Detection")
    
    # Equipment selector
    equipment_options = st.session_state.sensor_df['equipment_id'].unique()
    selected_equipment = st.selectbox("Select Equipment for Analysis", equipment_options, key="ts_equipment")
    
    # Parameter selector
    parameter_options = {
        'Pressure (PSI)': 'pressure_psi',
        'Temperature (°F)': 'temperature_f', 
        'Vibration (mm/s)': 'vibration_mm_s',
        'Flow Rate (BPM)': 'flow_rate_bpm'
    }
    selected_param_name = st.selectbox("Select Parameter", list(parameter_options.keys()), key="ts_param")
    selected_param = parameter_options[selected_param_name]
    
    # Get data for selected equipment
    eq_data = st.session_state.sensor_df[st.session_state.sensor_df['equipment_id'] == selected_equipment]
    
    # Get anomaly data for this equipment
    anomaly_mask = results['anomaly_results']['anomaly_flags']
    equipment_indices = st.session_state.sensor_df[st.session_state.sensor_df['equipment_id'] == selected_equipment].index
    equipment_anomalies = anomaly_mask[equipment_indices]
    
    # Create plot
    fig = go.Figure()
    
    # Add normal data
    normal_data = eq_data[~equipment_anomalies]
    fig.add_trace(go.Scatter(
        x=normal_data['timestamp'],
        y=normal_data[selected_param],
        mode='lines',
        name=f'Normal {selected_param_name}',
        line=dict(color='blue', width=1),
        opacity=0.7
    ))
    
    # Add anomaly points
    anomaly_data = eq_data[equipment_anomalies]
    if len(anomaly_data) > 0:
        # Get detection methods for anomalies
        anomaly_methods = [results['anomaly_results']['anomaly_methods'][i] for i in equipment_indices[equipment_anomalies]]
        
        # Color by detection method
        colors = {'IF': 'red', 'Z-Score': 'orange', 'LSTM': 'purple'}
        
        for method in ['IF', 'Z-Score', 'LSTM']:
            method_mask = [method in str(methods) for methods in anomaly_methods]
            if any(method_mask):
                method_data = anomaly_data[method_mask]
                fig.add_trace(go.Scatter(
                    x=method_data['timestamp'],
                    y=method_data[selected_param],
                    mode='markers',
                    name=f'{method} Anomalies',
                    marker=dict(
                        color=colors[method], 
                        size=10, 
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    )
                ))
    
    fig.update_layout(
        title=f"{selected_param_name} Analysis - {selected_equipment}",
        xaxis_title="Timestamp",
        yaxis_title=selected_param_name,
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detection method performance
    if len(anomaly_data) > 0:
        st.subheader("🎯 Detection Method Performance")
        method_counts = {}
        for methods in anomaly_methods:
            for method in str(methods).split(','):
                method = method.strip()
                if method and method != '':
                    method_counts[method] = method_counts.get(method, 0) + 1
        
        if method_counts:
            col1, col2, col3 = st.columns(3)
            for i, (method, count) in enumerate(method_counts.items()):
                with [col1, col2, col3][i % 3]:
                    st.metric(f"{method} Detections", count)




def display_correlation_analysis(results):
    """Display correlation analysis between anomalies and logs with semantic analysis"""
    st.subheader("🔗 Anomaly-Log Correlation Analysis with Semantic Similarity")
    
    correlations = results['correlations']
    
    if len(correlations) == 0:
        st.warning("No correlations found between anomalies and operator logs.")
        return
    
    # Correlation metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_correlation = correlations['correlation_score'].mean()
        st.metric("Avg Correlation Score", f"{avg_correlation:.3f}")
    
    with col2:
        if 'semantic_similarity' in correlations.columns:
            avg_semantic = correlations['semantic_similarity'].mean()
            st.metric("Avg Semantic Similarity", f"{avg_semantic:.3f}")
        else:
            st.metric("Semantic Analysis", "Not Available")
    
    with col3:
        if 'time_proximity' in correlations.columns:
            avg_time = correlations['time_proximity'].mean()
            st.metric("Avg Time Proximity", f"{avg_time:.3f}")
        else:
            st.metric("Time Proximity", "Calculated")
    
    with col4:
        high_quality_corr = len(correlations[correlations['correlation_score'] > 0.7])
        st.metric("High Quality Correlations", high_quality_corr)
    
    # Correlation score vs semantic similarity scatter plot
    if 'semantic_similarity' in correlations.columns:
        fig = px.scatter(
            correlations,
            x='semantic_similarity',
            y='correlation_score',
            color='equipment_id',
            size='time_proximity' if 'time_proximity' in correlations.columns else None,
            hover_data=['log_text', 'detection_method'],
            title="Correlation Score vs Semantic Similarity",
            labels={
                'semantic_similarity': 'Semantic Similarity (Cosine)',
                'correlation_score': 'Overall Correlation Score'
            }
        )
        fig.update_traces(marker=dict(opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            correlations,
            x='correlation_score',
            nbins=20,
            title="Distribution of Overall Correlation Scores",
            labels={'correlation_score': 'Correlation Score', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'semantic_similarity' in correlations.columns:
            fig = px.histogram(
                correlations,
                x='semantic_similarity',
                nbins=20,
                title="Distribution of Semantic Similarity Scores",
                labels={'semantic_similarity': 'Semantic Similarity', 'count': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Top correlations table with semantic information
    st.subheader("🏆 Top Correlations with Semantic Analysis")
    
    top_correlations = correlations.head(15)
    display_columns = [
        'anomaly_timestamp', 'equipment_id', 'log_text', 
        'correlation_score', 'detection_method'
    ]
    
    if 'semantic_similarity' in correlations.columns:
        display_columns.append('semantic_similarity')
    if 'time_proximity' in correlations.columns:
        display_columns.append('time_proximity')
    
    display_correlations = top_correlations[display_columns].copy()
    
    # Format the display
    display_correlations['anomaly_timestamp'] = display_correlations['anomaly_timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    display_correlations['correlation_score'] = display_correlations['correlation_score'].round(3)
    
    if 'semantic_similarity' in display_correlations.columns:
        display_correlations['semantic_similarity'] = display_correlations['semantic_similarity'].round(3)
    if 'time_proximity' in display_correlations.columns:
        display_correlations['time_proximity'] = display_correlations['time_proximity'].round(3)
    
    st.dataframe(display_correlations, use_container_width=True)
    
    # Semantic similarity analysis by equipment
    if 'semantic_similarity' in correlations.columns:
        st.subheader("📊 Semantic Similarity Analysis by Equipment")
        
        equipment_semantic = correlations.groupby('equipment_id').agg({
            'semantic_similarity': ['mean', 'std', 'count'],
            'correlation_score': 'mean'
        }).round(3)
        
        equipment_semantic.columns = ['Avg_Semantic_Sim', 'Std_Semantic_Sim', 'Count', 'Avg_Correlation']
        equipment_semantic = equipment_semantic.reset_index()
        
        fig = px.scatter(
            equipment_semantic,
            x='Avg_Semantic_Sim',
            y='Avg_Correlation',
            size='Count',
            hover_name='equipment_id',
            title="Equipment Performance: Semantic Similarity vs Overall Correlation",
            labels={
                'Avg_Semantic_Sim': 'Average Semantic Similarity',
                'Avg_Correlation': 'Average Correlation Score'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(equipment_semantic, use_container_width=True)
    
    # Detection method performance with semantic analysis
    st.subheader("🎯 Detection Method Performance with Semantic Context")
    
    if len(correlations) > 0:
        method_performance = []
        
        for method in ['LSTM', 'IF', 'Z-Score']:
            method_corrs = correlations[correlations['detection_method'].str.contains(method, na=False)]
            
            if len(method_corrs) > 0:
                perf_data = {
                    'Method': method,
                    'Count': len(method_corrs),
                    'Avg_Correlation': method_corrs['correlation_score'].mean(),
                    'High_Quality_Count': len(method_corrs[method_corrs['correlation_score'] > 0.7])
                }
                
                if 'semantic_similarity' in method_corrs.columns:
                    perf_data['Avg_Semantic_Sim'] = method_corrs['semantic_similarity'].mean()
                
                method_performance.append(perf_data)
        
        if method_performance:
            perf_df = pd.DataFrame(method_performance)
            for col in ['Avg_Correlation', 'Avg_Semantic_Sim']:
                if col in perf_df.columns:
                    perf_df[col] = perf_df[col].round(3)
            
            st.dataframe(perf_df, use_container_width=True)
            
            # Visualization
            if 'Avg_Semantic_Sim' in perf_df.columns:
                fig = px.bar(
                    perf_df,
                    x='Method',
                    y=['Avg_Correlation', 'Avg_Semantic_Sim'],
                    title="Detection Method Performance: Correlation vs Semantic Similarity",
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)



def display_insights_analysis(results):
    """Display AI-generated insights analysis"""
    st.subheader("💡 AI-Generated Insights & Recommendations")
    
    insights = results['insights']
    
    if not insights:
        st.warning("No insights generated.")
        return
    
    # Insights by severity
    severity_insights = {'high': [], 'medium': [], 'low': []}
    for insight in insights:
        if isinstance(insight, dict):
            severity = insight.get('severity', 'medium')
            severity_insights[severity].append(insight)
    
    # Display high severity first
    for severity, color, icon in [('high', 'error', '🔴'), ('medium', 'warning', '🟡'), ('low', 'info', '🟢')]:
        if severity_insights[severity]:
            st.subheader(f"{icon} {severity.title()} Severity Insights")
            
            for i, insight in enumerate(severity_insights[severity][:5], 1):  # Show top 5 per severity
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{i}. {insight.get('anomaly_type', 'Unknown').replace('_', ' ').title()} - {insight.get('equipment_id', 'Unknown')}**")
                        st.write(insight.get('insight', 'No insight available'))
                        
                        if insight.get('supporting_log'):
                            st.caption(f"*Supporting Log:* {insight['supporting_log']}")
                    
                    with col2:
                        if insight.get('confidence'):
                            st.metric("Confidence", f"{insight['confidence']:.2f}")
                        
                        if insight.get('detection_method'):
                            st.caption(f"Method: {insight['detection_method']}")
                    
                    # Recommendations
                    if insight.get('recommendations'):
                        with st.expander("📋 Recommendations", expanded=False):
                            for rec in insight['recommendations']:
                                st.write(f"• {rec}")
                    
                    st.markdown("---")

def display_model_performance():
    """Display model performance metrics"""
    st.header("🤖 Model Performance Metrics")
    
    detector = st.session_state.detector
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("LSTM Autoencoder Performance")
        if hasattr(detector, 'lstm_models'):
            performance_data = []
            for equipment, model in detector.lstm_models.items():
                threshold = detector.reconstruction_thresholds.get(equipment, 0)
                performance_data.append({
                    'Equipment': equipment,
                    'Model Trained': '✅',
                    'Reconstruction Threshold': f"{threshold:.4f}"
                })
            
            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                st.dataframe(perf_df, use_container_width=True)
            else:
                st.info("No LSTM models trained yet")
        else:
            st.info("LSTM models not available")
    
    # with col2:
    #     st.subheader("Detection Method Statistics")
    #     if 'enhanced_results' in st.session_state:
    #         results = st.session_state.enhanced_results
    #         if len(results['correlations']) > 0:
    #             method_stats = results['correlations']['detection_method'].value_counts()
                
    #             stats_data = []
    #             for method, count in method_stats.items():
    #                 stats_data.append({
    #                     'Method': method,
    #                     'Detections': count,
    #                     'Percentage': f"{count/len(results['correlations'])*100:.1f}%"
    #                 })
                
    #             stats_df = pd.DataFrame(stats_data)
    #             st.dataframe(stats_df, use_container_width=True)

if __name__ == "__main__":
    main()