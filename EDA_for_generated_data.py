import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings



warnings.filterwarnings('ignore')

class ExploratoryDataAnalysis:
    """Comprehensive EDA component for oil rig sensor and log data"""
    
    def __init__(self, sensor_df, logs_df, anomaly_df=None):
        self.sensor_df = sensor_df.copy()
        self.logs_df = logs_df.copy()
        self.anomaly_df = anomaly_df.copy() if anomaly_df is not None else pd.DataFrame()
        
    def display_eda_dashboard(self):
        """Main EDA dashboard for Streamlit"""
        st.header("📊 Exploratory Data Analysis")
        st.markdown("---")
        
        # Create tabs for different EDA sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Data Overview", 
            "🔍 Sensor Analysis", 
            "📝 Log Analysis", 
            "⚠️ Anomaly Patterns", 
            "🔗 Cross-Modal Analysis"
        ])
        
        with tab1:
            self._display_data_overview()
            
        with tab2:
            self._display_sensor_analysis()
            
        with tab3:
            self._display_log_analysis()
            
        with tab4:
            self._display_anomaly_analysis()
            
        with tab5:
            self._display_cross_modal_analysis()
    
    def _display_data_overview(self):
        """Display general data overview and statistics"""
        st.subheader("🎯 Data Summary")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sensor Records", f"{len(self.sensor_df):,}")
            
        with col2:
            st.metric("Total Log Entries", f"{len(self.logs_df):,}")
            
        with col3:
            date_range = (self.sensor_df['timestamp'].max() - self.sensor_df['timestamp'].min()).days
            st.metric("Date Range (Days)", f"{date_range}")
            
        with col4:
            st.metric("Equipment Types", f"{self.sensor_df['equipment_id'].nunique()}")
        
        # Data quality metrics
        st.subheader("📊 Data Quality Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Missing data analysis
            sensor_missing = self.sensor_df.isnull().sum()
            if sensor_missing.sum() > 0:
                fig_missing = px.bar(
                    x=sensor_missing.index,
                    y=sensor_missing.values,
                    title="Missing Values by Column",
                    labels={'x': 'Columns', 'y': 'Missing Count'}
                )
                fig_missing.update_layout(height=400)
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("✅ No missing values found in sensor data")
        
        with col2:
            # Data types and basic stats
            st.write("**Sensor Data Info:**")
            data_info = pd.DataFrame({
                'Column': self.sensor_df.columns,
                'Data Type': self.sensor_df.dtypes.astype(str),
                'Non-Null Count': self.sensor_df.count(),
                'Null Count': self.sensor_df.isnull().sum()
            })
            st.dataframe(data_info, use_container_width=True)
        
        # Time series coverage
        st.subheader("📅 Temporal Coverage")
        
        # Create hourly data coverage plot
        sensor_hourly = self.sensor_df.set_index('timestamp').resample('H').size()
        
        fig_coverage = go.Figure()
        fig_coverage.add_trace(go.Scatter(
            x=sensor_hourly.index,
            y=sensor_hourly.values,
            mode='lines',
            name='Records per Hour',
            line=dict(color='blue', width=2)
        ))
        
        fig_coverage.update_layout(
            title="Data Collection Coverage Over Time",
            xaxis_title="Time",
            yaxis_title="Records per Hour",
            height=400
        )
        
        st.plotly_chart(fig_coverage, use_container_width=True)
        
        # Equipment distribution
        st.subheader("🏭 Equipment Distribution")
        
        equipment_counts = self.sensor_df['equipment_id'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig_pie = px.pie(
                values=equipment_counts.values,
                names=equipment_counts.index,
                title="Data Distribution by Equipment"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = px.bar(
                x=equipment_counts.values,
                y=equipment_counts.index,
                orientation='h',
                title="Record Count by Equipment",
                labels={'x': 'Record Count', 'y': 'Equipment ID'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    def _display_sensor_analysis(self):
        """Display detailed sensor data analysis"""
        st.subheader("🔧 Sensor Data Analysis")
        
        # Sensor selection
        sensor_cols = ['pressure_psi', 'temperature_f', 'vibration_mm_s', 'flow_rate_bpm']
        selected_sensors = st.multiselect(
            "Select sensors to analyze:",
            sensor_cols,
            default=sensor_cols
        )
        
        if not selected_sensors:
            st.warning("Please select at least one sensor type.")
            return
        
        # Equipment selection
        equipment_list = ['All'] + list(self.sensor_df['equipment_id'].unique())
        selected_equipment = st.selectbox("Select equipment:", equipment_list)
        
        # Filter data
        if selected_equipment == 'All':
            filtered_data = self.sensor_df
        else:
            filtered_data = self.sensor_df[self.sensor_df['equipment_id'] == selected_equipment]
        
        # Statistical summary
        st.subheader("📈 Statistical Summary")
        
        summary_stats = filtered_data[selected_sensors].describe()
        st.dataframe(summary_stats, use_container_width=True)
        
        # Distribution plots
        st.subheader("📊 Sensor Value Distributions")
        
        # Create distribution plots for each sensor
        n_sensors = len(selected_sensors)
        n_cols = min(2, n_sensors)
        n_rows = (n_sensors + n_cols - 1) // n_cols
        
        fig_dist = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=selected_sensors,
            vertical_spacing=0.1
        )
        
        for i, sensor in enumerate(selected_sensors):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            # Create histogram
            fig_dist.add_trace(
                go.Histogram(
                    x=filtered_data[sensor],
                    name=sensor,
                    nbinsx=50,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig_dist.update_layout(
            title="Sensor Value Distributions",
            height=300 * n_rows
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Correlation analysis
        st.subheader("🔗 Sensor Correlation Analysis")
        
        correlation_matrix = filtered_data[selected_sensors].corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Sensor Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Time series analysis
        st.subheader("📈 Time Series Patterns")
        
        # Aggregate data by hour for visualization
        hourly_data = filtered_data.set_index('timestamp').resample('H')[selected_sensors].mean()
        
        fig_ts = go.Figure()
        
        for sensor in selected_sensors:
            fig_ts.add_trace(go.Scatter(
                x=hourly_data.index,
                y=hourly_data[sensor],
                mode='lines',
                name=sensor,
                line=dict(width=2)
            ))
        
        fig_ts.update_layout(
            title="Sensor Values Over Time (Hourly Averages)",
            xaxis_title="Time",
            yaxis_title="Sensor Value",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Daily patterns
        st.subheader("🕐 Daily and Weekly Patterns")
        
        # Add time features
        temp_data = filtered_data.copy()
        temp_data['hour'] = temp_data['timestamp'].dt.hour
        temp_data['day_of_week'] = temp_data['timestamp'].dt.day_name()
        temp_data['day_of_week_num'] = temp_data['timestamp'].dt.dayofweek
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly patterns
            hourly_avg = temp_data.groupby('hour')[selected_sensors].mean()
            
            fig_hourly = go.Figure()
            for sensor in selected_sensors:
                fig_hourly.add_trace(go.Scatter(
                    x=hourly_avg.index,
                    y=hourly_avg[sensor],
                    mode='lines+markers',
                    name=sensor
                ))
            
            fig_hourly.update_layout(
                title="Average Values by Hour of Day",
                xaxis_title="Hour",
                yaxis_title="Average Value"
            )
            
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Weekly patterns
            daily_avg = temp_data.groupby(['day_of_week_num', 'day_of_week'])[selected_sensors].mean().reset_index()
            daily_avg = daily_avg.sort_values('day_of_week_num')
            
            fig_weekly = go.Figure()
            for sensor in selected_sensors:
                fig_weekly.add_trace(go.Scatter(
                    x=daily_avg['day_of_week'],
                    y=daily_avg[sensor],
                    mode='lines+markers',
                    name=sensor
                ))
            
            fig_weekly.update_layout(
                title="Average Values by Day of Week",
                xaxis_title="Day of Week",
                yaxis_title="Average Value"
            )
            
            st.plotly_chart(fig_weekly, use_container_width=True)
        
        # Outlier detection visualization
        st.subheader("🚨 Outlier Detection (Statistical)")
        
        # Calculate Z-scores
        z_scores = np.abs((filtered_data[selected_sensors] - filtered_data[selected_sensors].mean()) / 
                         filtered_data[selected_sensors].std())
        
        outliers = (z_scores > 3).any(axis=1)
        outlier_percentage = (outliers.sum() / len(filtered_data)) * 100
        
        st.metric("Statistical Outliers (Z-score > 3)", f"{outliers.sum():,} ({outlier_percentage:.2f}%)")
        
        if outliers.sum() > 0:
            # Box plots to show outliers
            fig_box = go.Figure()
            
            for sensor in selected_sensors:
                fig_box.add_trace(go.Box(
                    y=filtered_data[sensor],
                    name=sensor,
                    boxpoints='outliers'
                ))
            
            fig_box.update_layout(
                title="Box Plots Showing Outliers",
                yaxis_title="Sensor Values",
                height=400
            )
            
            st.plotly_chart(fig_box, use_container_width=True)
    
    def _display_log_analysis(self):
        """Display operator log analysis"""
        st.subheader("📝 Operator Log Analysis")
        
        # Basic log statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Log Entries", f"{len(self.logs_df):,}")
        
        with col2:
            unique_operators = self.logs_df['operator_id'].nunique()
            st.metric("Unique Operators", f"{unique_operators}")
        
        with col3:
            avg_log_length = self.logs_df['log_text'].str.len().mean()
            st.metric("Avg Log Length", f"{avg_log_length:.0f} chars")
        
        # Log type distribution
        st.subheader("📊 Log Type Distribution")
        
        log_type_counts = self.logs_df['log_type'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_log_types = px.pie(
                values=log_type_counts.values,
                names=log_type_counts.index,
                title="Distribution of Log Types"
            )
            st.plotly_chart(fig_log_types, use_container_width=True)
        
        with col2:
            fig_log_bar = px.bar(
                x=log_type_counts.index,
                y=log_type_counts.values,
                title="Log Entries by Type",
                labels={'x': 'Log Type', 'y': 'Count'}
            )
            fig_log_bar.update_xaxes(tickangle=45)
            st.plotly_chart(fig_log_bar, use_container_width=True)
        
        # Temporal patterns of logs
        st.subheader("📅 Log Temporal Patterns")
        
        # Convert timestamp and create time features
        logs_temp = self.logs_df.copy()
        logs_temp['timestamp'] = pd.to_datetime(logs_temp['timestamp'])
        logs_temp['hour'] = logs_temp['timestamp'].dt.hour
        logs_temp['day_of_week'] = logs_temp['timestamp'].dt.day_name()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Logs by hour
            hourly_logs = logs_temp['hour'].value_counts().sort_index()
            
            fig_hourly_logs = px.bar(
                x=hourly_logs.index,
                y=hourly_logs.values,
                title="Log Entries by Hour of Day",
                labels={'x': 'Hour', 'y': 'Log Count'}
            )
            st.plotly_chart(fig_hourly_logs, use_container_width=True)
        
        with col2:
            # Logs by day of week
            daily_logs = logs_temp['day_of_week'].value_counts()
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_logs = daily_logs.reindex([day for day in days_order if day in daily_logs.index])
            
            fig_daily_logs = px.bar(
                x=daily_logs.index,
                y=daily_logs.values,
                title="Log Entries by Day of Week",
                labels={'x': 'Day', 'y': 'Log Count'}
            )
            fig_daily_logs.update_xaxes(tickangle=45)
            st.plotly_chart(fig_daily_logs, use_container_width=True)
        
        # Equipment-wise log distribution
        st.subheader("🏭 Equipment-wise Log Analysis")
        
        equipment_logs = self.logs_df['equipment_id'].value_counts()
        
        fig_eq_logs = px.bar(
            y=equipment_logs.index,
            x=equipment_logs.values,
            orientation='h',
            title="Log Entries by Equipment",
            labels={'x': 'Log Count', 'y': 'Equipment ID'}
        )
        st.plotly_chart(fig_eq_logs, use_container_width=True)
        
        # Operator activity analysis
        st.subheader("👥 Operator Activity Analysis")
        
        operator_activity = self.logs_df['operator_id'].value_counts().head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_operators = px.bar(
                x=operator_activity.values,
                y=operator_activity.index,
                orientation='h',
                title="Top 10 Most Active Operators",
                labels={'x': 'Log Entries', 'y': 'Operator ID'}
            )
            st.plotly_chart(fig_operators, use_container_width=True)
        
        with col2:
            # Log length analysis
            log_lengths = self.logs_df['log_text'].str.len()
            
            fig_length_dist = px.histogram(
                x=log_lengths,
                nbins=50,
                title="Distribution of Log Text Lengths",
                labels={'x': 'Character Count', 'y': 'Frequency'}
            )
            st.plotly_chart(fig_length_dist, use_container_width=True)
        
        # Text analysis - word frequency
        st.subheader("📝 Text Content Analysis")
        
        # Simple word frequency analysis
        all_text = ' '.join(self.logs_df['log_text'].astype(str)).lower()
        words = all_text.split()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        from collections import Counter
        word_freq = Counter(filtered_words).most_common(20)
        
        if word_freq:
            words_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
            
            fig_words = px.bar(
                words_df,
                x='Frequency',
                y='Word',
                orientation='h',
                title="Top 20 Most Frequent Words in Logs",
                labels={'x': 'Frequency', 'y': 'Word'}
            )
            fig_words.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_words, use_container_width=True)
        
        # Sample logs display
        st.subheader("📄 Sample Log Entries")
        
        # Show sample logs by type
        log_type = st.selectbox("Select log type to view samples:", self.logs_df['log_type'].unique())
        sample_logs = self.logs_df[self.logs_df['log_type'] == log_type].head(5)
        
        for idx, row in sample_logs.iterrows():
            with st.expander(f"Log {idx} - {row['timestamp']} - {row['operator_id']}"):
                st.write(f"**Equipment:** {row['equipment_id']}")
                st.write(f"**Text:** {row['log_text']}")
                st.write(f"**Type:** {row['log_type']}")
    
    def _display_anomaly_analysis(self):
        """Display anomaly pattern analysis"""
        st.subheader("⚠️ Anomaly Pattern Analysis")
        
        if self.anomaly_df.empty:
            st.info("No ground truth anomaly data available for analysis.")
            return
        
        # Basic anomaly statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Anomalies", f"{len(self.anomaly_df):,}")
        
        with col2:
            affected_equipment = self.anomaly_df['equipment_id'].nunique()
            st.metric("Affected Equipment", f"{affected_equipment}")
        
        with col3:
            anomaly_rate = (len(self.anomaly_df) / len(self.sensor_df)) * 100
            st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
        
        # Anomaly type distribution
        st.subheader("📊 Anomaly Type Distribution")
        
        anomaly_type_counts = self.anomaly_df['anomaly_type'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_anomaly_types = px.pie(
                values=anomaly_type_counts.values,
                names=anomaly_type_counts.index,
                title="Distribution of Anomaly Types"
            )
            st.plotly_chart(fig_anomaly_types, use_container_width=True)
        
        with col2:
            fig_anomaly_bar = px.bar(
                x=anomaly_type_counts.index,
                y=anomaly_type_counts.values,
                title="Anomaly Count by Type",
                labels={'x': 'Anomaly Type', 'y': 'Count'}
            )
            fig_anomaly_bar.update_xaxes(tickangle=45)
            st.plotly_chart(fig_anomaly_bar, use_container_width=True)
        
        # Severity analysis
        if 'severity' in self.anomaly_df.columns:
            st.subheader("🚨 Anomaly Severity Analysis")
            
            severity_counts = self.anomaly_df['severity'].value_counts()
            
            fig_severity = px.bar(
                x=severity_counts.index,
                y=severity_counts.values,
                title="Anomaly Distribution by Severity",
                labels={'x': 'Severity', 'y': 'Count'},
                color=severity_counts.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_severity, use_container_width=True)
        
        # Temporal anomaly patterns
        st.subheader("📅 Temporal Anomaly Patterns")
        
        anomaly_temp = self.anomaly_df.copy()
        anomaly_temp['timestamp'] = pd.to_datetime(anomaly_temp['timestamp'])
        anomaly_temp['hour'] = anomaly_temp['timestamp'].dt.hour
        anomaly_temp['day_of_week'] = anomaly_temp['timestamp'].dt.day_name()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Anomalies by hour
            hourly_anomalies = anomaly_temp['hour'].value_counts().sort_index()
            
            fig_hourly_anomalies = px.bar(
                x=hourly_anomalies.index,
                y=hourly_anomalies.values,
                title="Anomalies by Hour of Day",
                labels={'x': 'Hour', 'y': 'Anomaly Count'}
            )
            st.plotly_chart(fig_hourly_anomalies, use_container_width=True)
        
        with col2:
            # Equipment-wise anomaly distribution
            equipment_anomalies = self.anomaly_df['equipment_id'].value_counts()
            
            fig_eq_anomalies = px.bar(
                y=equipment_anomalies.index,
                x=equipment_anomalies.values,
                orientation='h',
                title="Anomalies by Equipment",
                labels={'x': 'Anomaly Count', 'y': 'Equipment ID'}
            )
            st.plotly_chart(fig_eq_anomalies, use_container_width=True)
        
        # Anomaly timeline
        st.subheader("📈 Anomaly Timeline")
        
        # Create daily anomaly counts
        daily_anomalies = anomaly_temp.set_index('timestamp').resample('D').size()
        
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=daily_anomalies.index,
            y=daily_anomalies.values,
            mode='lines+markers',
            name='Daily Anomaly Count',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ))
        
        fig_timeline.update_layout(
            title="Anomaly Occurrences Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Anomalies",
            height=400
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    def _display_cross_modal_analysis(self):
        """Display cross-modal analysis between sensors and logs"""
        st.subheader("🔗 Cross-Modal Analysis")
        
        # Time alignment analysis
        st.subheader("⏰ Temporal Alignment")
        
        # Convert timestamps to datetime
        sensor_times = pd.to_datetime(self.sensor_df['timestamp'])
        log_times = pd.to_datetime(self.logs_df['timestamp'])
        
        # Create hourly bins for comparison
        sensor_hourly = sensor_times.dt.floor('H').value_counts().sort_index()
        log_hourly = log_times.dt.floor('H').value_counts().sort_index()
        
        # Align the series
        all_hours = pd.date_range(
            start=min(sensor_hourly.index.min(), log_hourly.index.min()),
            end=max(sensor_hourly.index.max(), log_hourly.index.max()),
            freq='H'
        )
        
        sensor_aligned = sensor_hourly.reindex(all_hours, fill_value=0)
        log_aligned = log_hourly.reindex(all_hours, fill_value=0)
        
        fig_alignment = go.Figure()
        
        fig_alignment.add_trace(go.Scatter(
            x=all_hours,
            y=sensor_aligned.values,
            mode='lines',
            name='Sensor Records',
            line=dict(color='blue', width=2)
        ))
        
        fig_alignment.add_trace(go.Scatter(
            x=all_hours,
            y=log_aligned.values * 10,  # Scale up for visibility
            mode='lines',
            name='Log Entries (×10)',
            line=dict(color='orange', width=2),
            yaxis='y2'
        ))
        
        fig_alignment.update_layout(
            title="Sensor Records vs Log Entries Over Time",
            xaxis_title="Time",
            yaxis_title="Sensor Record Count",
            yaxis2=dict(
                title="Log Entry Count (×10)",
                overlaying='y',
                side='right'
            ),
            height=500
        )
        
        st.plotly_chart(fig_alignment, use_container_width=True)
        
        # Equipment coverage comparison
        st.subheader("🏭 Equipment Coverage Comparison")
        
        sensor_equipment = set(self.sensor_df['equipment_id'].unique())
        log_equipment = set(self.logs_df['equipment_id'].unique())
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Equipment in Sensors", len(sensor_equipment))
        
        with col2:
            st.metric("Equipment in Logs", len(log_equipment))
        
        with col3:
            overlap = len(sensor_equipment.intersection(log_equipment))
            st.metric("Equipment Overlap", f"{overlap} ({overlap/len(sensor_equipment)*100:.1f}%)")
        
        # Coverage visualization
        all_equipment = sensor_equipment.union(log_equipment)
        coverage_data = []
        
        for eq in all_equipment:
            coverage_data.append({
                'Equipment': eq,
                'Has_Sensors': eq in sensor_equipment,
                'Has_Logs': eq in log_equipment,
                'Coverage_Type': (
                    'Both' if eq in sensor_equipment and eq in log_equipment
                    else 'Sensors Only' if eq in sensor_equipment
                    else 'Logs Only'
                )
            })
        
        coverage_df = pd.DataFrame(coverage_data)
        coverage_counts = coverage_df['Coverage_Type'].value_counts()
        
        fig_coverage = px.pie(
            values=coverage_counts.values,
            names=coverage_counts.index,
            title="Equipment Data Coverage",
            color_discrete_map={
                'Both': 'green',
                'Sensors Only': 'blue', 
                'Logs Only': 'orange'
            }
        )
        
        st.plotly_chart(fig_coverage, use_container_width=True)
        
        # Log-Sensor correlation by equipment
        st.subheader("📊 Activity Correlation by Equipment")
        
        # Calculate daily activity for each equipment
        sensor_daily = self.sensor_df.groupby([
            self.sensor_df['timestamp'].dt.date,
            'equipment_id'
        ]).size().reset_index(name='sensor_count')
        sensor_daily.columns = ['date', 'equipment_id', 'sensor_count']
        
        log_daily = self.logs_df.groupby([
            pd.to_datetime(self.logs_df['timestamp']).dt.date,
            'equipment_id'
        ]).size().reset_index(name='log_count')
        log_daily.columns = ['date', 'equipment_id', 'log_count']
        
        # Merge the data
        activity_correlation = pd.merge(
            sensor_daily, log_daily,
            on=['date', 'equipment_id'],
            how='outer'
        ).fillna(0)
        
        if len(activity_correlation) > 0:
            # Calculate correlation for each equipment
            equipment_correlations = []
            
            for eq in activity_correlation['equipment_id'].unique():
                eq_data = activity_correlation[activity_correlation['equipment_id'] == eq]
                if len(eq_data) > 1:
                    corr = eq_data['sensor_count'].corr(eq_data['log_count'])
                    equipment_correlations.append({
                        'Equipment': eq,
                        'Correlation': corr if not pd.isna(corr) else 0,
                        'Data_Points': len(eq_data)
                    })
            
            if equipment_correlations:
                corr_df = pd.DataFrame(equipment_correlations)
                
                fig_corr_eq = px.bar(
                    corr_df,
                    x='Equipment',
                    y='Correlation',
                    title="Sensor-Log Activity Correlation by Equipment",
                    labels={'y': 'Correlation Coefficient'},
                    color='Correlation',
                    color_continuous_scale='RdYlBu'
                )
                fig_corr_eq.update_xaxes(tickangle=45)
                st.plotly_chart(fig_corr_eq, use_container_width=True)
                
                # Show correlation statistics
                avg_corr = corr_df['Correlation'].mean()
                st.metric("Average Sensor-Log Correlation", f"{avg_corr:.3f}")
        
        # Time lag analysis
        st.subheader("⏱️ Time Lag Analysis")
        
        st.info("""
        **Time Lag Analysis**: Examines the temporal relationship between sensor anomalies 
        and operator log entries to understand response times and reporting patterns.
        """)
        
        # If anomaly data is available, analyze lag between anomalies and logs
        if not self.anomaly_df.empty:
            # For each anomaly, find the closest log entry
            anomaly_log_lags = []
            
            for _, anomaly in self.anomaly_df.head(100).iterrows():  # Limit for performance
                anomaly_time = pd.to_datetime(anomaly['timestamp'])
                equipment = anomaly['equipment_id']
                
                # Find logs for the same equipment within ±4 hours
                equipment_logs = self.logs_df[
                    self.logs_df['equipment_id'] == equipment
                ].copy()
                
                if len(equipment_logs) > 0:
                    equipment_logs['log_time'] = pd.to_datetime(equipment_logs['timestamp'])
                    equipment_logs['time_diff'] = (equipment_logs['log_time'] - anomaly_time).dt.total_seconds() / 3600
                    
                    # Find logs within ±4 hours
                    nearby_logs = equipment_logs[
                        equipment_logs['time_diff'].abs() <= 4
                    ]
                    
                    if len(nearby_logs) > 0:
                        # Get the closest log
                        closest_log = nearby_logs.loc[nearby_logs['time_diff'].abs().idxmin()]
                        anomaly_log_lags.append({
                            'anomaly_time': anomaly_time,
                            'log_time': closest_log['log_time'],
                            'lag_hours': closest_log['time_diff'],
                            'equipment': equipment,
                            'log_type': closest_log['log_type']
                        })
            
            if anomaly_log_lags:
                lag_df = pd.DataFrame(anomaly_log_lags)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Lag distribution
                    fig_lag_dist = px.histogram(
                        lag_df,
                        x='lag_hours',
                        nbins=30,
                        title="Distribution of Anomaly-Log Time Lags",
                        labels={'x': 'Time Lag (hours)', 'y': 'Frequency'}
                    )
                    fig_lag_dist.add_vline(x=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_lag_dist, use_container_width=True)
                
                with col2:
                    # Average lag by log type
                    avg_lag_by_type = lag_df.groupby('log_type')['lag_hours'].mean().sort_values()
                    
                    fig_lag_type = px.bar(
                        x=avg_lag_by_type.values,
                        y=avg_lag_by_type.index,
                        orientation='h',
                        title="Average Time Lag by Log Type",
                        labels={'x': 'Average Lag (hours)', 'y': 'Log Type'}
                    )
                    st.plotly_chart(fig_lag_type, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    median_lag = lag_df['lag_hours'].median()
                    st.metric("Median Response Time", f"{median_lag:.1f} hours")
                
                with col2:
                    proactive_logs = (lag_df['lag_hours'] < 0).sum()
                    proactive_pct = (proactive_logs / len(lag_df)) * 100
                    st.metric("Proactive Logs", f"{proactive_pct:.1f}%")
                
                with col3:
                    reactive_logs = (lag_df['lag_hours'] > 0).sum()
                    reactive_pct = (reactive_logs / len(lag_df)) * 100
                    st.metric("Reactive Logs", f"{reactive_pct:.1f}%")
        
        # Data quality cross-check
        st.subheader("✅ Data Quality Cross-Check")
        
        # Check for equipment with sensors but no logs (and vice versa)
        sensors_only = sensor_equipment - log_equipment
        logs_only = log_equipment - sensor_equipment
        
        col1, col2 = st.columns(2)
        
        with col1:
            if sensors_only:
                st.warning(f"**Equipment with sensors but no logs:** {len(sensors_only)}")
                with st.expander("Show equipment list"):
                    for eq in sorted(sensors_only):
                        st.write(f"• {eq}")
            else:
                st.success("✅ All equipment with sensors have associated logs")
        
        with col2:
            if logs_only:
                st.warning(f"**Equipment with logs but no sensors:** {len(logs_only)}")
                with st.expander("Show equipment list"):
                    for eq in sorted(logs_only):
                        st.write(f"• {eq}")
            else:
                st.success("✅ All equipment with logs have associated sensors")
        
        # Temporal coverage gaps
        st.subheader("🕐 Temporal Coverage Analysis")
        
        # Find gaps in data collection
        sensor_time_range = (sensor_times.max() - sensor_times.min()).total_seconds() / 3600
        log_time_range = (log_times.max() - log_times.min()).total_seconds() / 3600
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Sensor Data Span", f"{sensor_time_range:.1f} hours")
            
            # Calculate data collection rate
            expected_records = sensor_time_range * len(sensor_equipment)  # Assuming hourly data
            actual_records = len(self.sensor_df)
            coverage_rate = (actual_records / expected_records) * 100 if expected_records > 0 else 0
            
            st.metric("Sensor Coverage Rate", f"{coverage_rate:.1f}%")
        
        with col2:
            st.metric("Log Data Span", f"{log_time_range:.1f} hours")
            
            # Calculate average logs per day
            log_days = (log_times.max() - log_times.min()).days + 1
            avg_logs_per_day = len(self.logs_df) / log_days if log_days > 0 else 0
            
            st.metric("Avg Logs per Day", f"{avg_logs_per_day:.1f}")

# Integration function to add EDA to main Streamlit app
def add_eda_to_main_app():
    """
    Integration function to add EDA functionality to the main Streamlit app.
    Add this to your main oil_rig_anomaly_system.py file.
    """
    
    # Add this to your main() function after data generation
    def display_eda_section():
        if st.sidebar.button("📊 Run Exploratory Data Analysis"):
            if 'sensor_df' in st.session_state and 'logs_df' in st.session_state:
                with st.spinner("Performing exploratory data analysis..."):
                    # Create EDA instance
                    eda = ExploratoryDataAnalysis(
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
    
    return display_eda_section