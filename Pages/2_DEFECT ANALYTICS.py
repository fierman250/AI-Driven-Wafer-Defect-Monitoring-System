"""
Analytics Page - Statistics & Charts
"""
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Add Repository to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Repository"))

from Repository.Data_Aggregator import DataAggregator
from Repository.config_LLM import DEFECT_PERCENTAGE_THRESHOLD

# Page code runs directly (no show() function needed for Streamlit pages)
st.title("📊 Analytics - Statistics & Charts")

# Initialize data aggregator
aggregator = DataAggregator()
aggregator.load_results()

# Ensure simulation_date is in the dataframe if data exists
if aggregator.df is not None and aggregator.data and 'simulation_date' not in aggregator.df.columns:
    sim_dates = [r.get('simulation_date') for r in aggregator.data]
    aggregator.df['simulation_date'] = sim_dates

# Get available simulation dates
available_dates = aggregator.get_available_simulation_dates()

# Debug: Show available dates in sidebar
if available_dates:
    st.sidebar.info(f"📅 Found {len(available_dates)} simulation date(s): {', '.join(available_dates[:3])}{'...' if len(available_dates) > 3 else ''}")
else:
    st.sidebar.warning("⚠️ No simulation dates found in data. Check if simulation_date field exists in results.")

# Date selection
if available_dates:
    selected_date = st.selectbox(
        "📅 Select Simulation Date",
        options=["All Dates"] + available_dates,
        index=0,
        help="Select which simulation date's data to analyze, or 'All Dates' for combined analysis"
    )
    
    if selected_date == "All Dates":
        filtered_data = aggregator.data if aggregator.data else []
        display_aggregator = aggregator
        # Ensure df has simulation_date column if data exists
        if display_aggregator.df is not None and filtered_data:
            if 'simulation_date' not in display_aggregator.df.columns:
                sim_dates = [r.get('simulation_date') for r in filtered_data]
                display_aggregator.df['simulation_date'] = sim_dates
    else:
        filtered_data = aggregator.filter_by_simulation_date(selected_date)
        # Create temporary aggregator with filtered data
        temp_aggregator = DataAggregator()
        temp_aggregator.data = filtered_data
        if filtered_data:
            temp_aggregator.df = pd.DataFrame(filtered_data)
            if 'timestamp' in temp_aggregator.df.columns:
                temp_aggregator.df['timestamp'] = pd.to_datetime(temp_aggregator.df['timestamp'])
            # Ensure simulation_date is in the dataframe
            if 'simulation_date' not in temp_aggregator.df.columns:
                sim_dates = [r.get('simulation_date') for r in filtered_data]
                temp_aggregator.df['simulation_date'] = sim_dates
        display_aggregator = temp_aggregator
else:
    selected_date = "All Dates"
    filtered_data = aggregator.data if aggregator.data else []
    display_aggregator = aggregator
    # Ensure df is properly set up with simulation_date
    if display_aggregator.df is not None and filtered_data and 'simulation_date' not in display_aggregator.df.columns:
        sim_dates = [r.get('simulation_date') for r in filtered_data]
        display_aggregator.df['simulation_date'] = sim_dates
    if not filtered_data:
        st.info("📅 No simulation dates available. Run a simulation first.")

# Sidebar filters
st.sidebar.markdown("### Filters")

# Date range filter (for timestamp-based filtering)
# Use simulation_date for date range, but allow filtering by actual timestamp if needed
date_range = None
original_min_date = None
original_max_date = None

# First, try to get date range from simulation_date (more meaningful for multi-day simulations)
if aggregator.data and aggregator.df is not None and 'simulation_date' in aggregator.df.columns:
    try:
        # Get unique simulation dates and convert to date objects
        sim_dates = aggregator.df['simulation_date'].dropna().unique()
        if len(sim_dates) > 0:
            # Convert simulation_date strings to date objects
            from datetime import datetime as dt
            date_objects = []
            for sd in sim_dates:
                try:
                    if isinstance(sd, str):
                        date_objects.append(dt.strptime(sd, '%Y-%m-%d').date())
                    elif hasattr(sd, 'date'):
                        date_objects.append(sd.date())
                except:
                    continue
            
            if date_objects:
                original_min_date = min(date_objects)
                original_max_date = max(date_objects)
            else:
                # Fallback to timestamp if simulation_date parsing fails
                if 'timestamp' in aggregator.df.columns:
                    original_min_date = aggregator.df['timestamp'].min().date()
                    original_max_date = aggregator.df['timestamp'].max().date()
        else:
            # Fallback to timestamp if no simulation_date
            if 'timestamp' in aggregator.df.columns:
                original_min_date = aggregator.df['timestamp'].min().date()
                original_max_date = aggregator.df['timestamp'].max().date()
    except Exception as e:
        # Fallback to timestamp
        try:
            if aggregator.df is not None and 'timestamp' in aggregator.df.columns:
                original_min_date = aggregator.df['timestamp'].min().date()
                original_max_date = aggregator.df['timestamp'].max().date()
        except:
            pass

# If we still don't have dates, try timestamp
if original_min_date is None or original_max_date is None:
    if aggregator.data and aggregator.df is not None and 'timestamp' in aggregator.df.columns:
        try:
            original_min_date = aggregator.df['timestamp'].min().date()
            original_max_date = aggregator.df['timestamp'].max().date()
        except Exception as e:
            st.sidebar.warning(f"⚠️ Error getting date range: {str(e)}")

# Set up the date input widget
if original_min_date and original_max_date:
    try:
        # Calculate the current filtered data's date range for default value
        if display_aggregator.data and display_aggregator.df is not None:
            # Try simulation_date first
            if 'simulation_date' in display_aggregator.df.columns:
                sim_dates = display_aggregator.df['simulation_date'].dropna().unique()
                if len(sim_dates) > 0:
                    from datetime import datetime as dt
                    date_objects = []
                    for sd in sim_dates:
                        try:
                            if isinstance(sd, str):
                                date_objects.append(dt.strptime(sd, '%Y-%m-%d').date())
                            elif hasattr(sd, 'date'):
                                date_objects.append(sd.date())
                        except:
                            continue
                    if date_objects:
                        current_min_date = min(date_objects)
                        current_max_date = max(date_objects)
                        default_value = (current_min_date, current_max_date)
                    else:
                        default_value = (original_min_date, original_max_date)
                else:
                    default_value = (original_min_date, original_max_date)
            elif 'timestamp' in display_aggregator.df.columns:
                current_min_date = display_aggregator.df['timestamp'].min().date()
                current_max_date = display_aggregator.df['timestamp'].max().date()
                default_value = (current_min_date, current_max_date)
            else:
                default_value = (original_min_date, original_max_date)
        else:
            default_value = (original_min_date, original_max_date)
        
        date_range = st.sidebar.date_input(
            "Timestamp Range",
            value=default_value,
            min_value=original_min_date,
            max_value=original_max_date,
            help="Filter by date. Uses simulation_date if available, otherwise uses timestamp. Select a date range or a single date."
        )
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error setting up date filter: {str(e)}")
        date_range = None
else:
    if not aggregator.data:
        st.sidebar.info("No date data available")

# Machine type filter
st.sidebar.markdown("### Machine Type")
filter_all = st.sidebar.checkbox("All", value=True)
filter_mechanical = st.sidebar.checkbox("Mechanical", disabled=filter_all)
filter_electrical = st.sidebar.checkbox("Electrical", disabled=filter_all)
filter_thermal = st.sidebar.checkbox("Thermal", disabled=filter_all)

# Apply machine type filter to already filtered data
if filter_all:
    machine_types = ["Mechanical", "Electrical", "Thermal"]
else:
    machine_types = []
    if filter_mechanical:
        machine_types.append("Mechanical")
    if filter_electrical:
        machine_types.append("Electrical")
    if filter_thermal:
        machine_types.append("Thermal")

if machine_types and filtered_data:
    filtered_data = [r for r in filtered_data if r.get('machine_type') in machine_types]

# Apply date range filter if specified
# Filter by simulation_date if available, otherwise by timestamp
if date_range and filtered_data:
    try:
        # Handle different date_range formats from st.date_input
        start_date = None
        end_date = None
        
        if isinstance(date_range, tuple) and len(date_range) == 2:
            # Range selected: (start_date, end_date)
            start_date, end_date = date_range
        elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
            # Single date selected (as tuple/list with one element)
            start_date = end_date = date_range[0]
        elif hasattr(date_range, 'date'):
            # Single date object
            start_date = end_date = date_range.date()
        elif isinstance(date_range, (list, tuple)) and len(date_range) > 0:
            # Fallback: use first element
            start_date = end_date = date_range[0] if hasattr(date_range[0], 'date') else date_range[0]
        
        if start_date and end_date:
            # Filter by simulation_date first (if available), otherwise by timestamp
            filtered_data_new = []
            for r in filtered_data:
                # Try simulation_date first
                sim_date = r.get('simulation_date')
                if sim_date:
                    try:
                        from datetime import datetime as dt
                        if isinstance(sim_date, str):
                            record_date = dt.strptime(sim_date, '%Y-%m-%d').date()
                        elif hasattr(sim_date, 'date'):
                            record_date = sim_date.date()
                        else:
                            record_date = None
                        
                        if record_date and start_date <= record_date <= end_date:
                            filtered_data_new.append(r)
                            continue
                    except:
                        pass
                
                # Fallback to timestamp if simulation_date not available or parsing failed
                if r.get('timestamp'):
                    try:
                        record_date = pd.to_datetime(r.get('timestamp')).date()
                        if start_date <= record_date <= end_date:
                            filtered_data_new.append(r)
                    except:
                        pass
            
            filtered_data = filtered_data_new
            
            # Update display_aggregator with filtered data
            if filtered_data:
                temp_aggregator = DataAggregator()
                temp_aggregator.data = filtered_data
                temp_aggregator.df = pd.DataFrame(filtered_data)
                if 'timestamp' in temp_aggregator.df.columns:
                    temp_aggregator.df['timestamp'] = pd.to_datetime(temp_aggregator.df['timestamp'])
                if 'simulation_date' not in temp_aggregator.df.columns:
                    sim_dates = [r.get('simulation_date') for r in filtered_data]
                    temp_aggregator.df['simulation_date'] = sim_dates
                display_aggregator = temp_aggregator
            else:
                st.warning("⚠️ No data available for the selected date range. Please adjust the filter.")
    except Exception as e:
        # If filtering fails, show warning but continue with unfiltered data
        st.sidebar.warning(f"⚠️ Error applying timestamp filter: {str(e)}")

if not filtered_data:
    st.warning("No data available with current filters.")
    st.info("Run the manufacturing simulation first or select a different date.")
    st.stop()

# Advanced Analytics Metrics Section
st.markdown("### 📈 Advanced Analytics Metrics")

if filtered_data and len(filtered_data) > 0:
    # Calculate statistical metrics
    defect_percentages = [r.get('defect_percentage', 0) for r in filtered_data if r.get('defect_percentage') is not None]
    confidences = [r.get('prediction', {}).get('Confidence Score', 0) for r in filtered_data if r.get('prediction', {}).get('Confidence Score') is not None]
    
    if defect_percentages:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Mean Defect %", f"{np.mean(defect_percentages):.2f}%")
        with col2:
            st.metric("Median Defect %", f"{np.median(defect_percentages):.2f}%")
        with col3:
            st.metric("Std Deviation", f"{np.std(defect_percentages):.2f}%")
        with col4:
            st.metric("Min Defect %", f"{np.min(defect_percentages):.2f}%")
        with col5:
            st.metric("Max Defect %", f"{np.max(defect_percentages):.2f}%")
        
        if confidences:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Confidence", f"{np.mean(confidences):.4f}")
            with col2:
                st.metric("Min Confidence", f"{np.min(confidences):.4f}")
            with col3:
                st.metric("Max Confidence", f"{np.max(confidences):.4f}")

st.markdown("---")

# Machine Performance Comparison
st.markdown("### Machine Performance Comparison")

machine_stats = display_aggregator.get_machine_statistics()
if machine_stats:
    # Prepare data for bar chart
    machines = []
    pass_rates = []
    
    for machine, stats in machine_stats.items():
        if not machine_types or any(mt in machine for mt in machine_types):
            machines.append(machine)
            pass_rates.append(stats.get('pass_rate', 0))
    
    if machines:
        fig = px.bar(
            x=machines,
            y=pass_rates,
            title="Pass Rate by Machine",
            labels={'x': 'Machine', 'y': 'Pass Rate (%)'},
            color=pass_rates,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Defect Percentage Over Time
st.markdown("### Defect Percentage Over Time")

if display_aggregator.df is not None and 'defect_percentage' in display_aggregator.df.columns:
    df_time = display_aggregator.df.copy()
    
    # Filter by machine type
    if machine_types:
        df_time = df_time[df_time['machine_type'].isin(machine_types)]
    
    # Determine grouping method based on date selection
    if selected_date == "All Dates" and 'simulation_date' in df_time.columns:
        # When "All Dates" is selected, group by simulation_date to show trends across days
        if not df_time.empty and df_time['simulation_date'].notna().any():
            # Filter out rows without simulation_date
            df_time = df_time[df_time['simulation_date'].notna()]
            if not df_time.empty:
                # Group by simulation_date and calculate average
                df_time_grouped = df_time.groupby('simulation_date')['defect_percentage'].mean().reset_index()
                df_time_grouped.columns = ['Date', 'Avg Defect %']
                df_time_grouped = df_time_grouped.sort_values('Date')  # Sort by date
                
                fig = px.line(
                    df_time_grouped,
                    x='Date',
                    y='Avg Defect %',
                    title="Average Defect Percentage by Simulation Date",
                    markers=True
                )
                fig.update_traces(line_color='#1E88E5', line_width=2)
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No simulation date data available for time series")
        else:
            st.info("No simulation date data available. Data may not have simulation_date field.")
    elif 'timestamp' in df_time.columns:
        # When a specific date is selected, use timestamp for intra-day analysis
        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            df_time = df_time[
                (df_time['timestamp'].dt.date >= date_range[0]) &
                (df_time['timestamp'].dt.date <= date_range[1])
            ]
        
        if not df_time.empty:
            # Group by timestamp date and calculate average
            df_time_grouped = df_time.groupby(df_time['timestamp'].dt.date)['defect_percentage'].mean().reset_index()
            df_time_grouped.columns = ['Date', 'Avg Defect %']
            df_time_grouped = df_time_grouped.sort_values('Date')
            
            fig = px.line(
                df_time_grouped,
                x='Date',
                y='Avg Defect %',
                title="Average Defect Percentage Over Time (Within Selected Date)",
                markers=True
            )
            fig.update_traces(line_color='#1E88E5', line_width=2)
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No time series data available for the selected filters")
    else:
        st.info("No timestamp or simulation_date data available for time series")
else:
    st.info("No defect percentage data available for time series")

# Correlation Analysis - Machine vs Defect Percentage
if filtered_data and len(filtered_data) > 0:
    st.markdown("---")
    st.markdown("### 🔗 Correlation Analysis: Machine vs Defect Percentage")
    
    # Prepare data for analysis
    machine_defect_data = []
    for result in filtered_data:
        defect_pct = result.get('defect_percentage', 0)
        machine_type = result.get('machine_type', 'Unknown')
        machine_id = result.get('machine_id', 'Unknown')
        machine_name = f"{machine_type} {machine_id}"
        
        if defect_pct is not None:
            machine_defect_data.append({
                'Machine Type': machine_type,
                'Machine': machine_name,
                'Defect %': defect_pct
            })
    
    if machine_defect_data and len(machine_defect_data) > 0:
        df_corr = pd.DataFrame(machine_defect_data)
        
        # Box plot: Defect % by Machine Type
        col1, col2 = st.columns(2)
        with col1:
            fig_box = px.box(
                df_corr,
                x='Machine Type',
                y='Defect %',
                color='Machine Type',
                title="Defect Percentage Distribution by Machine Type",
                labels={'Defect %': 'Defect Percentage (%)', 'Machine Type': 'Machine Type'}
            )
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Calculate average defect percentage by machine type
            machine_stats = df_corr.groupby('Machine Type')['Defect %'].agg(['mean', 'median', 'std', 'count']).reset_index()
            machine_stats.columns = ['Machine Type', 'Mean Defect %', 'Median Defect %', 'Std Deviation', 'Count']
            machine_stats = machine_stats.sort_values('Mean Defect %', ascending=False)
            
            st.markdown("#### Statistics by Machine Type")
            st.dataframe(
                machine_stats.style.format({
                    'Mean Defect %': '{:.2f}%',
                    'Median Defect %': '{:.2f}%',
                    'Std Deviation': '{:.2f}%',
                    'Count': '{:.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        
        # Violin plot for detailed distribution
        if len(df_corr['Machine Type'].unique()) > 1:
            fig_violin = px.violin(
                df_corr,
                x='Machine Type',
                y='Defect %',
                color='Machine Type',
                title="Defect Percentage Distribution (Violin Plot) by Machine Type",
                labels={'Defect %': 'Defect Percentage (%)', 'Machine Type': 'Machine Type'},
                box=True
            )
            fig_violin.update_layout(showlegend=False)
            st.plotly_chart(fig_violin, use_container_width=True)

st.markdown("---")

# Defect Distribution by Machine Type
st.markdown("### Defect Distribution by Machine Type")

if filtered_data:
    # Prepare data
    machine_defect_data = {}
    for result in filtered_data:
        machine_type = result.get('machine_type', 'Unknown')
        defect_class = result.get('prediction', {}).get('Defect Class', 'Unknown')
        
        if machine_type not in machine_defect_data:
            machine_defect_data[machine_type] = {}
        
        machine_defect_data[machine_type][defect_class] = machine_defect_data[machine_type].get(defect_class, 0) + 1
    
    # Create stacked bar chart
    if machine_defect_data:
        df_stack = pd.DataFrame(machine_defect_data).fillna(0)
        fig = px.bar(
            df_stack,
            title="Defect Class Distribution by Machine Type",
            labels={'value': 'Count', 'index': 'Defect Class'},
            barmode='stack'
        )
        st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Top Defect Classes
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Top Defect Classes")
    defect_dist = display_aggregator.get_defect_distribution()
    
    if defect_dist and 'counts' in defect_dist:
        # Sort by count
        sorted_defects = sorted(
            defect_dist['counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        defect_data = {
            "Class": [d[0] for d in sorted_defects],
            "Count": [d[1] for d in sorted_defects],
            "Percentage": [defect_dist['percentages'].get(d[0], 0) for d in sorted_defects]
        }
        
        df_defects = pd.DataFrame(defect_data)
        st.dataframe(df_defects, width='stretch', hide_index=True)
    else:
        st.info("No defect distribution data")

with col2:
    st.markdown(f"### Anomalies (>{DEFECT_PERCENTAGE_THRESHOLD}% defect)")
    anomalies = display_aggregator.get_anomalies(threshold_percentage=DEFECT_PERCENTAGE_THRESHOLD)
    
    if anomalies:
        anomaly_data = []
        for anomaly in anomalies[:20]:  # Show top 20
            anomaly_data.append({
                "Wafer ID": anomaly.get('wafer_id', 'N/A'),
                "Defect %": f"{anomaly.get('defect_percentage', 0):.2f}%",
                "Class": anomaly.get('prediction', {}).get('Defect Class', 'Unknown'),
                "Machine": anomaly.get('machine_type', 'Unknown')
            })
        
        df_anomalies = pd.DataFrame(anomaly_data)
        st.dataframe(df_anomalies, width='stretch', hide_index=True)
    else:
        st.info("No anomalies detected")

# Daily Comparison Section
if available_dates and len(available_dates) > 1:
    st.markdown("---")
    st.markdown("### 📊 Daily Performance Comparison")
    
    # Get statistics for each date
    daily_comparison = []
    for date in available_dates:
        daily_stats = aggregator.get_daily_statistics(date)
        if "error" not in daily_stats:
            daily_comparison.append({
                "Date": date,
                "Total Wafers": daily_stats.get('total_wafers', 0),
                "Pass Rate": f"{daily_stats.get('pass_rate', 0):.1f}%",
                "Fail Rate": f"{daily_stats.get('fail_rate', 0):.1f}%",
                "Avg Defect %": f"{daily_stats.get('average_defect_percentage', 0):.1f}%",
                "Avg Confidence": f"{daily_stats.get('average_confidence', 0):.4f}"
            })
    
    if daily_comparison:
        df_daily = pd.DataFrame(daily_comparison)
        st.dataframe(df_daily, use_container_width=True, hide_index=True)
        
        # Create trend chart for numeric values
        if len(daily_comparison) > 1:
            # Prepare data for chart (convert percentage strings to numeric)
            chart_data = []
            for row in daily_comparison:
                chart_data.append({
                    "Date": row["Date"],
                    "Pass Rate": float(row["Pass Rate"].replace("%", "")),
                    "Fail Rate": float(row["Fail Rate"].replace("%", "")),
                    "Avg Defect %": float(row["Avg Defect %"].replace("%", "")),
                    "Total Wafers": row["Total Wafers"]
                })
            df_chart = pd.DataFrame(chart_data)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['Pass Rate'], mode='lines+markers', name='Pass Rate (%)', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['Fail Rate'], mode='lines+markers', name='Fail Rate (%)', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['Avg Defect %'], mode='lines+markers', name='Avg Defect %', line=dict(color='blue')))
            fig.update_layout(
                title="Daily Performance Trends",
                xaxis_title="Date",
                yaxis_title="Percentage (%)",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    if available_dates:
        st.info("📊 No data available for the selected date. Try selecting a different date or 'All Dates'.")
    else:
        st.info("📊 No data available. Run a simulation first.")

# Export button
st.markdown("---")
if st.button("📥 Export Data to CSV"):
    if display_aggregator.df is not None:
        csv = display_aggregator.df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"wafer_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Navigation buttons at the bottom
st.markdown("---")
st.markdown("### 🔗 Page Navigation")
st.info("💡 Use the buttons below to navigate between pages")
col_nav = st.columns(3)
with col_nav[0]:
    if st.button("🏠 Welcome Page", use_container_width=True, type="secondary"):
        st.switch_page("WELCOME.py")
with col_nav[1]:
    if st.button("← Previous: 🏠 Dashboard", use_container_width=True, type="secondary"):
        st.switch_page("Pages/1_DASHBOARD.py")
with col_nav[2]:
    if st.button("Next: 🤖 AI Assistant →", use_container_width=True, type="primary"):
        st.switch_page("Pages/3_AI_ASSISTANT.py")

