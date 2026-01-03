"""
Dashboard Page - Real-Time Monitoring
"""
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import time
import threading

# Add Repository to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Repository"))

from Repository.Data_Aggregator import DataAggregator
from Repository.Manufacturing_Simulation import ManufacturingProcessController
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page code runs directly (no show() function needed for Streamlit pages)
st.title("🏠 Dashboard - Real-Time Monitoring")

# Initialize session state for simulation control
if 'simulation_controller' not in st.session_state:
    st.session_state.simulation_controller = None
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'simulation_thread' not in st.session_state:
    st.session_state.simulation_thread = None
if 'simulation_finished' not in st.session_state:
    st.session_state.simulation_finished = False
if 'simulation_start_time' not in st.session_state:
    st.session_state.simulation_start_time = None
if 'simulation_duration' not in st.session_state:
    st.session_state.simulation_duration = None


# Simulation Control Section - always visible
st.markdown("### Simulation Control")
    
# Simulation configuration
with st.expander("⚙️ Simulation Settings", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        num_mechanical = st.number_input("Mechanical Machines", min_value=1, max_value=10, value=2)
    with col2:
        num_electrical = st.number_input("Electrical Machines", min_value=1, max_value=10, value=2)
    with col3:
        num_thermal = st.number_input("Thermal Machines", min_value=1, max_value=10, value=2)
    
    col1, col2 = st.columns(2)
    with col1:
        sim_duration = st.number_input("Duration (seconds)", min_value=10, max_value=3600, value=300)
    with col2:
        max_wafers = st.number_input("Max Wafers (0 = unlimited)", min_value=0, value=0)
        max_wafers = None if max_wafers == 0 else max_wafers
    
    # Simulation date input
    sim_date = st.date_input(
        "Simulation Date",
        value=datetime.now().date(),
        help="Date to associate with this simulation run"
    )

# Quick Actions
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ Start Simulation", type="primary", width='stretch', disabled=st.session_state.simulation_running):
        try:
            # Create controller
            controller = ManufacturingProcessController(
                num_mechanical=int(num_mechanical),
                num_electrical=int(num_electrical),
                num_thermal=int(num_thermal)
            )
            
            # Store in session state
            st.session_state.simulation_controller = controller
            st.session_state.simulation_running = True
            st.session_state.simulation_start_time = time.time()
            st.session_state.simulation_duration = int(sim_duration)
            
            # Run simulation in background thread
            def run_sim():
                try:
                    controller.run_simulation(
                        duration_seconds=int(sim_duration),
                        max_wafers=max_wafers,
                        simulation_date=sim_date.strftime("%Y-%m-%d")
                    )
                except Exception as e:
                    st.error(f"Simulation error: {str(e)}")
                finally:
                    # Simulation finished - update session state
                    st.session_state.simulation_running = False
                    st.session_state.simulation_controller = None
                    st.session_state.simulation_thread = None
                    st.session_state.simulation_start_time = None
                    st.session_state.simulation_duration = None
                    st.session_state.simulation_finished = True  # Flag to trigger rerun
            
            thread = threading.Thread(target=run_sim, daemon=True)
            thread.start()
            st.session_state.simulation_thread = thread
            
            st.success("✅ Simulation started!")
            st.info("The simulation is running in the background. Results will appear as wafers are processed.")
            time.sleep(0.5)  # Brief pause to show message
            st.rerun()
        except Exception as e:
            st.error(f"Error starting simulation: {str(e)}")

with col2:
    if st.button("⏹️ Stop Simulation", width='stretch', disabled=not st.session_state.simulation_running):
        if st.session_state.simulation_controller:
            try:
                st.session_state.simulation_controller.stop_all_machines()
                st.session_state.simulation_running = False
                st.session_state.simulation_controller = None
                st.session_state.simulation_thread = None
                st.session_state.simulation_start_time = None
                st.session_state.simulation_duration = None
                st.success("✅ Simulation stopped!")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Error stopping simulation: {str(e)}")
        else:
            st.warning("No simulation running")

with col3:
    if st.button("🔄 Refresh Data", width='stretch'):
        st.rerun()

# Check if simulation duration has elapsed (auto-stop detection)
if st.session_state.simulation_running:
    should_stop = False
    
    # Check if duration has elapsed
    if (st.session_state.simulation_start_time is not None and 
        st.session_state.simulation_duration is not None):
        elapsed_time = time.time() - st.session_state.simulation_start_time
        if elapsed_time >= st.session_state.simulation_duration:
            should_stop = True
    
    # Also check if controller is no longer running (backup check)
    if not should_stop and st.session_state.simulation_controller:
        try:
            if not st.session_state.simulation_controller.is_running:
                should_stop = True
        except:
            pass
    
    # Also check if thread is no longer alive (backup check)
    if not should_stop and st.session_state.simulation_thread is not None:
        if not st.session_state.simulation_thread.is_alive():
            should_stop = True
    
    # Stop the simulation if any condition is met
    if should_stop:
        if st.session_state.simulation_controller:
            try:
                st.session_state.simulation_controller.stop_all_machines()
            except:
                pass
        st.session_state.simulation_running = False
        st.session_state.simulation_controller = None
        if st.session_state.simulation_thread:
            st.session_state.simulation_thread = None
        st.session_state.simulation_start_time = None
        st.session_state.simulation_duration = None
        st.session_state.simulation_finished = True

# Handle simulation finished flag
if st.session_state.simulation_finished:
    st.success("✅ **Simulation completed automatically** - Duration finished. Results are available below.")
    st.session_state.simulation_finished = False  # Reset flag
    # Small delay to show the message
    time.sleep(0.5)

# Simulation status
if st.session_state.simulation_running:
    st.info("🟢 **Simulation is running** - Processing wafers in the background...")
else:
    st.info("⚪ **Simulation stopped** - Click 'Start Simulation' to begin")
    
st.markdown("---")

st.markdown("### Key Performance Indicators")

# Initialize data aggregator
aggregator = DataAggregator()
aggregator.load_results()

# Get available simulation dates
available_dates = aggregator.get_available_simulation_dates()

# Date selection
if available_dates:
    selected_date = st.selectbox(
        "📅 Select Simulation Date",
        options=available_dates,
        index=0,
        help="Select which simulation date's data to view"
    )
    # Filter data by selected date
    filtered_data = aggregator.filter_by_simulation_date(selected_date)
    # Create temporary aggregator with filtered data
    temp_aggregator = DataAggregator()
    temp_aggregator.data = filtered_data if filtered_data else []
    if filtered_data:
        temp_aggregator.df = pd.DataFrame(filtered_data)
        if 'timestamp' in temp_aggregator.df.columns:
            temp_aggregator.df['timestamp'] = pd.to_datetime(temp_aggregator.df['timestamp'])
    # Use filtered aggregator for display
    display_aggregator = temp_aggregator
    if not filtered_data:
        st.info(f"📅 No data available for {selected_date}. Select a different date or run a new simulation.")
else:
    selected_date = None
    display_aggregator = aggregator
    if not aggregator.data:
        st.info("📅 No simulation dates available. Run a simulation first.")

# Auto-refresh control
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    auto_refresh = st.checkbox("🔄 Auto-refresh (5s)", value=True)
with col2:
    if st.button("🔄 Refresh Now"):
        st.rerun()

# Get summary statistics - handle no data gracefully
stats = display_aggregator.get_summary_statistics()
has_data = isinstance(stats, dict) and "error" not in stats and stats.get('total_wafers', 0) > 0

# Get worst performing machine for warning
worst_machine_name = None
worst_machine_pass_rate = None
if has_data:
    machine_stats = display_aggregator.get_machine_statistics()
    if machine_stats:
        # Find machine with lowest pass rate
        worst_machine = min(machine_stats.items(), key=lambda x: x[1].get('pass_rate', 100))
        worst_machine_name = worst_machine[0]
        worst_machine_pass_rate = worst_machine[1].get('pass_rate', 0)

# KPI Cards - always show, with zeros if no data
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Total Wafers",
        value=f"{stats.get('total_wafers', 0):,}" if has_data else "0",
        delta=None
    )

with col2:
    pass_rate = stats.get('pass_rate', 0) if has_data else 0
    st.metric(
        label="Pass Rate",
        value=f"{pass_rate:.1f}%",
        delta=f"{stats.get('pass_count', 0)} wafers" if has_data else "0 wafers"
    )

with col3:
    fail_rate = stats.get('fail_rate', 0) if has_data else 0
    st.metric(
        label="Fail Rate",
        value=f"{fail_rate:.1f}%",
        delta=f"{stats.get('fail_count', 0)} wafers" if has_data else "0 wafers",
        delta_color="inverse"
    )

with col4:
    st.metric(
        label="Avg Defect %",
        value=f"{stats.get('average_defect_percentage', 0):.1f}%" if has_data else "0.0%"
    )

with col5:
    st.metric(
        label="Avg Confidence",
        value=f"{stats.get('average_confidence', 0):.4f}" if has_data else "0.0000"
    )

# Warning box for worst performing machine
if worst_machine_name and worst_machine_pass_rate is not None and worst_machine_pass_rate < 70:  # Warning if pass rate below 70%
    st.warning(f"⚠️ **Performance Alert:** {worst_machine_name} has the lowest pass rate at {worst_machine_pass_rate:.1f}%. Consider investigating this machine.")

st.markdown("---")

# Charts Row - always show, with empty states if no data
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Defect Class Distribution")
    defect_dist = display_aggregator.get_defect_distribution()
    
    if has_data and defect_dist and 'counts' in defect_dist and len(defect_dist['counts']) > 0:
        # Create pie chart
        fig = px.pie(
            values=list(defect_dist['counts'].values()),
            names=list(defect_dist['counts'].keys()),
            title="Defect Classes",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, width='stretch')
    else:
        # Show empty state
        st.info("📊 No defect data available. Start the simulation to see defect distribution.")

with col2:
    st.markdown("### Machine Status")
    machine_stats = display_aggregator.get_machine_statistics()
    
    if has_data and machine_stats and len(machine_stats) > 0:
        # Create machine status table
        machine_data = []
        for machine, data in machine_stats.items():
            machine_data.append({
                "Machine": machine,
                "Total": data.get('total_wafers', 0),
                "Pass Rate": f"{data.get('pass_rate', 0):.1f}%",
                "Avg Defect %": f"{data.get('average_defect_percentage', 0):.1f}%"
            })
        
        df_machines = pd.DataFrame(machine_data)
        st.dataframe(df_machines, width='stretch', hide_index=True)
    else:
        # Show empty state
        st.info("📊 No machine data available. Start the simulation to see machine statistics.")

st.markdown("---")

# Recent Results Table - always show, with empty state if no data
st.markdown("### Recent Wafer Results (Last 20)")

if has_data and display_aggregator.data and len(display_aggregator.data) > 0:
    # Get last 20 results
    recent_data = display_aggregator.data[-20:]
    
    # Prepare table data
    table_data = []
    for result in recent_data:
        defect_class = result.get('prediction', {}).get('Defect Class', 'Unknown')
        defect_pct = result.get('defect_percentage', 0)
        status = result.get('quality_status', 'Unknown')
        
        # Status emoji
        status_emoji = "✅" if status == "PASS" else "❌"
        
        table_data.append({
            "Wafer ID": result.get('wafer_id', 'N/A'),
            "Machine": result.get('machine_type', 'N/A'),
            "Defect Class": defect_class,
            "Defect %": f"{defect_pct:.2f}%",
            "Status": f"{status_emoji} {status}"
        })
    
    df_recent = pd.DataFrame(table_data)
    st.dataframe(df_recent, width='stretch', hide_index=True)
else:
    # Show empty state with placeholder columns
    empty_df = pd.DataFrame({
        "Wafer ID": [],
        "Machine": [],
        "Defect Class": [],
        "Defect %": [],
        "Status": []
    })
    st.dataframe(empty_df, width='stretch', hide_index=True)
    st.info("📋 No wafer results available. Start the simulation to see wafer processing results.")

st.markdown("---")

# Clear Data Section
st.markdown("### 🗑️ Data Management")
col1, col2 = st.columns([3, 1])

with col1:
    st.info("⚠️ **Warning:** Clearing data will permanently delete all manufacturing results, processed images, and logs. This action cannot be undone.")

with col2:
    # Initialize confirmation state
    if 'confirm_clear' not in st.session_state:
        st.session_state.confirm_clear = False
    
    if st.session_state.confirm_clear:
        # Show confirmation buttons
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("✅ Confirm Clear", type="primary", width='stretch'):
                try:
                    from Repository.config_LLM import RESULTS_DIR, PROCESSED_IMAGES_DIR, LOGS_DIR
                    
                    deleted_count = 0
                    
                    # Delete all results JSON files
                    results_files = list(RESULTS_DIR.glob("results_*.json"))
                    for file in results_files:
                        file.unlink()
                        deleted_count += 1
                    
                    # Delete all processed images
                    if PROCESSED_IMAGES_DIR.exists():
                        image_files = list(PROCESSED_IMAGES_DIR.glob("*.jpg"))
                        for file in image_files:
                            file.unlink()
                            deleted_count += 1
                    
                    # Delete all log files
                    if LOGS_DIR.exists():
                        log_files = list(LOGS_DIR.glob("*.log"))
                        for file in log_files:
                            file.unlink()
                            deleted_count += 1
                    
                    st.session_state.confirm_clear = False
                    st.success(f"✅ Successfully cleared {deleted_count} files! All manufacturing data has been deleted.")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error clearing data: {str(e)}")
                    st.session_state.confirm_clear = False
        with col_no:
            if st.button("❌ Cancel", width='stretch'):
                st.session_state.confirm_clear = False
                st.rerun()
    else:
        if st.button("🗑️ Clear All Data", type="secondary", width='stretch'):
            st.session_state.confirm_clear = True
            st.rerun()

# Auto-refresh (also checks for simulation completion)
# Navigation buttons at the bottom (before auto-refresh)
st.markdown("---")
st.markdown("### 🔗 Page Navigation")
st.info("💡 Use the buttons below to navigate between pages")
col_nav = st.columns(3)
with col_nav[0]:
    if st.button("🏠 Welcome Page", use_container_width=True, type="secondary"):
        st.switch_page("WELCOME.py")
with col_nav[1]:
    st.empty()  # Empty column for spacing
with col_nav[2]:
    if st.button("Next: 📊 Defect Analytics →", use_container_width=True, type="primary"):
        st.switch_page("Pages/2_DEFECT ANALYTICS.py")

# Always refresh if simulation is running (to detect completion), or if auto-refresh is enabled
should_refresh = st.session_state.simulation_running or auto_refresh
if should_refresh:
    # If simulation is running, check more frequently (every 2 seconds) to detect completion
    refresh_interval = 2 if st.session_state.simulation_running else 5
    time.sleep(refresh_interval)
    st.rerun()

