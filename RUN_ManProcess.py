"""
Quick start script for the Semiconductor Manufacturing Simulation
Run this script to start the simulation with default settings.
"""

import sys
from pathlib import Path

# Add Repository to path for imports
sys.path.insert(0, str(Path(__file__).parent / "Repository"))

from Repository.Manufacturing_Simulation import ManufacturingProcessController
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Configuration - Modify these values as needed
    NUM_MECHANICAL = 2
    NUM_ELECTRICAL = 2
    NUM_THERMAL = 2
    SIMULATION_DURATION = 60  # seconds
    MAX_WAFERS = None  # Set to a number to limit, or None for unlimited
    
    print("="*70)
    print("SEMICONDUCTOR MANUFACTURING PROCESS SIMULATION")
    print("="*70)
    print("\nConfiguration:")
    print(f"  - Mechanical Machines: {NUM_MECHANICAL}")
    print(f"  - Electrical Machines: {NUM_ELECTRICAL}")
    print(f"  - Thermal Machines: {NUM_THERMAL}")
    print(f"  - Simulation Duration: {SIMULATION_DURATION} seconds")
    if MAX_WAFERS:
        print(f"  - Max Wafers: {MAX_WAFERS}")
    print("\nThe simulation will:")
    print("  1. Generate wafer images from test dataset")
    print("  2. Analyze each wafer for defects")
    print("  3. Save results to Manufacturing_Output/ directory")
    print("\n" + "="*70)
    
    # Create and run simulation
    try:
        controller = ManufacturingProcessController(
            num_mechanical=NUM_MECHANICAL,
            num_electrical=NUM_ELECTRICAL,
            num_thermal=NUM_THERMAL
        )
        
        controller.run_simulation(
            duration_seconds=SIMULATION_DURATION,
            max_wafers=MAX_WAFERS
        )
        
        print("\nSimulation completed successfully!")
        print(f"Check the Manufacturing_Output/ directory for results and logs.")
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()

