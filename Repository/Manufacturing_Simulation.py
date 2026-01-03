"""
Semiconductor Manufacturing Process Simulation
Simulates a manufacturing environment with Mechanical, Electrical, and Thermal machines.
Each machine generates wafer images that are analyzed using defect prediction.
"""

import os
import random
import time
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import threading
from queue import Queue
import logging

# Import the defect prediction module
from Repository.Defect_Prediction import WaferDefectPredictor, DefectCounter, main as predict_defect

# ------------------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------------------
# Base directory - point to AgentAI folder (parent of Repository)
BASE_DIR = Path(__file__).parent.parent

# Repository directory (where this file is located)
REPOSITORY_DIR = Path(__file__).parent

# Paths relative to Repository folder
TEST_DATASET_PATH = REPOSITORY_DIR / "Test"
MODEL_PATH = REPOSITORY_DIR / "MLModelv4.pth"
OUTPUT_DIR = BASE_DIR / "Manufacturing_Output"
PROCESSED_IMAGES_DIR = OUTPUT_DIR / "processed_images"
LOGS_DIR = OUTPUT_DIR / "logs"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOGS_DIR / f"manufacturing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file)),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------
# Image Generator Class
# ------------------------------------------------------------------------------------------
class WaferImageGenerator:
    """Generates wafer images by randomly copying from test dataset."""
    
    def __init__(self, test_dataset_path: str, output_dir: str):
        """
        Initialize the image generator.
        
        Args:
            test_dataset_path: Path to test dataset directory
            output_dir: Directory to save generated images
        """
        self.test_dataset_path = test_dataset_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all available images from test dataset, organized by class
        self.normal_images, self.defect_images = self._scan_test_dataset()
        logger.info(f"Found {len(self.normal_images)} Normal images and {len(self.defect_images)} defect images in test dataset")
    
    def _scan_test_dataset(self):
        """Scan test dataset and return Normal images and defect images separately."""
        normal_images = []
        defect_images = []
        if os.path.exists(self.test_dataset_path):
            for class_folder in os.listdir(self.test_dataset_path):
                class_path = os.path.join(self.test_dataset_path, class_folder)
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(class_path, img_file)
                            # Separate Normal class from defect classes
                            if class_folder.lower() == "normal":
                                normal_images.append(image_path)
                            else:
                                defect_images.append(image_path)
        return normal_images, defect_images
    
    def generate_image(self, wafer_id: str, machine_type: str, normal_probability: float = 0.7) -> Optional[str]:
        """
        Generate a wafer image by copying a random image from test dataset.
        Biased towards Normal class to increase PASS rate.
        
        Args:
            wafer_id: Unique identifier for the wafer
            machine_type: Type of machine generating the image (Mechanical, Electrical, Thermal)
            normal_probability: Probability of selecting a Normal image (default: 0.7 = 70%)
            
        Returns:
            Path to the generated image, or None if generation failed
        """
        # Check if we have images available
        if not self.normal_images and not self.defect_images:
            logger.error("No images available in test dataset")
            return None
        
        # Select image with bias towards Normal class
        if random.random() < normal_probability:
            # Select from Normal images (70% chance)
            if self.normal_images:
                source_image = random.choice(self.normal_images)
            elif self.defect_images:
                # Fallback to defect images if no Normal available
                source_image = random.choice(self.defect_images)
            else:
                return None
        else:
            # Select from defect images (30% chance)
            if self.defect_images:
                source_image = random.choice(self.defect_images)
            elif self.normal_images:
                # Fallback to Normal images if no defect images available
                source_image = random.choice(self.normal_images)
            else:
                return None
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{machine_type}_{wafer_id}_{timestamp}.jpg"
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            # Copy the image
            shutil.copy2(source_image, output_path)
            logger.info(f"Generated image: {output_path} (from {os.path.basename(source_image)})")
            return output_path
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return None

# ------------------------------------------------------------------------------------------
# Manufacturing Machine Base Class
# ------------------------------------------------------------------------------------------
class ManufacturingMachine:
    """Base class for manufacturing machines."""
    
    def __init__(self, machine_id: str, machine_type: str, image_generator: WaferImageGenerator):
        """
        Initialize a manufacturing machine.
        
        Args:
            machine_id: Unique identifier for the machine
            machine_type: Type of machine (Mechanical, Electrical, Thermal)
            image_generator: Image generator instance
        """
        self.machine_id = machine_id
        self.machine_type = machine_type
        self.image_generator = image_generator
        self.is_running = False
        self.processed_wafers = 0
        self.wafer_counter = 0
        self.min_interval = 2  # Minimum seconds between wafer generation
        self.max_interval = 10  # Maximum seconds between wafer generation
        
    def start(self):
        """Start the machine."""
        self.is_running = True
        logger.info(f"{self.machine_type} Machine {self.machine_id} started")
    
    def stop(self):
        """Stop the machine."""
        self.is_running = False
        logger.info(f"{self.machine_type} Machine {self.machine_id} stopped")
    
    def process_wafer(self) -> Optional[Dict]:
        """
        Process a wafer: generate image and return wafer information.
        
        Returns:
            Dictionary with wafer information, or None if processing failed
        """
        if not self.is_running:
            return None
        
        self.wafer_counter += 1
        wafer_id = f"{self.machine_type}_{self.machine_id}_W{self.wafer_counter:04d}"
        
        # Generate wafer image
        image_path = self.image_generator.generate_image(wafer_id, self.machine_type)
        
        if image_path is None:
            return None
        
        self.processed_wafers += 1
        
        return {
            "wafer_id": wafer_id,
            "machine_id": self.machine_id,
            "machine_type": self.machine_type,
            "image_path": image_path,
            "timestamp": datetime.now().isoformat(),
            "process_step": self._get_process_step()
        }
    
    def _get_process_step(self) -> str:
        """Get the process step description for this machine type."""
        steps = {
            "Mechanical": "Mechanical Processing (Dicing, Grinding, Polishing)",
            "Electrical": "Electrical Testing (Probe Testing, Parametric Testing)",
            "Thermal": "Thermal Processing (Annealing, Stress Relief, Burn-in)"
        }
        return steps.get(self.machine_type, "Unknown Process")
    
    def get_status(self) -> Dict:
        """Get current machine status."""
        return {
            "machine_id": self.machine_id,
            "machine_type": self.machine_type,
            "is_running": self.is_running,
            "processed_wafers": self.processed_wafers,
            "wafer_counter": self.wafer_counter
        }

# ------------------------------------------------------------------------------------------
# Specific Machine Types
# ------------------------------------------------------------------------------------------
class MechanicalMachine(ManufacturingMachine):
    """Mechanical processing machine (Dicing, Grinding, Polishing)."""
    
    def __init__(self, machine_id: str, image_generator: WaferImageGenerator):
        super().__init__(machine_id, "Mechanical", image_generator)
        self.min_interval = 5
        self.max_interval = 8

class ElectricalMachine(ManufacturingMachine):
    """Electrical testing machine (Probe Testing, Parametric Testing)."""
    
    def __init__(self, machine_id: str, image_generator: WaferImageGenerator):
        super().__init__(machine_id, "Electrical", image_generator)
        self.min_interval = 5
        self.max_interval = 6

class ThermalMachine(ManufacturingMachine):
    """Thermal processing machine (Annealing, Stress Relief, Burn-in)."""
    
    def __init__(self, machine_id: str, image_generator: WaferImageGenerator):
        super().__init__(machine_id, "Thermal", image_generator)
        self.min_interval = 5
        self.max_interval = 12

# ------------------------------------------------------------------------------------------
# Manufacturing Process Controller
# ------------------------------------------------------------------------------------------
class ManufacturingProcessController:
    """Controls the entire manufacturing process simulation."""
    
    def __init__(self, num_mechanical: int = 2, num_electrical: int = 2, num_thermal: int = 1):
        """
        Initialize the manufacturing process controller.
        
        Args:
            num_mechanical: Number of mechanical machines
            num_electrical: Number of electrical machines
            num_thermal: Number of thermal machines
        """
        # Initialize image generator
        self.image_generator = WaferImageGenerator(str(TEST_DATASET_PATH), str(PROCESSED_IMAGES_DIR))
        
        # Initialize defect predictor and counter
        try:
            logger.info(f"Initializing defect predictor with model: {MODEL_PATH}")
            logger.info(f"Model file exists: {MODEL_PATH.exists()}")
            self.predictor = WaferDefectPredictor(str(MODEL_PATH))
            self.defect_counter = DefectCounter()
            logger.info("Defect prediction system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing defect prediction: {e}", exc_info=True)
            self.predictor = None
            self.defect_counter = None
        
        # Initialize machines
        self.machines = []
        
        # Create mechanical machines
        for i in range(num_mechanical):
            machine = MechanicalMachine(f"MECH_{i+1:02d}", self.image_generator)
            self.machines.append(machine)
        
        # Create electrical machines
        for i in range(num_electrical):
            machine = ElectricalMachine(f"ELEC_{i+1:02d}", self.image_generator)
            self.machines.append(machine)
        
        # Create thermal machines
        for i in range(num_thermal):
            machine = ThermalMachine(f"THERM_{i+1:02d}", self.image_generator)
            self.machines.append(machine)
        
        logger.info(f"Initialized {len(self.machines)} machines: "
                   f"{num_mechanical} Mechanical, {num_electrical} Electrical, {num_thermal} Thermal")
        
        # Process queue and results
        self.process_queue = Queue()
        self.results = []
        self.results_lock = threading.Lock()
        self.is_running = False
        self.simulation_date = None  # Will be set when simulation starts
        self.results_file = OUTPUT_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def start_all_machines(self):
        """Start all manufacturing machines."""
        for machine in self.machines:
            machine.start()
        self.is_running = True
        logger.info("All machines started")
    
    def stop_all_machines(self):
        """Stop all manufacturing machines."""
        for machine in self.machines:
            machine.stop()
        self.is_running = False
        logger.info("All machines stopped")
    
    def process_wafer_with_analysis(self, wafer_info: Dict) -> Dict:
        """
        Process a wafer and perform defect analysis.
        
        Args:
            wafer_info: Dictionary containing wafer information
            
        Returns:
            Dictionary with complete analysis results
        """
        image_path = wafer_info.get("image_path")
        if not image_path or not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return {**wafer_info, "error": "Image not found"}
        
        # Perform defect prediction
        prediction_result = {}
        defect_count_result = {}
        
        if self.predictor:
            try:
                logger.debug(f"Running prediction on: {image_path}")
                prediction_result = self.predictor.predict(image_path)
                logger.debug(f"Prediction result: {prediction_result}")
                if not prediction_result or "Defect Class" not in prediction_result:
                    logger.warning(f"Prediction returned invalid result: {prediction_result}")
                    prediction_result = {"Defect Class": "Unknown", "Confidence Score": 0.0, "error": "Invalid prediction result"}
            except Exception as e:
                logger.error(f"Prediction error for {image_path}: {e}", exc_info=True)
                prediction_result = {"Defect Class": "Error", "Confidence Score": 0.0, "error": str(e)}
        else:
            logger.warning("Predictor not initialized, skipping prediction")
            prediction_result = {"Defect Class": "Unknown", "Confidence Score": 0.0, "error": "Predictor not initialized"}
        
        if self.defect_counter:
            try:
                logger.debug(f"Running defect counting on: {image_path}")
                defect_count_result = self.defect_counter.count_defects(image_path)
                logger.debug(f"Defect count result: {defect_count_result}")
                if not defect_count_result or "defect_percentage" not in defect_count_result:
                    logger.warning(f"Defect counting returned invalid result: {defect_count_result}")
                    defect_count_result = {"defect_percentage": 0.0, "error": "Invalid defect count result"}
            except Exception as e:
                logger.error(f"Defect counting error for {image_path}: {e}", exc_info=True)
                defect_count_result = {"defect_percentage": 0.0, "error": str(e)}
        else:
            logger.warning("Defect counter not initialized, skipping defect counting")
            defect_count_result = {"defect_percentage": 0.0, "error": "Defect counter not initialized"}
        
        # Combine results
        analysis_result = {
            **wafer_info,
            "prediction": prediction_result,
            "defect_count": defect_count_result,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Determine pass/fail status based on defect percentage
        defect_class = prediction_result.get("Defect Class", "Unknown")
        defect_percentage = defect_count_result.get("defect_percentage", 0.0)
        confidence = prediction_result.get("Confidence Score", 0.0)
        
        # Pass/fail logic: >40% defect = FAIL, <=40% defect = PASS
        defect_threshold = 40.0
        is_pass = defect_percentage <= defect_threshold
        analysis_result["quality_status"] = "PASS" if is_pass else "FAIL"
        analysis_result["quality_reason"] = (
            f"Defect Percentage: {defect_percentage}% "
            f"{'(>40% threshold)' if defect_percentage > defect_threshold else '(<=40% threshold)'}, "
            f"Defect Class: {defect_class}, "
            f"Confidence: {confidence:.2%}"
        )
        
        # Add defect threshold information to JSON
        analysis_result["defect_threshold"] = defect_threshold
        analysis_result["defect_percentage"] = defect_percentage
        analysis_result["threshold_exceeded"] = defect_percentage > defect_threshold
        
        return analysis_result
    
    def save_result(self, result: Dict):
        """Save analysis result to file."""
        with self.results_lock:
            # Add simulation date to result if available
            if self.simulation_date:
                result["simulation_date"] = self.simulation_date
            self.results.append(result)
            
            # Save to JSON file (append mode)
            try:
                with open(str(self.results_file), 'w') as f:
                    json.dump(self.results, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving result: {e}")
    
    def run_simulation(self, duration_seconds: int = 60, max_wafers: Optional[int] = None, simulation_date: Optional[str] = None):
        """
        Run the manufacturing simulation.
        
        Args:
            duration_seconds: How long to run the simulation (in seconds)
            max_wafers: Maximum number of wafers to process (None for unlimited)
            simulation_date: Date string (YYYY-MM-DD) for this simulation run, or None to use today's date
        """
        # Set simulation date
        if simulation_date is None:
            self.simulation_date = datetime.now().strftime("%Y-%m-%d")
        else:
            self.simulation_date = simulation_date
        
        logger.info(f"Starting manufacturing simulation for {duration_seconds} seconds (Date: {self.simulation_date})")
        self.start_all_machines()
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Machine threads for parallel processing
        machine_threads = []
        
        def machine_worker(machine: ManufacturingMachine):
            """Worker function for each machine thread."""
            while self.is_running and time.time() < end_time:
                # Check max_wafers limit
                with self.results_lock:
                    if max_wafers and len(self.results) >= max_wafers:
                        break
                
                # Process a wafer
                wafer_info = machine.process_wafer()
                if wafer_info:
                    # Analyze the wafer
                    analysis_result = self.process_wafer_with_analysis(wafer_info)
                    self.save_result(analysis_result)
                    
                    # Log the result
                    logger.info(f"Processed {wafer_info['wafer_id']}: "
                              f"Class={analysis_result['prediction'].get('Defect Class', 'N/A')}, "
                              f"Defect%={analysis_result['defect_count'].get('defect_percentage', 0):.2f}%, "
                              f"Status={analysis_result['quality_status']}")
                
                # Wait random interval before next wafer
                wait_time = random.uniform(machine.min_interval, machine.max_interval)
                time.sleep(wait_time)
        
        # Start machine threads
        for machine in self.machines:
            thread = threading.Thread(target=machine_worker, args=(machine,), daemon=True)
            thread.start()
            machine_threads.append(thread)
        
        # Wait for simulation to complete
        try:
            while time.time() < end_time:
                time.sleep(1)
                elapsed = time.time() - start_time
                if elapsed % 10 == 0:  # Log status every 10 seconds
                    total_processed = sum(m.processed_wafers for m in self.machines)
                    logger.info(f"Simulation running... Elapsed: {elapsed:.0f}s, Total wafers processed: {total_processed}")
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        
        # Stop all machines
        self.stop_all_machines()
        
        # Wait for threads to finish
        for thread in machine_threads:
            thread.join(timeout=5)
        
        total_processed = sum(m.processed_wafers for m in self.machines)
        logger.info(f"Simulation completed. Total wafers processed: {total_processed}")
        logger.info(f"Results saved to: {self.results_file}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary statistics of the simulation."""
        if not self.results:
            logger.info("No results to summarize")
            return
        
        total_wafers = len(self.results)
        pass_count = sum(1 for r in self.results if r.get("quality_status") == "PASS")
        fail_count = total_wafers - pass_count
        
        # Count by machine type
        machine_counts = {}
        defect_class_counts = {}
        
        for result in self.results:
            machine_type = result.get("machine_type", "Unknown")
            machine_counts[machine_type] = machine_counts.get(machine_type, 0) + 1
            
            defect_class = result.get("prediction", {}).get("Defect Class", "Unknown")
            defect_class_counts[defect_class] = defect_class_counts.get(defect_class, 0) + 1
        
        print("\n" + "="*70)
        print("MANUFACTURING SIMULATION SUMMARY")
        print("="*70)
        print(f"Total Wafers Processed: {total_wafers}")
        print(f"Pass: {pass_count} ({pass_count/total_wafers*100:.1f}%)")
        print(f"Fail: {fail_count} ({fail_count/total_wafers*100:.1f}%)")
        print("\nWafers by Machine Type:")
        for machine_type, count in machine_counts.items():
            print(f"  {machine_type}: {count}")
        print("\nDefect Class Distribution:")
        for defect_class, count in sorted(defect_class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {defect_class}: {count}")
        print("="*70)
        print(f"\nDetailed results saved to: {self.results_file}")

# ------------------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*70)
    print("SEMICONDUCTOR MANUFACTURING PROCESS SIMULATION")
    print("="*70)
    print("\nThis simulation will:")
    print("1. Simulate Mechanical, Electrical, and Thermal machines")
    print("2. Generate wafer images from test dataset at random intervals")
    print("3. Analyze each wafer using defect prediction and counting")
    print("4. Save all results to JSON files")
    print("\n" + "="*70)
    
    # Configuration
    NUM_MECHANICAL = 2
    NUM_ELECTRICAL = 2
    NUM_THERMAL = 2
    SIMULATION_DURATION = 60  # seconds
    MAX_WAFERS = None  # Set to a number to limit total wafers, or None for unlimited
    
    # Create and run simulation
    controller = ManufacturingProcessController(
        num_mechanical=NUM_MECHANICAL,
        num_electrical=NUM_ELECTRICAL,
        num_thermal=NUM_THERMAL
    )
    
    try:
        controller.run_simulation(
            duration_seconds=SIMULATION_DURATION,
            max_wafers=MAX_WAFERS
        )
    except Exception as e:
        logger.error(f"Simulation error: {e}", exc_info=True)
        print(f"\nError occurred: {e}")
        print("Check the log file for details.")

