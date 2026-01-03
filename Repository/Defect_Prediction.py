# Defect_Prediction.py
# This file is used to predict the defect of the wafer image.

import os
# Fix OpenMP duplicate library warning and prevent kernel crashes
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np
import cv2
import os
import logging

# Setup logging
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------
# Helper Function: Get Repository Path
# ------------------------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------------------------------
# Defect Counter Class (Integrated from Defect_Count.py)
# ------------------------------------------------------------------------------------------
class DefectCounter:
    def __init__(self):
        pass

    def count_defects(self, image_path):
        """
        Analyzes an image to count the percentage of defects on the wafer.
        Uses HSV color space to detect yellow defects and green wafer area.

        Args:
            image_path (str): Path to the wafer image.

        Returns:
            dict: A dictionary containing the defect percentage.
        """
        try:
            # Set OpenCV threads to prevent conflicts
            cv2.setNumThreads(1)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at {image_path}")

            # Convert image to RGB then HSV color space for better color detection
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

            # Define ranges for yellow (defects) and green (wafer area)
            # Yellow mask for defects: H: 20-30, S: 100-255, V: 100-255
            y_mask = cv2.inRange(image_hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
            
            # Green mask for wafer area: H: 35-85, S: 50-255, V: 50-255
            g_mask = cv2.inRange(image_hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))

            # Calculate metrics
            def_pix = np.count_nonzero(y_mask)  # Defect pixels (yellow)
            tot_pix = np.count_nonzero(g_mask) + def_pix  # Total wafer pixels (green + defects)

            # Calculate the percentage of defect pixels
            defect_percentage = (def_pix / tot_pix * 100) if tot_pix > 0 else 0

            return {"defect_percentage": round(defect_percentage, 2)}

        except Exception as e:
            logger.error(f"Error in defect counting: {e}", exc_info=True)
            return {"error": str(e), "defect_percentage": 0.0}

# ------------------------------------------------------------------------------------------
# Wafer Defect Predictor Class
# ------------------------------------------------------------------------------------------
class WaferDefectPredictor:
    def __init__(self, model_path, num_classes=9):
        """
        Initialize the wafer defect predictor with ResNet18 model.

        Args:
            model_path (str): Path to the trained model file (.pth)
            num_classes (int): Number of defect classes (default: 9)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize ResNet18 model
        self.model = models.resnet18(weights=None)  # We'll load our trained weights
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        # Load the trained model weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            logger.info(f"Checkpoint loaded, type: {type(checkpoint)}")
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # Check if it's a full checkpoint with 'model_state_dict' or 'state_dict' key
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    logger.info("Extracted 'model_state_dict' from checkpoint")
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    logger.info("Extracted 'state_dict' from checkpoint")
                else:
                    # Assume the dict itself is the state_dict
                    state_dict = checkpoint
                    logger.info("Using checkpoint dict directly as state_dict")
            else:
                # Direct state_dict
                state_dict = checkpoint
                logger.info("Checkpoint is direct state_dict")
            
            # Get model's expected keys
            model_keys = set(self.model.state_dict().keys())
            state_dict_keys = set(state_dict.keys())
            
            logger.info(f"Model expects {len(model_keys)} keys, state_dict has {len(state_dict_keys)} keys")
            
            # Try loading with strict=True first
            try:
                self.model.load_state_dict(state_dict, strict=True)
                logger.info("Model loaded successfully with strict=True")
            except RuntimeError as e:
                # If strict loading fails, try to fix key mismatches
                logger.warning(f"Strict loading failed: {str(e)[:200]}...")
                logger.info("Attempting to fix key mismatches...")
                
                # Check if keys have common prefixes that need to be stripped
                # Common prefixes: "model.", "module.", etc.
                prefixes_to_try = ["model.", "module.", "backbone."]
                
                fixed = False
                for prefix in prefixes_to_try:
                    if any(key.startswith(prefix) for key in state_dict_keys):
                        logger.info(f"Trying to strip '{prefix}' prefix from keys...")
                        new_state_dict = {}
                        for key, value in state_dict.items():
                            if key.startswith(prefix):
                                new_key = key[len(prefix):]
                                new_state_dict[new_key] = value
                            else:
                                new_state_dict[key] = value
                        
                        # Try loading with stripped keys
                        try:
                            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
                            if not missing_keys or len(missing_keys) < len(model_keys) * 0.1:  # Allow up to 10% missing
                                logger.info(f"Successfully loaded model after stripping '{prefix}' prefix")
                                logger.info(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                                if missing_keys:
                                    logger.warning(f"Some keys are missing: {missing_keys[:5]}...")
                                fixed = True
                                break
                        except Exception as e2:
                            logger.debug(f"Stripping '{prefix}' prefix didn't work: {str(e2)[:100]}")
                            continue
                
                # If prefix stripping didn't work, try loading with strict=False
                if not fixed:
                    logger.warning("Attempting to load with strict=False (some layers may not load)...")
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                    
                    if missing_keys:
                        logger.warning(f"Missing {len(missing_keys)} keys: {missing_keys[:5]}...")
                    if unexpected_keys:
                        logger.warning(f"Unexpected {len(unexpected_keys)} keys: {unexpected_keys[:5]}...")
                    
                    # Check if we have enough keys loaded
                    loaded_keys = model_keys - set(missing_keys)
                    if len(loaded_keys) < len(model_keys) * 0.5:  # Less than 50% loaded
                        error_msg = f"Too many keys missing ({len(missing_keys)}/{len(model_keys)}). Model may not work correctly."
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                    else:
                        logger.info(f"Loaded {len(loaded_keys)}/{len(model_keys)} keys. Model may work with reduced functionality.")
            
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            error_msg = f"Failed to load model from {model_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
        
        self.model.eval()
        self.model.to(self.device)

        # Image transformations (matching training preprocessing)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Class names based on the dataset structure
        self.class_names = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Local', 
                           'Near-Full', 'Normal', 'Random', 'Scratch']

    def predict(self, image_path):
        """
        Predict the defect class of a wafer image.

        Args:
            image_path (str): Path to the wafer image.

        Returns:
            dict: A dictionary containing "Defect Class" and "Confidence Score"
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Perform prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                confidence, predicted_idx = torch.max(probabilities, 0)

            defect_class = self.class_names[predicted_idx.item()]
            confidence_score = float(confidence.item())

            return {
                "Defect Class": defect_class,
                "Confidence Score": round(confidence_score, 4)
            }
        except Exception as e:
            logger.error(f"Error in prediction for {image_path}: {e}", exc_info=True)
            return {
                "Defect Class": "Error",
                "Confidence Score": 0.0,
                "error": str(e)
            }

# ------------------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------------------
def main(image_path, model_path=None):
    """
    Main function to perform both defect prediction and defect counting.

    Args:
        image_path (str): Path to the wafer image to analyze
        model_path (str, optional): Path to the model file. If None, uses default path.

    Returns:
        dict: Combined results containing prediction and defect count information
    """
    # Default model path - use script directory relative to script location
    if model_path is None:
        model_path = os.path.join(script_dir, "MLModelv4.pth")

    # Check if image exists
    if not os.path.exists(image_path):
        error_msg = f"Image file not found at {image_path}"
        print(f"Error: {error_msg}")
        return {
            "error": error_msg,
            "prediction": None,
            "defect_count": None
        }

    # Initialize predictor
    try:
        predictor = WaferDefectPredictor(model_path=model_path)
        prediction_result = predictor.predict(image_path)
        print("Prediction Result:")
        print(json.dumps(prediction_result, indent=4))
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        prediction_result = {
            "Defect Class": "Error",
            "Confidence Score": 0.0,
            "error": str(e)
        }

    # Initialize defect counter
    counter = DefectCounter()
    defect_count_result = counter.count_defects(image_path)
    print("\nDefect Count Result:")
    print(json.dumps(defect_count_result, indent=4))

    # Combine results
    combined_results = {
        "prediction": prediction_result,
        "defect_count": defect_count_result
    }

    print("\n" + "="*50)
    print("Combined Results:")
    print("="*50)
    print(json.dumps(combined_results, indent=4))

    return combined_results

# ------------------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage:
    # You can modify these paths to test with your own images
    example_image_path = os.path.join(script_dir, "wafer.jpg")  # Change this to your wafer image path
    example_model_path = os.path.join(script_dir, "MLModelv4.pth")

    # Check if example image exists, if not, try alternative paths
    if not os.path.exists(example_image_path):
        # Try other common locations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        alternative_paths = [
            os.path.join(parent_dir, "WaferDatasets", "test", "Normal"),
        ]
        print(f"Warning: Image not found at {example_image_path}")
        print(f"Script directory: {script_dir}")
        print("Please provide a valid image path as an argument or modify the example_image_path variable.")

    # Run the main function
    results = main(example_image_path, example_model_path)

