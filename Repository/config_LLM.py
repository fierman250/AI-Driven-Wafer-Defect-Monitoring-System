"""
Configuration file for LLM Monitoring Agent
Set your API keys and configuration here
"""

import os
from pathlib import Path

# ------------------------------------------------------------------------------------------
# LLM API Configuration
# ------------------------------------------------------------------------------------------

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("ReplaceAPIKeyHere")  # Set via environment variable or replace with your key
OPENAI_MODEL = "gpt-4.1-mini"  # Options: "gpt-4", "gpt-3.5-turbo", "gpt-4-turbo-preview" , "gpt-4.1-mini"

# ------------------------------------------------------------------------------------------
# Paths Configuration
# ------------------------------------------------------------------------------------------

# Base directory - point to AgentAI folder (parent of Repository)
BASE_DIR = Path(__file__).parent.parent

# Manufacturing output directory
MANUFACTURING_OUTPUT_DIR = BASE_DIR / "Manufacturing_Output"
RESULTS_DIR = MANUFACTURING_OUTPUT_DIR
PROCESSED_IMAGES_DIR = MANUFACTURING_OUTPUT_DIR / "processed_images"
LOGS_DIR = MANUFACTURING_OUTPUT_DIR / "logs"

# LLM output directory
LLM_OUTPUT_DIR = BASE_DIR / "LLM_Output"
LLM_OUTPUT_DIR.mkdir(exist_ok=True)
SUMMARIES_DIR = LLM_OUTPUT_DIR / "summaries"
SUMMARIES_DIR.mkdir(exist_ok=True)
REPORTS_DIR = LLM_OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
PDF_REPORTS_DIR = LLM_OUTPUT_DIR / "pdf_reports"
PDF_REPORTS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------------------------------
# LLM Agent Configuration
# ------------------------------------------------------------------------------------------

# Temperature for LLM responses (0.0 = deterministic, 1.0 = creative)
LLM_TEMPERATURE = 0.3  # Lower for more consistent, technical responses

# Maximum tokens for responses
MAX_TOKENS = 2000

# System prompt for the LLM agent
SYSTEM_PROMPT = """You are an AI monitoring agent for semiconductor manufacturing processes. 
Your role is to analyze wafer defect data, provide insights about multi-physics causes (thermal, mechanical, electrical), 
and recommend corrective actions. Be precise, technical, and data-driven in your responses."""

# ------------------------------------------------------------------------------------------
# Data Processing Configuration
# ------------------------------------------------------------------------------------------

# Date range for analysis (days)
DEFAULT_ANALYSIS_DAYS = 7

# Defect threshold for alerts
DEFECT_PERCENTAGE_THRESHOLD = 40.0

# Minimum confidence score for reliable predictions
MIN_CONFIDENCE_SCORE = 0.7

# ------------------------------------------------------------------------------------------
# Query Processing Configuration
# ------------------------------------------------------------------------------------------

# Supported query types
SUPPORTED_QUERIES = [
    "machine_performance",
    "defect_distribution",
    "trend_analysis",
    "root_cause",
    "recommendations",
    "summary",
    "comparison"
]

# ------------------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------------------

def validate_config():
    """Validate configuration settings."""
    errors = []
    
    # Validate OpenAI API key
    if not OPENAI_API_KEY or OPENAI_API_KEY.strip() == "":
        errors.append("OPENAI_API_KEY is not set. Please set it in config_llm.py or as environment variable.")
    elif not OPENAI_API_KEY.startswith('sk-'):
        errors.append(f"Invalid OpenAI API key format. Keys should start with 'sk-'. Got: {OPENAI_API_KEY[:10]}...")
    elif len(OPENAI_API_KEY) < 20:
        errors.append(f"OpenAI API key appears too short ({len(OPENAI_API_KEY)} chars). Valid keys are typically 51+ characters.")
    
    # Validate results directory
    if not RESULTS_DIR.exists():
        errors.append(f"Results directory not found: {RESULTS_DIR}")
    
    return errors

if __name__ == "__main__":
    errors = validate_config()
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid!")

