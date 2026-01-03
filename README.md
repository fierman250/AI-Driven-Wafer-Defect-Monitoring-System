# AI-Driven Wafer Defect Monitoring Framework using ML and LLM-POWERED AI AGENT

<img width="1918" height="904" alt="image" src="https://github.com/user-attachments/assets/121d4369-4d73-4d4f-9d68-21ea14f3c58f" />

DEMO Link in Youtube: https://www.youtube.com/watch?v=WJvnMq2CSCw

A comprehensive semiconductor manufacturing monitoring system that combines machine learning-based defect detection with LLM-powered intelligent analysis and reporting.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [System Workflow](#system-workflow)
- [Component Details](#component-details)
- [File Structure](#file-structure)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This system provides a two-layer AI framework for semiconductor wafer defect monitoring:

1. **Layer 1: ML-Based Defect Detection** - Uses ResNet18 CNN to classify wafer defects into 9 categories
2. **Layer 2: LLM-Powered AI Agent** - Provides intelligent analysis, natural language queries, and automated reporting

The system simulates a manufacturing environment with Mechanical, Electrical, and Thermal machines, processes wafer images, and generates comprehensive analysis reports.

**New in v2.0:** Interactive Streamlit web interface with real-time dashboard, chat-based AI assistant, and comprehensive analytics!

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Manufacturing Simulation                 │
│  (Mechanical, Electrical, Thermal Machines)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Layer 1: ML Defect Detection                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ ResNet18     │  │ Defect       │  │ Quality      │       │
│  │ Classifier   │  │ Counter      │  │ Assessment   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Results Storage (JSON)                         │
│  - Wafer ID, Machine Type, Defect Class                     │
│  - Defect Percentage, Confidence Score                      │
│  - Quality Status (PASS/FAIL)                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Layer 2: LLM-Powered AI Agent                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Data         │  │ Query        │  │ Summary      │       │
│  │ Aggregator   │  │ Processor    │  │ Generator    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │ Multi-       │  │ LLM          │                         │
│  │ Physics KB   │  │ Monitoring   │                         │
│  └──────────────┘  └──────────────┘                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Output: Reports & Analysis                     │
│  - Text Summaries, PDF Reports                              │
│  - Natural Language Answers                                 │
│  - Recommendations & Root Cause Analysis                    │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Features

### Layer 1: ML Defect Detection
- ✅ ResNet18 CNN model for defect classification
- ✅ 9 defect classes: Center, Donut, Edge-Loc, Edge-Ring, Local, Near-Full, Normal, Random, Scratch
- ✅ HSV-based defect percentage calculation
- ✅ Confidence scoring
- ✅ Quality status determination (PASS/FAIL based on defect threshold)

### Layer 2: LLM-Powered AI Agent
- ✅ OpenAI GPT integration for intelligent analysis
- ✅ Natural language query interface
- ✅ Daily summary generation
- ✅ Multi-physics root cause explanations (Thermal, Mechanical, Electrical)
- ✅ Corrective action recommendations
- ✅ PDF report generation
- ✅ Interactive CLI interface
- ✅ **Chat-based web interface** (ChatGPT-like experience)

### Manufacturing Simulation
- ✅ Multi-machine simulation (Mechanical, Electrical, Thermal)
- ✅ Parallel processing with threading
- ✅ Real-time defect analysis
- ✅ Configurable simulation parameters
- ✅ Comprehensive logging
- ✅ **Web-based simulation control** from dashboard

### Web Interface (Streamlit)
- ✅ **Real-time Dashboard** - Live monitoring with KPI cards, charts, and tables
- ✅ **AI Assistant Chat** - Interactive chat interface for natural language queries
- ✅ **Defect Analytics** - Comprehensive defect analysis and visualization
- ✅ **Simulation Control** - Start/stop simulations directly from the web interface
- ✅ **Auto-stop Simulation** - Automatically stops when duration completes
- ✅ **Date Filtering** - Filter data by simulation date across all pages
- ✅ **Data Management** - Clear data functionality with confirmation
- ✅ **Auto-refresh** - Automatic data updates every 5 seconds
- ✅ **Page Navigation** - Easy navigation between pages with Previous/Next buttons
- ✅ **Responsive Design** - Modern, user-friendly interface

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch (with CUDA support recommended)
- OpenAI API key (for LLM features)
- Streamlit (for web interface)

### Step 1: Install Dependencies

```bash
cd AgentAI/Repository
pip install -r requirements.txt
```

**Note:** The requirements.txt includes Streamlit. If you need to install it separately:

```bash
pip install streamlit
```

### Step 2: Configure API Key

Edit `AgentAI/Repository/config_LLM.py` and set your OpenAI API key:

```python
OPENAI_API_KEY = "sk-your-api-key-here"
```

Or set it as an environment variable:
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

### Step 3: Verify Model File

Ensure `MLModelv4.pth` is in `AgentAI/Repository/` directory.

### Step 4: Verify Test Dataset

Ensure test images are in `AgentAI/Repository/Test/` with subdirectories:
- Center/
- Donut/
- Edge-Loc/
- Edge-Ring/
- Local/
- Near-Full/
- Normal/
- Random/
- Scratch/

### Step 5: Launch Web Interface (Optional)

To use the web interface instead of CLI:

```bash
cd AgentAI
streamlit run WELCOME.py
```

The app will open in your default browser at `http://localhost:8501`

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)

Launch the Streamlit web application:

```bash
cd AgentAI
streamlit run WELCOME.py
```

This will open your browser to `http://localhost:8501` with the following pages:

1. **🏠 Welcome Page** - Landing page with system overview and navigation
2. **📊 Dashboard** - Real-time monitoring with:
   - KPI cards (Total Wafers, Pass Rate, Fail Rate, Avg Defect %, Avg Confidence)
   - Simulation control (start/stop simulation with configurable parameters)
   - Auto-stop functionality (stops automatically when duration completes)
   - Simulation date selection for filtering data
   - Warning box highlighting worst-performing machine
   - Defect class distribution chart
   - Machine status table
   - Recent wafer results table
   - Data management (clear all data with confirmation)

3. **🤖 AI Assistant** - Chat interface for:
   - Natural language queries (ChatGPT-like experience)
   - Conversation history in scrollable chat box (500px height)
   - Quick question buttons
   - Report generation:
     - Daily Summary Report (text format)
     - Comprehensive PDF Report (summary-only or full per-wafer details)
     - Date filtering for reports
   - Clear input and clear chat functionality

4. **📈 Defect Analytics** - Comprehensive analytics and visualizations:
   - Advanced metrics (Mean, Median, Std Dev, Min, Max)
   - Machine performance comparison
   - Defect percentage over time (by simulation date or timestamp)
   - Correlation Analysis: Machine vs Defect Percentage
   - Defect distribution by machine type
   - Top defect classes and anomalies
   - Daily performance comparison
   - Filters: Date range (timestamp/simulation_date), machine type
   - CSV export functionality

**Features:**
- Start/stop simulations directly from the dashboard
- Auto-stop simulation when duration completes
- Real-time data updates (auto-refresh every 5 seconds, 2 seconds during simulation)
- Chat-based AI assistant with persistent conversation history
- Interactive charts and visualizations
- Date-based filtering across all pages
- Page navigation with Previous/Next buttons
- PDF reports with defect images (summary-only or full details)

### Option 2: Command Line Interface

#### 1. Run Manufacturing Simulation

```bash
cd AgentAI
python RUN_ManProcess.py
```

This will:
- Start the manufacturing simulation
- Generate wafer images from test dataset
- Analyze each wafer for defects
- Save results to `Manufacturing_Output/results_*.json`

#### 2. Run LLM Monitoring Agent

```bash
cd AgentAI
python RUN_LLM_Agent.py
```

This provides an interactive menu to:
- Generate daily summaries
- Answer queries about manufacturing data
- Generate recommendations
- Create PDF reports

## 🔄 System Workflow

### Complete Workflow

```
1. MANUFACTURING SIMULATION
   │
   ├─> Machines generate wafer images
   │   ├─> Mechanical Machine → Random image from Test/
   │   ├─> Electrical Machine → Random image from Test/
   │   └─> Thermal Machine → Random image from Test/
   │
   ├─> Each wafer is analyzed:
   │   ├─> Defect Prediction (ResNet18)
   │   │   ├─> Load image
   │   │   ├─> Preprocess (resize, normalize)
   │   │   ├─> Run through ResNet18 model
   │   │   ├─> Get defect class (9 classes)
   │   │   └─> Get confidence score
   │   │
   │   └─> Defect Counting (HSV-based)
   │       ├─> Convert to HSV color space
   │       ├─> Detect yellow pixels (defects)
   │       ├─> Detect green pixels (wafer area)
   │       └─> Calculate defect percentage
   │
   └─> Save results to JSON
       ├─> Wafer ID, Machine Type, Timestamp
       ├─> Defect Class, Confidence Score
       ├─> Defect Percentage
       └─> Quality Status (PASS/FAIL)

2. DATA AGGREGATION
   │
   └─> DataAggregator loads results
       ├─> Scans Manufacturing_Output/ for results_*.json
       ├─> Parses JSON files
       ├─> Creates pandas DataFrame
       └─> Provides statistics and analysis

3. LLM ANALYSIS
   │
   ├─> Daily Summary Generation
   │   ├─> Aggregate statistics
   │   ├─> Identify trends
   │   ├─> Send to LLM for enhancement
   │   └─> Generate comprehensive summary
   │
   ├─> Query Processing
   │   ├─> Classify query type
   │   ├─> Extract relevant data
   │   ├─> Use LLM for intelligent answer
   │   └─> Return formatted response
   │
   └─> Report Generation
       ├─> Collect data and statistics
       ├─> Use Multi-Physics Knowledge Base
       ├─> Generate text/PDF report
       └─> Include LLM-enhanced insights

4. OUTPUT
   │
   ├─> Text Summaries → LLM_Output/summaries/
   ├─> JSON Reports → LLM_Output/reports/
   ├─> PDF Reports → LLM_Output/pdf_reports/
   └─> Logs → Manufacturing_Output/logs/
```

### Detailed Component Workflows

#### Manufacturing Simulation Workflow

```
RUN_ManProcess.py
    │
    ├─> Initialize ManufacturingProcessController
    │   ├─> Create WaferImageGenerator (scans Test/ folder)
    │   ├─> Initialize WaferDefectPredictor (loads MLModelv4.pth)
    │   ├─> Initialize DefectCounter
    │   └─> Create machines (Mechanical, Electrical, Thermal)
    │
    ├─> Start simulation
    │   ├─> Start all machines (threading)
    │   │
    │   ├─> Each machine thread:
    │   │   ├─> Generate wafer image (copy from Test/)
    │   │   ├─> Process wafer with analysis:
    │   │   │   ├─> Run defect prediction
    │   │   │   ├─> Run defect counting
    │   │   │   ├─> Determine quality status
    │   │   │   └─> Save result
    │   │   └─> Wait random interval (2-12 seconds)
    │   │
    │   └─> Continue for specified duration
    │
    └─> Generate summary statistics
        ├─> Total wafers processed
        ├─> Pass/Fail counts
        ├─> Machine type distribution
        └─> Defect class distribution
```

#### LLM Agent Workflow

```
RUN_LLM_Agent.py
    │
    ├─> Initialize components
    │   ├─> LLMMonitoringAgent
    │   │   ├─> Initialize OpenAI client
    │   │   ├─> Create DataAggregator
    │   │   └─> Load results from Manufacturing_Output/
    │   │
    │   ├─> QueryProcessor
    │   │   ├─> Create DataAggregator
    │   │   └─> Load results
    │   │
    │   └─> SummaryGenerator
    │       └─> Create DataAggregator
    │
    └─> Interactive menu loop
        ├─> Option 1: Generate Daily Summary
        │   └─> agent.generate_daily_summary()
        │       ├─> Aggregate data
        │       ├─> Call LLM for enhancement
        │       └─> Return formatted summary
        │
        ├─> Option 2: Answer Query
        │   └─> processor.process_query(query)
        │       ├─> Classify query type
        │       ├─> Extract relevant data
        │       ├─> Call LLM for answer
        │       └─> Return formatted answer
        │
        ├─> Option 3: Generate Recommendations
        │   └─> agent.generate_recommendations()
        │       ├─> Analyze defect patterns
        │       ├─> Use Multi-Physics KB
        │       ├─> Call LLM for recommendations
        │       └─> Return formatted recommendations
        │
        └─> Option 8: Generate PDF Report
            └─> generator.generate_pdf_report()
                ├─> Collect statistics
                ├─> Generate LLM summary
                ├─> Create PDF with ReportLab
                └─> Save to pdf_reports/
```

## 📁 Component Details

### Core Components

#### 1. Defect_Prediction.py
**Purpose:** ML-based defect classification and counting

**Classes:**
- `WaferDefectPredictor`: ResNet18 model for defect classification
  - `__init__(model_path)`: Loads model from checkpoint
  - `predict(image_path)`: Returns defect class and confidence
- `DefectCounter`: HSV-based defect percentage calculation
  - `count_defects(image_path)`: Returns defect percentage

**Key Features:**
- Handles multiple checkpoint formats
- Automatic key prefix stripping
- Comprehensive error handling

#### 2. Manufacturing_Simulation.py
**Purpose:** Simulates manufacturing process with multiple machines

**Classes:**
- `WaferImageGenerator`: Generates wafer images from test dataset
- `ManufacturingMachine`: Base class for machines
- `MechanicalMachine`, `ElectricalMachine`, `ThermalMachine`: Specific machine types
- `ManufacturingProcessController`: Main controller for simulation

**Key Features:**
- Multi-threaded parallel processing
- Real-time defect analysis integration
- Configurable machine counts and intervals
- Comprehensive logging

#### 3. LLM_Monitoring_Agent.py
**Purpose:** LLM-powered intelligent analysis

**Classes:**
- `LLMMonitoringAgent`: Main agent for LLM interactions
  - `generate_daily_summary()`: Creates daily analysis
  - `answer_query(query)`: Answers natural language queries
  - `generate_recommendations()`: Provides corrective actions

**Key Features:**
- OpenAI API integration
- Error handling with fallbacks
- Multi-physics knowledge integration

#### 4. Query_Processor.py
**Purpose:** Processes natural language queries

**Classes:**
- `QueryProcessor`: Classifies and processes queries
  - `process_query(query)`: Main processing function
  - Supports 10 query types (machine_performance, defect_distribution, etc.)

**Key Features:**
- Pattern-based query classification
- Intelligent routing to appropriate handlers
- LLM integration for complex queries

#### 5. Summary_Generator.py
**Purpose:** Generates formatted reports

**Classes:**
- `SummaryGenerator`: Creates various report formats
  - `generate_text_summary()`: Text format
  - `generate_json_summary()`: JSON format
  - `generate_pdf_report()`: PDF format with ReportLab

**Key Features:**
- LLM-enhanced summaries
- Fallback summaries when LLM unavailable
- Professional PDF formatting

#### 6. Data_Aggregator.py
**Purpose:** Aggregates and analyzes manufacturing results

**Classes:**
- `DataAggregator`: Data loading and analysis
  - `load_results()`: Loads JSON results
  - Various statistics methods (machine performance, defect distribution, etc.)

**Key Features:**
- Pandas DataFrame integration
- Time-series analysis
- Statistical calculations

#### 7. MultiPhysics_Knowledge_Base.py
**Purpose:** Maps defects to multi-physics root causes

**Functions:**
- `explain_defect(defect_class)`: Explains defect causes
- `get_defect_info(defect_class)`: Gets defect information
- `get_recommendations(defect_class)`: Gets corrective actions
- `get_machine_domain_info(machine_type)`: Gets domain information

**Key Features:**
- Thermal, Mechanical, Electrical domain mappings
- Process step identification
- Recommendation generation

## 📂 File Structure

```
AgentAI/
├── README.md                          # This file
├── WELCOME.py                         # Streamlit main app (landing page)
├── RUN_ManProcess.py                  # Manufacturing simulation entry point (CLI)
├── RUN_LLM_Agent.py                   # LLM agent entry point (CLI)
│
├── Pages/                             # Streamlit web pages
│   ├── 1_DASHBOARD.py                # Real-time monitoring dashboard
│   ├── 2_DEFECT ANALYTICS.py         # Defect analytics page
│   ├── 3_AI_ASSISTANT.py             # Chat-based AI assistant
│   └── LPBackgroung.png              # Landing page background image
│
├── Repository/                        # Core code modules
│   ├── config_LLM.py                 # Configuration (API keys, paths)
│   ├── Defect_Prediction.py          # ML defect detection
│   ├── Manufacturing_Simulation.py   # Manufacturing simulation
│   ├── LLM_Monitoring_Agent.py       # LLM agent
│   ├── Query_Processor.py            # Query processing
│   ├── Summary_Generator.py          # Report generation
│   ├── Data_Aggregator.py            # Data aggregation
│   ├── MultiPhysics_Knowledge_Base.py # Knowledge base
│   ├── TEST_API_Connection.py        # API connection test
│   ├── requirements.txt              # Python dependencies
│   ├── MLModelv4.pth                 # Trained ResNet18 model
│   │
│   └── Test/                          # Test dataset
│       ├── Center/
│       ├── Donut/
│       ├── Edge-Loc/
│       ├── Edge-Ring/
│       ├── Local/
│       ├── Near-Full/
│       ├── Normal/
│       ├── Random/
│       └── Scratch/
│
├── Manufacturing_Output/              # Simulation outputs
│   ├── results_*.json                 # Wafer analysis results
│   ├── processed_images/             # Generated wafer images
│   └── logs/                          # Log files
│
└── LLM_Output/                        # LLM agent outputs
    ├── summaries/                     # Text summaries
    ├── reports/                       # JSON reports
    └── pdf_reports/                   # PDF reports
```

## ⚙️ Configuration

### config_LLM.py

Main configuration file with:

```python
# API Configuration
OPENAI_API_KEY = "sk-..."              # Your OpenAI API key
OPENAI_MODEL = "gpt-4.1-mini"          # Model to use

# Paths (automatically configured)
BASE_DIR = Path(__file__).parent.parent
MANUFACTURING_OUTPUT_DIR = BASE_DIR / "Manufacturing_Output"
LLM_OUTPUT_DIR = BASE_DIR / "LLM_Output"

# LLM Settings
LLM_TEMPERATURE = 0.3                  # Response creativity (0.0-1.0)
MAX_TOKENS = 2000                      # Maximum response length
```

### RUN_ManProcess.py

Simulation configuration:

```python
NUM_MECHANICAL = 2                     # Number of mechanical machines
NUM_ELECTRICAL = 2                     # Number of electrical machines
NUM_THERMAL = 2                        # Number of thermal machines
SIMULATION_DURATION = 60               # Simulation duration (seconds)
MAX_WAFERS = None                      # Max wafers (None = unlimited)
```

## 💡 Usage Examples

### Example 1: Web Interface - Dashboard

```bash
cd AgentAI
streamlit run WELCOME.py
```

**Navigate to Dashboard:**
1. Click "📊 Go to Dashboard" from the welcome page
2. View real-time KPIs (Total Wafers, Pass Rate, Fail Rate, etc.)
3. Configure simulation settings:
   - Number of Mechanical/Electrical/Thermal machines
   - Simulation duration
   - Max wafers limit
4. Click "▶️ Start Simulation" to begin processing
5. Watch real-time updates as wafers are processed
6. View defect distribution charts and machine statistics

### Example 2: Web Interface - AI Assistant Chat

```bash
cd AgentAI
streamlit run WELCOME.py
```

**Navigate to AI Assistant:**
1. Click "🤖 AI Assistant" from the welcome page
2. Ask questions in natural language:
   - "Which machine has the highest defect rate?"
   - "What are the most common defect types?"
   - "Show me recent anomalies"
3. View conversation history in scrollable chat box
4. Ask follow-up questions without losing previous context
5. Generate reports directly from the interface

### Example 3: CLI - Run Manufacturing Simulation

```bash
cd AgentAI
python RUN_ManProcess.py
```

**Output:**
- Generates wafer images
- Analyzes each wafer
- Saves results to `Manufacturing_Output/results_*.json`
- Displays summary statistics

### Example 4: CLI - Generate Daily Summary

```bash
cd AgentAI
python RUN_LLM_Agent.py
# Select option 1: Generate Daily Summary Report
```

**Output:**
- Aggregates all wafer results
- Analyzes trends and patterns
- Generates LLM-enhanced summary
- Displays comprehensive report

### Example 5: CLI - Answer Query

```bash
cd AgentAI
python RUN_LLM_Agent.py
# Select option 2: Answer a Query (Interactive)
# Enter: "Which machine has the highest defect rate?"
```

**Output:**
- Processes query
- Extracts relevant data
- Uses LLM for intelligent answer
- Returns formatted response

### Example 6: CLI - Generate PDF Report

```bash
cd AgentAI
python RUN_LLM_Agent.py
# Select option 8: Generate PDF Report
```

**Output:**
- Creates comprehensive PDF report
- Includes statistics, trends, recommendations
- Saves to `LLM_Output/pdf_reports/`

## 🔧 Troubleshooting

### Model Loading Issues

**Problem:** "Error initializing defect prediction"

**Solutions:**
- Verify `MLModelv4.pth` exists in `AgentAI/Repository/`
- Check model file is not corrupted
- Review log files for detailed error messages

### LLM API Issues

**Problem:** "LLM client not initialized" or "429 Too Many Requests"

**Solutions:**
- Verify API key in `config_LLM.py`
- Check API quota/billing
- System will use fallback summaries if LLM unavailable

### Import Errors

**Problem:** "ModuleNotFoundError: No module named 'Repository'"

**Solutions:**
- Ensure you're running from `AgentAI/` directory
- Check `sys.path.insert()` in RUN scripts
- Verify Repository folder structure

### Path Issues

**Problem:** "File not found" errors

**Solutions:**
- Verify `Test/` folder exists in `Repository/`
- Check `MLModelv4.pth` is in `Repository/`
- Ensure output directories are created automatically

## 📊 Defect Classes

The system classifies defects into 9 categories:

1. **Center** - Concentric defects at wafer center (Thermal domain)
2. **Donut** - Ring-shaped defect pattern (Electrical domain)
3. **Edge-Loc** - Defects localized at wafer edges (Mechanical domain)
4. **Edge-Ring** - Ring pattern at wafer edges (Thermal domain)
5. **Local** - Localized defect clusters (Electrical domain)
6. **Near-Full** - Near-complete defect coverage (Multi-domain)
7. **Normal** - No significant defects
8. **Random** - Random defect distribution (Electrical domain)
9. **Scratch** - Linear scratch patterns (Mechanical domain)

## 🎓 Key Concepts

### Quality Status Determination

- **PASS**: Defect percentage ≤ 40%
- **FAIL**: Defect percentage > 40%

**Note:** The defect threshold is configurable in `Repository/config_LLM.py` (DEFECT_PERCENTAGE_THRESHOLD = 40.0)

### Multi-Physics Domains

- **Thermal**: Center, Edge-Ring defects (heating/cooling issues)
- **Mechanical**: Scratch, Edge-Loc defects (handling/stress issues)
- **Electrical**: Donut, Local, Random defects (plasma/field issues)

### Data Flow

1. **Simulation** → Generates wafer images → Analyzes defects
2. **Results** → Saved to JSON → Loaded by DataAggregator
3. **Analysis** → LLM Agent processes → Generates insights
4. **Output** → Reports generated → Saved to LLM_Output/

## 📝 Notes

- The system uses a 40% defect threshold for PASS/FAIL determination (configurable in `config_LLM.py`)
- LLM features require OpenAI API key and quota
- Model loading handles multiple checkpoint formats automatically
- All paths are relative and automatically configured
- Logging is comprehensive for debugging
- Simulation supports per-day tracking with `simulation_date` field
- PDF reports can be generated with summary-only or full per-wafer details
- Defect images are automatically included in PDF reports
- Navigation buttons are available at the bottom of each page (except Welcome page)

## 🔄 Recent Updates (v2.0)

✅ **Streamlit Web Interface**
- Real-time dashboard with live monitoring
- Chat-based AI assistant (ChatGPT-like experience)
- Defect analytics page with comprehensive visualizations
- Simulation control from web interface
- Auto-stop simulation functionality
- Auto-refresh functionality (5 seconds normal, 2 seconds during simulation)
- Data management tools with confirmation
- Page navigation with Previous/Next buttons

✅ **Enhanced Features**
- Normal class bias in simulation (70% Normal, 30% defects for higher PASS rate)
- Improved chat interface with scrollable conversation history (500px height)
- Date-based filtering (simulation_date) across all pages
- Timestamp range filtering in Defect Analytics
- Multi-date simulation support
- PDF report options:
  - Summary-only (faster, smaller file size, includes sample images)
  - Full report (with detailed per-wafer information)
- Defect images embedded in PDF reports
- Correlation Analysis: Machine vs Defect Percentage (replaced confidence score comparison)
- Warning box for worst-performing machine on Dashboard
- Better error handling and empty state management
- Modern UI with custom styling

## 🔄 Future Enhancements

Potential improvements (not yet implemented):
- Database integration (SQLite/PostgreSQL)
- Advanced time-series analysis
- Statistical Process Control (SPC) charts
- Unit and integration tests
- Docker containerization
- Export conversation history
- User authentication
- Multi-user support

## 📄 License

[Add your license information here]

## 👥 Authors

[Add author information here]

---

**Last Updated:** December 2025  
**Version:** 2.0  
**Status:** Production Ready ✅

## 🎓 Course Information

**Project:** AI-Driven Wafer Defect Monitoring Framework  
**Course:** Semiconductor Manufacturing Intelligence System  
**Institution:** National Cheng Kung University  
**Year:** 2025

**Team Members:**
- Iska (P86137210)
- Firman (M38147023)
- Indah Ayu (M38137028)

