"""
Multi-Physics Knowledge Base for Wafer Defect Analysis
Maps defect patterns to thermal, mechanical, and electrical root causes
Based on semiconductor packaging process knowledge
"""

# ------------------------------------------------------------------------------------------
# Defect Class to Multi-Physics Cause Mapping
# ------------------------------------------------------------------------------------------

DEFECT_KNOWLEDGE_BASE = {
    "Center": {
        "primary_domain": "Thermal",
        "secondary_domains": ["Electrical"],
        "description": "Concentric defects at wafer center",
        "causes": [
            "Non-uniform heating at wafer center during deposition or annealing",
            "Plasma heating imbalance causing radial temperature gradients",
            "Excessive thermal stress at center due to rapid heating/cooling",
            "Electric field concentration at wafer center during plasma processing"
        ],
        "process_steps": [
            "Annealing operations",
            "Plasma deposition",
            "Thermal cycling",
            "Stress relief processes"
        ],
        "recommendations": [
            "Check chamber temperature uniformity across wafer surface",
            "Verify plasma power distribution and electrode alignment",
            "Review annealing temperature profile and ramp rates",
            "Inspect thermal cycling parameters for excessive gradients",
            "Monitor center-to-edge temperature differential"
        ],
        "typical_machines": ["Thermal", "Electrical"]
    },
    
    "Donut": {
        "primary_domain": "Electrical",
        "secondary_domains": ["Thermal"],
        "description": "Ring-shaped defect pattern",
        "causes": [
            "Plasma density variations creating ring-shaped patterns",
            "Electric field non-uniformities during etching",
            "Localized plasma micro-loading effects",
            "Thermal gradients forming ring patterns"
        ],
        "process_steps": [
            "Plasma etching",
            "Lithography processes",
            "Deposition with plasma",
            "Electrical testing"
        ],
        "recommendations": [
            "Optimize plasma power distribution",
            "Check electrode configuration and spacing",
            "Review gas flow patterns in chamber",
            "Verify RF power uniformity",
            "Inspect plasma density measurements"
        ],
        "typical_machines": ["Electrical", "Thermal"]
    },
    
    "Edge-Loc": {
        "primary_domain": "Mechanical",
        "secondary_domains": ["Thermal"],
        "description": "Defects localized at wafer edges",
        "causes": [
            "Wafer warpage or bow causing edge stress",
            "Uneven chucking pressure at wafer edges",
            "Mechanical deformation during handling",
            "Non-uniform cooling at wafer periphery",
            "Edge contact during processing"
        ],
        "process_steps": [
            "Wafer handling and transport",
            "CMP (Chemical Mechanical Polishing)",
            "Chucking and clamping operations",
            "Thermal processing with edge effects"
        ],
        "recommendations": [
            "Inspect wafer handling equipment alignment",
            "Check chuck pressure uniformity",
            "Review CMP process parameters and pad condition",
            "Verify wafer flatness and warpage measurements",
            "Optimize edge cooling rates"
        ],
        "typical_machines": ["Mechanical", "Thermal"]
    },
    
    "Edge-Ring": {
        "primary_domain": "Thermal",
        "secondary_domains": ["Mechanical"],
        "description": "Ring pattern at wafer edges",
        "causes": [
            "Non-uniform cooling at wafer periphery",
            "Chamber flow distribution issues",
            "Cooling rate inconsistencies at edges",
            "Thermal cycling creating edge stress",
            "Edge contact during thermal processing"
        ],
        "process_steps": [
            "Thermal annealing",
            "Cooling processes",
            "Stress relief operations",
            "Thermal cycling"
        ],
        "recommendations": [
            "Optimize cooling rate and uniformity",
            "Check chamber gas flow distribution",
            "Review thermal cycling profile",
            "Verify edge temperature control",
            "Inspect chamber flow patterns"
        ],
        "typical_machines": ["Thermal"]
    },
    
    "Local": {
        "primary_domain": "Electrical",
        "secondary_domains": ["Mechanical"],
        "description": "Localized cluster defects",
        "causes": [
            "Plasma micro-loading effects in specific regions",
            "Localized electrical field variations",
            "Mechanical stress concentration points",
            "Contamination or particle issues",
            "Localized process variations"
        ],
        "process_steps": [
            "Plasma processing",
            "Localized etching",
            "Mechanical handling",
            "Electrical testing"
        ],
        "recommendations": [
            "Check for localized plasma density variations",
            "Inspect for contamination or particles",
            "Review process uniformity across wafer",
            "Verify electrical field distribution",
            "Check for mechanical stress points"
        ],
        "typical_machines": ["Electrical", "Mechanical"]
    },
    
    "Near-Full": {
        "primary_domain": "Thermal",
        "secondary_domains": ["Electrical", "Mechanical"],
        "description": "Defects covering most of wafer surface",
        "causes": [
            "Global thermal process issues",
            "Systematic electrical problems",
            "Process drift affecting entire wafer",
            "Equipment malfunction",
            "Process parameter out of specification"
        ],
        "process_steps": [
            "All thermal processes",
            "Global electrical testing",
            "System-wide process steps"
        ],
        "recommendations": [
            "Check overall process parameters",
            "Verify equipment calibration",
            "Review process recipe settings",
            "Inspect for systematic equipment issues",
            "Check process control limits"
        ],
        "typical_machines": ["Thermal", "Electrical", "Mechanical"]
    },
    
    "Normal": {
        "primary_domain": "None",
        "secondary_domains": [],
        "description": "No significant defects detected",
        "causes": [
            "Process operating within specifications",
            "No significant multi-physics issues"
        ],
        "process_steps": [
            "All processes when operating correctly"
        ],
        "recommendations": [
            "Continue monitoring",
            "Maintain current process parameters",
            "Regular equipment maintenance"
        ],
        "typical_machines": ["All"]
    },
    
    "Random": {
        "primary_domain": "Electrical",
        "secondary_domains": ["Mechanical"],
        "description": "Randomly distributed defects",
        "causes": [
            "Random plasma density fluctuations",
            "Stochastic electrical variations",
            "Random particle contamination",
            "Unpredictable process variations",
            "Equipment noise or instability"
        ],
        "process_steps": [
            "Plasma processing",
            "Electrical testing",
            "Any process with high variability"
        ],
        "recommendations": [
            "Investigate process stability",
            "Check for equipment noise or drift",
            "Review process control parameters",
            "Inspect for contamination sources",
            "Verify equipment calibration"
        ],
        "typical_machines": ["Electrical", "Mechanical"]
    },
    
    "Scratch": {
        "primary_domain": "Mechanical",
        "secondary_domains": [],
        "description": "Linear scratch defects",
        "causes": [
            "Frictional abrasion during polishing (CMP)",
            "Misaligned handling equipment",
            "Mechanical contact during transport",
            "Abrasive particles in process",
            "Improper wafer handling"
        ],
        "process_steps": [
            "CMP (Chemical Mechanical Polishing)",
            "Wafer handling and transport",
            "Dicing operations",
            "Grinding processes"
        ],
        "recommendations": [
            "Inspect handling equipment alignment",
            "Review CMP process parameters and pad condition",
            "Check for abrasive particles",
            "Verify wafer transport mechanisms",
            "Inspect dicing blade condition"
        ],
        "typical_machines": ["Mechanical"]
    }
}

# ------------------------------------------------------------------------------------------
# Machine Type to Process Domain Mapping
# ------------------------------------------------------------------------------------------

MACHINE_DOMAIN_MAPPING = {
    "Mechanical": {
        "primary_domain": "Mechanical",
        "processes": [
            "Dicing",
            "Grinding",
            "Polishing (CMP)",
            "Wafer handling",
            "Mechanical stress operations"
        ],
        "typical_defects": ["Scratch", "Edge-Loc", "Local"]
    },
    
    "Electrical": {
        "primary_domain": "Electrical",
        "processes": [
            "Probe testing",
            "Parametric testing",
            "Plasma processing",
            "Electrical characterization"
        ],
        "typical_defects": ["Donut", "Local", "Random", "Center"]
    },
    
    "Thermal": {
        "primary_domain": "Thermal",
        "processes": [
            "Annealing",
            "Stress relief",
            "Burn-in",
            "Thermal cycling"
        ],
        "typical_defects": ["Center", "Edge-Ring", "Edge-Loc", "Near-Full"]
    }
}

# ------------------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------------------

def get_defect_info(defect_class: str) -> dict:
    """
    Get information about a defect class.
    
    Args:
        defect_class: Name of the defect class
        
    Returns:
        Dictionary with defect information, or None if not found
    """
    return DEFECT_KNOWLEDGE_BASE.get(defect_class, None)

def get_root_causes(defect_class: str) -> list:
    """Get root causes for a defect class."""
    info = get_defect_info(defect_class)
    return info["causes"] if info else []

def get_recommendations(defect_class: str) -> list:
    """Get recommendations for a defect class."""
    info = get_defect_info(defect_class)
    return info["recommendations"] if info else []

def get_primary_domain(defect_class: str) -> str:
    """Get primary physics domain for a defect class."""
    info = get_defect_info(defect_class)
    return info["primary_domain"] if info else "Unknown"

def explain_defect(defect_class: str, machine_type: str = None) -> str:
    """
    Generate a natural language explanation for a defect.
    
    Args:
        defect_class: Name of the defect class
        machine_type: Optional machine type for context
        
    Returns:
        Formatted explanation string
    """
    info = get_defect_info(defect_class)
    if not info:
        return f"Unknown defect class: {defect_class}"
    
    explanation = f"**{defect_class} Defect Pattern**\n\n"
    explanation += f"Description: {info['description']}\n\n"
    explanation += f"Primary Physics Domain: {info['primary_domain']}\n\n"
    
    if machine_type:
        machine_info = MACHINE_DOMAIN_MAPPING.get(machine_type, {})
        if defect_class in machine_info.get("typical_defects", []):
            explanation += f"This defect is commonly associated with {machine_type} processes.\n\n"
    
    explanation += "**Likely Root Causes:**\n"
    for i, cause in enumerate(info['causes'], 1):
        explanation += f"{i}. {cause}\n"
    
    explanation += "\n**Recommended Actions:**\n"
    for i, rec in enumerate(info['recommendations'], 1):
        explanation += f"{i}. {rec}\n"
    
    return explanation

def get_machine_domain_info(machine_type: str) -> dict:
    """Get domain information for a machine type."""
    return MACHINE_DOMAIN_MAPPING.get(machine_type, {})

# ------------------------------------------------------------------------------------------
# Main Entry Point for Testing
# ------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Test the knowledge base
    print("="*70)
    print("MULTI-PHYSICS KNOWLEDGE BASE TEST")
    print("="*70)
    
    test_defects = ["Center", "Scratch", "Edge-Ring", "Donut"]
    
    for defect in test_defects:
        print(f"\n{explain_defect(defect)}")
        print("-"*70)
    
    print("\n\nMachine Domain Mapping:")
    for machine, info in MACHINE_DOMAIN_MAPPING.items():
        print(f"\n{machine}:")
        print(f"  Domain: {info['primary_domain']}")
        print(f"  Typical Defects: {', '.join(info['typical_defects'])}")

