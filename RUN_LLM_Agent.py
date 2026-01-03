"""
Main Entry Point for LLM Monitoring Agent
Interactive interface for querying and generating reportsd
"""

import sys
from pathlib import Path
from datetime import datetime

# Add Repository to path for imports
sys.path.insert(0, str(Path(__file__).parent / "Repository"))

from Repository.config_LLM import validate_config
from Repository.LLM_Monitoring_Agent import LLMMonitoringAgent
from Repository.Query_Processor import QueryProcessor
from Repository.Summary_Generator import SummaryGenerator


def print_banner():
    """Print welcome banner."""
    print("="*70)
    print("LLM-POWERED WAFER DEFECT MONITORING AGENT")
    print("="*70)
    print("AI-Driven Analysis for Semiconductor Manufacturing")
    print("="*70)
    print()

def print_menu():
    """Print main menu options."""
    print("\n" + "="*70)
    print("MAIN MENU")
    print("="*70)
    print("1. Generate Daily Summary Report")
    print("2. Answer a Query (Interactive)")
    print("3. Generate Recommendations")
    print("4. Analyze Specific Defect Type")
    print("5. Machine Performance Analysis")
    print("6. Save Summary Report to File")
    print("7. Test Query Examples")
    print("8. Generate PDF Report")
    print("0. Exit")
    print("="*70)


def generate_summary(agent: LLMMonitoringAgent):
    """Generate and display daily summary."""
    print("\nGenerating daily summary...")
    print("-"*70)
    summary = agent.generate_daily_summary()
    print(summary)


def interactive_query(processor: QueryProcessor):
    """Interactive query interface."""
    print("\n" + "="*70)
    print("INTERACTIVE QUERY INTERFACE")
    print("="*70)
    print("Type your question about the manufacturing process.")
    print("Type 'back' to return to main menu.")
    print("-"*70)
    
    while True:
        query = input("\nYour question: ").strip()
        
        if query.lower() in ['back', 'exit', 'quit']:
            break
        
        if not query:
            continue
        
        print("\nProcessing query...")
        print("-"*70)
        result = processor.process_query(query, use_llm=True)
        print(f"\nAnswer:\n{result['answer']}")
        print("-"*70)


def generate_recommendations(agent: LLMMonitoringAgent):
    """Generate and display recommendations."""
    print("\nGenerating recommendations...")
    print("-"*70)
    recommendations = agent.generate_recommendations()
    print(recommendations)


def analyze_defect(agent: LLMMonitoringAgent):
    """Analyze a specific defect type."""
    print("\nAvailable defect types:")
    defect_types = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Local", 
                    "Near-Full", "Normal", "Random", "Scratch"]
    for i, defect in enumerate(defect_types, 1):
        print(f"  {i}. {defect}")
    
    choice = input("\nEnter defect type name or number: ").strip()
    
    # Handle number input
    try:
        defect_num = int(choice)
        if 1 <= defect_num <= len(defect_types):
            defect_type = defect_types[defect_num - 1]
        else:
            print("Invalid number.")
            return
    except ValueError:
        defect_type = choice
    
    if defect_type not in defect_types:
        print(f"Invalid defect type: {defect_type}")
        return
    
    print(f"\nAnalyzing {defect_type} defects...")
    print("-"*70)
    explanation = agent.explain_defect_with_llm(defect_type)
    print(explanation)


def machine_performance(processor: QueryProcessor):
    """Display machine performance analysis."""
    print("\nMachine Performance Analysis")
    print("-"*70)
    result = processor.process_query("Which machine has the best and worst performance?", use_llm=True)
    print(result['answer'])


def save_summary(generator: SummaryGenerator):
    """Save summary to file."""
    print("\nSaving summary report...")
    filepath = generator.save_summary(use_llm=True)
    print(f"\nSummary saved to: {filepath}")
    
    # Also save JSON report
    json_path = generator.save_json_report()
    print(f"JSON report saved to: {json_path}")


def generate_pdf_report(generator: SummaryGenerator):
    """Generate PDF report."""
    print("\nGenerating PDF report...")
    print("This may take a moment, especially with LLM enhancement...")
    try:
        use_llm = input("Use LLM enhancement? (y/n, default=y): ").strip().lower()
        use_llm = use_llm != 'n'
        
        filepath = generator.generate_pdf_report(use_llm=use_llm)
        print(f"\nPDF report saved to: {filepath}")
    except ImportError as e:
        print(f"\nError: {e}")
        print("Please install reportlab: pip install reportlab")
    except Exception as e:
        print(f"\nError generating PDF: {e}")
        import traceback
        traceback.print_exc()


def test_examples(processor: QueryProcessor):
    """Run test query examples."""
    print("\n" + "="*70)
    print("TEST QUERY EXAMPLES")
    print("="*70)
    
    test_queries = [
        "Which machine has the highest defect rate?",
        "What are the most common defect types?",
        "Why do we see Center defects?",
        "What recommendations do you have for improving yield?",
        "Show me a summary of today's manufacturing results"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Example {i}: {query}")
        print(f"{'='*70}")
        result = processor.process_query(query, use_llm=True)
        print(f"\nAnswer:\n{result['answer'][:500]}...")  # Show first 500 chars
        input("\nPress Enter to continue to next example...")


def main():
    """Main function."""
    print_banner()
    
    # Validate configuration
    print("Validating configuration...")
    errors = validate_config()
    if errors:
        print("\n⚠️  Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease fix configuration before using the LLM agent.")
        print("You can still use basic features without LLM (set use_llm=False).")
        response = input("\nContinue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return
    
    # Initialize components
    print("\nInitializing components...")
    try:
        agent = LLMMonitoringAgent()
        processor = QueryProcessor()
        generator = SummaryGenerator()
        print("✓ Components initialized successfully")
    except Exception as e:
        print(f"⚠️  Error initializing components: {e}")
        print("Some features may not be available.")
        agent = None
        processor = None
        generator = None
    
    # Main loop
    while True:
        print_menu()
        choice = input("\nEnter your choice: ").strip()
        
        if choice == "0":
            print("\nExiting... Goodbye!")
            break
        
        elif choice == "1":
            if agent:
                generate_summary(agent)
            else:
                print("LLM agent not available.")
        
        elif choice == "2":
            if processor:
                interactive_query(processor)
            else:
                print("Query processor not available.")
        
        elif choice == "3":
            if agent:
                generate_recommendations(agent)
            else:
                print("LLM agent not available.")
        
        elif choice == "4":
            if agent:
                analyze_defect(agent)
            else:
                print("LLM agent not available.")
        
        elif choice == "5":
            if processor:
                machine_performance(processor)
            else:
                print("Query processor not available.")
        
        elif choice == "6":
            if generator:
                save_summary(generator)
            else:
                print("Summary generator not available.")
        
        elif choice == "7":
            if processor:
                test_examples(processor)
            else:
                print("Query processor not available.")
        
        elif choice == "8":
            if generator:
                generate_pdf_report(generator)
            else:
                print("Summary generator not available.")
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

