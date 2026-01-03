"""
Diagnostic script to test LLM connection and configuration
Run this to diagnose LLM initialization issues
"""

import sys
from pathlib import Path

# Add Repository to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Repository.config_LLM import (
    OPENAI_API_KEY, OPENAI_MODEL,
    validate_config
)
from Repository.LLM_Monitoring_Agent import LLMMonitoringAgent


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(title)
    print("="*70)


def test_configuration():
    """Test configuration settings."""
    print_section("CONFIGURATION CHECK")
    
    print(f"OpenAI Model: {OPENAI_MODEL}")
    
    # Check API key
    print("\nAPI Key Status:")
    if OPENAI_API_KEY:
        masked_key = OPENAI_API_KEY[:10] + "..." + OPENAI_API_KEY[-5:] if len(OPENAI_API_KEY) > 15 else "***"
        print(f"  OpenAI API Key: {masked_key}")
        print(f"  Key Length: {len(OPENAI_API_KEY)} characters")
        print(f"  Key Format Valid: {OPENAI_API_KEY.startswith('sk-')}")
    else:
        print("  OpenAI API Key: NOT SET")
    
    # Validate config
    print("\nConfiguration Validation:")
    errors = validate_config()
    if errors:
        print("  [X] Configuration Errors Found:")
        for error in errors:
            print(f"    - {error}")
    else:
        print("  [OK] Configuration is valid")


def test_package_installation():
    """Test if required packages are installed."""
    print_section("PACKAGE INSTALLATION CHECK")
    
    try:
        import openai
        print(f"  [OK] OpenAI package installed (version: {openai.__version__})")
        return True
    except ImportError:
        print("  [X] OpenAI package NOT installed")
        print("     Install with: pip install openai")
        return False


def test_llm_initialization():
    """Test LLM agent initialization."""
    print_section("LLM AGENT INITIALIZATION TEST")
    
    try:
        print("Attempting to initialize LLM agent...")
        agent = LLMMonitoringAgent()
        
        if agent.client:
            print("  [OK] LLM client initialized successfully")
        else:
            print("  [X] LLM client NOT initialized")
            if agent.initialization_error:
                print(f"     Error: {agent.initialization_error}")
        
        return agent
        
    except Exception as e:
        print(f"  [X] Failed to initialize LLM agent: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_connection(agent: LLMMonitoringAgent):
    """Test the actual API connection."""
    print_section("API CONNECTION TEST")
    
    if not agent or not agent.client:
        print("  [WARNING] Cannot test connection - client not initialized")
        return
    
    try:
        print("Testing API connection with a simple query...")
        result = agent.test_connection()
        
        print(f"\nTest Results:")
        print(f"  Provider: OpenAI")
        print(f"  Client Initialized: {result['client_initialized']}")
        print(f"  API Key Present: {result['api_key_present']}")
        print(f"  API Key Format Valid: {result['api_key_format_valid']}")
        print(f"  Connection Test: {result['connection_test']}")
        
        if result['error']:
            print(f"  Error: {result['error']}")
        
        print(f"  Message: {result['message']}")
        
        if result['connection_test']:
            print("\n  [OK] Connection test PASSED!")
        else:
            print("\n  [X] Connection test FAILED")
            
    except Exception as e:
        print(f"  [X] Connection test error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all diagnostic tests."""
    print("="*70)
    print("LLM CONNECTION DIAGNOSTIC TOOL")
    print("="*70)
    
    # Test configuration
    test_configuration()
    
    # Test package installation
    packages_ok = test_package_installation()
    
    if not packages_ok:
        print("\n[WARNING] Please install the required packages before continuing.")
        return
    
    # Test initialization
    agent = test_llm_initialization()
    
    # Test connection
    if agent:
        test_connection(agent)
    
    print_section("DIAGNOSTIC COMPLETE")
    print("\nIf you're still experiencing issues:")
    print("1. Verify your API key is correct and not expired")
    print("2. Check your internet connection")
    print("3. Ensure you have sufficient API credits/quota")
    print("4. Review the error messages above for specific issues")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

