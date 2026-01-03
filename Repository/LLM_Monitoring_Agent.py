"""
LLM-Powered AI Monitoring Agent for Semiconductor Manufacturing
Integrates with OpenAI API to provide intelligent analysis and reporting
"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from Repository.config_LLM import (
    OPENAI_API_KEY, OPENAI_MODEL,
    LLM_TEMPERATURE, MAX_TOKENS, SYSTEM_PROMPT
)
from Repository.Data_Aggregator import DataAggregator
from Repository.MultiPhysics_Knowledge_Base import (
    explain_defect, get_defect_info, get_recommendations,
    get_machine_domain_info
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMMonitoringAgent:
    """LLM-powered monitoring agent for wafer defect analysis."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM monitoring agent.
        
        Args:
            api_key: Optional API key (if not set in config)
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.aggregator = DataAggregator()
        self.client = None
        self.initialization_error = None  # Store initialization error for debugging
        
        # Initialize LLM client
        self._initialize_client()
        
        # Load data
        self.aggregator.load_results()
    
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
            
            if not self.api_key or self.api_key.strip() == "":
                error_msg = "OpenAI API key not set. Please set OPENAI_API_KEY in config_llm.py or as environment variable."
                logger.error(error_msg)
                self.initialization_error = error_msg
                self.client = None
                return
            
            # Validate API key format (should start with 'sk-')
            if not self.api_key.startswith('sk-'):
                error_msg = f"Invalid OpenAI API key format. API keys should start with 'sk-'. Current key starts with: {self.api_key[:5]}..."
                logger.error(error_msg)
                self.initialization_error = error_msg
                self.client = None
                return
            
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI client with model: {OPENAI_MODEL}")
            self.initialization_error = None  # Clear any previous errors
            
            # Just verify the client can be created
            logger.info("OpenAI client created successfully")
                        
        except ImportError:
            error_msg = "OpenAI package not installed. Run: pip install openai"
            logger.error(error_msg)
            self.initialization_error = error_msg
            self.client = None
        except Exception as e:
            error_msg = f"Failed to initialize OpenAI client: {str(e)}"
            logger.error(error_msg)
            self.initialization_error = error_msg
            self.client = None
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call the LLM API with a prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            LLM response text
        """
        if not self.client:
            error_details = self.initialization_error or "Unknown error during initialization"
            error_msg = f"Error: LLM client not initialized.\n"
            error_msg += f"Details: {error_details}\n"
            error_msg += f"Provider: {self.provider}\n"
            error_msg += f"API Key Present: {'Yes' if self.api_key else 'No'}\n"
            error_msg += "\nTroubleshooting:\n"
            error_msg += "1. Check your API key in config_llm.py\n"
            error_msg += "2. Verify the API key is valid and not expired\n"
            error_msg += "3. Ensure the OpenAI package is installed (pip install openai)\n"
            error_msg += "4. Check your internet connection\n"
            error_msg += "5. Review the logs above for more details"
            logger.error(error_msg)
            return error_msg
        
        system_prompt = system_prompt or SYSTEM_PROMPT
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            return response.choices[0].message.content
        
        except Exception as e:
            error_str = str(e)
            logger.error(f"Error calling LLM: {error_str}")
            
            # Provide more helpful error messages
            if "429" in error_str or "quota" in error_str.lower() or "insufficient_quota" in error_str.lower():
                return f"Error: API quota exceeded. Your OpenAI API key has exceeded its quota limit.\n" \
                       f"Please check your OpenAI account billing and usage at https://platform.openai.com/usage\n" \
                       f"Original error: {error_str}"
            elif "401" in error_str or "unauthorized" in error_str.lower() or "invalid_api_key" in error_str.lower():
                return f"Error: Invalid API key. Please verify your OpenAI API key is correct and not expired.\n" \
                       f"Original error: {error_str}"
            elif "rate_limit" in error_str.lower():
                return f"Error: Rate limit exceeded. Please wait a moment and try again.\n" \
                       f"Original error: {error_str}"
            else:
                return f"Error calling LLM API: {error_str}\n" \
                       f"If this persists, check your API key, internet connection, and account status."
    
    def generate_daily_summary(self, date: Optional[str] = None) -> str:
        """
        Generate a daily summary report.
        
        Args:
            date: Specific date (YYYY-MM-DD) or None for latest data
            
        Returns:
            Formatted daily summary
        """
        # Reload data to get latest
        self.aggregator.load_results()
        
        summary_stats = self.aggregator.get_summary_statistics()
        machine_stats = self.aggregator.get_machine_statistics()
        defect_dist = self.aggregator.get_defect_distribution()
        anomalies = self.aggregator.get_anomalies()
        
        # Format data for LLM
        data_summary = self.aggregator.format_for_llm()
        
        # Get date statistics for context
        date_stats = self.aggregator.get_date_statistics()
        date_context = ""
        if date_stats:
            date_context = "\n\nDATE-BASED ANALYSIS:\n"
            date_context += "The data includes results from multiple simulation dates. Pay special attention to:\n"
            for date in sorted(date_stats.keys(), reverse=True):
                stats = date_stats[date]
                date_context += f"- {date}: {stats['total_wafers']} wafers, {stats['pass_rate']:.2f}% pass rate, "
                date_context += f"{stats['anomalies']} anomalies\n"
        
        prompt = f"""Analyze the following semiconductor manufacturing wafer defect data and generate a comprehensive daily summary report.

{data_summary}
{date_context}

Please provide:
1. **Executive Summary**: Overall performance metrics and key highlights, including date-specific insights if multiple dates are present
2. **Defect Analysis**: Breakdown of defect types with multi-physics explanations (thermal, mechanical, electrical causes)
3. **Machine Performance**: Analysis of each machine type's performance and any issues
4. **Anomalies & Alerts**: Highlight any wafers exceeding thresholds and their potential root causes, including which dates have the most anomalies
5. **Date-Based Trends**: If multiple dates are present, identify trends over time and which dates had the most defects
6. **Recommendations**: Specific corrective actions based on the defect patterns observed

Be technical, data-driven, and provide actionable insights. Reference the multi-physics domains (thermal, mechanical, electrical) when explaining defect causes. Always include specific dates when discussing temporal patterns."""

        response = self._call_llm(prompt)
        return response
    
    def answer_query(self, query: str) -> str:
        """
        Answer an operator's natural language query.
        
        Args:
            query: Natural language question
            
        Returns:
            Answer to the query
        """
        # Reload data
        self.aggregator.load_results()
        
        # Get relevant data based on query type
        data_context = self._get_query_context(query)
        
        prompt = f"""You are an AI monitoring agent for semiconductor manufacturing. Answer the operator's question about the manufacturing process.

Operator's Question: {query}

Available Manufacturing Data:
{data_context}

IMPORTANT: The manufacturing data includes simulation_date information for each wafer. When answering questions about dates, trends over time, or which date has the most defects, use the date statistics provided above. Each wafer record includes a simulation_date field (format: YYYY-MM-DD) that indicates when the wafer was processed.

Provide a clear, technical answer with specific numbers, statistics, dates, and actionable recommendations. If the question relates to defect causes, explain the multi-physics aspects (thermal, mechanical, electrical domains). Always reference specific dates when discussing temporal patterns or date-specific data."""

        response = self._call_llm(prompt)
        return response
    
    def _get_query_context(self, query: str) -> str:
        """Get relevant data context based on query."""
        query_lower = query.lower()
        context = ""
        
        from Repository.config_LLM import DEFECT_PERCENTAGE_THRESHOLD
        
        # Date-based statistics (always include for date-related queries)
        if any(word in query_lower for word in ["date", "day", "when", "which date", "most defect"]):
            date_stats = self.aggregator.get_date_statistics()
            if date_stats:
                context += "STATISTICS BY SIMULATION DATE:\n"
                # Sort by total wafers or anomalies to show most relevant dates first
                sorted_dates = sorted(
                    date_stats.items(), 
                    key=lambda x: (x[1]['anomalies'], x[1]['total_wafers']), 
                    reverse=True
                )
                for date, stats in sorted_dates:
                    context += f"  Date: {date}\n"
                    context += f"    Total Wafers: {stats['total_wafers']}\n"
                    context += f"    Pass Rate: {stats['pass_rate']:.2f}% ({stats['pass_count']} pass, {stats['fail_count']} fail)\n"
                    context += f"    Avg Defect %: {stats['avg_defect_percentage']:.2f}%\n"
                    context += f"    Anomalies (>{DEFECT_PERCENTAGE_THRESHOLD}%): {stats['anomalies']} wafers\n"
                context += "\n"
        
        # Machine performance queries
        if any(word in query_lower for word in ["machine", "tool", "equipment", "which"]):
            ranking = self.aggregator.get_machine_performance_ranking()
            context += "Machine Performance Ranking:\n"
            for i, machine in enumerate(ranking, 1):
                context += f"  {i}. {machine['machine']}: {machine['pass_rate']:.2f}% pass rate "
                context += f"({machine['total_wafers']} wafers, avg defect: {machine['average_defect_percentage']:.2f}%)\n"
            context += "\n"
        
        # Defect distribution queries
        if any(word in query_lower for word in ["defect", "pattern", "type", "common"]):
            defect_dist = self.aggregator.get_defect_distribution()
            context += "Defect Class Distribution:\n"
            for defect_class, count in sorted(defect_dist['counts'].items(), key=lambda x: x[1], reverse=True):
                pct = defect_dist['percentages'].get(defect_class, 0)
                context += f"  {defect_class}: {count} wafers ({pct}%)\n"
            context += "\n"
        
        # Summary statistics
        summary = self.aggregator.get_summary_statistics()
        context += f"Overall Statistics:\n"
        context += f"  Total Wafers: {summary['total_wafers']}\n"
        context += f"  Pass Rate: {summary['pass_rate']:.2f}%\n"
        context += f"  Average Defect Percentage: {summary['average_defect_percentage']:.2f}%\n"
        context += "\n"
        
        # Anomalies with dates
        anomalies = self.aggregator.get_anomalies()
        if anomalies:
            context += f"Anomalies (>{DEFECT_PERCENTAGE_THRESHOLD}% defect): {len(anomalies)} wafers\n"
            for i, anomaly in enumerate(anomalies[:5], 1):
                sim_date = anomaly.get('simulation_date', 'Unknown')
                context += f"  {i}. {anomaly.get('wafer_id')}: {anomaly.get('defect_percentage', 0):.2f}% "
                context += f"({anomaly.get('prediction', {}).get('Defect Class', 'Unknown')}) "
                context += f"[Date: {sim_date}]\n"
        
        return context
    
    def explain_defect_with_llm(self, defect_class: str, machine_type: str = None, 
                                defect_percentage: float = None) -> str:
        """
        Get LLM-enhanced explanation for a defect.
        
        Args:
            defect_class: Defect class name
            machine_type: Optional machine type
            defect_percentage: Optional defect percentage
            
        Returns:
            Enhanced explanation
        """
        # Get base knowledge
        base_explanation = explain_defect(defect_class, machine_type)
        defect_info = get_defect_info(defect_class)
        
        prompt = f"""Based on the following defect information, provide a detailed technical explanation:

Defect Class: {defect_class}
Machine Type: {machine_type or "Unknown"}
Defect Percentage: {defect_percentage or "Unknown"}%

Base Knowledge:
{base_explanation}

Provide an enhanced explanation that:
1. Explains the multi-physics root causes (thermal, mechanical, electrical)
2. Relates the defect to specific semiconductor packaging processes
3. Provides context about why this defect pattern occurs
4. Suggests specific process parameters to check

Be technical and specific to semiconductor manufacturing."""

        response = self._call_llm(prompt)
        return response
    
    def generate_recommendations(self) -> str:
        """
        Generate actionable recommendations based on current data.
        
        Returns:
            Formatted recommendations
        """
        self.aggregator.load_results()
        
        summary = self.aggregator.get_summary_statistics()
        machine_stats = self.aggregator.get_machine_statistics()
        defect_dist = self.get_defect_distribution()
        anomalies = self.aggregator.get_anomalies()
        
        # Get most common defects
        most_common_defects = sorted(
            defect_dist['counts'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        prompt = f"""Based on the following manufacturing data, generate specific, actionable recommendations:

Summary:
- Total Wafers: {summary['total_wafers']}
- Pass Rate: {summary['pass_rate']:.2f}%
- Average Defect Percentage: {summary['average_defect_percentage']:.2f}%

Most Common Defect Types:
{chr(10).join([f"- {defect[0]}: {defect[1]} occurrences" for defect in most_common_defects])}

Machine Performance Issues:
{self._format_machine_issues(machine_stats)}

Anomalies: {len(anomalies)} wafers exceeding 40% defect threshold

Provide prioritized recommendations that:
1. Address the most critical issues first
2. Include specific process parameters to check
3. Reference multi-physics root causes (thermal, mechanical, electrical)
4. Suggest preventive measures
5. Include equipment maintenance suggestions

Format as a numbered list with clear action items."""

        response = self._call_llm(prompt)
        return response
    
    def _format_machine_issues(self, machine_stats: Dict) -> str:
        """Format machine statistics for prompt."""
        issues = []
        for machine, stats in machine_stats.items():
            if stats['pass_rate'] < 80:  # Flag machines with <80% pass rate
                issues.append(f"- {machine}: {stats['pass_rate']:.2f}% pass rate "
                            f"(avg defect: {stats['average_defect_percentage']:.2f}%)")
        
        if not issues:
            return "No significant machine performance issues detected."
        
        return "\n".join(issues)
    
    def get_defect_distribution(self) -> Dict:
        """Get defect distribution (wrapper for aggregator)."""
        return self.aggregator.get_defect_distribution()
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the OpenAI connection and API key.
        
        Returns:
            Dictionary with test results
        """
        result = {
            "provider": "openai",
            "client_initialized": self.client is not None,
            "api_key_present": bool(self.api_key),
            "api_key_format_valid": False,
            "connection_test": False,
            "error": self.initialization_error,
            "message": ""
        }
        
        if not self.api_key:
            result["message"] = "API key is not set"
            return result
        
        # Check API key format
        result["api_key_format_valid"] = self.api_key.startswith('sk-')
        if not result["api_key_format_valid"]:
            result["message"] = f"Invalid API key format. OpenAI keys should start with 'sk-'. Got: {self.api_key[:10]}..."
            return result
        
        # Test actual connection
        if self.client:
            try:
                # Make a minimal test call
                test_response = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": "Say 'test'"}],
                    max_tokens=5
                )
                result["connection_test"] = True
                result["message"] = "Connection test successful!"
            except Exception as e:
                result["connection_test"] = False
                result["error"] = str(e)
                result["message"] = f"Connection test failed: {str(e)}"
        else:
            result["message"] = "Client not initialized. Check initialization error."
        
        return result


# ------------------------------------------------------------------------------------------
# Main Entry Point for Testing
# ------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*70)
    print("LLM MONITORING AGENT TEST")
    print("="*70)
    
    # Check configuration
    from Repository.config_LLM import validate_config
    errors = validate_config()
    if errors:
        print("\nConfiguration Errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease fix configuration before running the agent.")
    else:
        print("\nInitializing LLM Monitoring Agent...")
        try:
            agent = LLMMonitoringAgent()
            
            print("\n" + "="*70)
            print("GENERATING DAILY SUMMARY")
            print("="*70)
            summary = agent.generate_daily_summary()
            print(summary)
            
            print("\n" + "="*70)
            print("TESTING QUERY: Which machine has the highest defect rate?")
            print("="*70)
            answer = agent.answer_query("Which machine has the highest defect rate?")
            print(answer)
            
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

