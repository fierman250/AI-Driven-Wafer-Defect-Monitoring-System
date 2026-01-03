"""
Query Processor for Natural Language Queries
Processes operator queries and routes to appropriate handlers
"""

import re
from typing import Dict, List, Optional, Tuple
from enum import Enum

from Repository.Data_Aggregator import DataAggregator
from Repository.LLM_Monitoring_Agent import LLMMonitoringAgent
from Repository.MultiPhysics_Knowledge_Base import explain_defect, get_defect_info


class QueryType(Enum):
    """Types of queries that can be processed."""
    MACHINE_PERFORMANCE = "machine_performance"
    DEFECT_DISTRIBUTION = "defect_distribution"
    TREND_ANALYSIS = "trend_analysis"
    ROOT_CAUSE = "root_cause"
    RECOMMENDATIONS = "recommendations"
    SUMMARY = "summary"
    COMPARISON = "comparison"
    SPECIFIC_DEFECT = "specific_defect"
    ANOMALY_ANALYSIS = "anomaly_analysis"
    GENERAL = "general"


class QueryProcessor:
    """Processes natural language queries about manufacturing data."""
    
    def __init__(self):
        """Initialize the query processor."""
        self.aggregator = DataAggregator()
        self.llm_agent = None  # Initialize on demand
        self.aggregator.load_results()
        
        # Query patterns for classification
        self.query_patterns = {
            QueryType.MACHINE_PERFORMANCE: [
                r"which.*machine.*(?:best|worst|highest|lowest|most|least)",
                r"machine.*performance",
                r"which.*tool.*defect",
                r"machine.*rate",
                r"equipment.*performance"
            ],
            QueryType.DEFECT_DISTRIBUTION: [
                r"defect.*distribution",
                r"most.*common.*defect",
                r"defect.*type",
                r"what.*defect",
                r"defect.*pattern"
            ],
            QueryType.TREND_ANALYSIS: [
                r"trend",
                r"over.*time",
                r"last.*(?:day|week|month)",
                r"improving|worsening",
                r"change.*time"
            ],
            QueryType.ROOT_CAUSE: [
                r"why.*defect",
                r"cause.*defect",
                r"root.*cause",
                r"reason.*defect",
                r"what.*cause"
            ],
            QueryType.RECOMMENDATIONS: [
                r"recommend",
                r"what.*should.*do",
                r"action",
                r"suggest",
                r"how.*fix"
            ],
            QueryType.SUMMARY: [
                r"summary",
                r"overview",
                r"overall",
                r"status",
                r"report"
            ],
            QueryType.SPECIFIC_DEFECT: [
                r"center.*defect",
                r"scratch.*defect",
                r"edge.*defect",
                r"donut.*defect",
                r"local.*defect"
            ],
            QueryType.ANOMALY_ANALYSIS: [
                r"anomal",
                r"outlier",
                r"exceed",
                r"threshold",
                r"problem.*wafer"
            ]
        }
    
    def _get_llm_agent(self) -> Optional[LLMMonitoringAgent]:
        """Get or create LLM agent."""
        if self.llm_agent is None:
            try:
                self.llm_agent = LLMMonitoringAgent()
            except Exception as e:
                print(f"Warning: Could not initialize LLM agent: {e}")
        return self.llm_agent
    
    def classify_query(self, query: str) -> QueryType:
        """
        Classify the type of query.
        
        Args:
            query: Natural language query
            
        Returns:
            QueryType enum
        """
        query_lower = query.lower()
        
        # Check each pattern
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type
        
        return QueryType.GENERAL
    
    def process_query(self, query: str, use_llm: bool = True) -> Dict:
        """
        Process a natural language query.
        
        Args:
            query: Natural language query
            use_llm: Whether to use LLM for response
            
        Returns:
            Dictionary with query result
        """
        # Reload data
        self.aggregator.load_results()
        
        # Classify query
        query_type = self.classify_query(query)
        
        # Process based on type
        if query_type == QueryType.MACHINE_PERFORMANCE:
            answer = self._answer_machine_performance(query, use_llm)
        elif query_type == QueryType.DEFECT_DISTRIBUTION:
            answer = self._answer_defect_distribution(query, use_llm)
        elif query_type == QueryType.ROOT_CAUSE:
            answer = self._answer_root_cause(query, use_llm)
        elif query_type == QueryType.RECOMMENDATIONS:
            answer = self._answer_recommendations(query, use_llm)
        elif query_type == QueryType.SUMMARY:
            answer = self._answer_summary(query, use_llm)
        elif query_type == QueryType.SPECIFIC_DEFECT:
            answer = self._answer_specific_defect(query, use_llm)
        elif query_type == QueryType.ANOMALY_ANALYSIS:
            answer = self._answer_anomaly_analysis(query, use_llm)
        else:
            # General query - use LLM
            answer = self._answer_general(query, use_llm)
        
        return {
            "query": query,
            "query_type": query_type.value,
            "answer": answer
        }
    
    def _answer_machine_performance(self, query: str, use_llm: bool) -> str:
        """Answer machine performance queries."""
        ranking = self.aggregator.get_machine_performance_ranking()
        
        if not ranking:
            return "No machine performance data available."
        
        # Simple answer
        answer = "Machine Performance Ranking:\n\n"
        for i, machine in enumerate(ranking, 1):
            answer += f"{i}. {machine['machine']}: {machine['pass_rate']:.2f}% pass rate "
            answer += f"({machine['total_wafers']} wafers, "
            answer += f"avg defect: {machine['average_defect_percentage']:.2f}%)\n"
        
        # LLM enhancement
        if use_llm:
            agent = self._get_llm_agent()
            if agent:
                llm_answer = agent.answer_query(query)
                # answer += "\n" + "="*50 + "\n"
                answer += "Detailed Analysis:\n"
                answer += "="*50 + "\n" + llm_answer
        
        return answer
    
    def _answer_defect_distribution(self, query: str, use_llm: bool) -> str:
        """Answer defect distribution queries."""
        defect_dist = self.aggregator.get_defect_distribution()
        
        answer = "Defect Class Distribution:\n\n"
        for defect_class, count in sorted(defect_dist['counts'].items(), key=lambda x: x[1], reverse=True):
            pct = defect_dist['percentages'].get(defect_class, 0)
            answer += f"{defect_class:20s}: {count:4d} wafers ({pct:5.2f}%)\n"
        
        if use_llm:
            agent = self._get_llm_agent()
            if agent:
                llm_answer = agent.answer_query(query)
                # answer += "\n" + "="*50 + "\n"
                answer += "Analysis:\n"
                answer += "="*50 + "\n" + llm_answer
        
        return answer
    
    def _answer_root_cause(self, query: str, use_llm: bool) -> str:
        """Answer root cause queries."""
        # Extract defect class from query if possible
        defect_classes = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Local", 
                         "Near-Full", "Normal", "Random", "Scratch"]
        mentioned_defect = None
        for defect in defect_classes:
            if defect.lower() in query.lower():
                mentioned_defect = defect
                break
        
        if mentioned_defect:
            answer = explain_defect(mentioned_defect)
            if use_llm:
                agent = self._get_llm_agent()
                if agent:
                    llm_answer = agent.explain_defect_with_llm(mentioned_defect)
                    # answer += "\n" + "="*50 + "\n"
                    answer += "Enhanced Analysis:\n"
                    answer += "="*50 + "\n" + llm_answer
        else:
            # General root cause analysis
            if use_llm:
                agent = self._get_llm_agent()
                if agent:
                    answer = agent.answer_query(query)
                else:
                    answer = "Please specify a defect type for root cause analysis."
            else:
                answer = "Please specify a defect type for root cause analysis."
        
        return answer
    
    def _answer_recommendations(self, query: str, use_llm: bool) -> str:
        """Answer recommendation queries."""
        if use_llm:
            agent = self._get_llm_agent()
            if agent:
                return agent.generate_recommendations()
        
        # Fallback without LLM
        answer = "Recommendations based on current data:\n\n"
        anomalies = self.aggregator.get_anomalies()
        if anomalies:
            answer += f"1. Address {len(anomalies)} wafers exceeding 40% defect threshold\n"
        
        machine_stats = self.aggregator.get_machine_statistics()
        for machine, stats in machine_stats.items():
            if stats['pass_rate'] < 80:
                answer += f"2. Investigate {machine} with {stats['pass_rate']:.2f}% pass rate\n"
        
        return answer
    
    def _answer_summary(self, query: str, use_llm: bool) -> str:
        """Answer summary queries."""
        if use_llm:
            agent = self._get_llm_agent()
            if agent:
                return agent.generate_daily_summary()
        
        # Fallback
        return self.aggregator.format_for_llm()
    
    def _answer_specific_defect(self, query: str, use_llm: bool) -> str:
        """Answer queries about specific defect types."""
        defect_classes = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Local", 
                         "Near-Full", "Normal", "Random", "Scratch"]
        
        for defect in defect_classes:
            if defect.lower() in query.lower():
                answer = explain_defect(defect)
                if use_llm:
                    agent = self._get_llm_agent()
                    if agent:
                        llm_answer = agent.explain_defect_with_llm(defect)
                        # answer += "\n" + "="*50 + "\n"
                        answer += "Enhanced Analysis:\n"
                        answer += "="*50 + "\n" + llm_answer
                return answer
        
        return "Please specify a defect type (Center, Scratch, Edge-Loc, etc.)"
    
    def _answer_anomaly_analysis(self, query: str, use_llm: bool) -> str:
        """Answer anomaly analysis queries."""
        anomalies = self.aggregator.get_anomalies()
        
        answer = f"Anomaly Analysis:\n\n"
        answer += f"Total Anomalies (>40% defect): {len(anomalies)}\n\n"
        
        if anomalies:
            answer += "Top Anomalies:\n"
            for i, anomaly in enumerate(anomalies[:10], 1):
                sim_date = anomaly.get('simulation_date', 'Unknown')
                answer += f"{i}. {anomaly.get('wafer_id')}: "
                answer += f"{anomaly.get('defect_percentage', 0):.2f}% defect, "
                answer += f"Class: {anomaly.get('prediction', {}).get('Defect Class', 'Unknown')}, "
                answer += f"Machine: {anomaly.get('machine_type')} {anomaly.get('machine_id')}, "
                answer += f"Date: {sim_date}\n"
        
        if use_llm:
            agent = self._get_llm_agent()
            if agent:
                llm_answer = agent.answer_query(query)
                # answer += "\n" + "="*50 + "\n"
                answer += "Analysis:\n"
                answer += "="*50 + "\n" + llm_answer
        
        return answer
    
    def _answer_general(self, query: str, use_llm: bool) -> str:
        """Answer general queries using LLM."""
        if use_llm:
            agent = self._get_llm_agent()
            if agent:
                return agent.answer_query(query)
        
        return "I can help you with questions about machine performance, defect distribution, root causes, and recommendations. Please try rephrasing your question."


# ------------------------------------------------------------------------------------------
# Main Entry Point for Testing
# ------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*70)
    print("QUERY PROCESSOR TEST")
    print("="*70)
    
    processor = QueryProcessor()
    
    test_queries = [
        "Which machine has the highest defect rate?",
        "What are the most common defect types?",
        "Why do we see Center defects?",
        "What recommendations do you have?",
        "Give me a summary of today's results"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        result = processor.process_query(query, use_llm=False)  # Test without LLM first
        print(f"Type: {result['query_type']}")
        print(f"\nAnswer:\n{result['answer']}")

