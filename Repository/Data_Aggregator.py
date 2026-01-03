"""
Data Aggregator for Manufacturing Results
Parses JSON results files and provides aggregated statistics
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import pandas as pd

from Repository.config_LLM import RESULTS_DIR


class DataAggregator:
    """Aggregates and analyzes manufacturing results from JSON files."""
    
    def __init__(self, results_dir: Optional[Path] = None):
        """
        Initialize the data aggregator.
        
        Args:
            results_dir: Directory containing results JSON files
        """
        self.results_dir = results_dir or RESULTS_DIR
        self.data = []
        self.df = None
        
    def load_results(self, file_path: Optional[Path] = None) -> List[Dict]:
        """
        Load results from a specific JSON file or scan directory.
        
        Args:
            file_path: Specific file to load, or None to load latest
            
        Returns:
            List of wafer result dictionaries
        """
        if file_path:
            files_to_load = [file_path]
        else:
            # Find all results JSON files
            files_to_load = sorted(
                self.results_dir.glob("results_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
        
        all_results = []
        for file_path in files_to_load:
            try:
                with open(file_path, 'r') as f:
                    results = json.load(f)
                    if isinstance(results, list):
                        all_results.extend(results)
                    else:
                        all_results.append(results)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        self.data = all_results
        if self.data:
            self.df = pd.DataFrame(self.data)
            # Convert timestamp to datetime
            if 'timestamp' in self.df.columns:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            # Ensure simulation_date is in the dataframe (add if missing from some records)
            if 'simulation_date' not in self.df.columns:
                sim_dates = [r.get('simulation_date') for r in self.data]
                self.df['simulation_date'] = sim_dates
        
        return all_results
    
    def filter_by_simulation_date(self, simulation_date: str) -> List[Dict]:
        """
        Filter results by simulation date.
        
        Args:
            simulation_date: Date string (YYYY-MM-DD)
            
        Returns:
            Filtered list of results
        """
        if not self.data:
            return []
        return [r for r in self.data if r.get('simulation_date') == simulation_date]
    
    def get_available_simulation_dates(self) -> List[str]:
        """
        Get list of available simulation dates.
        
        Returns:
            Sorted list of unique simulation dates (YYYY-MM-DD format)
        """
        if not self.data:
            return []
        dates = set()
        for r in self.data:
            sim_date = r.get('simulation_date')
            if sim_date:
                dates.add(sim_date)
        return sorted(list(dates), reverse=True)  # Most recent first
    
    def get_daily_statistics(self, simulation_date: str) -> Dict:
        """
        Get statistics for a specific simulation date.
        
        Args:
            simulation_date: Date string (YYYY-MM-DD)
            
        Returns:
            Dictionary with daily statistics
        """
        daily_data = self.filter_by_simulation_date(simulation_date)
        if not daily_data:
            return {"error": f"No data found for date {simulation_date}"}
        
        # Create temporary aggregator with filtered data
        temp_aggregator = DataAggregator()
        temp_aggregator.data = daily_data
        if daily_data:
            temp_aggregator.df = pd.DataFrame(daily_data)
            if 'timestamp' in temp_aggregator.df.columns:
                temp_aggregator.df['timestamp'] = pd.to_datetime(temp_aggregator.df['timestamp'])
        
        return temp_aggregator.get_summary_statistics()
    
    def get_summary_statistics(self) -> Dict:
        """
        Get overall summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.data:
            return {"error": "No data loaded"}
        
        total_wafers = len(self.data)
        pass_count = sum(1 for r in self.data if r.get("quality_status") == "PASS")
        fail_count = total_wafers - pass_count
        
        # Average defect percentage
        defect_percentages = [
            r.get("defect_percentage", 0) 
            for r in self.data 
            if r.get("defect_percentage") is not None
        ]
        avg_defect_percentage = sum(defect_percentages) / len(defect_percentages) if defect_percentages else 0
        
        # Average confidence
        confidences = [
            r.get("prediction", {}).get("Confidence Score", 0)
            for r in self.data
            if r.get("prediction", {}).get("Confidence Score") is not None
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "total_wafers": total_wafers,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pass_rate": (pass_count / total_wafers * 100) if total_wafers > 0 else 0,
            "fail_rate": (fail_count / total_wafers * 100) if total_wafers > 0 else 0,
            "average_defect_percentage": round(avg_defect_percentage, 2),
            "average_confidence": round(avg_confidence, 4)
        }
    
    def get_machine_statistics(self) -> Dict:
        """
        Get statistics grouped by machine.
        
        Returns:
            Dictionary with machine-level statistics
        """
        if not self.data:
            return {}
        
        machine_stats = defaultdict(lambda: {
            "total": 0,
            "pass": 0,
            "fail": 0,
            "defect_percentages": [],
            "defect_classes": defaultdict(int)
        })
        
        for result in self.data:
            machine_id = result.get("machine_id", "Unknown")
            machine_type = result.get("machine_type", "Unknown")
            key = f"{machine_type}_{machine_id}"
            
            machine_stats[key]["total"] += 1
            if result.get("quality_status") == "PASS":
                machine_stats[key]["pass"] += 1
            else:
                machine_stats[key]["fail"] += 1
            
            defect_pct = result.get("defect_percentage")
            if defect_pct is not None:
                machine_stats[key]["defect_percentages"].append(defect_pct)
            
            defect_class = result.get("prediction", {}).get("Defect Class", "Unknown")
            machine_stats[key]["defect_classes"][defect_class] += 1
        
        # Calculate averages and rates
        formatted_stats = {}
        for key, stats in machine_stats.items():
            defect_pcts = stats["defect_percentages"]
            formatted_stats[key] = {
                "machine_id": key,
                "total_wafers": stats["total"],
                "pass_count": stats["pass"],
                "fail_count": stats["fail"],
                "pass_rate": round((stats["pass"] / stats["total"] * 100), 2) if stats["total"] > 0 else 0,
                "average_defect_percentage": round(sum(defect_pcts) / len(defect_pcts), 2) if defect_pcts else 0,
                "defect_class_distribution": dict(stats["defect_classes"])
            }
        
        return formatted_stats
    
    def get_defect_distribution(self) -> Dict:
        """
        Get defect class distribution.
        
        Returns:
            Dictionary with defect class counts and percentages
        """
        if not self.data:
            return {}
        
        defect_counts = defaultdict(int)
        for result in self.data:
            defect_class = result.get("prediction", {}).get("Defect Class", "Unknown")
            defect_counts[defect_class] += 1
        
        total = sum(defect_counts.values())
        distribution = {
            "counts": dict(defect_counts),
            "percentages": {
                k: round((v / total * 100), 2) 
                for k, v in defect_counts.items()
            } if total > 0 else {}
        }
        
        return distribution
    
    def get_time_series_data(self, days: int = 7) -> Dict:
        """
        Get time series data for trend analysis.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with time series statistics
        """
        if not self.data:
            return {}
        
        if self.df is None:
            return {}
        
        # Filter by date range
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered_df = self.df[self.df['timestamp'] >= cutoff_date].copy()
        
        if len(filtered_df) == 0:
            return {"error": f"No data in last {days} days"}
        
        # Group by date
        filtered_df['date'] = filtered_df['timestamp'].dt.date
        daily_stats = filtered_df.groupby('date').agg({
            'wafer_id': 'count',
            'defect_percentage': 'mean',
            'quality_status': lambda x: (x == 'PASS').sum()
        }).reset_index()
        
        daily_stats.columns = ['date', 'total_wafers', 'avg_defect_percentage', 'pass_count']
        daily_stats['fail_count'] = daily_stats['total_wafers'] - daily_stats['pass_count']
        daily_stats['pass_rate'] = (daily_stats['pass_count'] / daily_stats['total_wafers'] * 100).round(2)
        
        return {
            "period_days": days,
            "total_wafers": len(filtered_df),
            "daily_breakdown": daily_stats.to_dict('records')
        }
    
    def get_anomalies(self, threshold_percentage: float = 50.0) -> List[Dict]:
        """
        Get wafers that exceed defect threshold.
        
        Args:
            threshold_percentage: Defect percentage threshold
            
        Returns:
            List of anomalous wafer results
        """
        if not self.data:
            return []
        
        anomalies = []
        for result in self.data:
            defect_pct = result.get("defect_percentage", 0)
            if defect_pct > threshold_percentage:
                anomalies.append(result)
        
        # Sort by defect percentage (highest first)
        anomalies.sort(key=lambda x: x.get("defect_percentage", 0), reverse=True)
        
        return anomalies
    
    def get_machine_performance_ranking(self) -> List[Dict]:
        """
        Rank machines by performance (pass rate).
        
        Returns:
            List of machines sorted by pass rate
        """
        machine_stats = self.get_machine_statistics()
        
        ranking = []
        for key, stats in machine_stats.items():
            ranking.append({
                "machine": key,
                "pass_rate": stats["pass_rate"],
                "total_wafers": stats["total_wafers"],
                "average_defect_percentage": stats["average_defect_percentage"]
            })
        
        # Sort by pass rate (highest first)
        ranking.sort(key=lambda x: x["pass_rate"], reverse=True)
        
        return ranking
    
    def get_date_statistics(self) -> Dict:
        """
        Get statistics grouped by simulation date.
        
        Returns:
            Dictionary with date-based statistics
        """
        if not self.data:
            return {}
        
        date_stats = defaultdict(lambda: {
            'total_wafers': 0,
            'pass_count': 0,
            'fail_count': 0,
            'defect_percentages': [],
            'anomalies': 0
        })
        
        from Repository.config_LLM import DEFECT_PERCENTAGE_THRESHOLD
        
        for result in self.data:
            sim_date = result.get('simulation_date')
            if not sim_date:
                continue
            
            date_stats[sim_date]['total_wafers'] += 1
            if result.get('quality_status') == 'PASS':
                date_stats[sim_date]['pass_count'] += 1
            else:
                date_stats[sim_date]['fail_count'] += 1
            
            defect_pct = result.get('defect_percentage', 0)
            if defect_pct is not None:
                date_stats[sim_date]['defect_percentages'].append(defect_pct)
            
            if defect_pct > DEFECT_PERCENTAGE_THRESHOLD:
                date_stats[sim_date]['anomalies'] += 1
        
        # Format statistics
        formatted_stats = {}
        for date, stats in date_stats.items():
            avg_defect = sum(stats['defect_percentages']) / len(stats['defect_percentages']) if stats['defect_percentages'] else 0
            pass_rate = (stats['pass_count'] / stats['total_wafers'] * 100) if stats['total_wafers'] > 0 else 0
            
            formatted_stats[date] = {
                'total_wafers': stats['total_wafers'],
                'pass_count': stats['pass_count'],
                'fail_count': stats['fail_count'],
                'pass_rate': round(pass_rate, 2),
                'avg_defect_percentage': round(avg_defect, 2),
                'anomalies': stats['anomalies']
            }
        
        return formatted_stats
    
    def format_for_llm(self) -> str:
        """
        Format aggregated data as a string for LLM processing.
        
        Returns:
            Formatted string with key statistics
        """
        summary = self.get_summary_statistics()
        machine_stats = self.get_machine_statistics()
        defect_dist = self.get_defect_distribution()
        anomalies = self.get_anomalies()
        date_stats = self.get_date_statistics()
        
        from Repository.config_LLM import DEFECT_PERCENTAGE_THRESHOLD
        
        formatted = "="*70 + "\n"
        formatted += "MANUFACTURING RESULTS SUMMARY\n"
        formatted += "="*70 + "\n\n"
        
        # Date-based statistics
        if date_stats:
            formatted += "STATISTICS BY SIMULATION DATE:\n"
            for date in sorted(date_stats.keys(), reverse=True):
                stats = date_stats[date]
                formatted += f"  Date: {date}\n"
                formatted += f"    Total Wafers: {stats['total_wafers']}\n"
                formatted += f"    Pass Rate: {stats['pass_rate']:.2f}% ({stats['pass_count']} pass, {stats['fail_count']} fail)\n"
                formatted += f"    Avg Defect %: {stats['avg_defect_percentage']:.2f}%\n"
                formatted += f"    Anomalies (>{DEFECT_PERCENTAGE_THRESHOLD}%): {stats['anomalies']} wafers\n"
            formatted += "\n"
        
        formatted += f"Total Wafers Processed: {summary['total_wafers']}\n"
        formatted += f"Pass Rate: {summary['pass_rate']:.2f}% ({summary['pass_count']} wafers)\n"
        formatted += f"Fail Rate: {summary['fail_rate']:.2f}% ({summary['fail_count']} wafers)\n"
        formatted += f"Average Defect Percentage: {summary['average_defect_percentage']:.2f}%\n"
        formatted += f"Average Confidence Score: {summary['average_confidence']:.4f}\n\n"
        
        formatted += "DEFECT CLASS DISTRIBUTION:\n"
        for defect_class, count in sorted(defect_dist['counts'].items(), key=lambda x: x[1], reverse=True):
            pct = defect_dist['percentages'].get(defect_class, 0)
            formatted += f"  {defect_class}: {count} ({pct}%)\n"
        formatted += "\n"
        
        formatted += "MACHINE PERFORMANCE:\n"
        for machine, stats in machine_stats.items():
            formatted += f"  {machine}:\n"
            formatted += f"    Total: {stats['total_wafers']}, Pass Rate: {stats['pass_rate']:.2f}%\n"
            formatted += f"    Avg Defect %: {stats['average_defect_percentage']:.2f}%\n"
        formatted += "\n"
        
        if anomalies:
            formatted += f"ANOMALIES (>{DEFECT_PERCENTAGE_THRESHOLD}% defect): {len(anomalies)} wafers\n"
            formatted += "Top 5 Anomalies:\n"
            for i, anomaly in enumerate(anomalies[:5], 1):
                sim_date = anomaly.get('simulation_date', 'Unknown')
                formatted += f"  {i}. {anomaly.get('wafer_id')}: {anomaly.get('defect_percentage', 0):.2f}% "
                formatted += f"({anomaly.get('prediction', {}).get('Defect Class', 'Unknown')}) "
                formatted += f"[Date: {sim_date}]\n"
        
        return formatted


# ------------------------------------------------------------------------------------------
# Main Entry Point for Testing
# ------------------------------------------------------------------------------------------

if __name__ == "__main__":
    aggregator = DataAggregator()
    
    print("Loading results...")
    results = aggregator.load_results()
    print(f"Loaded {len(results)} wafer results\n")
    
    if results:
        print(aggregator.format_for_llm())
        
        print("\n" + "="*70)
        print("MACHINE PERFORMANCE RANKING")
        print("="*70)
        ranking = aggregator.get_machine_performance_ranking()
        for i, machine in enumerate(ranking, 1):
            print(f"{i}. {machine['machine']}: {machine['pass_rate']:.2f}% pass rate "
                  f"({machine['total_wafers']} wafers)")

