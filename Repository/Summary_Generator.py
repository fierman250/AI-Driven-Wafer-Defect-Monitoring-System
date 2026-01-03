"""
Summary Generator for Manufacturing Reports
Generates formatted summaries and reports with LLM enhancement
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from Repository.config_LLM import SUMMARIES_DIR, REPORTS_DIR, PDF_REPORTS_DIR, OPENAI_MODEL, PROCESSED_IMAGES_DIR
from Repository.Data_Aggregator import DataAggregator
from Repository.LLM_Monitoring_Agent import LLMMonitoringAgent
from Repository.MultiPhysics_Knowledge_Base import explain_defect, get_defect_info, get_recommendations

# PDF generation imports
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not installed. PDF generation will be unavailable.")
    print("Install with: pip install reportlab")


class SummaryGenerator:
    """Generates formatted summaries and reports."""
    
    def __init__(self):
        """Initialize the summary generator."""
        self.aggregator = DataAggregator()
        self.llm_agent = None  # Initialize on demand
        
    def _get_llm_agent(self) -> LLMMonitoringAgent:
        """Get or create LLM agent."""
        if self.llm_agent is None:
            try:
                self.llm_agent = LLMMonitoringAgent()
            except Exception as e:
                print(f"Warning: Could not initialize LLM agent: {e}")
                print("Generating summary without LLM enhancement...")
        return self.llm_agent
    
    def generate_text_summary(self, use_llm: bool = True) -> str:
        """
        Generate a text summary report.
        
        Args:
            use_llm: Whether to use LLM for enhanced summary
            
        Returns:
            Formatted text summary
        """
        self.aggregator.load_results()
        
        summary = "="*70 + "\n"
        summary += "MANUFACTURING PROCESS SUMMARY REPORT\n"
        summary += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += "="*70 + "\n\n"
        
        # Basic statistics
        stats = self.aggregator.get_summary_statistics()
        summary += "OVERALL STATISTICS\n"
        summary += "-"*70 + "\n"
        summary += f"Total Wafers Processed: {stats['total_wafers']}\n"
        summary += f"Pass Rate: {stats['pass_rate']:.2f}% ({stats['pass_count']} wafers)\n"
        summary += f"Fail Rate: {stats['fail_rate']:.2f}% ({stats['fail_count']} wafers)\n"
        summary += f"Average Defect Percentage: {stats['average_defect_percentage']:.2f}%\n"
        summary += f"Average Confidence Score: {stats['average_confidence']:.4f}\n\n"
        
        # Defect distribution
        defect_dist = self.aggregator.get_defect_distribution()
        summary += "DEFECT CLASS DISTRIBUTION\n"
        summary += "-"*70 + "\n"
        for defect_class, count in sorted(defect_dist['counts'].items(), key=lambda x: x[1], reverse=True):
            pct = defect_dist['percentages'].get(defect_class, 0)
            summary += f"{defect_class:20s}: {count:4d} wafers ({pct:5.2f}%)\n"
        summary += "\n"
        
        # Machine performance
        machine_stats = self.aggregator.get_machine_statistics()
        summary += "MACHINE PERFORMANCE\n"
        summary += "-"*70 + "\n"
        for machine, stats in machine_stats.items():
            summary += f"{machine}:\n"
            summary += f"  Total Wafers: {stats['total_wafers']}\n"
            summary += f"  Pass Rate: {stats['pass_rate']:.2f}%\n"
            summary += f"  Average Defect %: {stats['average_defect_percentage']:.2f}%\n"
            summary += f"  Top Defect Classes: {', '.join(list(stats['defect_class_distribution'].keys())[:3])}\n"
        summary += "\n"
        
        # Anomalies
        anomalies = self.aggregator.get_anomalies()
        if anomalies:
            summary += f"ANOMALIES (>40% defect): {len(anomalies)} wafers\n"
            summary += "-"*70 + "\n"
            for i, anomaly in enumerate(anomalies[:10], 1):
                summary += f"{i}. {anomaly.get('wafer_id')}: "
                summary += f"{anomaly.get('defect_percentage', 0):.2f}% defect "
                summary += f"({anomaly.get('prediction', {}).get('Defect Class', 'Unknown')}) "
                summary += f"from {anomaly.get('machine_type', 'Unknown')} machine\n"
            summary += "\n"
        
        # LLM-enhanced analysis
        if use_llm:
            try:
                agent = self._get_llm_agent()
                if agent:
                    summary += "="*70 + "\n"
                    summary += "AI-ENHANCED ANALYSIS\n"
                    summary += "="*70 + "\n\n"
                    llm_summary = agent.generate_daily_summary()
                    summary += llm_summary + "\n"
            except Exception as e:
                summary += f"\nNote: LLM enhancement unavailable ({e})\n"
        
        return summary
    
    def generate_json_summary(self) -> Dict:
        """
        Generate a JSON summary report.
        
        Returns:
            Dictionary with summary data
        """
        self.aggregator.load_results()
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "summary_statistics": self.aggregator.get_summary_statistics(),
            "machine_statistics": self.aggregator.get_machine_statistics(),
            "defect_distribution": self.aggregator.get_defect_distribution(),
            "machine_ranking": self.aggregator.get_machine_performance_ranking(),
            "anomalies_count": len(self.aggregator.get_anomalies()),
            "top_anomalies": [
                {
                    "wafer_id": a.get("wafer_id"),
                    "defect_percentage": a.get("defect_percentage"),
                    "defect_class": a.get("prediction", {}).get("Defect Class"),
                    "machine_type": a.get("machine_type")
                }
                for a in self.aggregator.get_anomalies()[:10]
            ]
        }
        
        return summary
    
    def save_summary(self, filename: Optional[str] = None, use_llm: bool = True) -> Path:
        """
        Save summary to file.
        
        Args:
            filename: Optional filename, or None for auto-generated
            use_llm: Whether to use LLM enhancement
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_{timestamp}.txt"
        
        filepath = SUMMARIES_DIR / filename
        
        summary_text = self.generate_text_summary(use_llm=use_llm)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print(f"Summary saved to: {filepath}")
        return filepath
    
    def save_json_report(self, filename: Optional[str] = None) -> Path:
        """
        Save JSON report to file.
        
        Args:
            filename: Optional filename, or None for auto-generated
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.json"
        
        filepath = REPORTS_DIR / filename
        
        summary_json = self.generate_json_summary()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_json, f, indent=2)
        
        print(f"JSON report saved to: {filepath}")
        return filepath
    
    def generate_defect_analysis_report(self, defect_class: str) -> str:
        """
        Generate detailed analysis report for a specific defect class.
        
        Args:
            defect_class: Defect class to analyze
            
        Returns:
            Formatted analysis report
        """
        self.aggregator.load_results()
        
        report = "="*70 + "\n"
        report = f"DETAILED ANALYSIS: {defect_class.upper()} DEFECTS\n"
        report += "="*70 + "\n\n"
        
        # Get defect information
        defect_info = get_defect_info(defect_class)
        if defect_info:
            report += explain_defect(defect_class) + "\n\n"
        
        # Get statistics for this defect
        defect_dist = self.aggregator.get_defect_distribution()
        count = defect_dist['counts'].get(defect_class, 0)
        percentage = defect_dist['percentages'].get(defect_class, 0)
        
        report += f"OCCURRENCE STATISTICS\n"
        report += "-"*70 + "\n"
        report += f"Total Occurrences: {count}\n"
        report += f"Percentage of All Defects: {percentage:.2f}%\n\n"
        
        # Find wafers with this defect
        wafers_with_defect = [
            r for r in self.aggregator.data
            if r.get("prediction", {}).get("Defect Class") == defect_class
        ]
        
        if wafers_with_defect:
            report += f"WAFERS WITH {defect_class.upper()} DEFECT\n"
            report += "-"*70 + "\n"
            for i, wafer in enumerate(wafers_with_defect[:20], 1):
                report += f"{i}. {wafer.get('wafer_id')}: "
                report += f"{wafer.get('defect_percentage', 0):.2f}% defect, "
                report += f"Machine: {wafer.get('machine_type')} {wafer.get('machine_id')}\n"
        
        # LLM-enhanced explanation
        try:
            agent = self._get_llm_agent()
            if agent:
                report += "\n" + "="*70 + "\n"
                report += "AI-ENHANCED EXPLANATION\n"
                report += "="*70 + "\n\n"
                explanation = agent.explain_defect_with_llm(
                    defect_class,
                    machine_type=None,
                    defect_percentage=None
                )
                report += explanation + "\n"
        except Exception as e:
            report += f"\nNote: LLM enhancement unavailable ({e})\n"
        
        return report
    
    def _esc(self, s) -> str:
        """Basic HTML escaping for ReportLab Paragraph."""
        if s is None:
            return ""
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
    
    def _get_pred(self, record: Dict, key: str, default: str = ""):
        """Get prediction value from record."""
        return record.get("prediction", {}).get(key, default)
    
    def _get_defect_pct(self, record: Dict):
        """Get defect percentage from record."""
        return record.get("defect_count", {}).get("defect_percentage", None)
    
    def _clean_llm_output(self, text: str) -> str:
        """Remove ``` / ```json fences if the model adds them."""
        t = text.strip()
        if t.startswith("```"):
            first_newline = t.find("\n")
            if first_newline != -1:
                t = t[first_newline + 1:]
        t = t.strip()
        if t.endswith("```"):
            t = t[:-3]
        return t.strip()
    
    def _generate_fallback_summary(self, batch_stats: dict) -> dict:
        """
        Generate a fallback summary using rule-based analysis when LLM is unavailable.
        
        Args:
            batch_stats: Dictionary with batch statistics
            
        Returns:
            Dictionary with summary
        """
        counts = batch_stats.get("counts", {})
        total = counts.get("total_wafers", 0)
        pass_count = counts.get("pass", 0)
        fail_count = counts.get("fail", 0)
        pass_rate = counts.get("pass_rate_percent", 0)
        
        distribution = batch_stats.get("distribution", {})
        by_defect_class = distribution.get("by_defect_class", {})
        worst_defects = batch_stats.get("worst_defect_percentages_top5", [])
        
        # Determine yield impact
        if pass_rate >= 90:
            impact = "Low"
        elif pass_rate >= 70:
            impact = "Medium"
        else:
            impact = "High"
        
        # Generate summary text
        summary_parts = []
        summary_parts.append(f"This batch processed {total} wafers with a pass rate of {pass_rate:.2f}% ({pass_count} PASS, {fail_count} FAIL).")
        
        if by_defect_class:
            top_defect = max(by_defect_class.items(), key=lambda x: x[1])
            summary_parts.append(f"The most common defect type is {top_defect[0]} with {top_defect[1]} occurrences.")
        
        if worst_defects:
            worst = worst_defects[0]
            summary_parts.append(f"The highest defect percentage observed is {worst.get('defect_percentage', 0):.2f}% on wafer {worst.get('wafer_id', 'Unknown')}.")
        
        summary_parts.append(f"Overall batch yield impact is estimated as {impact} based on pass rate and defect distribution.")
        
        summary_text = " ".join(summary_parts)
        
        # Generate key risks
        key_risks = []
        if pass_rate < 80:
            key_risks.append(f"Low pass rate ({pass_rate:.2f}%) indicates potential process issues requiring investigation")
        if fail_count > total * 0.2:
            key_risks.append(f"High failure count ({fail_count} wafers) suggests systematic defect patterns")
        if worst_defects and worst_defects[0].get('defect_percentage', 0) > 40:
            key_risks.append("Some wafers exceed 40% defect threshold, indicating severe process deviations")
        
        # Generate recommendations using knowledge base
        recommended_actions = []
        if by_defect_class:
            # Get recommendations for top defect classes
            for defect_class, count in sorted(by_defect_class.items(), key=lambda x: x[1], reverse=True)[:3]:
                recs = get_recommendations(defect_class)
                if recs:
                    recommended_actions.extend(recs[:2])  # Top 2 recommendations per defect type
        
        # Remove duplicates
        recommended_actions = list(dict.fromkeys(recommended_actions))[:5]  # Limit to 5
        
        if not recommended_actions:
            recommended_actions = [
                "Review process parameters for machines with highest defect rates",
                "Check equipment calibration and maintenance schedules",
                "Analyze defect patterns for systematic root causes",
                "Verify material quality and handling procedures",
                "Monitor thermal, mechanical, and electrical process conditions"
            ]
        
        return {
            "summary_text": summary_text,
            "estimated_batch_yield_impact": impact,
            "key_risks": key_risks,
            "recommended_actions": recommended_actions
        }
    
    def _get_llm_batch_summary(self, batch_stats: dict) -> dict:
        """
        Get LLM-generated batch summary with fallback to rule-based analysis.
        
        Args:
            batch_stats: Dictionary with batch statistics
            
        Returns:
            Dictionary with LLM summary or fallback summary
        """
        agent = self._get_llm_agent()
        if not agent or not agent.client:
            # Use fallback summary
            return self._generate_fallback_summary(batch_stats)
        
        system_prompt = """You are a senior semiconductor yield engineer.
Write a short batch-level engineering summary from the provided batch statistics.

Return ONLY valid JSON with keys:
- summary_text: string (2–5 sentences)
- estimated_batch_yield_impact: "Low" | "Medium" | "High"
- key_risks: array of short strings
- recommended_actions: array of short strings

Do NOT wrap output in ``` fences."""
        
        user_prompt = f"Batch statistics JSON:\n{json.dumps(batch_stats, indent=2)}"
        
        try:
            response = agent.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            cleaned = self._clean_llm_output(response.choices[0].message.content)
            return json.loads(cleaned)
        except Exception as e:
            error_str = str(e)
            # Check if it's a quota/API error
            if "429" in error_str or "quota" in error_str.lower() or "insufficient_quota" in error_str.lower():
                # Use fallback summary instead of showing error
                return self._generate_fallback_summary(batch_stats)
            elif "401" in error_str or "unauthorized" in error_str.lower():
                return self._generate_fallback_summary(batch_stats)
            else:
                # For other errors, still use fallback but could log the error
                return self._generate_fallback_summary(batch_stats)
    
    def generate_pdf_report(self, filename: Optional[str] = None, use_llm: bool = True, include_per_wafer_details: bool = True, simulation_date: Optional[str] = None) -> Path:
        """
        Generate a PDF report with per-wafer details and batch summary.
        
        Args:
            filename: Optional filename, or None for auto-generated
            use_llm: Whether to use LLM for batch summary
            include_per_wafer_details: If True, include detailed per-wafer information. 
                                      If False, only include summary statistics.
            simulation_date: Optional simulation date (YYYY-MM-DD) to filter data. 
                            If None, includes all dates.
            
        Returns:
            Path to saved PDF file
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError("reportlab is not installed. Install with: pip install reportlab")
        
        self.aggregator.load_results()
        
        # Filter data by simulation date if provided
        if simulation_date:
            filtered_data = self.aggregator.filter_by_simulation_date(simulation_date)
            if not filtered_data:
                raise ValueError(f"No data found for simulation date: {simulation_date}")
            # Create temporary aggregator with filtered data
            temp_aggregator = DataAggregator()
            temp_aggregator.data = filtered_data
            if filtered_data:
                import pandas as pd
                temp_aggregator.df = pd.DataFrame(filtered_data)
                if 'timestamp' in temp_aggregator.df.columns:
                    temp_aggregator.df['timestamp'] = pd.to_datetime(temp_aggregator.df['timestamp'])
            records_sorted = sorted(filtered_data, key=lambda r: r.get("timestamp", ""))
            data_source = temp_aggregator
        else:
            records_sorted = sorted(self.aggregator.data, key=lambda r: r.get("timestamp", ""))
            data_source = self.aggregator
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            date_suffix = f"_{simulation_date.replace('-', '')}" if simulation_date else ""
            filename = f"batch_report{date_suffix}_{timestamp}.pdf"
        
        filepath = PDF_REPORTS_DIR / filename
        
        # Initialize PDF document
        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(str(filepath), pagesize=A4)
        story = []
        
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Title
        if include_per_wafer_details:
            story.append(Paragraph("<b>Batch Wafer Quality Report</b>", styles["Title"]))
        else:
            story.append(Paragraph("<b>Batch Wafer Quality Summary Report</b>", styles["Title"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>Report Generated:</b> {self._esc(report_time)}", styles["Normal"]))
        if simulation_date:
            story.append(Paragraph(f"<b>Simulation Date:</b> {self._esc(simulation_date)}", styles["Normal"]))
        else:
            story.append(Paragraph(f"<b>Simulation Date:</b> All Dates", styles["Normal"]))
        if not include_per_wafer_details:
            story.append(Paragraph(f"<b>Report Type:</b> Summary Only (Per-Wafer Details Excluded)", styles["Normal"]))
        story.append(Spacer(1, 12))
        
        pass_count = 0
        fail_count = 0
        by_machine_type = {}
        by_defect_class = {}
        defect_pcts = []
        
        # Process each wafer to collect statistics
        for i, r in enumerate(records_sorted, start=1):
            wafer_id = r.get("wafer_id", "unknown")
            machine_type = r.get("machine_type", "")
            machine_id = r.get("machine_id", "")
            ts = r.get("timestamp", "")
            
            defect_class = self._get_pred(r, "Defect Class", "")
            conf = self._get_pred(r, "Confidence Score", None)
            defect_pct = self._get_defect_pct(r)
            
            decision = str(r.get("quality_status", "")).upper().strip()
            reason = r.get("quality_reason", "")
            
            if decision == "PASS":
                pass_count += 1
            else:
                fail_count += 1
            
            by_machine_type[machine_type] = by_machine_type.get(machine_type, 0) + 1
            by_defect_class[defect_class] = by_defect_class.get(defect_class, 0) + 1
            
            if isinstance(defect_pct, (int, float)):
                defect_pcts.append((wafer_id, defect_pct, machine_type, defect_class))
        
        # Batch Summary Section (always shown first)
        story.append(Paragraph("<b>Batch Summary</b>", styles["Heading2"]))
        story.append(Spacer(1, 6))
        
        total = len(records_sorted)
        pass_rate = round((pass_count / total) * 100, 2) if total else 0.0
        
        story.append(Paragraph(f"<b>Total wafers:</b> {total}", styles["Normal"]))
        if simulation_date:
            story.append(Paragraph(f"<b>Date Range:</b> {self._esc(simulation_date)} (Single Day)", styles["Normal"]))
        else:
            # Show date range if multiple dates exist
            dates_in_data = set()
            for r in records_sorted:
                sim_date = r.get("simulation_date")
                if sim_date:
                    dates_in_data.add(sim_date)
            if dates_in_data:
                sorted_dates = sorted(dates_in_data)
                if len(sorted_dates) > 1:
                    story.append(Paragraph(f"<b>Date Range:</b> {self._esc(sorted_dates[0])} to {self._esc(sorted_dates[-1])} ({len(sorted_dates)} days)", styles["Normal"]))
                else:
                    story.append(Paragraph(f"<b>Date Range:</b> {self._esc(sorted_dates[0])} (Single Day)", styles["Normal"]))
        story.append(Paragraph(f"<b>PASS:</b> {pass_count} | <b>FAIL:</b> {fail_count}", styles["Normal"]))
        story.append(Paragraph(f"<b>PASS rate:</b> {pass_rate}%", styles["Normal"]))
        story.append(Spacer(1, 12))
        
        # Add sample images in summary-only mode (show top 6 worst defects + 2 good examples)
        if not include_per_wafer_details:
            # Get top worst defects (highest defect percentage)
            defect_pcts_sorted = sorted(defect_pcts, key=lambda x: x[1], reverse=True)
            
            # Select images to show: top 6 worst defects + 2 good examples (if available)
            images_to_show = []
            
            # Add top 6 worst defects
            worst_defects = defect_pcts_sorted[:6]
            for w_id, pct, mtype, dclass in worst_defects:
                # Find the record for this wafer
                for r in records_sorted:
                    if r.get("wafer_id") == w_id:
                        image_path = r.get("image_path")
                        if image_path:
                            images_to_show.append({
                                "path": image_path,
                                "wafer_id": w_id,
                                "defect_pct": pct,
                                "defect_class": dclass,
                                "status": "FAIL" if pct > 40 else "PASS"
                            })
                        break
            
            # Add 2 good examples (lowest defect percentage, if available)
            if len(defect_pcts_sorted) > 6:
                good_examples = defect_pcts_sorted[-2:] if len(defect_pcts_sorted) >= 2 else []
                for w_id, pct, mtype, dclass in good_examples:
                    # Find the record for this wafer
                    for r in records_sorted:
                        if r.get("wafer_id") == w_id:
                            image_path = r.get("image_path")
                            if image_path:
                                images_to_show.append({
                                    "path": image_path,
                                    "wafer_id": w_id,
                                    "defect_pct": pct,
                                    "defect_class": dclass,
                                    "status": "PASS"
                                })
                            break
            
            # Display selected images
            if images_to_show:
                story.append(Paragraph("<b>Sample Wafer Images</b>", styles["Heading3"]))
                story.append(Paragraph("<i>Showing representative samples: Top defects and good examples</i>", styles["Normal"]))
                story.append(Spacer(1, 6))
                
                # Display images (up to 8 total: 6 worst + 2 good examples)
                for img_info in images_to_show:
                    image_path = img_info["path"]
                    wafer_id = img_info["wafer_id"]
                    defect_pct = img_info["defect_pct"]
                    defect_class = img_info["defect_class"]
                    status = img_info["status"]
                    
                    # Resolve image path
                    if os.path.exists(image_path):
                        img_path = image_path
                    elif PROCESSED_IMAGES_DIR and os.path.exists(PROCESSED_IMAGES_DIR / Path(image_path).name):
                        img_path = str(PROCESSED_IMAGES_DIR / Path(image_path).name)
                    elif PROCESSED_IMAGES_DIR and os.path.exists(str(PROCESSED_IMAGES_DIR / os.path.basename(image_path))):
                        img_path = str(PROCESSED_IMAGES_DIR / os.path.basename(image_path))
                    else:
                        img_path = None
                    
                    if img_path and os.path.exists(img_path):
                        try:
                            # Create image with smaller size for summary (3 inches wide)
                            img = Image(img_path)
                            max_width = 3 * inch
                            if img.imageWidth > max_width:
                                ratio = max_width / img.imageWidth
                                img.drawWidth = max_width
                                img.drawHeight = img.imageHeight * ratio
                            else:
                                img.drawWidth = img.imageWidth
                                img.drawHeight = img.imageHeight
                            
                            # Add image with caption
                            story.append(Paragraph(
                                f"<b>{self._esc(wafer_id)}</b> - {self._esc(defect_class)} "
                                f"({defect_pct:.2f}%) - <b>{status}</b>",
                                styles["Normal"]
                            ))
                            story.append(img)
                            story.append(Spacer(1, 8))
                        except Exception as e:
                            # Skip if image fails to load
                            pass
                
                story.append(Spacer(1, 12))
        
        # Top 5 worst defect percentages
        defect_pcts_sorted = sorted(defect_pcts, key=lambda x: x[1], reverse=True)[:5]
        if defect_pcts_sorted:
            story.append(Paragraph("<b>Top 5 Highest Defect Percentages</b>", styles["Heading3"]))
            for w_id, pct, mtype, dclass in defect_pcts_sorted:
                story.append(Paragraph(
                    f"• {self._esc(w_id)} ({self._esc(mtype)}): {pct:.2f}% [{self._esc(dclass)}]",
                    styles["Normal"]
                ))
            story.append(Spacer(1, 12))
        
        # Machine type distribution
        if by_machine_type:
            story.append(Paragraph("<b>Distribution by Machine Type</b>", styles["Heading3"]))
            for mtype, count in sorted(by_machine_type.items(), key=lambda x: x[1], reverse=True):
                story.append(Paragraph(f"• {self._esc(mtype)}: {count} wafers", styles["Normal"]))
            story.append(Spacer(1, 12))
        
        # Defect class distribution
        if by_defect_class:
            story.append(Paragraph("<b>Distribution by Defect Class</b>", styles["Heading3"]))
            for dclass, count in sorted(by_defect_class.items(), key=lambda x: x[1], reverse=True):
                story.append(Paragraph(f"• {self._esc(dclass)}: {count} wafers", styles["Normal"]))
            story.append(Spacer(1, 12))
        
        # LLM Batch Summary
        if use_llm:
            batch_stats = {
                "counts": {
                    "total_wafers": total,
                    "pass": pass_count,
                    "fail": fail_count,
                    "pass_rate_percent": pass_rate
                },
                "distribution": {
                    "by_machine_type": by_machine_type,
                    "by_defect_class": by_defect_class
                },
                "worst_defect_percentages_top5": [
                    {"wafer_id": w_id, "defect_percentage": pct, "machine_type": mtype, "defect_class": dclass}
                    for (w_id, pct, mtype, dclass) in defect_pcts_sorted
                ]
            }
            
            # Get summary (LLM or fallback)
            llm_sum = self._get_llm_batch_summary(batch_stats)
            
            # Determine if this is LLM-generated or fallback
            # Check if summary looks like it came from LLM (more detailed) or fallback (structured)
            agent = self._get_llm_agent()
            is_llm_generated = (
                agent and agent.client and 
                "This batch processed" not in llm_sum.get("summary_text", "")[:50]
            )
            
            if is_llm_generated:
                story.append(Paragraph("<b>AI-Enhanced Engineering Summary</b>", styles["Heading2"]))
            else:
                story.append(Paragraph("<b>Engineering Summary</b>", styles["Heading2"]))
            
            story.append(Spacer(1, 6))
            story.append(Paragraph(self._esc(llm_sum.get("summary_text", "")), styles["Normal"]))
            story.append(Spacer(1, 6))
            story.append(Paragraph(
                f"<b>Estimated batch yield impact:</b> {self._esc(llm_sum.get('estimated_batch_yield_impact', 'Medium'))}",
                styles["Normal"]
            ))
            
            risks = llm_sum.get("key_risks", [])
            if isinstance(risks, list) and risks:
                story.append(Spacer(1, 8))
                story.append(Paragraph("<b>Key Risks</b>", styles["Heading3"]))
                for x in risks:
                    story.append(Paragraph(f"• {self._esc(x)}", styles["Normal"]))
            
            actions = llm_sum.get("recommended_actions", [])
            if isinstance(actions, list) and actions:
                story.append(Spacer(1, 8))
                story.append(Paragraph("<b>Recommended Actions</b>", styles["Heading3"]))
                for x in actions:
                    story.append(Paragraph(f"• {self._esc(x)}", styles["Normal"]))
        
        # Per-Wafer Decisions Section (only if include_per_wafer_details is True, shown after summary)
        if include_per_wafer_details:
            story.append(PageBreak())
            story.append(Paragraph("<b>Per-Wafer Decisions</b>", styles["Heading2"]))
            story.append(Spacer(1, 8))
            
            # Process each wafer for detailed display
            for i, r in enumerate(records_sorted, start=1):
                wafer_id = r.get("wafer_id", "unknown")
                machine_type = r.get("machine_type", "")
                machine_id = r.get("machine_id", "")
                ts = r.get("timestamp", "")
                
                defect_class = self._get_pred(r, "Defect Class", "")
                conf = self._get_pred(r, "Confidence Score", None)
                defect_pct = self._get_defect_pct(r)
                
                decision = str(r.get("quality_status", "")).upper().strip()
                reason = r.get("quality_reason", "")
                
                # Wafer block
                story.append(Paragraph(f"<b>Wafer {i} Inspection Decision</b>", styles["Heading3"]))
                story.append(Paragraph(f"<b>Wafer ID:</b> {self._esc(wafer_id)}", styles["Normal"]))
                story.append(Paragraph(f"<b>Decision:</b> {self._esc(decision)}", styles["Normal"]))
                story.append(Spacer(1, 6))
                
                story.append(Paragraph(f"<b>Machine Type:</b> {self._esc(machine_type)}", styles["Normal"]))
                story.append(Paragraph(f"<b>Machine ID:</b> {self._esc(machine_id)}", styles["Normal"]))
                story.append(Paragraph(f"<b>Timestamp:</b> {self._esc(ts)}", styles["Normal"]))
                # Show simulation date if available
                sim_date = r.get("simulation_date")
                if sim_date:
                    story.append(Paragraph(f"<b>Simulation Date:</b> {self._esc(sim_date)}", styles["Normal"]))
                story.append(Spacer(1, 6))
                
                story.append(Paragraph(f"<b>Defect Class:</b> {self._esc(defect_class)}", styles["Normal"]))
                if isinstance(conf, (int, float)):
                    story.append(Paragraph(f"<b>Confidence Score:</b> {conf:.4f}", styles["Normal"]))
                else:
                    story.append(Paragraph(f"<b>Confidence Score:</b> {self._esc(conf)}", styles["Normal"]))
                
                if isinstance(defect_pct, (int, float)):
                    story.append(Paragraph(f"<b>Defect Percentage:</b> {defect_pct:.2f}%", styles["Normal"]))
                else:
                    story.append(Paragraph(f"<b>Defect Percentage:</b> {self._esc(defect_pct)}", styles["Normal"]))
                
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"<b>Reason:</b> {self._esc(reason)}", styles["Normal"]))
                
                # Add wafer defect image if available
                image_path = r.get("image_path")
                if image_path:
                    # Check if image file exists (handle both absolute and relative paths)
                    # Try absolute path first
                    if os.path.exists(image_path):
                        img_path = image_path
                    # Try relative to PROCESSED_IMAGES_DIR
                    elif PROCESSED_IMAGES_DIR and os.path.exists(PROCESSED_IMAGES_DIR / Path(image_path).name):
                        img_path = str(PROCESSED_IMAGES_DIR / Path(image_path).name)
                    # Try just the filename in PROCESSED_IMAGES_DIR
                    elif PROCESSED_IMAGES_DIR and os.path.exists(str(PROCESSED_IMAGES_DIR / os.path.basename(image_path))):
                        img_path = str(PROCESSED_IMAGES_DIR / os.path.basename(image_path))
                    else:
                        img_path = None
                    
                    if img_path and os.path.exists(img_path):
                        try:
                            story.append(Spacer(1, 6))
                            story.append(Paragraph("<b>Wafer Defect Image:</b>", styles["Normal"]))
                            story.append(Spacer(1, 3))
                            
                            # Create image with appropriate size (max 4 inches wide, maintain aspect ratio)
                            img = Image(img_path)
                            # Calculate size to fit within 4 inches width while maintaining aspect ratio
                            max_width = 4 * inch
                            if img.imageWidth > max_width:
                                ratio = max_width / img.imageWidth
                                img.drawWidth = max_width
                                img.drawHeight = img.imageHeight * ratio
                            else:
                                # Keep original size if smaller than max
                                img.drawWidth = img.imageWidth
                                img.drawHeight = img.imageHeight
                            
                            story.append(img)
                            story.append(Spacer(1, 6))
                        except Exception as e:
                            # If image loading fails, just skip it
                            story.append(Paragraph(f"<i>Image unavailable: {self._esc(str(e))}</i>", styles["Normal"]))
                            story.append(Spacer(1, 6))
                
                story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        print(f"PDF report saved to: {filepath}")
        return filepath


# ------------------------------------------------------------------------------------------
# Main Entry Point for Testing
# ------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*70)
    print("SUMMARY GENERATOR TEST")
    print("="*70)
    
    generator = SummaryGenerator()
    
    print("\nGenerating text summary...")
    summary = generator.generate_text_summary(use_llm=False)  # Test without LLM first
    print(summary)
    
    print("\nSaving summary to file...")
    generator.save_summary(use_llm=False)
    
    print("\nGenerating JSON report...")
    generator.save_json_report()

