from typing import Dict, Any
import json
import os

class ReportGenerator:
    """Generates an HTML report from evaluation results."""
    
    def generate(self, data: Dict[str, Any], output_path: str):
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>VIRK Evaluation Report</title>
            <style>
                body {{ font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1, h2 {{ color: #333; }}
                .metric {{ background: #f4f4f4; padding: 10px; margin-bottom: 10px; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .good {{ color: green; font-weight: bold; }}
                .bad {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>VIRK Evaluation Report</h1>
            
            <h2>1. Drift Detection Performance</h2>
            <div class="metric">
                <strong>Overall AUROC:</strong> {data['detection_overall']['auroc']:.4f}<br>
                Clean Mean Score: {data['detection_overall']['clean_mean']:.4f} (std: {data['detection_overall']['clean_std']:.4f})<br>
                Drifted Mean Score: {data['detection_overall']['shifted_mean']:.4f} (std: {data['detection_overall']['shifted_std']:.4f})
            </div>
            
            <h3>Per-Shift Performance</h3>
            <table>
                <tr>
                    <th>Shift</th>
                    <th>AUROC</th>
                    <th>Clean Mean</th>
                    <th>Shift Mean</th>
                </tr>
        """
        
        for shift, metrics in data['per_shift_detection'].items():
            html += f"""
                <tr>
                    <td>{shift}</td>
                    <td>{metrics['auroc']:.4f}</td>
                    <td>{metrics['clean_mean']:.4f}</td>
                    <td>{metrics['shifted_mean']:.4f}</td>
                </tr>
            """
            
        html += """
            </table>
            
            <h2>2. Fingerprinting Accuracy</h2>
        """
        
        acc = data['fingerprint_overall']['accuracy']
        cm_labels = data['fingerprint_overall']['labels']
        cm_data = data['fingerprint_overall']['confusion_matrix']
        
        html += f"""
            <div class="metric">
                <strong>Overall Accuracy:</strong> {acc:.4f}
            </div>
            
            <h3>Confusion Matrix</h3>
            <table>
                <tr>
                    <th>True \ Pred</th>
        """
        
        for label in cm_labels:
            html += f"<th>{label}</th>"
        html += "</tr>"
        
        for idx, row in enumerate(cm_data):
            html += f"<tr><td><b>{cm_labels[idx]}</b></td>"
            for val in row:
                style = "background-color: #e6fffa;" if val > 0 else ""
                html += f"<td style='{style}'>{val}</td>"
            html += "</tr>"
            
        html += """
            </table>
        </body>
        </html>
        """
        
        with open(output_path, "w") as f:
            f.write(html)
        print(f"Report saved to {output_path}")
