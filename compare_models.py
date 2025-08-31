#!/usr/bin/env python3
"""
Model comparison script for analyzing multiple trained models
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.evaluation.comparison import ModelComparison

def main():
    parser = argparse.ArgumentParser(description="Compare trained sentiment analysis models")
    parser.add_argument("--experiments", nargs='+', required=True,
                       help="List of experiment names to compare")
    parser.add_argument("--results_dir", type=str, default="experiments",
                       help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default="comparison_results",
                       help="Directory to save comparison results")
    parser.add_argument("--report", action="store_true",
                       help="Generate HTML comparison report")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize comparison tool
    comparator = ModelComparison(results_dir=args.results_dir)
    
    print(f"Loading results from {len(args.experiments)} experiments...")
    
    # Load experiment results
    results = comparator.load_experiment_results(args.experiments)
    
    if not results:
        print("No valid experiment results found!")
        return
    
    print(f"Successfully loaded results for: {list(results.keys())}")
    
    # Create comparison table
    print("\nCreating comparison table...")
    comparison_df = comparator.create_comparison_table(results)
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison table
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    
    # Generate visualizations
    print("\nGenerating comparison plots...")
    
    # Training curves
    comparator.plot_training_curves(results, save_path=output_dir / "training_curves.png")
    
    # Performance comparison
    comparator.plot_performance_comparison(results, save_path=output_dir / "performance_comparison.png")
    
    # Confusion matrices
    comparator.plot_confusion_matrices(results, save_path=output_dir / "confusion_matrices.png")
    
    # Generate HTML report if requested
    if args.report:
        print("\nGenerating HTML comparison report...")
        report_path = output_dir / "comparison_report.html"
        comparator.generate_comparison_report(results, save_path=str(report_path))
        print(f"Report saved to: {report_path}")
    
    print(f"\nComparison results saved to: {output_dir}")

if __name__ == "__main__":
    main()