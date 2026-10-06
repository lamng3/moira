"""
OOT Explorer - CLI for exploring thought graphs from DynamoDB

This tool provides a simple command-line interface for the enhanced OOTReconstructor.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from moira.agents import OOTReconstructor


"""
Usage:
    python oot_explorer.py --help
    python oot_explorer.py --table my-table --list-runs
    python oot_explorer.py --table my-table --load latest
    python oot_explorer.py --table my-table --load first-5
    python oot_explorer.py --table my-table --load run-id-123
    python oot_explorer.py --table my-table --search "environment"
    python oot_explorer.py --table my-table --visualize --output graph.png
"""
def main():
    parser = argparse.ArgumentParser(description="OOT Explorer - Explore thought graphs from DynamoDB")
    
    # Required arguments
    parser.add_argument("--table", type=str, required=True,
                       help="DynamoDB table name")
    
    # Action arguments
    parser.add_argument("--list-runs", action="store_true",
                       help="List available run IDs in the table")
    parser.add_argument("--load", type=str, 
                       choices=["latest", "first-5", "all"],
                       help="Load thought graph: latest, first-5, or all runs")
    parser.add_argument("--load-run", type=str,
                       help="Load specific run ID")
    parser.add_argument("--search", type=str,
                       help="Search for thoughts containing this string")
    parser.add_argument("--visualize", action="store_true",
                       help="Create a visualization of the thought graph")
    parser.add_argument("--output", type=str,
                       help="Output file for visualization")
    parser.add_argument("--max-nodes", type=int, default=50,
                       help="Maximum number of nodes for visualization")
    parser.add_argument("--layout", type=str, default="spring",
                       choices=["spring", "hierarchical"],
                       help="Layout algorithm for visualization")
    parser.add_argument("--stats", action="store_true",
                       help="Print detailed statistics")
    parser.add_argument("--summary", action="store_true",
                       help="Print summary of loaded thought graph")
    
    # AWS configuration
    parser.add_argument("--aws-region", type=str,
                       help="AWS region for DynamoDB")
    parser.add_argument("--aws-profile", type=str,
                       help="AWS profile for DynamoDB")
    
    args = parser.parse_args()
    
    # Create reconstructor
    try:
        reconstructor = OOTReconstructor(
            table=args.table,
            region=args.aws_region,
            profile=args.aws_profile
        )
        print(f"Connected to DynamoDB table: {args.table}")
    except Exception as e:
        print(f"Error connecting to DynamoDB: {e}")
        return 1
    
    # List runs if requested
    if args.list_runs:
        reconstructor.print_available_runs(limit=20)
        return 0
    
    # Load thought graph if requested
    if args.load or args.load_run:
        try:
            if args.load_run:
                print(f"Loading specific run: {args.load_run}")
                reconstructor.load_by_run_id(args.load_run)
            elif args.load == "latest":
                print("Loading latest run...")
                reconstructor.load_latest()
            elif args.load == "first-5":
                print("Loading first 5 runs...")
                reconstructor.load_latest_n(5)
            elif args.load == "all":
                print("Loading all available runs...")
                run_ids = reconstructor.list_available_run_ids(limit=100)
                if run_ids:
                    reconstructor.load_by_run_ids(run_ids)
                else:
                    print("No runs found")
                    return 1
            
            print("✅ Thought graph loaded successfully!")
            
        except Exception as e:
            print(f"Error loading thought graph: {e}")
            return 1
    
    # If no load action was specified, try to load latest by default
    if not (args.load or args.load_run) and (args.search or args.visualize or args.stats or args.summary):
        try:
            print("No load action specified, loading latest run by default...")
            reconstructor.load_latest()
            print("✅ Latest thought graph loaded successfully!")
        except Exception as e:
            print(f"Error loading latest thought graph: {e}")
            return 1
    
    # Print summary if requested
    if args.summary:
        reconstructor.print_summary()
    
    # Print statistics if requested
    if args.stats:
        reconstructor.print_statistics()
    
    # Search if requested
    if args.search:
        print(f"\nSearching for: '{args.search}'")
        results = reconstructor.search_thoughts(args.search)
        reconstructor.print_search_results(results)
    
    # Visualize if requested
    if args.visualize:
        print(f"Creating visualization...")
        if args.output:
            output_file = reconstructor.visualize(output_file=args.output, max_nodes=args.max_nodes, layout=args.layout)
        else:
            output_file = reconstructor.visualize(max_nodes=args.max_nodes, layout=args.layout)
        
        if output_file:
            print(f"Visualization saved to: {output_file}")
        else:
            print("Visualization creation failed")
    
    # If no specific action was requested, show summary
    if not any([args.list_runs, args.load, args.load_run, args.search, args.visualize, args.stats, args.summary]):
        print("No specific action requested. Available actions:")
        print("  --list-runs     : List available run IDs")
        print("  --load latest   : Load latest run")
        print("  --load first-5  : Load first 5 runs")
        print("  --load-run ID   : Load specific run ID")
        print("  --search QUERY  : Search for thoughts")
        print("  --visualize     : Create graph visualization")
        print("  --stats         : Print detailed statistics")
        print("  --summary       : Print graph summary")
        print("\nUse --help for more information.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
