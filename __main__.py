"""
QuantumCore Nexus - Main Entry Point
Scientific-grade quantum simulation platform
"""

import argparse
import json
import numpy as np
from datetime import datetime
import sys

from src.demonstrations.qubit_demos import QuantumDemonstrationSuite
from src.integration.module_bridge import QuantumModuleIntegrator
from src.integration.report_generator import QuantumReportGenerator

def main():
    parser = argparse.ArgumentParser(
        description="QuantumCore Nexus - Scientific Quantum Simulation Platform"
    )
    parser.add_argument("--demo", choices=["qubit", "qudit", "all"], 
                       default="qubit", help="Type of demonstration to run")
    parser.add_argument("--max-qubits", type=int, default=8,
                       help="Maximum number of qubits for demonstrations")
    parser.add_argument("--validate", action="store_true",
                       help="Run quantum principle validation")
    parser.add_argument("--integrate", action="store_true",
                       help="Check for and integrate with existing modules")
    parser.add_argument("--output", type=str, default="report.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("QUANTUMCORE NEXUS - Scientific Quantum Simulation Platform")
    print("="*70)
    
    # Check for module integration
    if args.integrate:
        print("\n🔍 Checking for available quantum modules...")
        integrator = QuantumModuleIntegrator()
        print(f"  Found {sum(1 for m in integrator.available_modules.values() if m.get('available'))} modules")
    
    # Run demonstrations
    print(f"\n🚀 Running {args.demo} demonstrations...")
    
    if args.demo in ["qubit", "all"]:
        demo_suite = QuantumDemonstrationSuite(system_type="qubit")
        results = demo_suite.run_all_demos(max_qubits=args.max_qubits)
        
        # Generate report
        print("\n📊 Generating scientific report...")
        report_generator = QuantumReportGenerator()
        report = report_generator.generate_experiment_report(
            demo_results=results,
            validation_results={}  # Would include validation results
        )
        
        # Save report
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, cls=report_generator.NumpyEncoder)
        
        print(f"  Report saved to {args.output}")
    
    print("\n" + "="*70)
    print("✅ QuantumCore Nexus execution completed")
    print("="*70)

if __name__ == "__main__":
    main()