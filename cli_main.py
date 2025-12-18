#!/usr/bin/env python3
"""
QuantumCore Nexus - Main Entry Point
Integration of all CLI components
"""

import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from cli_demos import QuantumCoreNexusDemos

class QuantumCoreNexusCLI(QuantumCoreNexusDemos):
    """Complete QuantumCore Nexus CLI with all functionality"""
    
    def main_menu(self):
        """Main menu loop"""
        while True:
            self.print_banner()
            
            # Status indicators
            system_status = self.colorize("Ready", 'green')
            module_status = self.colorize("Basic", 'yellow')
            results_status = f"{len(self.results_history)}"
            demo_status = f"{len(self.demonstration_results)}"
            
            options = [
                {
                    'description': 'Quantum Demonstrations',
                    'action': 'demo_menu',
                    'shortcut': 'd',
                    'category': 'Core Features',
                    'status': f'{demo_status} sets'
                },
                {
                    'description': 'Module Management',
                    'action': 'module_management',
                    'shortcut': 'm',
                    'category': 'Integration',
                    'status': module_status
                },
                {
                    'description': 'System Configuration',
                    'action': 'configuration',
                    'shortcut': 'c',
                    'category': 'Settings',
                    'status': 'Settings'
                },
                {
                    'description': 'View Documentation',
                    'action': 'documentation',
                    'shortcut': 'h',
                    'category': 'Help',
                    'status': 'Info'
                },
                {
                    'description': 'Exit QuantumCore Nexus',
                    'action': 'exit',
                    'shortcut': 'x',
                    'category': 'Navigation',
                    'status': self.colorize('Exit', 'red')
                }
            ]
            
            choice = self.print_menu("QUANTUMCORE NEXUS MAIN MENU", options, "Select option: ")
            
            if choice == 'demo_menu':
                self.demo_menu()
            elif choice == 'module_management':
                self.module_management()
            elif choice == 'configuration':
                self.system_configuration()
            elif choice == 'documentation':
                self.show_documentation()
            elif choice == 'exit':
                print(f"\n{self.colorize('=' * 70, 'cyan')}")
                print(self.colorize("Thank you for using QuantumCore Nexus!", 'green', True))
                print(self.colorize("Scientific Quantum Simulation Platform", 'cyan'))
                print(self.colorize(f"Session: {self.session_id}", 'yellow'))
                print(self.colorize("=" * 70, 'cyan'))
                
                # Save session summary
                self._save_session_summary()
                break
    
    def module_management(self):
        """Module management menu"""
        self.print_banner()
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("MODULE MANAGEMENT", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        print(f"\n{self.colorize('External module integration for advanced features:', 'yellow')}")
        print("  • sentiflow: Quantum circuit framework")
        print("  • quantum_core_engine: Core quantum operations")
        print("  • qybrik: Circuit building blocks")
        print("  • qylintos: Circuit optimization")
        print("  • cognition_core: Quantum machine learning")
        
        print(f"\n{self.colorize('To download modules, run:', 'cyan')}")
        print("  python scripts/download_modules.py")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def system_configuration(self):
        """System configuration menu"""
        self.print_banner()
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("SYSTEM CONFIGURATION", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        print(f"\n{self.colorize('Current Configuration:', 'yellow')}")
        print(f"  {self.colorize('Working Directory:', 'cyan')} {Path(__file__).parent}")
        print(f"  {self.colorize('Log Directory:', 'cyan')} {self.log_dir}")
        print(f"  {self.colorize('Data Directory:', 'cyan')} {self.data_dir}")
        print(f"  {self.colorize('Session ID:', 'cyan')} {self.session_id}")
        print(f"  {self.colorize('Python Version:', 'cyan')} {self.system_info['python_version'].split()[0]}")
        print(f"  {self.colorize('Available RAM:', 'cyan')} {self.system_info['available_ram_gb']:.1f} GB")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def show_documentation(self):
        """Show documentation"""
        self.print_banner()
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("QUANTUMCORE NEXUS DOCUMENTATION", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        docs = f"""
{self.colorize('QUICK START:', 'yellow', True)}
  1. Run {self.colorize('Quantum Demonstrations', 'cyan')} to see quantum effects
  2. Use {self.colorize('Custom Quantum System', 'cyan')} for interactive experiments
  3. Run {self.colorize('Performance Benchmark', 'cyan')} to test system capabilities

{self.colorize('KEY FEATURES:', 'yellow', True)}
  • {self.colorize('Qubit Demonstrations:', 'cyan')} Bell states, teleportation, QFT
  • {self.colorize('Qudit Systems:', 'cyan')} Multi-level quantum systems
  • {self.colorize('Quantum Walk:', 'cyan')} Quantum dynamics simulation
  • {self.colorize('Performance Analysis:', 'cyan')} System benchmarking

{self.colorize('DEMONSTRATIONS:', 'yellow', True)}
  • {self.colorize('Bell State:', 'cyan')} 2-qubit entanglement
  • {self.colorize('GHZ State:', 'cyan')} Multi-qubit entanglement
  • {self.colorize('Quantum Teleportation:', 'cyan')} State transfer protocol
  • {self.colorize('Superdense Coding:', 'cyan')} 2 classical bits in 1 qubit
  • {self.colorize('Quantum Fourier Transform:', 'cyan')} Quantum algorithm
  • {self.colorize('Quantum Walk:', 'cyan')} Quantum dynamics

{self.colorize('COMMAND LINE USAGE:', 'yellow', True)}
  {self.colorize('python cli_main.py', 'cyan')} - Interactive mode
  {self.colorize('python cli_main.py --demo qubit --qubits 8', 'cyan')}
  {self.colorize('python cli_main.py --benchmark', 'cyan')}

{self.colorize('TROUBLESHOOTING:', 'yellow', True)}
  • Install requirements: pip install numpy scipy
  • For visualization: pip install matplotlib
  • Memory issues: Reduce qubit count in benchmarks
        """
        
        print(docs)
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def _save_session_summary(self):
        """Save session summary before exit"""
        try:
            summary = {
                'session_id': self.session_id,
                'start_time': self.session_id,
                'end_time': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'results_count': len(self.results_history),
                'demonstration_count': len(self.demonstration_results),
                'system_info': self.system_info
            }
            
            summary_file = self.log_dir / f"session_summary_{self.session_id}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n{self.colorize('Session summary saved:', 'cyan')} {summary_file}")
            
        except Exception as e:
            print(f"{self.colorize('⚠️  Could not save session summary:', 'yellow')} {e}")
    
    def run(self):
        """Run the CLI"""
        try:
            # Initialize
            print(f"\n{self.colorize('Initializing QuantumCore Nexus...', 'cyan')}")
            
            # Show welcome
            self.print_banner()
            
            print(f"\n{self.colorize('Welcome to QuantumCore Nexus!', 'green', True)}")
            print(f"{self.colorize('Scientific Quantum Simulation Platform', 'cyan')}")
            print(f"\n{self.colorize('Session:', 'yellow')} {self.session_id}")
            print(f"{self.colorize('System:', 'yellow')} {self.system_info['platform'].split('-')[0]}")
            print(f"{self.colorize('Python:', 'yellow')} {self.system_info['python_version'].split()[0]}")
            print(f"{self.colorize('Available RAM:', 'yellow')} {self.system_info['available_ram_gb']:.1f} GB")
            
            input(f"\n{self.colorize('Press Enter to continue to main menu...', 'yellow')}")
            
            # Start main menu
            self.main_menu()
            
        except KeyboardInterrupt:
            print(f"\n\n{self.colorize('Interrupted by user. Exiting gracefully...', 'yellow')}")
            self._save_session_summary()
        except Exception as e:
            print(f"\n{self.colorize('❌ Fatal error:', 'red')} {e}")
            import traceback
            traceback.print_exc()
            self._save_session_summary()

def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(
        description="QuantumCore Nexus - Scientific Quantum Simulation Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_main.py                     # Interactive mode
  python cli_main.py --demo qubit        # Run qubit demonstrations
  python cli_main.py --benchmark         # Run performance benchmark
  python cli_main.py --qubits 10         # Test with 10 qubits
        
For interactive use, run without arguments.
        """
    )
    
    parser.add_argument('--demo', type=str, choices=['qubit', 'qudit', 'all'],
                       help='Run specific demonstration non-interactively')
    parser.add_argument('--qubits', type=int, default=8,
                       help='Number of qubits for demonstrations')
    parser.add_argument('--output', type=str,
                       help='Output file for results')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--version', action='version', 
                       version='QuantumCore Nexus v1.0.0')
    
    args = parser.parse_args()
    
    # Handle command line arguments
    if args.demo or args.benchmark:
        # Non-interactive mode
        cli = QuantumCoreNexusCLI()
        
        if args.benchmark:
            cli.performance_benchmark()
        elif args.demo:
            # Run demonstrations
            if args.demo == 'qubit':
                print(f"{cli.colorize('Running qubit demonstrations...', 'cyan')}")
                print(f"{cli.colorize(f'Maximum qubits: {args.qubits}', 'yellow')}")
                
                # Run qubit demos
                cli.qubit_demonstrations()
            
            elif args.demo == 'qudit':
                print(f"{cli.colorize('Running qudit demonstrations...', 'cyan')}")
                cli.qudit_demonstrations()
        
        return
    
    # Interactive mode
    cli = QuantumCoreNexusCLI()
    cli.run()

if __name__ == "__main__":
    main()