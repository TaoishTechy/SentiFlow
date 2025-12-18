#!/usr/bin/env python3
"""
QuantumCore Nexus - Scientific-Grade Quantum Simulation Platform
Main Entry Point with Interactive Menu System
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# In the imports section of main.py, update to:
try:
    from quantum_core_nexus.demonstrations.qubit_demos import QuantumDemonstrationSuite
    from quantum_core_nexus.demonstrations.qudit_demos import QuditDemonstrationSuite  # Add this
    from quantum_core_nexus.demonstrations.advanced_demos import AdvancedQuantumApplications
    from quantum_core_nexus.integration.module_bridge import AdvancedQuantumModuleIntegrator
    from quantum_core_nexus.integration.report_generator import QuantumReportGenerator
    from quantum_core_nexus.integration.visualization_engine import QuantumVisualizationEngine
    from quantum_core_nexus.core.qubit_system import QubitSystem
    from quantum_core_nexus.core.qudit_system import QuditSystem  # Add this
    from quantum_core_nexus.validation.scientific_validator import QuantumValidator
    from quantum_core_nexus.utils.numpy_encoder import NumpyEncoder
except ImportError as e:
    print(f"Error importing QuantumCore Nexus modules: {e}")
    print("Make sure you have installed the package or are running from the correct directory.")
    sys.exit(1)

class QuantumCoreNexusCLI:
    """
    Interactive Command Line Interface for QuantumCore Nexus
    """
    
    def __init__(self):
        self.integrator = None
        self.current_system = None
        self.results_history = []
        self.config = self._load_config()
        self._setup_logging()
        
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        config_path = Path(__file__).parent / "config" / "modules.yaml"
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {
            "module_priorities": {
                "high": ["sentiflow", "quantum_core_engine", "qybrik"],
                "medium": ["bumpy", "flumpy", "qylintos"],
                "low": ["laser", "bugginrace", "cognition_core"]
            }
        }
    
    def _setup_logging(self):
        """Setup logging directory"""
        self.log_dir = Path(__file__).parent / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_banner(self):
        """Print application banner"""
        self.clear_screen()
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ╔═╗┬ ┬┌─┐┌─┐┌┬┐┌─┐┌─┐  ╔═╗┌┐┌┌─┐┬ ┬┌─┐┌┐┌┌┬┐┌─┐  ╔═╗┌─┐┌┬┐┌─┐┬─┐┌┬┐┌─┐  ║
║   ║ ╦│ │├┤ ├─┤ │ ├┤ └─┐  ╠╣ ││││ ││ │├┤ │││ │ ├┤   ╠═╝├─┤ │ ├┤ ├┬┘│││└─┐  ║
║   ╚═╝└─┘└─┘┴ ┴ ┴ └─┘└─┘  ╚  ┘└┘└─┘└─┘└─┘┘└┘ ┴ └─┘  ╩  ┴ ┴ ┴ └─┘┴└─┴ ┴└─┘  ║
║                                                                              ║
║   Scientific-Grade Quantum Simulation Platform v1.0.0                        ║
║   "From Demonstration to Discovery"                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print(" " * 20 + "=" * 50)
        print(" " * 20 + "MODULAR QUANTUM SIMULATION WITH SCIENTIFIC VALIDATION")
        print(" " * 20 + "=" * 50)
        print()
    
    def print_menu(self, title: str, options: List[Dict[str, Any]], prompt: str = "Select an option: "):
        """Print a formatted menu"""
        print(f"\n{title}")
        print("-" * 70)
        
        for i, option in enumerate(options, 1):
            shortcut = option.get('shortcut', '')
            description = option['description']
            status = option.get('status', '')
            
            if status:
                print(f"  [{i:2d}] {description:50} [{status:10}] {shortcut}")
            else:
                print(f"  [{i:2d}] {description:50} {shortcut}")
        
        print("-" * 70)
        
        while True:
            try:
                choice = input(f"\n{prompt}").strip()
                if choice == '':
                    return None
                
                # Check for shortcut keys
                for option in options:
                    if option.get('shortcut', '').lower() == choice.lower():
                        return option.get('action')
                
                # Check numeric choice
                choice_num = int(choice)
                if 1 <= choice_num <= len(options):
                    return options[choice_num - 1].get('action')
                else:
                    print(f"Please enter a number between 1 and {len(options)}")
            except ValueError:
                print("Invalid input. Please enter a number or shortcut.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return None
    
    def initialize_integration(self):
        """Initialize module integration"""
        print("\n" + "=" * 70)
        print("MODULE INTEGRATION INITIALIZATION")
        print("=" * 70)
        
        try:
            self.integrator = AdvancedQuantumModuleIntegrator()
            
            # Count available modules
            available = [name for name, mod in self.integrator.modules.items() 
                        if mod.get('available', False)]
            
            print(f"\n✅ Integration initialized successfully!")
            print(f"   Available modules: {len(available)}/{len(self.integrator.modules)}")
            
            if available:
                print("\nDetected modules:")
                for name in available:
                    mod_info = self.integrator.modules[name]
                    print(f"  • {name:20} v{mod_info.get('version', '?.?.?')}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to initialize integration: {e}")
            return False
    
    def demo_menu(self):
        """Main demonstration menu"""
        while True:
            self.print_banner()
            
            options = [
                {
                    'description': 'Qubit Demonstrations',
                    'action': 'qubit_demos',
                    'shortcut': 'q'
                },
                {
                    'description': 'Qudit Demonstrations',
                    'action': 'qudit_demos',
                    'shortcut': 'd'
                },
                {
                    'description': 'Advanced Quantum Applications',
                    'action': 'advanced_demos',
                    'shortcut': 'a'
                },
                {
                    'description': 'Custom Quantum System',
                    'action': 'custom_system',
                    'shortcut': 'c'
                },
                {
                    'description': 'Performance Benchmarking',
                    'action': 'benchmark',
                    'shortcut': 'b'
                },
                {
                    'description': 'Return to Main Menu',
                    'action': 'main_menu',
                    'shortcut': 'm'
                }
            ]
            
            choice = self.print_menu("DEMONSTRATION SUITE", options, "Select demonstration type: ")
            
            if choice == 'qubit_demos':
                self.qubit_demonstrations()
            elif choice == 'qudit_demos':
                self.qudit_demonstrations()
            elif choice == 'advanced_demos':
                self.advanced_demonstrations()
            elif choice == 'custom_system':
                self.custom_quantum_system()
            elif choice == 'benchmark':
                self.performance_benchmark()
            elif choice == 'main_menu':
                break
            else:
                break
    
    def qubit_demonstrations(self):
        """Run qubit demonstrations"""
        self.print_banner()
        print("\n" + "=" * 70)
        print("QUBIT DEMONSTRATIONS")
        print("=" * 70)
        
        # Get system size
        while True:
            try:
                max_qubits = input("\nEnter maximum qubits to test (2-16, default=8): ").strip()
                if max_qubits == '':
                    max_qubits = 8
                else:
                    max_qubits = int(max_qubits)
                    if 2 <= max_qubits <= 16:
                        break
                    else:
                        print("Please enter a number between 2 and 16")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        # Get number of repetitions
        while True:
            try:
                repetitions = input("Enter measurement repetitions (default=10000): ").strip()
                if repetitions == '':
                    repetitions = 10000
                else:
                    repetitions = int(repetitions)
                    if repetitions > 0:
                        break
                    else:
                        print("Please enter a positive number")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        # Select demonstrations
        print("\nSelect demonstrations to run:")
        print("  1. Bell State (2-qubit entanglement)")
        print("  2. GHZ State (multi-qubit entanglement)")
        print("  3. Quantum Teleportation")
        print("  4. Superdense Coding")
        print("  5. Quantum Fourier Transform")
        print("  6. All of the above")
        
        demo_choice = input("\nEnter choice (1-6, default=6): ").strip()
        if demo_choice == '':
            demo_choice = '6'
        
        # Run demonstrations
        print("\n" + "=" * 70)
        print("RUNNING QUBIT DEMONSTRATIONS")
        print("=" * 70)
        
        start_time = time.time()
        demo_suite = QuantumDemonstrationSuite(system_type="qubit")
        
        if demo_choice in ['1', '6']:
            print("\n▶ Running Bell State Demonstration...")
            bell_results = demo_suite.demo_bell_state()
            self.results_history.append(('bell_state', bell_results))
            print("  ✓ Bell state demonstration complete")
        
        if demo_choice in ['2', '6']:
            print("\n▶ Running GHZ State Demonstration...")
            ghz_results = demo_suite.demo_ghz_state(max_qubits)
            self.results_history.append(('ghz_state', ghz_results))
            print("  ✓ GHZ state demonstration complete")
        
        if demo_choice in ['3', '6'] and max_qubits >= 3:
            print("\n▶ Running Quantum Teleportation...")
            teleport_results = demo_suite.demo_teleportation()
            self.results_history.append(('teleportation', teleport_results))
            print("  ✓ Quantum teleportation demonstration complete")
        
        if demo_choice in ['4', '6']:
            print("\n▶ Running Superdense Coding...")
            coding_results = demo_suite.demo_superdense_coding()
            self.results_history.append(('superdense_coding', coding_results))
            print("  ✓ Superdense coding demonstration complete")
        
        elapsed = time.time() - start_time
        print(f"\n✅ All demonstrations completed in {elapsed:.2f} seconds")
        
        # Ask to save results
        self.save_results_menu("qubit")
        
        input("\nPress Enter to continue...")
    
    def qudit_demonstrations(self):
        """Run qudit demonstrations"""
        self.print_banner()
        print("\n" + "=" * 70)
        print("QUDIT DEMONSTRATIONS")
        print("=" * 70)
        
        # Get qudit parameters
        while True:
            try:
                dimension = input("\nEnter qudit dimension (d > 2, default=3): ").strip()
                if dimension == '':
                    dimension = 3
                else:
                    dimension = int(dimension)
                    if dimension > 2:
                        break
                    else:
                        print("Qudits must have dimension > 2")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        while True:
            try:
                num_qudits = input("Enter number of qudits (default=2): ").strip()
                if num_qudits == '':
                    num_qudits = 2
                else:
                    num_qudits = int(num_qudits)
                    if num_qudits > 0:
                        break
                    else:
                        print("Please enter a positive number")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        print("\n" + "=" * 70)
        print(f"RUNNING QUDIT DEMONSTRATIONS (d={dimension}, n={num_qudits})")
        print("=" * 70)
        
        try:
            demo_suite = QuditDemonstrationSuite(dimension=dimension)
            
            print("\n▶ Running Generalized Bell States...")
            bell_results = demo_suite.demo_generalized_bell_states()
            self.results_history.append(('qudit_bell', bell_results))
            
            print("\n▶ Running Quantum Fourier Transform...")
            qft_results = demo_suite.demo_quantum_fourier_transform_qudit(num_qudits)
            self.results_history.append(('qudit_qft', qft_results))
            
            print("\n✅ Qudit demonstrations completed")
            
            # Save results
            self.save_results_menu("qudit")
            
        except Exception as e:
            print(f"❌ Error in qudit demonstration: {e}")
        
        input("\nPress Enter to continue...")
    
    def advanced_demonstrations(self):
        """Run advanced quantum applications"""
        if not self.integrator:
            print("❌ Module integration not initialized. Please initialize first.")
            input("\nPress Enter to continue...")
            return
        
        self.print_banner()
        print("\n" + "=" * 70)
        print("ADVANCED QUANTUM APPLICATIONS")
        print("=" * 70)
        
        # Check if cognition_core is available
        if not self.integrator.modules.get("cognition_core", {}).get("available"):
            print("\n⚠️  cognition_core module not available")
            print("   Advanced demonstrations require external modules.")
            print("   Run 'Download Modules' from main menu first.")
            input("\nPress Enter to continue...")
            return
        
        print("\nAvailable advanced demonstrations:")
        print("  1. Quantum Pattern Recognition")
        print("  2. Quantum Machine Learning")
        print("  3. Quantum Error Correction")
        print("  4. Quantum Walks")
        print("  5. Quantum Chemistry Simulation")
        
        choice = input("\nSelect demonstration (1-5): ").strip()
        
        if choice == '1':
            self.quantum_pattern_recognition()
        elif choice == '2':
            self.quantum_machine_learning()
        else:
            print("Demonstration not yet implemented")
        
        input("\nPress Enter to continue...")
    
    def quantum_pattern_recognition(self):
        """Demonstrate quantum pattern recognition"""
        print("\n" + "=" * 70)
        print("QUANTUM PATTERN RECOGNITION")
        print("=" * 70)
        
        try:
            from quantum_core_nexus.demonstrations.advanced_demos import AdvancedQuantumApplications
            
            apps = AdvancedQuantumApplications(self.integrator)
            
            # Create sample pattern data
            import numpy as np
            patterns = np.array([
                [1, 0, 0, 1, 0, 1, 1, 0],  # Pattern A
                [0, 1, 1, 0, 1, 0, 0, 1],  # Pattern B (inverse of A)
                [1, 1, 0, 0, 1, 1, 0, 0],  # Pattern C
            ])
            
            print("\nSample patterns:")
            for i, pattern in enumerate(patterns):
                print(f"  Pattern {i+1}: {pattern}")
            
            # Test pattern to recognize
            test_pattern = np.array([1, 0, 0, 1, 0, 1, 1, 0])  # Should match Pattern A
            
            print(f"\nTest pattern: {test_pattern}")
            print("\n▶ Running quantum pattern recognition...")
            
            result = apps.quantum_pattern_recognition(test_pattern)
            
            print(f"\n✅ Recognition complete:")
            print(f"   Best match: Pattern {result.get('best_match', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 0):.2%}")
            
            self.results_history.append(('pattern_recognition', result))
            
        except Exception as e:
            print(f"❌ Error in pattern recognition: {e}")
    
    def custom_quantum_system(self):
        """Create and manipulate a custom quantum system"""
        self.print_banner()
        print("\n" + "=" * 70)
        print("CUSTOM QUANTUM SYSTEM")
        print("=" * 70)
        
        while True:
            print("\n1. Create Qubit System")
            print("2. Create Qudit System")
            print("3. Apply Gates")
            print("4. Measure")
            print("5. Calculate Metrics")
            print("6. Save System")
            print("7. Load System")
            print("8. Return to Demo Menu")
            
            choice = input("\nSelect option (1-8): ").strip()
            
            if choice == '1':
                self.create_qubit_system()
            elif choice == '2':
                self.create_qudit_system()
            elif choice == '3' and self.current_system:
                self.apply_gates_menu()
            elif choice == '4' and self.current_system:
                self.measure_system()
            elif choice == '5' and self.current_system:
                self.calculate_metrics()
            elif choice == '6' and self.current_system:
                self.save_system()
            elif choice == '7':
                self.load_system()
            elif choice == '8':
                break
            else:
                print("Invalid choice or no system created")
    
    def create_qubit_system(self):
        """Create a custom qubit system"""
        try:
            num_qubits = int(input("Enter number of qubits: "))
            validation = input("Enable strict validation? (y/n, default=n): ").strip().lower()
            
            self.current_system = QubitSystem(
                num_qubits=num_qubits,
                validation_level="strict" if validation == 'y' else "warn"
            )
            
            print(f"✅ Created {num_qubits}-qubit system")
            print(f"   Hilbert dimension: {self.current_system.hilbert_dimension}")
            print(f"   State norm: {self.current_system.state_norm:.12f}")
            
        except Exception as e:
            print(f"❌ Error creating system: {e}")
    
    def apply_gates_menu(self):
        """Apply gates to current system"""
        if not self.current_system:
            print("No system created. Please create a system first.")
            return
        
        print("\nAvailable Gates:")
        print("  H - Hadamard")
        print("  X - Pauli-X")
        print("  Y - Pauli-Y")
        print("  Z - Pauli-Z")
        print("  CX - CNOT (control target)")
        
        gate_input = input("\nEnter gate and targets (e.g., 'H 0' or 'CX 0 1'): ").strip().split()
        
        if not gate_input:
            return
        
        gate_name = gate_input[0].upper()
        targets = list(map(int, gate_input[1:]))
        
        try:
            if gate_name == 'H':
                gate = self.current_system.HADAMARD
            elif gate_name == 'X':
                gate = self.current_system.PAULI_X
            elif gate_name == 'Y':
                gate = self.current_system.PAULI_Y
            elif gate_name == 'Z':
                gate = self.current_system.PAULI_Z
            elif gate_name == 'CX' and len(targets) == 2:
                # Create CNOT gate
                cnot_matrix = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]
                ])
                from quantum_core_nexus.core.unified_quantum_system import QuantumGate
                gate = QuantumGate(cnot_matrix, "CNOT")
            else:
                print(f"Unknown gate or invalid targets: {gate_name}")
                return
            
            success = self.current_system.apply_gate(gate, targets)
            if success:
                print(f"✅ Applied {gate_name} gate to qubits {targets}")
                print(f"   State norm: {self.current_system.state_norm:.12f}")
            else:
                print(f"❌ Failed to apply gate")
                
        except Exception as e:
            print(f"❌ Error applying gate: {e}")
    
    def measure_system(self):
        """Measure the current system"""
        if not self.current_system:
            print("No system created.")
            return
        
        try:
            repetitions = input("Enter number of measurements (default=1000): ").strip()
            repetitions = int(repetitions) if repetitions else 1000
            
            print(f"\n▶ Measuring system {repetitions} times...")
            results = self.current_system.measure(repetitions=repetitions)
            
            print("\n📊 Measurement Results:")
            print("-" * 40)
            
            # Show top 10 results
            sorted_results = sorted(results['probabilities'].items(), 
                                  key=lambda x: x[1], reverse=True)
            
            for i, (state, prob) in enumerate(sorted_results[:10]):
                print(f"  {state}: {prob:.4f} ({results['counts'].get(state, 0)} counts)")
            
            if len(sorted_results) > 10:
                print(f"  ... and {len(sorted_results) - 10} more outcomes")
            
            self.results_history.append(('custom_measurement', results))
            
        except Exception as e:
            print(f"❌ Error measuring system: {e}")
    
    def calculate_metrics(self):
        """Calculate quantum metrics for current system"""
        if not self.current_system:
            print("No system created.")
            return
        
        try:
            from quantum_core_nexus.validation.metric_calculator import QuantumMetricCalculator
            
            state_vector = self.current_system.get_state_vector()
            density_matrix = self.current_system.get_density_matrix()
            
            print("\n📈 Quantum Metrics:")
            print("-" * 40)
            
            # Calculate various metrics
            entropy = QuantumMetricCalculator.calculate_von_neumann_entropy(density_matrix)
            coherence = QuantumMetricCalculator.calculate_coherence(density_matrix)
            purity = QuantumMetricCalculator.calculate_purity(density_matrix)
            
            print(f"  Von Neumann Entropy: {entropy:.6f}")
            print(f"  Quantum Coherence: {coherence:.6f}")
            print(f"  Purity: {purity:.6f}")
            
            # Calculate concurrence for 2-qubit systems
            if self.current_system.config.num_subsystems == 2:
                concurrence = QuantumMetricCalculator.calculate_concurrence(
                    state_vector, (2, 2)
                )
                print(f"  Concurrence: {concurrence:.6f}")
            
            # Calculate subsystem entropies
            n = self.current_system.config.num_subsystems
            if n > 1:
                print(f"\n  Subsystem Entropies:")
                for k in range(1, n):
                    subsystem = list(range(k))
                    entropy = self.current_system.calculate_entropy(subsystem)
                    print(f"    First {k} qubits: {entropy:.6f}")
            
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
    
    def performance_benchmark(self):
        """Run performance benchmarking"""
        self.print_banner()
        print("\n" + "=" * 70)
        print("PERFORMANCE BENCHMARKING")
        print("=" * 70)
        
        print("\nBenchmarking quantum operations...")
        
        import time
        import numpy as np
        
        benchmarks = []
        
        # Benchmark different system sizes
        system_sizes = [2, 4, 6, 8, 10, 12]
        
        for n in system_sizes:
            print(f"\n▶ Benchmarking {n}-qubit system...")
            
            try:
                # Create system
                start = time.time()
                system = QubitSystem(n)
                create_time = time.time() - start
                
                # Benchmark state initialization
                start = time.time()
                system._initialize_state()
                init_time = time.time() - start
                
                # Benchmark gate application
                start = time.time()
                for i in range(min(n, 5)):  # Apply 5 gates
                    system.apply_gate(system.HADAMARD, [i])
                gate_time = time.time() - start
                
                # Benchmark measurement
                start = time.time()
                system.measure(repetitions=1000)
                measure_time = time.time() - start
                
                benchmark = {
                    'qubits': n,
                    'hilbert_dim': 2**n,
                    'create_time': create_time,
                    'init_time': init_time,
                    'gate_time': gate_time,
                    'measure_time': measure_time,
                    'total_time': create_time + init_time + gate_time + measure_time
                }
                
                benchmarks.append(benchmark)
                
                print(f"  ✓ Hilbert dimension: {2**n}")
                print(f"  ✓ Total time: {benchmark['total_time']:.3f}s")
                
            except MemoryError:
                print(f"  ✗ Memory error at {n} qubits")
                break
            except Exception as e:
                print(f"  ✗ Error: {e}")
                break
        
        # Generate benchmark report
        if benchmarks:
            print("\n" + "=" * 70)
            print("BENCHMARK RESULTS")
            print("=" * 70)
            
            print("\nSystem Size vs Time:")
            print("-" * 50)
            print("Qubits | Hilbert Dim | Total Time | Memory Estimate")
            print("-" * 50)
            
            for b in benchmarks:
                memory_estimate = (2**b['qubits']) * 16 / 1024 / 1024  # MB
                print(f"{b['qubits']:6d} | {2**b['qubits']:11d} | {b['total_time']:10.3f}s | {memory_estimate:7.1f} MB")
            
            # Save benchmark results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.json"
            
            with open(self.log_dir / filename, 'w') as f:
                json.dump(benchmarks, f, indent=2, cls=NumpyEncoder)
            
            print(f"\n✅ Benchmark results saved to: {self.log_dir/filename}")
        
        input("\nPress Enter to continue...")
    
    def module_management(self):
        """Module management menu"""
        while True:
            self.print_banner()
            
            if not self.integrator:
                status = "Not Initialized"
            else:
                available = sum(1 for m in self.integrator.modules.values() if m.get('available'))
                status = f"{available}/{len(self.integrator.modules)} Available"
            
            options = [
                {
                    'description': 'Initialize Module Integration',
                    'action': 'init_integration',
                    'shortcut': 'i',
                    'status': 'Ready' if not self.integrator else 'Done'
                },
                {
                    'description': 'Download External Modules',
                    'action': 'download_modules',
                    'shortcut': 'd',
                    'status': 'External'
                },
                {
                    'description': 'List Available Modules',
                    'action': 'list_modules',
                    'shortcut': 'l',
                    'status': status
                },
                {
                    'description': 'Test Module Integration',
                    'action': 'test_modules',
                    'shortcut': 't',
                    'status': 'Test'
                },
                {
                    'description': 'Configure Module Priorities',
                    'action': 'configure_modules',
                    'shortcut': 'c',
                    'status': 'Config'
                },
                {
                    'description': 'Return to Main Menu',
                    'action': 'main_menu',
                    'shortcut': 'm'
                }
            ]
            
            choice = self.print_menu("MODULE MANAGEMENT", options, "Select action: ")
            
            if choice == 'init_integration':
                self.initialize_integration()
                input("\nPress Enter to continue...")
            elif choice == 'download_modules':
                self.download_modules()
            elif choice == 'list_modules':
                self.list_modules()
                input("\nPress Enter to continue...")
            elif choice == 'test_modules':
                self.test_module_integration()
                input("\nPress Enter to continue...")
            elif choice == 'configure_modules':
                self.configure_modules()
            elif choice == 'main_menu':
                break
            else:
                break
    
    def download_modules(self):
        """Download external modules"""
        self.print_banner()
        print("\n" + "=" * 70)
        print("DOWNLOAD EXTERNAL MODULES")
        print("=" * 70)
        
        print("\nThis will download modules from GitHub to the local 'external' directory.")
        print("Modules to download:")
        print("  • sentiflow.py")
        print("  • quantum_core_engine.py")
        print("  • qybrik.py")
        print("  • qylintos.py")
        print("  • bumpy.py")
        print("  • flumpy.py")
        print("  • laser.py")
        print("  • bugginrace.py")
        print("  • cognition_core.py")
        
        confirm = input("\nProceed with download? (y/n): ").strip().lower()
        
        if confirm == 'y':
            try:
                # Run download script
                script_path = Path(__file__).parent / "scripts" / "download_modules.py"
                
                if script_path.exists():
                    result = subprocess.run(
                        [sys.executable, str(script_path)],
                        capture_output=True,
                        text=True
                    )
                    
                    print("\n" + result.stdout)
                    if result.stderr:
                        print("Errors:", result.stderr)
                else:
                    print("❌ Download script not found at:", script_path)
                    
            except Exception as e:
                print(f"❌ Error downloading modules: {e}")
        
        input("\nPress Enter to continue...")
    
    def list_modules(self):
        """List available modules"""
        if not self.integrator:
            print("❌ Module integration not initialized.")
            return
        
        print("\n" + "=" * 70)
        print("AVAILABLE MODULES")
        print("=" * 70)
        
        modules_by_status = {'available': [], 'unavailable': []}
        
        for name, info in self.integrator.modules.items():
            if info.get('available'):
                modules_by_status['available'].append((name, info))
            else:
                modules_by_status['unavailable'].append((name, info))
        
        print("\n✅ Available Modules:")
        print("-" * 60)
        for name, info in modules_by_status['available']:
            version = info.get('version', '?.?.?')
            purpose = info.get('purpose', 'Unknown')
            print(f"  • {name:20} v{version}")
            print(f"    {purpose}")
        
        print("\n❌ Unavailable Modules:")
        print("-" * 60)
        for name, info in modules_by_status['unavailable']:
            purpose = info.get('purpose', 'Unknown')
            print(f"  • {name:20} - {purpose}")
        
        print(f"\nTotal: {len(modules_by_status['available'])}/{len(self.integrator.modules)} modules available")
    
    def test_module_integration(self):
        """Test module integration"""
        if not self.integrator:
            print("❌ Module integration not initialized.")
            return
        
        print("\n" + "=" * 70)
        print("MODULE INTEGRATION TEST")
        print("=" * 70)
        
        test_results = []
        
        # Test each available module
        for name, info in self.integrator.modules.items():
            if info.get('available'):
                print(f"\n▶ Testing {name}...")
                
                try:
                    module = info['module']
                    
                    # Basic test - check if module has expected attributes
                    if hasattr(module, '__version__'):
                        version = module.__version__
                        test_results.append((name, True, f"v{version}"))
                        print(f"  ✓ Version: {version}")
                    else:
                        test_results.append((name, True, "No version"))
                        print(f"  ✓ Module loaded (no version info)")
                    
                except Exception as e:
                    test_results.append((name, False, str(e)))
                    print(f"  ✗ Error: {e}")
        
        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for _, success, _ in test_results if success)
        total = len(test_results)
        
        print(f"\nTests passed: {passed}/{total}")
        
        if passed == total:
            print("✅ All modules integrated successfully!")
        else:
            print("⚠️  Some modules failed integration")
            print("\nFailed modules:")
            for name, success, error in test_results:
                if not success:
                    print(f"  • {name}: {error}")
    
    def save_results_menu(self, demo_type: str):
        """Menu for saving demonstration results"""
        save = input("\nSave results to file? (y/n): ").strip().lower()
        
        if save == 'y':
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{demo_type}_demo_{timestamp}.json"
            
            # Get recent results
            recent_results = [r for r in self.results_history 
                            if r[0].startswith(demo_type)]
            
            if recent_results:
                results_dict = {
                    'demo_type': demo_type,
                    'timestamp': timestamp,
                    'results': dict(recent_results)
                }
                
                # Save to file
                save_path = self.log_dir / filename
                with open(save_path, 'w') as f:
                    json.dump(results_dict, f, indent=2, cls=NumpyEncoder)
                
                print(f"✅ Results saved to: {save_path}")
                
                # Ask about generating report
                report = input("\nGenerate detailed report? (y/n): ").strip().lower()
                if report == 'y':
                    self.generate_report(demo_type, recent_results)
            else:
                print("❌ No recent results to save")
    
    def generate_report(self, demo_type: str, results: List):
        """Generate detailed report"""
        try:
            from quantum_core_nexus.integration.report_generator import QuantumReportGenerator
            
            generator = QuantumReportGenerator()
            
            # Convert results to appropriate format
            demo_dict = {}
            for name, result in results:
                demo_dict[name] = result
            
            # Generate report
            report = generator.generate_experiment_report(
                demo_results=demo_dict,
                validation_results={}  # Would include validation
            )
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"{demo_type}_report_{timestamp}.json"
            report_path = self.log_dir / report_file
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)
            
            print(f"✅ Report generated: {report_path}")
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
    
    def visualization_menu(self):
        """Data visualization menu"""
        if not self.results_history:
            print("❌ No results available for visualization.")
            print("   Run some demonstrations first.")
            input("\nPress Enter to continue...")
            return
        
        self.print_banner()
        print("\n" + "=" * 70)
        print("DATA VISUALIZATION")
        print("=" * 70)
        
        try:
            from quantum_core_nexus.integration.visualization_engine import QuantumVisualizationEngine
            
            viz = QuantumVisualizationEngine()
            
            print("\nAvailable visualizations:")
            print("  1. Bloch Sphere (for single qubit states)")
            print("  2. Entanglement Entropy Scaling")
            print("  3. Measurement Probability Distribution")
            print("  4. Circuit Diagram")
            print("  5. Return to Main Menu")
            
            choice = input("\nSelect visualization (1-5): ").strip()
            
            if choice == '1':
                self.visualize_bloch_sphere(viz)
            elif choice == '2':
                self.visualize_entanglement_scaling(viz)
            elif choice == '3':
                self.visualize_probability_distribution(viz)
            elif choice == '4':
                self.visualize_circuit_diagram(viz)
            else:
                return
                
        except ImportError as e:
            print(f"❌ Visualization engine not available: {e}")
        
        input("\nPress Enter to continue...")
    
    def visualize_bloch_sphere(self, viz):
        """Visualize state on Bloch sphere"""
        if not self.current_system:
            print("No quantum system available.")
            print("Create a system or run demonstrations first.")
            return
        
        if self.current_system.config.num_subsystems < 1:
            print("System has no qubits.")
            return
        
        try:
            state_vector = self.current_system.get_state_vector()
            fig = viz.plot_bloch_sphere(state_vector, qubit_index=0)
            
            # Save figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bloch_sphere_{timestamp}.png"
            save_path = self.log_dir / filename
            
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Bloch sphere saved to: {save_path}")
            
            # Try to show plot
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except:
                pass
                
        except Exception as e:
            print(f"❌ Error visualizing Bloch sphere: {e}")
    
    def save_system(self):
        """Save current quantum system to file"""
        if not self.current_system:
            print("No system to save.")
            return
        
        try:
            system_data = self.current_system.to_dict()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quantum_system_{timestamp}.json"
            save_path = self.log_dir / filename
            
            with open(save_path, 'w') as f:
                json.dump(system_data, f, indent=2, cls=NumpyEncoder)
            
            print(f"✅ System saved to: {save_path}")
            
        except Exception as e:
            print(f"❌ Error saving system: {e}")
    
    def load_system(self):
        """Load quantum system from file"""
        try:
            import glob
            
            # List saved systems
            system_files = list(self.log_dir.glob("quantum_system_*.json"))
            
            if not system_files:
                print("No saved systems found.")
                return
            
            print("\nSaved systems:")
            for i, file in enumerate(system_files[:10], 1):
                print(f"  {i}. {file.name}")
            
            if len(system_files) > 10:
                print(f"  ... and {len(system_files) - 10} more")
            
            choice = input("\nSelect system to load (number): ").strip()
            
            if choice and choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(system_files):
                    file_path = system_files[idx]
                    
                    with open(file_path, 'r') as f:
                        system_data = json.load(f)
                    
                    # Recreate system (simplified - in practice would need full recreation)
                    print(f"✅ Loaded system from: {file_path.name}")
                    print(f"   Note: Full system recreation not yet implemented.")
                else:
                    print("Invalid selection.")
            else:
                print("No selection made.")
                
        except Exception as e:
            print(f"❌ Error loading system: {e}")
    
    def main_menu(self):
        """Main menu loop"""
        while True:
            self.print_banner()
            
            # System status
            system_status = f"{self.current_system.config.num_subsystems} qubits" \
                          if self.current_system else "None"
            
            # Module status
            if self.integrator:
                available = sum(1 for m in self.integrator.modules.values() 
                              if m.get('available'))
                module_status = f"{available} modules"
            else:
                module_status = "Not initialized"
            
            # Results status
            results_status = f"{len(self.results_history)} sets"
            
            options = [
                {
                    'description': 'Run Quantum Demonstrations',
                    'action': 'demo_menu',
                    'shortcut': 'd',
                    'status': 'Ready'
                },
                {
                    'description': 'Module Management',
                    'action': 'module_management',
                    'shortcut': 'm',
                    'status': module_status
                },
                {
                    'description': 'Data Visualization',
                    'action': 'visualization',
                    'shortcut': 'v',
                    'status': results_status
                },
                {
                    'description': 'Scientific Validation',
                    'action': 'validation',
                    'shortcut': 's',
                    'status': 'Ready'
                },
                {
                    'description': 'Performance Benchmark',
                    'action': 'benchmark',
                    'shortcut': 'b',
                    'status': 'Test'
                },
                {
                    'description': 'View Documentation',
                    'action': 'documentation',
                    'shortcut': 'h'
                },
                {
                    'description': 'System Configuration',
                    'action': 'configuration',
                    'shortcut': 'c',
                    'status': system_status
                },
                {
                    'description': 'Exit QuantumCore Nexus',
                    'action': 'exit',
                    'shortcut': 'x'
                }
            ]
            
            choice = self.print_menu("QUANTUMCORE NEXUS MAIN MENU", options, "Select option: ")
            
            if choice == 'demo_menu':
                self.demo_menu()
            elif choice == 'module_management':
                self.module_management()
            elif choice == 'visualization':
                self.visualization_menu()
            elif choice == 'validation':
                self.scientific_validation()
            elif choice == 'benchmark':
                self.performance_benchmark()
            elif choice == 'documentation':
                self.show_documentation()
            elif choice == 'configuration':
                self.system_configuration()
            elif choice == 'exit':
                print("\n" + "=" * 70)
                print("Thank you for using QuantumCore Nexus!")
                print("Scientific Quantum Simulation Platform")
                print("=" * 70)
                break
            else:
                # Default action
                pass
    
    def scientific_validation(self):
        """Run scientific validation suite"""
        self.print_banner()
        print("\n" + "=" * 70)
        print("SCIENTIFIC VALIDATION SUITE")
        print("=" * 70)
        
        if not self.current_system:
            print("No quantum system available for validation.")
            print("Create a system or run demonstrations first.")
            input("\nPress Enter to continue...")
            return
        
        try:
            validator = QuantumValidator(self.current_system)
            
            print("\n▶ Running quantum principle validation...")
            results = validator.run_all_tests()
            
            print("\n✅ Validation Results:")
            print("-" * 60)
            
            passed = 0
            total = len(results)
            
            for test_name, result in results.items():
                status = "✓ PASS" if result['passed'] else "✗ FAIL"
                print(f"  {test_name:30} {status}")
                
                if not result['passed']:
                    print(f"    {result.get('message', result.get('error', 'No details'))}")
                
                if result['passed']:
                    passed += 1
            
            print("-" * 60)
            print(f"  Summary: {passed}/{total} tests passed")
            
            if passed == total:
                print("  ✅ All quantum principles validated!")
            else:
                print(f"  ⚠️  {total - passed} principle violations detected")
                print("\n  Check the violation log for details.")
            
            # Save validation results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_{timestamp}.json"
            save_path = self.log_dir / filename
            
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
            
            print(f"\n✅ Validation report saved to: {save_path}")
            
        except Exception as e:
            print(f"❌ Error during validation: {e}")
        
        input("\nPress Enter to continue...")
    
    def show_documentation(self):
        """Show documentation"""
        self.print_banner()
        print("\n" + "=" * 70)
        print("QUANTUMCORE NEXUS DOCUMENTATION")
        print("=" * 70)
        
        docs = """
        QUICK START:
        1. Run 'Module Management' to initialize and download modules
        2. Run 'Quantum Demonstrations' to see quantum effects
        3. Use 'Data Visualization' to view results
        
        KEY FEATURES:
        • Unified qubit/qudit system interface
        • Scientific validation of quantum principles
        • Integration with SentiFlow ecosystem
        • Performance benchmarking
        • Custom quantum system manipulation
        
        DEMONSTRATIONS:
        • Bell State: 2-qubit entanglement
        • GHZ State: Multi-qubit entanglement
        • Quantum Teleportation: State transfer protocol
        • Superdense Coding: 2 classical bits in 1 qubit
        • Quantum Fourier Transform: Quantum algorithm
        
        MODULE INTEGRATION:
        The system integrates with these external modules:
        • sentiflow: Quantum circuit framework
        • quantum_core_engine: Low-level quantum operations
        • qybrik: Circuit building blocks
        • qylintos: Circuit optimization
        • bumpy/flumpy: Array operations
        • laser: Quantum control
        • cognition_core: Quantum ML
        • bugginrace: Circuit debugging
        
        COMMANDS:
        • d: Run demonstrations
        • m: Module management
        • v: Data visualization
        • b: Performance benchmark
        • s: Scientific validation
        • h: This documentation
        • x: Exit
        
        For more information, see the README.md file.
        """
        
        print(docs)
        input("\nPress Enter to continue...")
    
    def system_configuration(self):
        """System configuration menu"""
        self.print_banner()
        print("\n" + "=" * 70)
        print("SYSTEM CONFIGURATION")
        print("=" * 70)
        
        print("\nCurrent Configuration:")
        print(f"  • Working Directory: {Path(__file__).parent}")
        print(f"  • Log Directory: {self.log_dir}")
        print(f"  • Current System: {self.current_system}")
        
        print("\nConfiguration Options:")
        print("  1. Change Log Directory")
        print("  2. Clear Results History")
        print("  3. View System Information")
        print("  4. Return to Main Menu")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            new_dir = input("Enter new log directory: ").strip()
            if new_dir:
                new_path = Path(new_dir)
                try:
                    new_path.mkdir(exist_ok=True, parents=True)
                    self.log_dir = new_path
                    print(f"✅ Log directory changed to: {new_path}")
                except Exception as e:
                    print(f"❌ Error changing directory: {e}")
        elif choice == '2':
            confirm = input("Clear all results history? (y/n): ").strip().lower()
            if confirm == 'y':
                self.results_history = []
                print("✅ Results history cleared")
        elif choice == '3':
            self.show_system_info()
        
        input("\nPress Enter to continue...")
    
    def show_system_info(self):
        """Show system information"""
        import platform
        import psutil
        
        print("\n" + "=" * 70)
        print("SYSTEM INFORMATION")
        print("=" * 70)
        
        info = {
            "Platform": platform.platform(),
            "Python Version": sys.version,
            "CPU Cores": psutil.cpu_count(logical=True),
            "Physical Cores": psutil.cpu_count(logical=False),
            "Total RAM": f"{psutil.virtual_memory().total / 1e9:.2f} GB",
            "Available RAM": f"{psutil.virtual_memory().available / 1e9:.2f} GB",
            "Python Path": sys.executable,
            "Current Directory": str(Path.cwd()),
            "QuantumCore Nexus Path": str(Path(__file__).parent)
        }
        
        for key, value in info.items():
            print(f"  {key:20}: {value}")
        
        # Check for required packages
        print("\nRequired Packages:")
        required = ['numpy', 'scipy', 'matplotlib', 'pyyaml', 'requests']
        
        for package in required:
            try:
                __import__(package)
                print(f"  • {package:15} ✓ Installed")
            except ImportError:
                print(f"  • {package:15} ✗ Missing")
    
    def run(self):
        """Run the CLI"""
        try:
            self.initialize_integration()
            self.main_menu()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting gracefully...")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="QuantumCore Nexus - Scientific Quantum Simulation Platform"
    )
    parser.add_argument('--download', action='store_true',
                       help='Download external modules and exit')
    parser.add_argument('--demo', type=str, choices=['qubit', 'qudit', 'all'],
                       help='Run specific demonstration non-interactively')
    parser.add_argument('--qubits', type=int, default=8,
                       help='Number of qubits for demonstrations')
    parser.add_argument('--output', type=str,
                       help='Output file for results')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    
    args = parser.parse_args()
    
    # Handle command line arguments
    if args.download:
        # Download modules
        script_path = Path(__file__).parent / "scripts" / "download_modules.py"
        if script_path.exists():
            subprocess.run([sys.executable, str(script_path)])
        else:
            print("Download script not found.")
        return
    
    if args.demo or args.benchmark:
        # Non-interactive mode
        cli = QuantumCoreNexusCLI()
        cli.initialize_integration()
        
        if args.benchmark:
            cli.performance_benchmark()
        elif args.demo == 'qubit':
            # Create demo suite and run
            demo_suite = QuantumDemonstrationSuite()
            results = demo_suite.run_all_demos(max_qubits=args.qubits)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, cls=NumpyEncoder)
                print(f"Results saved to {args.output}")
        return
    
    # Interactive mode
    cli = QuantumCoreNexusCLI()
    cli.run()

if __name__ == "__main__":
    main()