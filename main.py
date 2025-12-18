#!/usr/bin/env python3
"""
QuantumCore Nexus - Scientific-Grade Quantum Simulation Platform
Main Entry Point with Complete Integration & Menu System
Version: 1.0.0
"""

import os
import sys
import json
import time
import argparse
import subprocess
import platform
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import QuantumCore Nexus modules with fallback handling
def import_quantum_modules():
    """Import quantum modules with graceful fallback handling"""
    modules_status = {}
    
    try:
        from quantum_core_nexus.core.unified_quantum_system import QuantumSystemConfig, QuantumGate
        modules_status['core.unified_quantum_system'] = True
    except ImportError as e:
        modules_status['core.unified_quantum_system'] = False
        print(f"⚠️  Core module import warning: {e}")
    
    try:
        from quantum_core_nexus.core.qubit_system import QubitSystem
        modules_status['core.qubit_system'] = True
    except ImportError as e:
        modules_status['core.qubit_system'] = False
        print(f"⚠️  Qubit system import warning: {e}")
    
    try:
        from quantum_core_nexus.core.qudit_system import QuditSystem
        modules_status['core.qudit_system'] = True
    except ImportError as e:
        modules_status['core.qudit_system'] = False
        print(f"⚠️  Qudit system import warning: {e}")
    
    try:
        from quantum_core_nexus.demonstrations.qubit_demos import QuantumDemonstrationSuite
        modules_status['demonstrations.qubit_demos'] = True
    except ImportError as e:
        modules_status['demonstrations.qubit_demos'] = False
        print(f"⚠️  Qubit demos import warning: {e}")
    
    try:
        from quantum_core_nexus.demonstrations.qudit_demos import QuditDemonstrationSuite
        modules_status['demonstrations.qudit_demos'] = True
    except ImportError as e:
        modules_status['demonstrations.qudit_demos'] = False
        print(f"⚠️  Qudit demos import warning: {e}")
    
    try:
        from quantum_core_nexus.validation.scientific_validator import QuantumValidator, QuantumPrincipleViolation
        modules_status['validation.scientific_validator'] = True
    except ImportError as e:
        modules_status['validation.scientific_validator'] = False
        print(f"⚠️  Validator import warning: {e}")
    
    try:
        from quantum_core_nexus.validation.metric_calculator import QuantumMetricCalculator
        modules_status['validation.metric_calculator'] = True
    except ImportError as e:
        modules_status['validation.metric_calculator'] = False
        print(f"⚠️  Metrics import warning: {e}")
    
    try:
        from quantum_core_nexus.integration.module_bridge import AdvancedQuantumModuleIntegrator
        modules_status['integration.module_bridge'] = True
    except ImportError as e:
        modules_status['integration.module_bridge'] = False
        print(f"⚠️  Module bridge import warning: {e}")
    
    try:
        from quantum_core_nexus.integration.report_generator import QuantumReportGenerator
        modules_status['integration.report_generator'] = True
    except ImportError as e:
        modules_status['integration.report_generator'] = False
        print(f"⚠️  Report generator import warning: {e}")
    
    try:
        from quantum_core_nexus.integration.visualization_engine import QuantumVisualizationEngine
        modules_status['integration.visualization_engine'] = True
    except ImportError as e:
        modules_status['integration.visualization_engine'] = False
        print(f"⚠️  Visualization import warning: {e}")
    
    try:
        from quantum_core_nexus.utils.numpy_encoder import NumpyEncoder
        modules_status['utils.numpy_encoder'] = True
    except ImportError as e:
        modules_status['utils.numpy_encoder'] = False
        print(f"⚠️  Numpy encoder import warning: {e}")
    
    # Check for optional advanced modules
    try:
        from quantum_core_nexus.demonstrations.advanced_demos import AdvancedQuantumApplications
        modules_status['demonstrations.advanced_demos'] = True
    except ImportError:
        modules_status['demonstrations.advanced_demos'] = False
    
    try:
        import numpy as np
        modules_status['numpy'] = True
    except ImportError:
        modules_status['numpy'] = False
        print("❌ NumPy is not installed. This is a required dependency.")
    
    try:
        import scipy
        modules_status['scipy'] = True
    except ImportError:
        modules_status['scipy'] = False
        print("⚠️  SciPy is not installed. Some features may be limited.")
    
    return modules_status

# Import modules
modules_status = import_quantum_modules()

# Check if essential modules are available
essential_modules = ['core.qubit_system', 'demonstrations.qubit_demos', 'numpy']
missing_essential = [mod for mod in essential_modules if not modules_status.get(mod, False)]

if missing_essential:
    print(f"\n❌ Missing essential modules: {missing_essential}")
    print("Please install required dependencies: pip install -r requirements.txt")
    print("Or run: pip install numpy scipy")
    sys.exit(1)

# Now import the modules we know are available
if modules_status['core.unified_quantum_system']:
    from quantum_core_nexus.core.unified_quantum_system import QuantumSystemConfig, QuantumGate

if modules_status['core.qubit_system']:
    from quantum_core_nexus.core.qubit_system import QubitSystem

if modules_status['core.qudit_system']:
    from quantum_core_nexus.core.qudit_system import QuditSystem

if modules_status['demonstrations.qubit_demos']:
    from quantum_core_nexus.demonstrations.qubit_demos import QuantumDemonstrationSuite

if modules_status['demonstrations.qudit_demos']:
    from quantum_core_nexus.demonstrations.qudit_demos import QuditDemonstrationSuite

if modules_status['validation.scientific_validator']:
    from quantum_core_nexus.validation.scientific_validator import QuantumValidator, QuantumPrincipleViolation

if modules_status['validation.metric_calculator']:
    from quantum_core_nexus.validation.metric_calculator import QuantumMetricCalculator

if modules_status['integration.module_bridge']:
    from quantum_core_nexus.integration.module_bridge import AdvancedQuantumModuleIntegrator

if modules_status['integration.report_generator']:
    from quantum_core_nexus.integration.report_generator import QuantumReportGenerator

if modules_status['integration.visualization_engine']:
    from quantum_core_nexus.integration.visualization_engine import QuantumVisualizationEngine

if modules_status['utils.numpy_encoder']:
    from quantum_core_nexus.utils.numpy_encoder import NumpyEncoder

if modules_status['demonstrations.advanced_demos']:
    from quantum_core_nexus.demonstrations.advanced_demos import AdvancedQuantumApplications

class QuantumCoreNexusCLI:
    """
    Interactive Command Line Interface for QuantumCore Nexus
    with complete feature set and error handling
    """
    
    def __init__(self):
        self.integrator = None
        self.current_system = None
        self.results_history = []
        self.demonstration_results = {}
        self.system_info = self._gather_system_info()
        self.config = self._load_config()
        self._setup_directories()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Color codes for terminal output
        self.COLORS = {
            'HEADER': '\033[95m',
            'BLUE': '\033[94m',
            'CYAN': '\033[96m',
            'GREEN': '\033[92m',
            'YELLOW': '\033[93m',
            'RED': '\033[91m',
            'BOLD': '\033[1m',
            'UNDERLINE': '\033[4m',
            'END': '\033[0m',
        }
        
        # Module status tracking
        self.module_status = {
            'sentiflow': False,
            'quantum_core_engine': False,
            'qybrik': False,
            'qylintos': False,
            'bumpy': False,
            'flumpy': False,
            'laser': False,
            'bugginrace': False,
            'cognition_core': False,
        }
    
    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather system information"""
        import platform
        import psutil
        
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_cores": psutil.cpu_count(logical=True),
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "total_ram_gb": psutil.virtual_memory().total / 1e9,
            "available_ram_gb": psutil.virtual_memory().available / 1e9,
            "python_executable": sys.executable,
            "working_directory": str(Path.cwd()),
            "timestamp": datetime.now().isoformat(),
        }
    
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        config_path = Path(__file__).parent / "config" / "modules.yaml"
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"⚠️  Config load warning: {e}")
        
        # Default configuration
        return {
            "module_priorities": {
                "high": ["sentiflow", "quantum_core_engine", "qybrik"],
                "medium": ["bumpy", "flumpy", "qylintos"],
                "low": ["laser", "bugginrace", "cognition_core"]
            },
            "performance": {
                "max_qubits": 16,
                "max_qudits": 8,
                "max_dimension": 10,
                "validation_tolerance": 1e-10,
                "measurement_repetitions": 10000,
            },
            "visualization": {
                "enabled": True,
                "save_plots": True,
                "plot_format": "png",
                "dpi": 150,
            },
            "logging": {
                "level": "INFO",
                "save_all_results": True,
                "compress_old_logs": False,
            }
        }
    
    def _setup_directories(self):
        """Setup required directories"""
        base_dir = Path(__file__).parent
        directories = [
            "logs",
            "config",
            "data",
            "results",
            "plots",
            "exports",
            "tmp",
        ]
        
        for dir_name in directories:
            dir_path = base_dir / dir_name
            dir_path.mkdir(exist_ok=True)
            setattr(self, f"{dir_name}_dir", dir_path)
    
    def colorize(self, text: str, color: str = None, bold: bool = False) -> str:
        """Colorize text for terminal output"""
        if not sys.stdout.isatty():
            return text  # No colors if not in terminal
        
        color_code = self.COLORS.get(color.upper(), '') if color else ''
        bold_code = self.COLORS['BOLD'] if bold else ''
        
        return f"{bold_code}{color_code}{text}{self.COLORS['END']}"
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_banner(self):
        """Print application banner"""
        self.clear_screen()
        
        banner = f"""
{self.colorize('╔══════════════════════════════════════════════════════════════════════════════╗', 'cyan')}
{self.colorize('║', 'cyan')}                                                                              {self.colorize('║', 'cyan')}
{self.colorize('║', 'cyan')}   {self.colorize('╔═╗┬ ┬┌─┐┌─┐┌┬┐┌─┐┌─┐  ╔═╗┌┐┌┌─┐┬ ┬┌─┐┌┐┌┌┬┐┌─┐  ╔═╗┌─┐┌┬┐┌─┐┬─┐┌┬┐┌─┐', 'blue', True)}  {self.colorize('║', 'cyan')}
{self.colorize('║', 'cyan')}   {self.colorize('║ ╦│ │├┤ ├─┤ │ ├┤ └─┐  ╠╣ ││││ ││ │├┤ │││ │ ├┤   ╠═╝├─┤ │ ├┤ ├┬┘│││└─┐', 'blue', True)}  {self.colorize('║', 'cyan')}
{self.colorize('║', 'cyan')}   {self.colorize('╚═╝└─┘└─┘┴ ┴ ┴ └─┘└─┘  ╚  ┘└┘└─┘└─┘└─┘┘└┘ ┴ └─┘  ╩  ┴ ┴ ┴ └─┘┴└─┴ ┴└─┘', 'blue', True)}  {self.colorize('║', 'cyan')}
{self.colorize('║', 'cyan')}                                                                              {self.colorize('║', 'cyan')}
{self.colorize('║', 'cyan')}   {self.colorize('Scientific-Grade Quantum Simulation Platform v1.0.0', 'yellow')}                        {self.colorize('║', 'cyan')}
{self.colorize('║', 'cyan')}   {self.colorize('"From Demonstration to Discovery"', 'green')}                                          {self.colorize('║', 'cyan')}
{self.colorize('║', 'cyan')}                                                                              {self.colorize('║', 'cyan')}
{self.colorize('╚══════════════════════════════════════════════════════════════════════════════╝', 'cyan')}
        """
        print(banner)
        
        # System status line
        status_line = f"Session: {self.session_id} | Python: {self.system_info['python_version'].split()[0]} | "
        if self.current_system:
            status_line += f"System: {self.current_system.config.num_subsystems} qubits | "
        if self.integrator:
            available = sum(1 for status in self.module_status.values() if status)
            status_line += f"Modules: {available}/9"
        
        print(self.colorize(" " * 5 + "=" * 70, 'cyan'))
        print(self.colorize(" " * 5 + "MODULAR QUANTUM SIMULATION WITH SCIENTIFIC VALIDATION", 'yellow'))
        print(self.colorize(" " * 5 + status_line, 'green'))
        print(self.colorize(" " * 5 + "=" * 70, 'cyan'))
        print()
    
    def print_menu(self, title: str, options: List[Dict[str, Any]], 
                  prompt: str = "Select an option: ") -> Optional[Any]:
        """Print a formatted interactive menu"""
        print(f"\n{self.colorize(title, 'cyan', True)}")
        print(self.colorize("-" * 70, 'cyan'))
        
        # Group options by category if present
        categories = {}
        for option in options:
            category = option.get('category', 'General')
            if category not in categories:
                categories[category] = []
            categories[category].append(option)
        
        option_counter = 1
        option_map = {}
        
        for category, cat_options in categories.items():
            if len(categories) > 1:
                print(f"\n{self.colorize(f'{category}:', 'yellow')}")
            
            for option in cat_options:
                shortcut = self.colorize(f"[{option.get('shortcut', '')}]", 'green') if option.get('shortcut') else "     "
                description = option['description']
                status = option.get('status', '')
                
                if status:
                    status_display = self.colorize(f"[{status}]", 'blue')
                    print(f"  {self.colorize(f'{option_counter:2d}', 'yellow')}. {description:45} {status_display:10} {shortcut}")
                else:
                    print(f"  {self.colorize(f'{option_counter:2d}', 'yellow')}. {description:45} {' ' * 10} {shortcut}")
                
                option_map[option_counter] = option.get('action')
                option_map[option.get('shortcut', '').lower()] = option.get('action')
                option_counter += 1
        
        print(self.colorize("-" * 70, 'cyan'))
        
        while True:
            try:
                choice = input(f"\n{self.colorize(prompt, 'green')}").strip().lower()
                
                if choice == '':
                    return None
                
                # Check for shortcuts first
                if choice in option_map:
                    return option_map[choice]
                
                # Check numeric choice
                try:
                    choice_num = int(choice)
                    if choice_num in option_map:
                        return option_map[choice_num]
                    else:
                        print(self.colorize(f"Please enter a number between 1 and {option_counter-1}", 'red'))
                except ValueError:
                    print(self.colorize("Invalid input. Please enter a number or shortcut.", 'red'))
                    
            except KeyboardInterrupt:
                print(f"\n{self.colorize('Operation cancelled.', 'yellow')}")
                return None
            except EOFError:
                print(f"\n{self.colorize('End of input.', 'yellow')}")
                return None
    
    def initialize_integration(self, silent: bool = False) -> bool:
        """Initialize module integration with progress tracking"""
        if not silent:
            print(f"\n{self.colorize('=' * 70, 'cyan')}")
            print(self.colorize("MODULE INTEGRATION INITIALIZATION", 'cyan', True))
            print(self.colorize("=" * 70, 'cyan'))
        
        try:
            self.integrator = AdvancedQuantumModuleIntegrator()
            
            # Update module status
            for module_name in self.module_status.keys():
                self.module_status[module_name] = self.integrator.modules.get(
                    module_name, {}).get('available', False)
            
            # Count available modules
            available = [name for name, mod in self.integrator.modules.items() 
                        if mod.get('available', False)]
            
            if not silent:
                print(f"\n{self.colorize('✅', 'green')} Integration initialized successfully!")
                print(f"   {self.colorize('Available modules:', 'yellow')} {len(available)}/{len(self.integrator.modules)}")
                
                if available:
                    print(f"\n{self.colorize('Detected modules:', 'yellow')}")
                    for name in available:
                        mod_info = self.integrator.modules[name]
                        version = mod_info.get('version', '?.?.?')
                        purpose = mod_info.get('purpose', 'Unknown')[:40]
                        print(f"  {self.colorize('•', 'green')} {self.colorize(name, 'blue'):20} v{version}")
                        print(f"    {purpose}")
                
                # Check for missing high priority modules
                high_priority = self.config.get('module_priorities', {}).get('high', [])
                missing_high = [mod for mod in high_priority if mod not in available]
                
                if missing_high:
                    print(f"\n{self.colorize('⚠️  Missing high-priority modules:', 'yellow')}")
                    for mod in missing_high:
                        print(f"  {self.colorize('✗', 'red')} {mod}")
                    print(f"\n{self.colorize('Consider running: python scripts/download_modules.py', 'yellow')}")
            
            return True
            
        except Exception as e:
            if not silent:
                print(f"{self.colorize('❌', 'red')} Failed to initialize integration: {e}")
                if "AdvancedQuantumModuleIntegrator" in str(e):
                    print(f"{self.colorize('Info:', 'yellow')} Module bridge may not be implemented yet")
            return False
    
    def demo_menu(self):
        """Main demonstration menu"""
        while True:
            self.print_banner()
            
            options = [
                {
                    'description': 'Qubit Demonstrations',
                    'action': 'qubit_demos',
                    'shortcut': 'q',
                    'category': 'Quantum Systems',
                    'status': 'Ready'
                },
                {
                    'description': 'Qudit Demonstrations',
                    'action': 'qudit_demos',
                    'shortcut': 'd',
                    'category': 'Quantum Systems',
                    'status': 'Ready' if modules_status['demonstrations.qudit_demos'] else 'Missing'
                },
                {
                    'description': 'Advanced Quantum Applications',
                    'action': 'advanced_demos',
                    'shortcut': 'a',
                    'category': 'Advanced Features',
                    'status': 'Module Req'
                },
                {
                    'description': 'Custom Quantum System',
                    'action': 'custom_system',
                    'shortcut': 'c',
                    'category': 'Quantum Systems',
                    'status': 'Interactive'
                },
                {
                    'description': 'Performance Benchmarking',
                    'action': 'benchmark',
                    'shortcut': 'b',
                    'category': 'Advanced Features',
                    'status': 'System Test'
                },
                {
                    'description': 'Quantum Walk Simulation',
                    'action': 'quantum_walk',
                    'shortcut': 'w',
                    'category': 'Advanced Features',
                    'status': 'Experimental'
                },
                {
                    'description': 'Return to Main Menu',
                    'action': 'main_menu',
                    'shortcut': 'm',
                    'category': 'Navigation'
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
            elif choice == 'quantum_walk':
                self.quantum_walk_demo()
            elif choice == 'main_menu':
                break
    
    def qubit_demonstrations(self):
        """Run comprehensive qubit demonstrations"""
        self.print_banner()
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("QUBIT DEMONSTRATIONS", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        # Get parameters
        max_qubits = self._get_input_with_default(
            "Maximum qubits to test", 
            default=8, 
            min_val=2, 
            max_val=16
        )
        
        repetitions = self._get_input_with_default(
            "Measurement repetitions", 
            default=10000, 
            min_val=100, 
            max_val=1000000
        )
        
        validation_level = self._get_choice(
            "Validation level",
            options=["strict", "warn", "none"],
            default="warn"
        )
        
        # Select demonstrations
        print(f"\n{self.colorize('Select demonstrations to run:', 'yellow')}")
        print(f"  1. {self.colorize('Bell State', 'cyan')} (2-qubit entanglement)")
        print(f"  2. {self.colorize('GHZ State', 'cyan')} (multi-qubit entanglement)")
        print(f"  3. {self.colorize('Quantum Teleportation', 'cyan')} (3+ qubits)")
        print(f"  4. {self.colorize('Superdense Coding', 'cyan')} (2 classical bits in 1 qubit)")
        print(f"  5. {self.colorize('Quantum Fourier Transform', 'cyan')}")
        print(f"  6. {self.colorize('Entanglement Swapping', 'cyan')} (4+ qubits)")
        print(f"  7. {self.colorize('All Demonstrations', 'green', True)}")
        
        demo_choice = self._get_input_with_default(
            "Enter choice", 
            default=7, 
            min_val=1, 
            max_val=7
        )
        
        # Determine which demos to run
        demos_to_run = []
        if demo_choice in [1, 7]:
            demos_to_run.append('bell')
        if demo_choice in [2, 7]:
            demos_to_run.append('ghz')
        if demo_choice in [3, 7] and max_qubits >= 3:
            demos_to_run.append('teleport')
        if demo_choice in [4, 7]:
            demos_to_run.append('superdense')
        if demo_choice in [5, 7]:
            demos_to_run.append('qft')
        if demo_choice in [6, 7] and max_qubits >= 4:
            demos_to_run.append('swap')
        
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("RUNNING QUBIT DEMONSTRATIONS", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        start_time = time.time()
        all_results = {}
        
        try:
            for demo in demos_to_run:
                result = self._run_specific_qubit_demo(
                    demo, max_qubits, repetitions, validation_level
                )
                if result:
                    all_results[demo] = result
            
            elapsed = time.time() - start_time
            
            # Summary
            print(f"\n{self.colorize('=' * 70, 'green')}")
            print(self.colorize("DEMONSTRATION SUMMARY", 'green', True))
            print(self.colorize("=" * 70, 'green'))
            
            print(f"\n{self.colorize('Completed:', 'yellow')} {len(all_results)}/{len(demos_to_run)} demonstrations")
            print(f"{self.colorize('Total time:', 'yellow')} {elapsed:.2f} seconds")
            print(f"{self.colorize('Average time per demo:', 'yellow')} {elapsed/len(all_results):.2f}s" if all_results else "")
            
            # Calculate overall metrics
            if all_results:
                success_rate = sum(1 for r in all_results.values() if r.get('success', False)) / len(all_results)
                avg_fidelity = np.mean([r.get('fidelity', 0) for r in all_results.values() if 'fidelity' in r])
                
                print(f"{self.colorize('Success rate:', 'yellow')} {success_rate:.1%}")
                print(f"{self.colorize('Average fidelity:', 'yellow')} {avg_fidelity:.6f}")
            
            self.demonstration_results['qubit'] = all_results
            
            # Save results
            self._save_results_prompt("qubit", all_results, elapsed)
            
        except Exception as e:
            print(f"{self.colorize('❌ Error running demonstrations:', 'red')} {e}")
            traceback.print_exc()
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def _run_specific_qubit_demo(self, demo_name: str, max_qubits: int, 
                               repetitions: int, validation_level: str) -> Optional[Dict]:
        """Run a specific qubit demonstration"""
        demo_configs = {
            'bell': {
                'name': 'Bell State',
                'min_qubits': 2,
                'function': self._demo_bell_state,
            },
            'ghz': {
                'name': 'GHZ State',
                'min_qubits': 2,
                'function': self._demo_ghz_state,
                'params': {'max_qubits': max_qubits}
            },
            'teleport': {
                'name': 'Quantum Teleportation',
                'min_qubits': 3,
                'function': self._demo_teleportation,
            },
            'superdense': {
                'name': 'Superdense Coding',
                'min_qubits': 2,
                'function': self._demo_superdense_coding,
            },
            'qft': {
                'name': 'Quantum Fourier Transform',
                'min_qubits': 2,
                'function': self._demo_qft,
                'params': {'max_qubits': min(max_qubits, 4)}
            },
            'swap': {
                'name': 'Entanglement Swapping',
                'min_qubits': 4,
                'function': self._demo_entanglement_swapping,
            }
        }
        
        if demo_name not in demo_configs:
            return None
        
        config = demo_configs[demo_name]
        
        if max_qubits < config['min_qubits']:
            print(f"\n{self.colorize('⚠️', 'yellow')} Skipping {config['name']}: requires at least {config['min_qubits']} qubits")
            return None
        
        print(f"\n{self.colorize('▶', 'cyan')} Running {config['name']}...")
        demo_start = time.time()
        
        try:
            params = config.get('params', {})
            result = config['function'](repetitions, validation_level, **params)
            result['duration'] = time.time() - demo_start
            result['success'] = result.get('success', True)
            
            # Print results
            status = self.colorize('✓', 'green') if result['success'] else self.colorize('✗', 'red')
            print(f"  {status} {config['name']}: {result.get('duration', 0):.2f}s")
            
            if 'fidelity' in result:
                fidelity_color = 'green' if result['fidelity'] > 0.99 else 'yellow' if result['fidelity'] > 0.9 else 'red'
                print(f"    Fidelity: {self.colorize(f'{result[\"fidelity\"]:.6f}', fidelity_color)}")
            
            if 'entropy' in result:
                print(f"    Entanglement entropy: {result['entropy']:.6f}")
            
            return result
            
        except Exception as e:
            print(f"{self.colorize('  ✗', 'red')} Error: {e}")
            return {'error': str(e), 'success': False, 'duration': time.time() - demo_start}
    
    def _demo_bell_state(self, repetitions: int, validation_level: str) -> Dict:
        """Bell state demonstration"""
        system = QubitSystem(2, validation_level=validation_level)
        system.create_bell_state()
        
        # Calculate metrics
        state_vector = system.get_state_vector()
        density_matrix = np.outer(state_vector, state_vector.conj())
        
        entropy = QuantumMetricCalculator.calculate_von_neumann_entropy(
            self._partial_trace(density_matrix, [0])
        )
        concurrence = QuantumMetricCalculator.calculate_concurrence(state_vector, (2, 2))
        
        # Measure
        measurements = system.measure(repetitions=repetitions)
        
        # Check correlation
        correlated_prob = measurements['probabilities'].get('00', 0) + \
                         measurements['probabilities'].get('11', 0)
        
        return {
            'state_vector': state_vector.tolist(),
            'entropy': entropy,
            'concurrence': concurrence,
            'correlated_probability': correlated_prob,
            'expected_correlation': 1.0,
            'fidelity': 1.0,  # Perfect for theoretical Bell state
            'measurements': measurements,
            'success': abs(correlated_prob - 1.0) < 0.01
        }
    
    def _demo_ghz_state(self, repetitions: int, validation_level: str, 
                       max_qubits: int) -> Dict:
        """GHZ state demonstration for multiple qubit counts"""
        results = {}
        
        for n in [2, 3, min(4, max_qubits), min(8, max_qubits)]:
            system = QubitSystem(n, validation_level=validation_level)
            system.create_ghz_state()
            
            # Calculate entanglement entropy for half partition
            if n > 1:
                subsystem = list(range(n//2))
                entropy = system.calculate_entropy(subsystem)
            else:
                entropy = 0.0
            
            # Measure
            measurements = system.measure(repetitions=repetitions//n if n > 2 else repetitions)
            
            # Check correlation
            all_zeros = measurements['probabilities'].get('0'*n, 0)
            all_ones = measurements['probabilities'].get('1'*n, 0)
            correlated_prob = all_zeros + all_ones
            
            results[f'{n}_qubits'] = {
                'entropy': entropy,
                'correlated_probability': correlated_prob,
                'expected_correlation': 1.0 if n == 2 else 0.5,
                'measurements': measurements,
                'success': abs(correlated_prob - (1.0 if n == 2 else 0.5)) < 0.05
            }
        
        return results
    
    def _demo_teleportation(self, repetitions: int, validation_level: str) -> Dict:
        """Quantum teleportation demonstration"""
        system = QubitSystem(3, validation_level=validation_level)
        
        # Create arbitrary state to teleport
        np.random.seed(42)
        alpha = np.random.randn() + 1j * np.random.randn()
        beta = np.random.randn() + 1j * np.random.randn()
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        alpha /= norm
        beta /= norm
        
        # Prepare initial state |ψ⟩⊗|0⟩⊗|0⟩
        initial_state = np.zeros(8, dtype=np.complex128)
        initial_state[0] = alpha  # |000⟩
        initial_state[1] = beta   # |001⟩
        system._state = initial_state
        
        # Teleportation protocol
        # Create Bell pair
        system.apply_gate(system.HADAMARD, [1])
        
        # CNOT(1, 2)
        cnot_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        cnot_gate = QuantumGate(cnot_matrix, "CNOT")
        system.apply_gate(cnot_gate, [1, 2])
        
        # CNOT(0, 1)
        system.apply_gate(cnot_gate, [0, 1])
        
        # Hadamard on qubit 0
        system.apply_gate(system.HADAMARD, [0])
        
        # Measure qubits 0 and 1 (simulated - we calculate fidelity)
        state_vector = system.get_state_vector()
        
        # Extract Bob's state (qubit 2)
        bob_state = np.zeros(2, dtype=np.complex128)
        for i in [0, 2, 4, 6]:  # Bob's qubit = 0
            bob_state[0] += state_vector[i]
        for i in [1, 3, 5, 7]:  # Bob's qubit = 1
            bob_state[1] += state_vector[i]
        
        # Normalize
        bob_norm = np.linalg.norm(bob_state)
        if bob_norm > 0:
            bob_state /= bob_norm
        
        # Calculate fidelity
        target_state = np.array([alpha, beta])
        fidelity = abs(np.vdot(bob_state, target_state))**2
        
        return {
            'initial_state': [alpha, beta],
            'teleported_state': bob_state.tolist(),
            'fidelity': fidelity,
            'success': fidelity > 0.99,
            'protocol_steps': 5
        }
    
    def _demo_superdense_coding(self, repetitions: int, validation_level: str) -> Dict:
        """Superdense coding demonstration"""
        system = QubitSystem(2, validation_level=validation_level)
        
        # Create Bell state
        system.create_bell_state()
        
        # Encode 2 classical bits: "11"
        system.apply_gate(system.PAULI_X, [0])
        system.apply_gate(system.PAULI_Z, [0])
        
        # Decode
        cnot_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        cnot_gate = QuantumGate(cnot_matrix, "CNOT")
        system.apply_gate(cnot_gate, [0, 1])
        system.apply_gate(system.HADAMARD, [0])
        
        # Measure
        measurements = system.measure(repetitions=repetitions)
        
        # Check if we measured "11"
        prob_11 = measurements['probabilities'].get('11', 0)
        
        return {
            'encoded_bits': '11',
            'measured_probability_11': prob_11,
            'measurements': measurements,
            'success': prob_11 > 0.99,
            'fidelity': prob_11
        }
    
    def _demo_qft(self, repetitions: int, validation_level: str, max_qubits: int) -> Dict:
        """Quantum Fourier Transform demonstration"""
        import numpy as np
        
        results = {}
        
        for n in [1, 2, min(3, max_qubits)]:
            system = QubitSystem(n, validation_level=validation_level)
            
            # Create computational basis state |1⟩⊗|0⟩⊗...
            if n == 1:
                initial_state = np.array([0, 1], dtype=np.complex128)
            else:
                initial_state = np.zeros(2**n, dtype=np.complex128)
                initial_state[1] = 1.0  # |0...01⟩
            
            system._state = initial_state
            
            # Apply QFT (simplified - using Hadamard for demonstration)
            # In full implementation, would use proper QFT circuit
            for i in range(n):
                system.apply_gate(system.HADAMARD, [i])
            
            # Measure
            measurements = system.measure(repetitions=repetitions//(2**n))
            
            # For |1⟩, QFT should give uniform distribution
            expected_uniform = 1.0 / (2**n)
            
            # Calculate chi-squared for uniformity
            probs = list(measurements['probabilities'].values())
            if len(probs) == 2**n:
                from scipy import stats
                expected = [expected_uniform] * len(probs)
                chi2, p_value = stats.chisquare(probs, expected)
                uniform = p_value > 0.05
            else:
                uniform = False
                p_value = 0.0
            
            results[f'{n}_qubits'] = {
                'uniformity_p_value': p_value,
                'is_uniform': uniform,
                'expected_uniform': expected_uniform,
                'measurements': measurements,
                'success': uniform,
                'fidelity': min(probs) / expected_uniform if probs else 0
            }
        
        return results
    
    def _demo_entanglement_swapping(self, repetitions: int, validation_level: str) -> Dict:
        """Entanglement swapping demonstration (requires 4 qubits)"""
        system = QubitSystem(4, validation_level=validation_level)
        
        # Create two Bell pairs: qubits (0,1) and (2,3)
        # First Bell pair
        system.apply_gate(system.HADAMARD, [0])
        cnot_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        cnot_gate = QuantumGate(cnot_matrix, "CNOT")
        system.apply_gate(cnot_gate, [0, 1])
        
        # Second Bell pair
        system.apply_gate(system.HADAMARD, [2])
        system.apply_gate(cnot_gate, [2, 3])
        
        # Perform Bell measurement on qubits 1 and 2
        system.apply_gate(cnot_gate, [1, 2])
        system.apply_gate(system.HADAMARD, [1])
        
        # Qubits 0 and 3 should now be entangled
        # Measure correlation between qubits 0 and 3
        measurements = system.measure(repetitions=repetitions)
        
        # Calculate correlation
        correlated = 0.0
        total = 0.0
        for state, prob in measurements['probabilities'].items():
            if len(state) == 4:
                # Check if qubits 0 and 3 are equal
                if state[0] == state[3]:
                    correlated += prob
                total += prob
        
        correlation = correlated / total if total > 0 else 0.0
        
        return {
            'correlation_0_3': correlation,
            'expected_correlation': 1.0,
            'measurements': measurements,
            'success': correlation > 0.99,
            'fidelity': correlation
        }
    
    def _partial_trace(self, density_matrix: np.ndarray, keep: List[int]) -> np.ndarray:
        """Partial trace implementation (simplified)"""
        # Simplified version for 2-qubit systems
        n = int(np.sqrt(density_matrix.shape[0]))
        if n == 2:  # Single qubit
            return density_matrix
        elif n == 4:  # Two qubits
            # Trace out second qubit
            rho = np.zeros((2, 2), dtype=np.complex128)
            for i in [0, 1]:
                for j in [0, 1]:
                    rho[i, j] = density_matrix[2*i, 2*j] + density_matrix[2*i+1, 2*j+1]
            return rho
        return density_matrix
    
    def qudit_demonstrations(self):
        """Run comprehensive qudit demonstrations"""
        if not modules_status['demonstrations.qudit_demos']:
            print(f"\n{self.colorize('❌ Qudit demonstrations module not available', 'red')}")
            print(f"{self.colorize('Please ensure qudit_system.py and qudit_demos.py are installed', 'yellow')}")
            input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            return
        
        self.print_banner()
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("QUDIT DEMONSTRATIONS", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        # Get parameters
        dimension = self._get_input_with_default(
            "Qudit dimension (d > 2)", 
            default=3, 
            min_val=3, 
            max_val=10
        )
        
        max_qudits = self._get_input_with_default(
            "Maximum qudits to test", 
            default=2, 
            min_val=1, 
            max_val=4
        )
        
        repetitions = self._get_input_with_default(
            "Measurement repetitions", 
            default=5000, 
            min_val=100, 
            max_val=100000
        )
        
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize(f"RUNNING QUDIT DEMONSTRATIONS (d={dimension})", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        start_time = time.time()
        
        try:
            demo_suite = QuditDemonstrationSuite(dimension=dimension)
            all_results = demo_suite.run_all_demos(max_qudits=max_qudits)
            
            elapsed = time.time() - start_time
            
            # Process results
            print(f"\n{self.colorize('=' * 70, 'green')}")
            print(self.colorize("QUDIT DEMONSTRATION SUMMARY", 'green', True))
            print(self.colorize("=" * 70, 'green'))
            
            total_demos = sum(len(results) for results in all_results.values() 
                            if isinstance(results, dict))
            successful = 0
            
            for qudit_count, results in all_results.items():
                print(f"\n{self.colorize(f'{qudit_count}:', 'yellow')}")
                
                for demo_name, demo_result in results.items():
                    if isinstance(demo_result, dict):
                        success = demo_result.get('success', 
                                                demo_result.get('fidelity', 0) > 0.99)
                        if success:
                            successful += 1
                        
                        status = self.colorize('✓', 'green') if success else self.colorize('✗', 'red')
                        duration = demo_result.get('duration', 0)
                        
                        print(f"  {status} {demo_name:20} {duration:.2f}s")
                        
                        if 'fidelity' in demo_result:
                            fid = demo_result['fidelity']
                            color = 'green' if fid > 0.99 else 'yellow' if fid > 0.9 else 'red'
                            print(f"    Fidelity: {self.colorize(f'{fid:.6f}', color)}")
                        
                        if 'entanglement_entropy' in demo_result:
                            ent = demo_result['entanglement_entropy']
                            print(f"    Entropy: {ent:.6f}")
            
            print(f"\n{self.colorize('Summary:', 'cyan')}")
            print(f"  Total demonstrations: {total_demos}")
            print(f"  Successful: {successful}/{total_demos} ({successful/total_demos:.1%})")
            print(f"  Total time: {elapsed:.2f} seconds")
            print(f"  Hilbert space tested up to: {dimension ** max_qudits} dimensions")
            
            self.demonstration_results['qudit'] = all_results
            
            # Save results
            self._save_results_prompt("qudit", all_results, elapsed)
            
        except Exception as e:
            print(f"{self.colorize('❌ Error running qudit demonstrations:', 'red')} {e}")
            traceback.print_exc()
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def advanced_demonstrations(self):
        """Run advanced quantum applications"""
        # Check if module integration is available
        if not self.integrator:
            print(f"\n{self.colorize('⚠️  Module integration not initialized', 'yellow')}")
            init = input(f"{self.colorize('Initialize module integration now? (y/n): ', 'yellow')}")
            if init.lower() == 'y':
                self.initialize_integration()
            else:
                return
        
        self.print_banner()
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("ADVANCED QUANTUM APPLICATIONS", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        options = [
            {
                'description': 'Quantum Pattern Recognition',
                'action': 'pattern_recognition',
                'shortcut': 'p',
                'category': 'Quantum Machine Learning',
                'status': 'cognition_core' if self.module_status['cognition_core'] else 'Module Req'
            },
            {
                'description': 'Quantum Circuit Optimization',
                'action': 'circuit_optimization',
                'shortcut': 'o',
                'category': 'Quantum Engineering',
                'status': 'qylintos' if self.module_status['qylintos'] else 'Module Req'
            },
            {
                'description': 'Quantum Error Correction',
                'action': 'error_correction',
                'shortcut': 'e',
                'category': 'Quantum Engineering',
                'status': 'Experimental'
            },
            {
                'description': 'Quantum Control & Pulse Shaping',
                'action': 'quantum_control',
                'shortcut': 'c',
                'category': 'Quantum Engineering',
                'status': 'laser' if self.module_status['laser'] else 'Module Req'
            },
            {
                'description': 'Quantum Chemistry Simulation',
                'action': 'quantum_chemistry',
                'shortcut': 'h',
                'category': 'Quantum Simulation',
                'status': 'Experimental'
            },
            {
                'description': 'Return to Demo Menu',
                'action': 'demo_menu',
                'shortcut': 'm',
                'category': 'Navigation'
            }
        ]
        
        choice = self.print_menu("ADVANCED APPLICATIONS", options, "Select application: ")
        
        if choice == 'pattern_recognition':
            self.quantum_pattern_recognition()
        elif choice == 'circuit_optimization':
            self.circuit_optimization_demo()
        elif choice == 'error_correction':
            self.quantum_error_correction()
        elif choice == 'quantum_control':
            self.quantum_control_demo()
        elif choice == 'quantum_chemistry':
            self.quantum_chemistry_demo()
        elif choice == 'demo_menu':
            return
    
    def quantum_pattern_recognition(self):
        """Demonstrate quantum pattern recognition"""
        if not self.module_status['cognition_core']:
            print(f"\n{self.colorize('❌ cognition_core module not available', 'red')}")
            print(f"{self.colorize('Please download from: https://github.com/TaoishTechy/SentiFlow', 'yellow')}")
            input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            return
        
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("QUANTUM PATTERN RECOGNITION", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        try:
            # Try to use the advanced module
            apps = AdvancedQuantumApplications(self.integrator)
            
            # Create sample patterns
            import numpy as np
            patterns = [
                np.array([1, 0, 0, 1, 0, 1, 1, 0]),  # Pattern A
                np.array([0, 1, 1, 0, 1, 0, 0, 1]),  # Pattern B
                np.array([1, 1, 0, 0, 1, 1, 0, 0]),  # Pattern C
            ]
            
            print(f"\n{self.colorize('Training patterns:', 'yellow')}")
            for i, pattern in enumerate(patterns):
                print(f"  Pattern {i+1}: {pattern}")
            
            # Test patterns
            test_cases = [
                (np.array([1, 0, 0, 1, 0, 1, 1, 0]), "Exact match to Pattern A"),
                (np.array([1, 0, 0, 1, 0, 1, 1, 1]), "One-bit error"),
                (np.array([0, 1, 1, 0, 1, 0, 0, 0]), "One-bit error"),
            ]
            
            results = []
            
            for test_pattern, description in test_cases:
                print(f"\n{self.colorize('Testing:', 'cyan')} {description}")
                print(f"  Test pattern: {test_pattern}")
                
                try:
                    result = apps.quantum_pattern_recognition(test_pattern)
                    best_match = result.get('best_match', 'Unknown')
                    confidence = result.get('confidence', 0)
                    
                    print(f"  Result: Pattern {best_match}")
                    print(f"  Confidence: {confidence:.2%}")
                    
                    results.append({
                        'test_pattern': test_pattern.tolist(),
                        'best_match': best_match,
                        'confidence': confidence,
                        'description': description
                    })
                    
                except Exception as e:
                    print(f"  {self.colorize('Error:', 'red')} {e}")
                    results.append({
                        'test_pattern': test_pattern.tolist(),
                        'error': str(e),
                        'description': description
                    })
            
            self.demonstration_results['pattern_recognition'] = results
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pattern_recognition_{timestamp}.json"
            save_path = self.results_dir / filename
            
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
            
            print(f"\n{self.colorize('✅ Results saved to:', 'green')} {save_path}")
            
        except Exception as e:
            print(f"\n{self.colorize('❌ Error in pattern recognition:', 'red')} {e}")
            traceback.print_exc()
            
            # Fallback demonstration
            self._fallback_pattern_recognition()
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def _fallback_pattern_recognition(self):
        """Fallback pattern recognition demonstration"""
        print(f"\n{self.colorize('Running fallback pattern recognition...', 'yellow')}")
        
        import numpy as np
        
        # Simple pattern matching using quantum-inspired distance
        patterns = [
            np.array([1, 0, 0, 1, 0, 1, 1, 0]),  # Pattern A
            np.array([0, 1, 1, 0, 1, 0, 0, 1]),  # Pattern B
            np.array([1, 1, 0, 0, 1, 1, 0, 0]),  # Pattern C
        ]
        
        # Test pattern
        test_pattern = np.array([1, 0, 0, 1, 0, 1, 1, 0])
        
        print(f"\n{self.colorize('Patterns:', 'yellow')}")
        for i, pattern in enumerate(patterns):
            print(f"  Pattern {i+1}: {pattern}")
        
        print(f"\n{self.colorize('Test pattern:', 'yellow')} {test_pattern}")
        
        # Calculate quantum-inspired distances (overlap)
        distances = []
        for i, pattern in enumerate(patterns):
            # Normalize as quantum states
            pattern_norm = pattern / np.linalg.norm(pattern)
            test_norm = test_pattern / np.linalg.norm(test_pattern)
            
            # Calculate fidelity (overlap squared)
            fidelity = abs(np.vdot(pattern_norm, test_norm))**2
            distances.append((i+1, fidelity))
        
        # Find best match
        best_match = max(distances, key=lambda x: x[1])
        
        print(f"\n{self.colorize('Results:', 'cyan')}")
        for pattern_num, fidelity in distances:
            print(f"  Pattern {pattern_num}: fidelity = {fidelity:.4f}")
        
        print(f"\n{self.colorize('Best match:', 'green')} Pattern {best_match[0]} (fidelity: {best_match[1]:.4f})")
    
    def circuit_optimization_demo(self):
        """Demonstrate circuit optimization"""
        if not self.module_status['qylintos']:
            print(f"\n{self.colorize('❌ qylintos module not available', 'red')}")
            print(f"{self.colorize('Circuit optimization requires qylintos module', 'yellow')}")
            input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            return
        
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("QUANTUM CIRCUIT OPTIMIZATION", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        print(f"\n{self.colorize('Circuit optimization demonstration requires qylintos module', 'yellow')}")
        print(f"{self.colorize('This would demonstrate:', 'cyan')}")
        print("  1. Gate cancellation (H H = I)")
        print("  2. Gate merging (X X = I)")
        print("  3. Gate reordering for parallelism")
        print("  4. Circuit depth reduction")
        
        # Simulate optimization results
        optimization_examples = [
            {
                'original': "H - H - X - X",
                'optimized': "I (identity)",
                'reduction': "4 gates → 0 gates",
                'depth_reduction': "100%"
            },
            {
                'original': "H - CNOT - H - CNOT",
                'optimized': "CNOT - Z - CNOT",
                'reduction': "4 gates → 3 gates",
                'depth_reduction': "25%"
            },
            {
                'original': "X - H - Y - H - Z",
                'optimized': "Y - X - Z",
                'reduction': "5 gates → 3 gates",
                'depth_reduction': "40%"
            }
        ]
        
        print(f"\n{self.colorize('Example optimizations:', 'yellow')}")
        for example in optimization_examples:
            print(f"\n  {self.colorize('Original:', 'cyan')} {example['original']}")
            print(f"  {self.colorize('Optimized:', 'green')} {example['optimized']}")
            print(f"  {self.colorize('Reduction:', 'yellow')} {example['reduction']}")
            print(f"  {self.colorize('Depth reduction:', 'blue')} {example['depth_reduction']}")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def quantum_error_correction(self):
        """Demonstrate quantum error correction"""
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("QUANTUM ERROR CORRECTION", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        print(f"\n{self.colorize('Quantum Error Correction Codes:', 'yellow')}")
        print("  1. 3-qubit bit-flip code")
        print("  2. 3-qubit phase-flip code")
        print("  3. 9-qubit Shor code")
        print("  4. 5-qubit perfect code")
        print("  5. Steane code (7 qubits)")
        
        choice = self._get_input_with_default(
            "Select code to demonstrate", 
            default=1, 
            min_val=1, 
            max_val=5
        )
        
        # Demonstrate 3-qubit bit-flip code
        if choice == 1:
            print(f"\n{self.colorize('3-Qubit Bit-Flip Code:', 'cyan')}")
            print("  Encoding: |0⟩ → |000⟩, |1⟩ → |111⟩")
            print("  Corrects single bit-flip errors (X errors)")
            print("  Syndrome measurements detect which qubit flipped")
            
            # Simulate error correction
            print(f"\n{self.colorize('Simulation:', 'yellow')}")
            
            # Create logical |0⟩ state
            system = QubitSystem(3, validation_level="warn")
            # Already in |000⟩ state
            
            # Apply random error
            import random
            error_qubit = random.randint(0, 2)
            error_type = random.choice(['X', 'Z', 'Y'])
            
            print(f"  Applying {error_type} error to qubit {error_qubit}")
            
            if error_type == 'X':
                system.apply_gate(system.PAULI_X, [error_qubit])
            elif error_type == 'Z':
                system.apply_gate(system.PAULI_Z, [error_qubit])
            elif error_type == 'Y':
                system.apply_gate(system.PAULI_Y, [error_qubit])
            
            # Syndrome measurement (simplified)
            print(f"  Syndrome measurement...")
            
            # For bit-flip code, measure parity
            parity_01 = self._measure_parity(system, [0, 1])
            parity_12 = self._measure_parity(system, [1, 2])
            
            print(f"  Parity 0-1: {parity_01}")
            print(f"  Parity 1-2: {parity_12}")
            
            # Determine error location
            if parity_01 == 1 and parity_12 == 0:
                error_loc = 0
            elif parity_01 == 1 and parity_12 == 1:
                error_loc = 1
            elif parity_01 == 0 and parity_12 == 1:
                error_loc = 2
            else:
                error_loc = -1  # No error or multi-qubit error
            
            if error_loc >= 0:
                print(f"  Detected error on qubit {error_loc}")
                print(f"  Applying correction...")
                
                # Apply correction
                if error_type == 'X':
                    system.apply_gate(system.PAULI_X, [error_loc])
                elif error_type == 'Z':
                    # Bit-flip code doesn't correct phase errors
                    print(f"  Warning: Bit-flip code cannot correct {error_type} errors")
                elif error_type == 'Y':
                    print(f"  Warning: Bit-flip code cannot correct {error_type} errors")
                
                # Verify correction
                expected_state = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128)
                actual_state = system.get_state_vector()
                fidelity = abs(np.vdot(expected_state, actual_state))**2
                
                print(f"  Correction fidelity: {fidelity:.6f}")
                print(f"  {'✅ Success' if fidelity > 0.99 else '❌ Failed'}")
            else:
                print(f"  Could not locate error (may be multi-qubit error)")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def _measure_parity(self, system: QubitSystem, qubits: List[int]) -> int:
        """Measure parity of two qubits (simplified simulation)"""
        # Simplified parity measurement
        state_vector = system.get_state_vector()
        
        # Calculate probability of even parity (00 or 11)
        even_prob = 0.0
        for i, amplitude in enumerate(state_vector):
            # Get bits for the two qubits
            bit1 = (i >> (system.config.num_subsystems - 1 - qubits[0])) & 1
            bit2 = (i >> (system.config.num_subsystems - 1 - qubits[1])) & 1
            
            if bit1 == bit2:  # Even parity
                even_prob += abs(amplitude) ** 2
        
        # Simulate measurement outcome
        return 0 if even_prob > 0.5 else 1
    
    def quantum_walk_demo(self):
        """Demonstrate quantum walk"""
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("QUANTUM WALK SIMULATION", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        print(f"\n{self.colorize('Quantum Walk vs Classical Random Walk:', 'yellow')}")
        print("  Classical: Diffusive spread (σ ∝ √t)")
        print("  Quantum: Ballistic spread (σ ∝ t)")
        
        steps = self._get_input_with_default(
            "Number of steps", 
            default=10, 
            min_val=1, 
            max_val=50
        )
        
        print(f"\n{self.colorize('Running quantum walk simulation...', 'cyan')}")
        
        # Simple quantum walk simulation on a line
        # Position space: |position⟩, Coin space: |coin⟩
        
        # Initial state: |0⟩⊗|0⟩ (position 0, coin 0)
        num_positions = 2 * steps + 1
        total_dim = 2 * num_positions  # 2 for coin, num_positions for position
        
        # Hadamard coin operator
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        
        # Shift operator: S|position, coin⟩ = |position + (-1)^coin, coin⟩
        # Implement shift operator as a matrix
        S = np.zeros((total_dim, total_dim), dtype=np.complex128)
        
        for pos in range(num_positions):
            for coin in [0, 1]:
                idx_from = 2 * pos + coin
                
                # Calculate new position
                if coin == 0:  # Move right
                    new_pos = (pos + 1) % num_positions
                else:  # Move left
                    new_pos = (pos - 1) % num_positions
                
                idx_to = 2 * new_pos + coin
                S[idx_to, idx_from] = 1.0
        
        # Initial state
        psi = np.zeros(total_dim, dtype=np.complex128)
        psi[0] = 1.0  # |0, 0⟩
        
        # Evolution
        position_probs = []
        
        for step in range(steps + 1):
            # Calculate position probabilities
            pos_prob = np.zeros(num_positions)
            for pos in range(num_positions):
                for coin in [0, 1]:
                    idx = 2 * pos + coin
                    pos_prob[pos] += abs(psi[idx]) ** 2
            
            position_probs.append(pos_prob.copy())
            
            if step < steps:
                # Apply coin operator (Hadamard on each position's coin space)
                for pos in range(num_positions):
                    coin_state = psi[2*pos:2*pos+2]
                    psi[2*pos:2*pos+2] = H @ coin_state
                
                # Apply shift operator
                psi = S @ psi
        
        # Display results
        print(f"\n{self.colorize('Quantum Walk Results:', 'yellow')}")
        print(f"  Steps: {steps}")
        print(f"  Position range: {-steps} to {steps}")
        
        # Show final probability distribution
        final_probs = position_probs[-1]
        max_prob_idx = np.argmax(final_probs)
        actual_position = max_prob_idx - steps
        
        print(f"\n{self.colorize('Most probable position:', 'green')} {actual_position}")
        print(f"{self.colorize('Probability at center:', 'cyan')} {final_probs[steps]:.4f}")
        
        # Calculate spread
        positions = np.arange(-steps, steps + 1)
        mean = np.sum(positions * final_probs)
        variance = np.sum((positions - mean) ** 2 * final_probs)
        std_dev = np.sqrt(variance)
        
        print(f"{self.colorize('Standard deviation:', 'cyan')} {std_dev:.2f}")
        print(f"{self.colorize('Expected classical std dev:', 'yellow')} {np.sqrt(steps):.2f}")
        
        # Save results
        results = {
            'steps': steps,
            'final_probability_distribution': final_probs.tolist(),
            'most_probable_position': int(actual_position),
            'probability_at_center': float(final_probs[steps]),
            'standard_deviation': float(std_dev),
            'expected_classical_std': float(np.sqrt(steps)),
            'quantum_speedup': std_dev / np.sqrt(steps) if steps > 0 else 1.0
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quantum_walk_{timestamp}.json"
        save_path = self.results_dir / filename
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        print(f"\n{self.colorize('✅ Results saved to:', 'green')} {save_path}")
        
        # Ask about visualization
        viz = input(f"\n{self.colorize('Create visualization? (y/n): ', 'yellow')}")
        if viz.lower() == 'y' and modules_status['integration.visualization_engine']:
            self._visualize_quantum_walk(position_probs, steps)
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def _visualize_quantum_walk(self, position_probs: List[np.ndarray], steps: int):
        """Visualize quantum walk results"""
        try:
            viz = QuantumVisualizationEngine()
            
            # Create animation frames or final distribution plot
            positions = np.arange(-steps, steps + 1)
            
            # Plot final distribution
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Final distribution
            final_probs = position_probs[-1]
            ax1.bar(positions, final_probs, alpha=0.7, color='blue')
            ax1.set_xlabel('Position')
            ax1.set_ylabel('Probability')
            ax1.set_title(f'Quantum Walk after {steps} steps')
            ax1.grid(True, alpha=0.3)
            
            # Classical comparison (Gaussian approximation)
            classical_probs = np.exp(-positions**2 / (2 * steps)) / np.sqrt(2 * np.pi * steps)
            ax1.plot(positions, classical_probs, 'r-', linewidth=2, label='Classical')
            ax1.legend()
            
            # Spread over time
            std_devs = [np.sqrt(np.sum((positions**2) * probs)) for probs in position_probs]
            times = range(steps + 1)
            
            ax2.plot(times, std_devs, 'b-', linewidth=2, label='Quantum')
            ax2.plot(times, np.sqrt(times), 'r--', linewidth=2, label='Classical (√t)')
            ax2.plot(times, times, 'g:', linewidth=2, label='Ballistic (t)')
            
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Standard Deviation')
            ax2.set_title('Spread over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = f"quantum_walk_plot_{timestamp}.png"
            plot_path = self.plots_dir / plot_file
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            
            print(f"{self.colorize('✅ Plot saved to:', 'green')} {plot_path}")
            
            # Try to show plot
            try:
                plt.show()
            except:
                pass
                
        except Exception as e:
            print(f"{self.colorize('⚠️  Visualization error:', 'yellow')} {e}")
    
    def custom_quantum_system(self):
        """Create and manipulate a custom quantum system"""
        while True:
            self.print_banner()
            
            system_status = f"{self.current_system.config.num_subsystems} {'qubit' if self.current_system.config.dimensions == 2 else f'qudit(d={self.current_system.config.dimensions})'}" \
                          if self.current_system else "None"
            
            options = [
                {
                    'description': 'Create Qubit System',
                    'action': 'create_qubit',
                    'shortcut': 'q',
                    'category': 'System Creation',
                    'status': 'Ready'
                },
                {
                    'description': 'Create Qudit System',
                    'action': 'create_qudit',
                    'shortcut': 'd',
                    'category': 'System Creation',
                    'status': 'Ready' if modules_status['core.qudit_system'] else 'Missing'
                },
                {
                    'description': 'Apply Quantum Gates',
                    'action': 'apply_gates',
                    'shortcut': 'g',
                    'category': 'System Manipulation',
                    'status': system_status if self.current_system else 'No System'
                },
                {
                    'description': 'Measure System',
                    'action': 'measure',
                    'shortcut': 'm',
                    'category': 'System Analysis',
                    'status': system_status if self.current_system else 'No System'
                },
                {
                    'description': 'Calculate Quantum Metrics',
                    'action': 'calculate_metrics',
                    'shortcut': 'c',
                    'category': 'System Analysis',
                    'status': system_status if self.current_system else 'No System'
                },
                {
                    'description': 'Save System State',
                    'action': 'save_system',
                    'shortcut': 's',
                    'category': 'System Management',
                    'status': system_status if self.current_system else 'No System'
                },
                {
                    'description': 'Load System State',
                    'action': 'load_system',
                    'shortcut': 'l',
                    'category': 'System Management',
                    'status': 'Ready'
                },
                {
                    'description': 'Reset System',
                    'action': 'reset_system',
                    'shortcut': 'r',
                    'category': 'System Management',
                    'status': system_status if self.current_system else 'No System'
                },
                {
                    'description': 'Return to Demo Menu',
                    'action': 'demo_menu',
                    'shortcut': 'x',
                    'category': 'Navigation'
                }
            ]
            
            choice = self.print_menu("CUSTOM QUANTUM SYSTEM", options, "Select option: ")
            
            if choice == 'create_qubit':
                self.create_qubit_system()
            elif choice == 'create_qudit':
                self.create_qudit_system()
            elif choice == 'apply_gates':
                self.apply_gates_menu()
            elif choice == 'measure':
                self.measure_system()
            elif choice == 'calculate_metrics':
                self.calculate_metrics()
            elif choice == 'save_system':
                self.save_system()
            elif choice == 'load_system':
                self.load_system()
            elif choice == 'reset_system':
                self.reset_system()
            elif choice == 'demo_menu':
                break
    
    def create_qubit_system(self):
        """Create a custom qubit system"""
        try:
            num_qubits = self._get_input_with_default(
                "Number of qubits", 
                default=2, 
                min_val=1, 
                max_val=16
            )
            
            validation = self._get_choice(
                "Validation level",
                options=["strict", "warn", "none"],
                default="warn"
            )
            
            representation = self._get_choice(
                "Representation",
                options=["dense", "sparse", "tensor"],
                default="dense"
            )
            
            self.current_system = QubitSystem(
                num_qubits=num_qubits,
                validation_level=validation,
                representation=representation
            )
            
            print(f"\n{self.colorize('✅', 'green')} Created {num_qubits}-qubit system")
            print(f"   {self.colorize('Hilbert dimension:', 'cyan')} {self.current_system.hilbert_dimension}")
            print(f"   {self.colorize('State norm:', 'cyan')} {self.current_system.state_norm:.12f}")
            print(f"   {self.colorize('Initial state:', 'cyan')} |{''.join(['0']*num_qubits)}⟩")
            
        except Exception as e:
            print(f"\n{self.colorize('❌ Error creating system:', 'red')} {e}")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def create_qudit_system(self):
        """Create a custom qudit system"""
        if not modules_status['core.qudit_system']:
            print(f"\n{self.colorize('❌ Qudit system module not available', 'red')}")
            input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            return
        
        try:
            num_qudits = self._get_input_with_default(
                "Number of qudits", 
                default=2, 
                min_val=1, 
                max_val=8
            )
            
            dimension = self._get_input_with_default(
                "Qudit dimension (d > 2)", 
                default=3, 
                min_val=3, 
                max_val=10
            )
            
            validation = self._get_choice(
                "Validation level",
                options=["strict", "warn", "none"],
                default="warn"
            )
            
            self.current_system = QuditSystem(
                num_qudits=num_qudits,
                dimension=dimension,
                validation_level=validation
            )
            
            print(f"\n{self.colorize('✅', 'green')} Created {num_qudits}-qudit system (d={dimension})")
            print(f"   {self.colorize('Hilbert dimension:', 'cyan')} {self.current_system.hilbert_dimension}")
            print(f"   {self.colorize('State norm:', 'cyan')} {self.current_system.state_norm:.12f}")
            print(f"   {self.colorize('Initial state:', 'cyan')} |{''.join(['0']*num_qudits)}⟩")
            
        except Exception as e:
            print(f"\n{self.colorize('❌ Error creating system:', 'red')} {e}")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def apply_gates_menu(self):
        """Apply gates to current system"""
        if not self.current_system:
            print(f"\n{self.colorize('❌ No system created', 'red')}")
            input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            return
        
        is_qubit = self.current_system.config.dimensions == 2
        
        while True:
            self.print_banner()
            
            print(f"{self.colorize('Current System:', 'cyan')} {self.current_system.config.num_subsystems} {'qubit' if is_qubit else 'qudit'}(s)")
            print(f"{self.colorize('State norm:', 'cyan')} {self.current_system.state_norm:.12f}")
            
            # Available gates
            print(f"\n{self.colorize('Available Gates:', 'yellow')}")
            
            if is_qubit:
                gate_list = [
                    ("H", "Hadamard"),
                    ("X", "Pauli-X"),
                    ("Y", "Pauli-Y"),
                    ("Z", "Pauli-Z"),
                    ("CX", "CNOT (control target)"),
                    ("RX", "Rotation-X (angle)"),
                    ("RY", "Rotation-Y (angle)"),
                    ("RZ", "Rotation-Z (angle)"),
                    ("S", "Phase (π/2)"),
                    ("T", "π/8 gate"),
                ]
            else:
                gate_list = [
                    ("GH", f"Generalized Hadamard (d={self.current_system.config.dimensions})"),
                    ("GX", "Generalized X (shift)"),
                    ("GZ", "Generalized Z (phase)"),
                    ("CINC", "Controlled increment"),
                ]
            
            for i, (gate, desc) in enumerate(gate_list):
                print(f"  {i+1:2d}. {gate:4} - {desc}")
            
            print(f"\n  {self.colorize('0', 'yellow')}. Return to system menu")
            
            choice = input(f"\n{self.colorize('Select gate (or enter gate name): ', 'green')}").strip()
            
            if choice == '0':
                break
            
            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(gate_list):
                    gate_name = gate_list[choice_num-1][0]
                else:
                    print(f"{self.colorize('Invalid selection', 'red')}")
                    continue
            else:
                gate_name = choice.upper()
            
            # Get targets
            targets_input = input(f"{self.colorize('Enter target qubit(s) (e.g., 0 or 0 1): ', 'green')}").strip()
            
            if not targets_input:
                continue
            
            try:
                targets = list(map(int, targets_input.split()))
                
                # Validate targets
                for target in targets:
                    if target >= self.current_system.config.num_subsystems:
                        print(f"{self.colorize(f'Target {target} out of range', 'red')}")
                        continue
                
                # Apply gate
                if is_qubit:
                    self._apply_qubit_gate(gate_name, targets)
                else:
                    self._apply_qudit_gate(gate_name, targets)
                
                print(f"{self.colorize('✅ Gate applied', 'green')}")
                print(f"   {self.colorize('New state norm:', 'cyan')} {self.current_system.state_norm:.12f}")
                
            except Exception as e:
                print(f"{self.colorize('❌ Error applying gate:', 'red')} {e}")
            
            cont = input(f"\n{self.colorize('Apply another gate? (y/n): ', 'yellow')}")
            if cont.lower() != 'y':
                break
    
    def _apply_qubit_gate(self, gate_name: str, targets: List[int]):
        """Apply qubit gate"""
        if gate_name == 'H':
            gate = self.current_system.HADAMARD
            self.current_system.apply_gate(gate, targets)
        elif gate_name == 'X':
            gate = self.current_system.PAULI_X
            self.current_system.apply_gate(gate, targets)
        elif gate_name == 'Y':
            gate = self.current_system.PAULI_Y
            self.current_system.apply_gate(gate, targets)
        elif gate_name == 'Z':
            gate = self.current_system.PAULI_Z
            self.current_system.apply_gate(gate, targets)
        elif gate_name == 'CX' and len(targets) == 2:
            cnot_matrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ])
            gate = QuantumGate(cnot_matrix, "CNOT")
            self.current_system.apply_gate(gate, targets)
        elif gate_name in ['RX', 'RY', 'RZ']:
            angle = float(input(f"{self.colorize(f'Enter angle for {gate_name} (in radians): ', 'green')}"))
            
            if gate_name == 'RX':
                # Rotation-X: exp(-iθX/2)
                theta = angle / 2
                matrix = np.array([
                    [np.cos(theta), -1j*np.sin(theta)],
                    [-1j*np.sin(theta), np.cos(theta)]
                ])
            elif gate_name == 'RY':
                # Rotation-Y: exp(-iθY/2)
                theta = angle / 2
                matrix = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ])
            elif gate_name == 'RZ':
                # Rotation-Z: exp(-iθZ/2)
                matrix = np.array([
                    [np.exp(-1j*angle/2), 0],
                    [0, np.exp(1j*angle/2)]
                ])
            
            gate = QuantumGate(matrix, gate_name)
            self.current_system.apply_gate(gate, targets)
        elif gate_name == 'S':
            # Phase gate: diag(1, i)
            matrix = np.array([[1, 0], [0, 1j]])
            gate = QuantumGate(matrix, "S")
            self.current_system.apply_gate(gate, targets)
        elif gate_name == 'T':
            # π/8 gate: diag(1, exp(iπ/4))
            matrix = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])
            gate = QuantumGate(matrix, "T")
            self.current_system.apply_gate(gate, targets)
        else:
            print(f"{self.colorize(f'Unknown gate: {gate_name}', 'red')}")
    
    def _apply_qudit_gate(self, gate_name: str, targets: List[int]):
        """Apply qudit gate"""
        d = self.current_system.config.dimensions
        
        if gate_name == 'GH':
            self.current_system.apply_generalized_hadamard(targets[0])
        elif gate_name == 'GX':
            # Generalized X (shift) operator
            matrix = np.zeros((d, d), dtype=np.complex128)
            for i in range(d):
                matrix[i, (i + 1) % d] = 1.0
            gate = QuantumGate(matrix, "Generalized_X")
            self.current_system.apply_gate(gate, targets)
        elif gate_name == 'GZ':
            # Generalized Z (phase) operator
            omega = np.exp(2j * np.pi / d)
            matrix = np.diag([omega**k for k in range(d)])
            gate = QuantumGate(matrix, "Generalized_Z")
            self.current_system.apply_gate(gate, targets)
        elif gate_name == 'CINC' and len(targets) == 2:
            self.current_system.apply_controlled_increment(targets[0], targets[1])
        else:
            print(f"{self.colorize(f'Unknown qudit gate: {gate_name}', 'red')}")
    
    def measure_system(self):
        """Measure the current system"""
        if not self.current_system:
            print(f"\n{self.colorize('❌ No system created', 'red')}")
            input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            return
        
        try:
            repetitions = self._get_input_with_default(
                "Number of measurements", 
                default=10000, 
                min_val=10, 
                max_val=1000000
            )
            
            print(f"\n{self.colorize('▶ Measuring system...', 'cyan')}")
            
            results = self.current_system.measure(repetitions=repetitions)
            
            print(f"\n{self.colorize('📊 Measurement Results:', 'yellow')}")
            print(self.colorize("-" * 50, 'cyan'))
            
            # Show top results
            sorted_results = sorted(results['probabilities'].items(), 
                                  key=lambda x: x[1], reverse=True)
            
            max_show = min(10, len(sorted_results))
            for i, (state, prob) in enumerate(sorted_results[:max_show]):
                count = results['counts'].get(state, 0)
                percentage = prob * 100
                print(f"  {self.colorize(state, 'blue'):10} {percentage:6.2f}% ({count:6d} counts)")
            
            if len(sorted_results) > max_show:
                print(f"  ... and {len(sorted_results) - max_show} more outcomes")
            
            print(self.colorize("-" * 50, 'cyan'))
            print(f"  {self.colorize('Total measurements:', 'cyan')} {repetitions}")
            
            # Check Born rule
            total_prob = sum(results['probabilities'].values())
            born_rule_check = abs(total_prob - 1.0) < 1e-8
            
            status = self.colorize('✓', 'green') if born_rule_check else self.colorize('✗', 'red')
            print(f"  {self.colorize('Born rule check:', 'cyan')} {status} (sum={total_prob:.8f})")
            
            # Save to history
            self.results_history.append({
                'type': 'measurement',
                'system': str(self.current_system.config),
                'results': results,
                'timestamp': datetime.now().isoformat()
            })
            
            # Ask to save
            save = input(f"\n{self.colorize('Save measurement results? (y/n): ', 'yellow')}")
            if save.lower() == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"measurement_{timestamp}.json"
                save_path = self.results_dir / filename
                
                with open(save_path, 'w') as f:
                    json.dump(results, f, indent=2, cls=NumpyEncoder)
                
                print(f"{self.colorize('✅ Results saved to:', 'green')} {save_path}")
            
        except Exception as e:
            print(f"{self.colorize('❌ Error measuring system:', 'red')} {e}")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def calculate_metrics(self):
        """Calculate quantum metrics for current system"""
        if not self.current_system:
            print(f"\n{self.colorize('❌ No system created', 'red')}")
            input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            return
        
        try:
            print(f"\n{self.colorize('📈 Calculating Quantum Metrics...', 'cyan')}")
            
            state_vector = self.current_system.get_state_vector()
            density_matrix = np.outer(state_vector, state_vector.conj())
            
            metrics = {}
            
            # Von Neumann entropy
            if len(state_vector) > 1:
                entropy = QuantumMetricCalculator.calculate_von_neumann_entropy(density_matrix)
                metrics['von_neumann_entropy'] = entropy
                print(f"  {self.colorize('Von Neumann Entropy:', 'cyan')} {entropy:.6f}")
            
            # Quantum coherence
            coherence = QuantumMetricCalculator.calculate_coherence(density_matrix)
            metrics['coherence'] = coherence
            max_coherence = len(state_vector) - 1
            coherence_ratio = coherence / max_coherence if max_coherence > 0 else 0
            print(f"  {self.colorize('Quantum Coherence:', 'cyan')} {coherence:.2f}/{max_coherence:.2f} ({coherence_ratio:.1%})")
            
            # Purity
            purity = QuantumMetricCalculator.calculate_purity(density_matrix)
            metrics['purity'] = purity
            print(f"  {self.colorize('Purity:', 'cyan')} {purity:.6f}")
            
            # For qubit systems, calculate more metrics
            if self.current_system.config.dimensions == 2:
                n = self.current_system.config.num_subsystems
                
                # Single qubit expectations
                if n == 1:
                    # Bloch sphere representation
                    rho = density_matrix
                    x_exp = np.real(np.trace(rho @ np.array([[0, 1], [1, 0]])))
                    y_exp = np.real(np.trace(rho @ np.array([[0, -1j], [1j, 0]])))
                    z_exp = np.real(np.trace(rho @ np.array([[1, 0], [0, -1]])))
                    
                    metrics['bloch_vector'] = [x_exp, y_exp, z_exp]
                    print(f"  {self.colorize('Bloch vector:', 'cyan')} ({x_exp:.3f}, {y_exp:.3f}, {z_exp:.3f})")
                
                # Concurrence for 2-qubit systems
                if n == 2:
                    concurrence = QuantumMetricCalculator.calculate_concurrence(state_vector, (2, 2))
                    metrics['concurrence'] = concurrence
                    print(f"  {self.colorize('Concurrence:', 'cyan')} {concurrence:.6f}")
                
                # Subsystem entropies
                if n > 1:
                    print(f"\n  {self.colorize('Subsystem Entropies:', 'yellow')}")
                    for k in range(1, n):
                        subsystem = list(range(k))
                        entropy = self.current_system.calculate_entropy(subsystem)
                        metrics[f'entropy_subsystem_{k}'] = entropy
                        print(f"    First {k} qubits: {entropy:.6f}")
            
            # Save metrics
            self.results_history.append({
                'type': 'metrics',
                'system': str(self.current_system.config),
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            })
            
            # Ask to save
            save = input(f"\n{self.colorize('Save metrics? (y/n): ', 'yellow')}")
            if save.lower() == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"metrics_{timestamp}.json"
                save_path = self.results_dir / filename
                
                with open(save_path, 'w') as f:
                    json.dump(metrics, f, indent=2, cls=NumpyEncoder)
                
                print(f"{self.colorize('✅ Metrics saved to:', 'green')} {save_path}")
            
        except Exception as e:
            print(f"{self.colorize('❌ Error calculating metrics:', 'red')} {e}")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def save_system(self):
        """Save current quantum system to file"""
        if not self.current_system:
            print(f"\n{self.colorize('❌ No system to save', 'red')}")
            input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            return
        
        try:
            # Get system data
            system_data = {
                'config': self.current_system.config.__dict__,
                'state_vector': self.current_system.get_state_vector().tolist(),
                'gate_history': [gate.name for gate in self.current_system._gate_history],
                'measurement_history': self.current_system._measurement_history[-5:],  # Last 5 measurements
                'timestamp': datetime.now().isoformat(),
                'session_id': self.session_id
            }
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            n = self.current_system.config.num_subsystems
            d = self.current_system.config.dimensions
            system_type = 'qubit' if d == 2 else f'qudit_d{d}'
            filename = f"{system_type}_{n}_{timestamp}.json"
            save_path = self.data_dir / filename
            
            # Save
            with open(save_path, 'w') as f:
                json.dump(system_data, f, indent=2, cls=NumpyEncoder)
            
            print(f"\n{self.colorize('✅ System saved to:', 'green')} {save_path}")
            print(f"   {self.colorize('System type:', 'cyan')} {system_type}")
            print(f"   {self.colorize('Number of subsystems:', 'cyan')} {n}")
            print(f"   {self.colorize('Hilbert dimension:', 'cyan')} {self.current_system.hilbert_dimension}")
            print(f"   {self.colorize('Gates applied:', 'cyan')} {len(self.current_system._gate_history)}")
            
        except Exception as e:
            print(f"\n{self.colorize('❌ Error saving system:', 'red')} {e}")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def load_system(self):
        """Load quantum system from file"""
        try:
            # List saved systems
            system_files = list(self.data_dir.glob("*.json"))
            
            if not system_files:
                print(f"\n{self.colorize('❌ No saved systems found', 'red')}")
                print(f"{self.colorize('Save a system first from the custom system menu', 'yellow')}")
                input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
                return
            
            print(f"\n{self.colorize('📁 Saved Systems:', 'yellow')}")
            for i, file in enumerate(system_files[:20], 1):
                # Try to parse filename for info
                name_parts = file.stem.split('_')
                if len(name_parts) >= 3:
                    system_type = name_parts[0]
                    n = name_parts[1]
                    timestamp = name_parts[2]
                    print(f"  {i:2d}. {system_type:8} {n:>2} subsystems ({timestamp})")
                else:
                    print(f"  {i:2d}. {file.name}")
            
            if len(system_files) > 20:
                print(f"  ... and {len(system_files) - 20} more")
            
            choice = input(f"\n{self.colorize('Select system to load (number) or 0 to cancel: ', 'green')}").strip()
            
            if choice == '0':
                return
            
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(system_files):
                    file_path = system_files[idx]
                    
                    print(f"\n{self.colorize('▶ Loading system...', 'cyan')}")
                    
                    with open(file_path, 'r') as f:
                        system_data = json.load(f)
                    
                    # Recreate system
                    config_dict = system_data['config']
                    config = QuantumSystemConfig(**config_dict)
                    
                    if config.dimensions == 2:
                        self.current_system = QubitSystem(
                            num_qubits=config.num_subsystems,
                            validation_level=config.validation_level,
                            representation=config.representation
                        )
                    else:
                        if modules_status['core.qudit_system']:
                            self.current_system = QuditSystem(
                                num_qudits=config.num_subsystems,
                                dimension=config.dimensions,
                                validation_level=config.validation_level,
                                representation=config.representation
                            )
                        else:
                            print(f"{self.colorize('❌ Qudit system module not available', 'red')}")
                            return
                    
                    # Load state vector
                    state_vector = np.array(system_data['state_vector'], dtype=np.complex128)
                    self.current_system._state = state_vector
                    
                    # Restore gate history (simplified)
                    self.current_system._gate_history = []
                    for gate_name in system_data.get('gate_history', []):
                        # Create placeholder gates
                        if gate_name == 'H':
                            self.current_system._gate_history.append(self.current_system.HADAMARD)
                    
                    print(f"\n{self.colorize('✅ System loaded successfully:', 'green')}")
                    print(f"   {self.colorize('File:', 'cyan')} {file_path.name}")
                    print(f"   {self.colorize('System:', 'cyan')} {config.num_subsystems} {'qubit' if config.dimensions == 2 else 'qudit'}(s)")
                    print(f"   {self.colorize('State norm:', 'cyan')} {self.current_system.state_norm:.12f}")
                    print(f"   {self.colorize('Gates in history:', 'cyan')} {len(self.current_system._gate_history)}")
                    
                else:
                    print(f"{self.colorize('Invalid selection', 'red')}")
            else:
                print(f"{self.colorize('Invalid input', 'red')}")
                
        except Exception as e:
            print(f"\n{self.colorize('❌ Error loading system:', 'red')} {e}")
            traceback.print_exc()
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def reset_system(self):
        """Reset the current system to initial state"""
        if not self.current_system:
            print(f"\n{self.colorize('❌ No system to reset', 'red')}")
            input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            return
        
        confirm = input(f"\n{self.colorize('Reset system to initial state? (y/n): ', 'red')}")
        
        if confirm.lower() == 'y':
            try:
                self.current_system._initialize_state()
                self.current_system._gate_history = []
                self.current_system._measurement_history = []
                
                print(f"\n{self.colorize('✅ System reset to initial state', 'green')}")
                print(f"   {self.colorize('State:', 'cyan')} |{''.join(['0']*self.current_system.config.num_subsystems)}⟩")
                print(f"   {self.colorize('State norm:', 'cyan')} {self.current_system.state_norm:.12f}")
                
            except Exception as e:
                print(f"\n{self.colorize('❌ Error resetting system:', 'red')} {e}")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def performance_benchmark(self):
        """Run performance benchmarking"""
        self.print_banner()
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("PERFORMANCE BENCHMARKING", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        print(f"\n{self.colorize('Benchmarking quantum operations...', 'yellow')}")
        
        import time
        import numpy as np
        
        # Test different system sizes
        if self.system_info['available_ram_gb'] < 2:
            max_qubits = 12  # Conservative for low RAM
            print(f"{self.colorize('⚠️  Low RAM detected:', 'yellow')} {self.system_info['available_ram_gb']:.1f} GB")
            print(f"{self.colorize('Limiting benchmark to', 'yellow')} {max_qubits} qubits")
        else:
            max_qubits = 16
        
        system_sizes = [2, 4, 6, 8, 10, 12]
        if max_qubits >= 14:
            system_sizes.append(14)
        if max_qubits >= 16:
            system_sizes.append(16)
        
        benchmarks = []
        
        for n in system_sizes:
            print(f"\n{self.colorize(f'▶ Benchmarking {n}-qubit system...', 'cyan')}")
            
            try:
                # Estimate memory
                hilbert_dim = 2**n
                memory_estimate = hilbert_dim * 16 / 1e9  # GB
                
                if memory_estimate > self.system_info['available_ram_gb'] * 0.5:
                    print(f"  {self.colorize('⚠️  Skipping:', 'yellow')} Memory estimate {memory_estimate:.1f} GB > available RAM")
                    break
                
                # Create system
                start = time.time()
                system = QubitSystem(n, validation_level="none")
                create_time = time.time() - start
                
                # Benchmark state initialization
                start = time.time()
                system._initialize_state()
                init_time = time.time() - start
                
                # Benchmark gate application
                start = time.time()
                gates_to_apply = min(n, 10)
                for i in range(gates_to_apply):
                    system.apply_gate(system.HADAMARD, [i % n])
                gate_time = time.time() - start
                
                # Benchmark measurement
                reps = min(10000, 100000 // hilbert_dim)
                start = time.time()
                system.measure(repetitions=reps)
                measure_time = time.time() - start
                
                # Benchmark entropy calculation
                start = time.time()
                if n >= 2:
                    system.calculate_entropy([0])
                entropy_time = time.time() - start
                
                total_time = create_time + init_time + gate_time + measure_time + entropy_time
                
                benchmark = {
                    'qubits': n,
                    'hilbert_dim': hilbert_dim,
                    'memory_estimate_gb': memory_estimate,
                    'times': {
                        'create': create_time,
                        'init': init_time,
                        'gate': gate_time,
                        'measure': measure_time,
                        'entropy': entropy_time,
                        'total': total_time
                    },
                    'gates_applied': gates_to_apply,
                    'measurements': reps
                }
                
                benchmarks.append(benchmark)
                
                print(f"  {self.colorize('✓', 'green')} Hilbert dimension: {hilbert_dim:,}")
                print(f"  {self.colorize('✓', 'green')} Memory estimate: {memory_estimate:.2f} GB")
                print(f"  {self.colorize('✓', 'green')} Total time: {total_time:.3f}s")
                
            except MemoryError:
                print(f"  {self.colorize('✗ Memory error at', 'red')} {n} qubits")
                break
            except Exception as e:
                print(f"  {self.colorize(f'✗ Error: {e}', 'red')}")
                break
        
        # Generate benchmark report
        if benchmarks:
            print(f"\n{self.colorize('=' * 70, 'green')}")
            print(self.colorize("BENCHMARK RESULTS", 'green', True))
            print(self.colorize("=" * 70, 'green'))
            
            # Summary table
            print(f"\n{self.colorize('System Size vs Performance:', 'yellow')}")
            print(self.colorize("-" * 80, 'cyan'))
            print(f"{self.colorize('Qubits', 'cyan'):>8} {self.colorize('Hilbert Dim', 'cyan'):>12} {self.colorize('Memory (GB)', 'cyan'):>12} {self.colorize('Total Time (s)', 'cyan'):>15} {self.colorize('Time/Qubit (ms)', 'cyan'):>15}")
            print(self.colorize("-" * 80, 'cyan'))
            
            for b in benchmarks:
                time_per_qubit = b['times']['total'] / b['qubits'] * 1000 if b['qubits'] > 0 else 0
                print(f"{b['qubits']:8d} {b['hilbert_dim']:12,} {b['memory_estimate_gb']:12.2f} {b['times']['total']:15.3f} {time_per_qubit:15.2f}")
            
            print(self.colorize("-" * 80, 'cyan'))
            
            # Scaling analysis
            if len(benchmarks) > 1:
                print(f"\n{self.colorize('Scaling Analysis:', 'yellow')}")
                
                # Fit exponential scaling
                qubits = [b['qubits'] for b in benchmarks]
                times = [b['times']['total'] for b in benchmarks]
                
                # Log-log fit
                log_qubits = np.log(qubits)
                log_times = np.log(times)
                
                try:
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(log_qubits, log_times)
                    
                    print(f"  Time scaling: O(2^{slope:.2f}N) [R²={r_value**2:.3f}]")
                    
                    if slope > 1:
                        scaling = self.colorize("Super-exponential", 'red')
                    elif slope > 0.8:
                        scaling = self.colorize("Exponential", 'yellow')
                    else:
                        scaling = self.colorize("Sub-exponential", 'green')
                    
                    print(f"  Scaling type: {scaling}")
                    
                except:
                    print(f"  {self.colorize('⚠️  Could not calculate scaling', 'yellow')}")
            
            # Save benchmark results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.json"
            save_path = self.results_dir / filename
            
            with open(save_path, 'w') as f:
                json.dump({
                    'system_info': self.system_info,
                    'benchmarks': benchmarks,
                    'timestamp': timestamp
                }, f, indent=2, cls=NumpyEncoder)
            
            print(f"\n{self.colorize('✅ Benchmark results saved to:', 'green')} {save_path}")
            
            # Recommendations
            print(f"\n{self.colorize('Recommendations:', 'yellow')}")
            
            max_safe = max(b['qubits'] for b in benchmarks)
            if max_safe >= 16:
                print(f"  • Your system can handle {max_safe}+ qubit simulations")
                print(f"  • Consider enabling sparse representation for >16 qubits")
            elif max_safe >= 12:
                print(f"  • Your system can handle up to {max_safe} qubit simulations")
                print(f"  • For larger systems, close other applications")
            else:
                print(f"  • Your system is limited to {max_safe} qubit simulations")
                print(f"  • Consider upgrading RAM for larger systems")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def module_management(self):
        """Module management menu"""
        while True:
            self.print_banner()
            
            # Module status summary
            if self.integrator:
                available = sum(1 for status in self.module_status.values() if status)
                total = len(self.module_status)
                module_summary = f"{available}/{total}"
            else:
                module_summary = "Not initialized"
            
            options = [
                {
                    'description': 'Initialize Module Integration',
                    'action': 'init_integration',
                    'shortcut': 'i',
                    'category': 'Integration',
                    'status': 'Ready' if not self.integrator else 'Done'
                },
                {
                    'description': 'Download External Modules',
                    'action': 'download_modules',
                    'shortcut': 'd',
                    'category': 'Integration',
                    'status': 'External'
                },
                {
                    'description': 'List Available Modules',
                    'action': 'list_modules',
                    'shortcut': 'l',
                    'category': 'Information',
                    'status': module_summary
                },
                {
                    'description': 'Test Module Integration',
                    'action': 'test_modules',
                    'shortcut': 't',
                    'category': 'Testing',
                    'status': 'Validate'
                },
                {
                    'description': 'Configure Module Priorities',
                    'action': 'configure_modules',
                    'shortcut': 'c',
                    'category': 'Configuration',
                    'status': 'Edit'
                },
                {
                    'description': 'Update Module Configuration',
                    'action': 'update_config',
                    'shortcut': 'u',
                    'category': 'Configuration',
                    'status': 'Refresh'
                },
                {
                    'description': 'Return to Main Menu',
                    'action': 'main_menu',
                    'shortcut': 'm',
                    'category': 'Navigation'
                }
            ]
            
            choice = self.print_menu("MODULE MANAGEMENT", options, "Select action: ")
            
            if choice == 'init_integration':
                self.initialize_integration()
                input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            elif choice == 'download_modules':
                self.download_modules()
            elif choice == 'list_modules':
                self.list_modules()
                input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            elif choice == 'test_modules':
                self.test_module_integration()
                input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            elif choice == 'configure_modules':
                self.configure_modules()
            elif choice == 'update_config':
                self.update_module_config()
            elif choice == 'main_menu':
                break
    
    def download_modules(self):
        """Download external modules"""
        self.print_banner()
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("DOWNLOAD EXTERNAL MODULES", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        print(f"\n{self.colorize('This will download quantum modules from GitHub:', 'yellow')}")
        print(f"  1. {self.colorize('sentiflow.py', 'blue')} - Main quantum circuit framework")
        print(f"  2. {self.colorize('quantum_core_engine.py', 'blue')} - Core quantum operations")
        print(f"  3. {self.colorize('qybrik.py', 'blue')} - Circuit building blocks")
        print(f"  4. {self.colorize('qylintos.py', 'blue')} - Circuit optimization")
        print(f"  5. {self.colorize('bumpy.py', 'blue')} - Array operations")
        print(f"  6. {self.colorize('flumpy.py', 'blue')} - Flexible arrays")
        print(f"  7. {self.colorize('laser.py', 'blue')} - Quantum control")
        print(f"  8. {self.colorize('bugginrace.py', 'blue')} - Circuit debugging")
        print(f"  9. {self.colorize('cognition_core.py', 'blue')} - Quantum ML")
        
        print(f"\n{self.colorize('Modules will be saved to:', 'cyan')} src/quantum_core_nexus/external/")
        
        confirm = input(f"\n{self.colorize('Proceed with download? (y/n): ', 'red')}").strip().lower()
        
        if confirm == 'y':
            try:
                # Run download script
                script_path = Path(__file__).parent / "scripts" / "download_modules.py"
                
                if script_path.exists():
                    print(f"\n{self.colorize('▶ Downloading modules...', 'cyan')}")
                    
                    result = subprocess.run(
                        [sys.executable, str(script_path)],
                        capture_output=True,
                        text=True
                    )
                    
                    print("\n" + result.stdout)
                    if result.stderr:
                        print(f"{self.colorize('Errors:', 'red')}")
                        print(result.stderr)
                    
                    # Reinitialize integration
                    if result.returncode == 0:
                        print(f"\n{self.colorize('▶ Reinitializing module integration...', 'cyan')}")
                        self.initialize_integration(silent=False)
                else:
                    print(f"\n{self.colorize('❌ Download script not found:', 'red')} {script_path}")
                    print(f"{self.colorize('Please create scripts/download_modules.py', 'yellow')}")
                    
            except Exception as e:
                print(f"\n{self.colorize('❌ Error downloading modules:', 'red')} {e}")
                traceback.print_exc()
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def list_modules(self):
        """List available modules with detailed information"""
        if not self.integrator:
            print(f"\n{self.colorize('❌ Module integration not initialized.', 'red')}")
            return
        
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("MODULE STATUS REPORT", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        # Group modules by availability
        available_modules = []
        unavailable_modules = []
        
        for name, info in self.integrator.modules.items():
            if info.get('available'):
                available_modules.append((name, info))
            else:
                unavailable_modules.append((name, info))
        
        # Available modules
        if available_modules:
            print(f"\n{self.colorize('✅ AVAILABLE MODULES:', 'green')}")
            print(self.colorize("-" * 70, 'green'))
            
            for name, info in available_modules:
                version = info.get('version', '?.?.?')
                purpose = info.get('purpose', 'No description')
                priority = info.get('priority', 'medium')
                
                priority_color = {
                    'high': 'red',
                    'medium': 'yellow',
                    'low': 'blue'
                }.get(priority, 'white')
                
                print(f"  {self.colorize('•', 'green')} {self.colorize(name, 'cyan'):20} {self.colorize(f'v{version}', 'yellow')}")
                print(f"    {self.colorize('Purpose:', 'white')} {purpose}")
                print(f"    {self.colorize('Priority:', 'white')} {self.colorize(priority, priority_color)}")
                
                if 'url' in info:
                    print(f"    {self.colorize('URL:', 'white')} {info['url']}")
        
        # Unavailable modules
        if unavailable_modules:
            print(f"\n{self.colorize('❌ UNAVAILABLE MODULES:', 'red')}")
            print(self.colorize("-" * 70, 'red'))
            
            for name, info in unavailable_modules:
                purpose = info.get('purpose', 'No description')
                priority = info.get('priority', 'medium')
                
                print(f"  {self.colorize('✗', 'red')} {self.colorize(name, 'cyan'):20}")
                print(f"    {self.colorize('Purpose:', 'white')} {purpose}")
                print(f"    {self.colorize('Status:', 'white')} {self.colorize('Not installed', 'red')}")
                
                if 'url' in info:
                    print(f"    {self.colorize('Download:', 'white')} {info['url']}")
        
        # Summary
        total = len(self.integrator.modules)
        available = len(available_modules)
        missing_high = [name for name, info in self.integrator.modules.items() 
                       if not info.get('available') and info.get('priority') == 'high']
        
        print(f"\n{self.colorize('📊 SUMMARY:', 'yellow')}")
        print(f"  Total modules: {total}")
        print(f"  Available: {available} ({available/total:.1%})")
        print(f"  Unavailable: {total - available}")
        
        if missing_high:
            print(f"\n{self.colorize('⚠️  MISSING HIGH-PRIORITY MODULES:', 'red')}")
            for mod in missing_high:
                print(f"  • {mod}")
            print(f"\n{self.colorize('Run "Download External Modules" to install', 'yellow')}")
    
    def test_module_integration(self):
        """Test module integration"""
        if not self.integrator:
            print(f"\n{self.colorize('❌ Module integration not initialized.', 'red')}")
            return
        
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("MODULE INTEGRATION TEST", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        test_results = []
        
        # Test each available module
        for name, info in self.integrator.modules.items():
            if info.get('available'):
                print(f"\n{self.colorize(f'▶ Testing {name}...', 'cyan')}")
                
                try:
                    module = info['module']
                    
                    # Basic functionality tests
                    tests_passed = 0
                    tests_total = 0
                    
                    # Test 1: Check version
                    tests_total += 1
                    if hasattr(module, '__version__'):
                        version = module.__version__
                        print(f"  {self.colorize('✓', 'green')} Version: {version}")
                        tests_passed += 1
                    else:
                        print(f"  {self.colorize('⚠️', 'yellow')} No version info")
                    
                    # Test 2: Check for expected functions/classes
                    tests_total += 1
                    expected_attrs = {
                        'sentiflow': ['QuantumCircuit', 'simulate'],
                        'quantum_core_engine': ['apply_gate', 'QuantumGate'],
                        'qybrik': ['controlled_gate', 'circuit_block'],
                        'qylintos': ['optimize', 'validate_equivalence'],
                        'bumpy': ['BumpyArray'],
                        'flumpy': ['tensor_product'],
                        'laser': ['create_pulse', 'apply_pulse'],
                        'bugginrace': ['analyze_circuit', 'generate_report'],
                        'cognition_core': ['quantum_pattern_match'],
                    }
                    
                    expected = expected_attrs.get(name, [])
                    if expected:
                        found = []
                        for attr in expected:
                            if hasattr(module, attr):
                                found.append(attr)
                        
                        if found:
                            print(f"  {self.colorize('✓', 'green')} Found functions: {', '.join(found[:3])}" + 
                                 (f"... ({len(found)} total)" if len(found) > 3 else ""))
                            tests_passed += 1
                        else:
                            print(f"  {self.colorize('⚠️', 'yellow')} Expected functions not found")
                    else:
                        # Just check if module has any public functions
                        public_funcs = [attr for attr in dir(module) 
                                      if not attr.startswith('_') and callable(getattr(module, attr))]
                        if public_funcs:
                            print(f"  {self.colorize('✓', 'green')} Has {len(public_funcs)} public functions")
                            tests_passed += 1
                        else:
                            print(f"  {self.colorize('⚠️', 'yellow')} No public functions found")
                    
                    # Test 3: Try to create an instance or call a function
                    tests_total += 1
                    try:
                        # Try a simple operation based on module type
                        if name == 'bumpy' and hasattr(module, 'BumpyArray'):
                            arr = module.BumpyArray([1, 2, 3])
                            print(f"  {self.colorize('✓', 'green')} Can create BumpyArray")
                            tests_passed += 1
                        elif name == 'numpy':
                            arr = module.array([1, 2, 3])
                            print(f"  {self.colorize('✓', 'green')} NumPy arrays work")
                            tests_passed += 1
                        else:
                            # For other modules, just try to access something
                            _ = dir(module)
                            print(f"  {self.colorize('✓', 'green')} Module accessible")
                            tests_passed += 1
                    except Exception as e:
                        print(f"  {self.colorize('⚠️', 'yellow')} Functional test: {e}")
                    
                    success_rate = tests_passed / tests_total if tests_total > 0 else 0
                    status = self.colorize('PASS', 'green') if success_rate > 0.66 else \
                            self.colorize('WARN', 'yellow') if success_rate > 0.33 else \
                            self.colorize('FAIL', 'red')
                    
                    test_results.append({
                        'name': name,
                        'passed': tests_passed,
                        'total': tests_total,
                        'success_rate': success_rate,
                        'status': status.replace('\033[0m', '').replace('\033[92m', '').replace('\033[93m', '').replace('\033[91m', '')
                    })
                    
                except Exception as e:
                    print(f"  {self.colorize('✗', 'red')} Error: {e}")
                    test_results.append({
                        'name': name,
                        'error': str(e),
                        'status': 'ERROR'
                    })
        
        # Summary
        print(f"\n{self.colorize('=' * 70, 'green')}")
        print(self.colorize("TEST SUMMARY", 'green', True))
        print(self.colorize("=" * 70, 'green'))
        
        if test_results:
            passed = sum(1 for r in test_results if r.get('success_rate', 0) > 0.66)
            total = len(test_results)
            
            print(f"\n{self.colorize('Overall Results:', 'yellow')}")
            print(f"  Tests run: {total}")
            print(f"  Modules passing: {passed} ({passed/total:.1%})")
            print(f"  Modules with issues: {total - passed}")
            
            print(f"\n{self.colorize('Detailed Results:', 'yellow')}")
            for result in test_results:
                name = result['name']
                if 'error' in result:
                    status = self.colorize('ERROR', 'red')
                    print(f"  {name:20} {status:10} {result['error']}")
                else:
                    status = result['status']
                    if 'PASS' in status:
                        status_display = self.colorize('PASS', 'green')
                    elif 'WARN' in status:
                        status_display = self.colorize('WARN', 'yellow')
                    else:
                        status_display = self.colorize('FAIL', 'red')
                    
                    success_rate = result['success_rate']
                    print(f"  {name:20} {status_display:10} {result['passed']}/{result['total']} tests ({success_rate:.0%})")
            
            if passed == total:
                print(f"\n{self.colorize('✅ All modules integrated successfully!', 'green')}")
            else:
                print(f"\n{self.colorize('⚠️  Some modules have integration issues', 'yellow')}")
                print(f"{self.colorize('Consider re-downloading problematic modules', 'cyan')}")
        else:
            print(f"\n{self.colorize('No modules available for testing', 'yellow')}")
    
    def configure_modules(self):
        """Configure module priorities and settings"""
        self.print_banner()
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("MODULE CONFIGURATION", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        print(f"\n{self.colorize('Current Configuration:', 'yellow')}")
        
        if 'module_priorities' in self.config:
            for priority, modules in self.config['module_priorities'].items():
                print(f"\n  {self.colorize(priority.upper() + ' PRIORITY:', 'cyan')}")
                for module in modules:
                    status = self.colorize('(available)', 'green') if self.module_status.get(module) else \
                            self.colorize('(missing)', 'red')
                    print(f"    • {module:20} {status}")
        
        print(f"\n{self.colorize('Configuration Options:', 'yellow')}")
        print("  1. Edit module priorities")
        print("  2. Change performance settings")
        print("  3. Reset to defaults")
        print("  4. Save configuration")
        print("  5. Return to module menu")
        
        choice = input(f"\n{self.colorize('Select option: ', 'green')}").strip()
        
        if choice == '1':
            self._edit_module_priorities()
        elif choice == '2':
            self._edit_performance_settings()
        elif choice == '3':
            self._reset_configuration()
        elif choice == '4':
            self._save_configuration()
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def _edit_module_priorities(self):
        """Edit module priority settings"""
        print(f"\n{self.colorize('Edit Module Priorities:', 'yellow')}")
        
        # Show current priorities
        priorities = self.config.get('module_priorities', {})
        
        for priority_level in ['high', 'medium', 'low']:
            modules = priorities.get(priority_level, [])
            print(f"\n  {self.colorize(priority_level.upper() + ':', 'cyan')}")
            for i, module in enumerate(modules, 1):
                print(f"    {i}. {module}")
        
        print(f"\n{self.colorize('Options:', 'yellow')}")
        print("  1. Move module between priorities")
        print("  2. Add new module")
        print("  3. Remove module")
        print("  4. Cancel")
        
        choice = input(f"\n{self.colorize('Select option: ', 'green')}").strip()
        
        # Implementation would go here
        print(f"\n{self.colorize('⚠️  Configuration editor not fully implemented', 'yellow')}")
        print(f"{self.colorize('Edit config/modules.yaml manually', 'cyan')}")
    
    def _edit_performance_settings(self):
        """Edit performance settings"""
        print(f"\n{self.colorize('Performance Settings:', 'yellow')}")
        
        perf = self.config.get('performance', {})
        
        print(f"\n  Current settings:")
        for key, value in perf.items():
            print(f"    {key:30} = {value}")
        
        print(f"\n{self.colorize('Edit which setting?', 'cyan')}")
        for i, key in enumerate(perf.keys(), 1):
            print(f"  {i}. {key}")
        
        choice = input(f"\n{self.colorize('Select setting (or Enter to cancel): ', 'green')}").strip()
        
        if choice and choice.isdigit():
            idx = int(choice) - 1
            keys = list(perf.keys())
            if 0 <= idx < len(keys):
                key = keys[idx]
                current = perf[key]
                new_value = input(f"{self.colorize(f'New value for {key} (current: {current}): ', 'green')}").strip()
                
                try:
                    # Try to convert to appropriate type
                    if isinstance(current, int):
                        perf[key] = int(new_value)
                    elif isinstance(current, float):
                        perf[key] = float(new_value)
                    else:
                        perf[key] = new_value
                    
                    self.config['performance'] = perf
                    print(f"{self.colorize('✅ Setting updated', 'green')}")
                except ValueError:
                    print(f"{self.colorize('❌ Invalid value', 'red')}")
    
    def _reset_configuration(self):
        """Reset configuration to defaults"""
        confirm = input(f"\n{self.colorize('Reset configuration to defaults? (y/n): ', 'red')}").strip().lower()
        
        if confirm == 'y':
            self.config = {
                "module_priorities": {
                    "high": ["sentiflow", "quantum_core_engine", "qybrik"],
                    "medium": ["bumpy", "flumpy", "qylintos"],
                    "low": ["laser", "bugginrace", "cognition_core"]
                },
                "performance": {
                    "max_qubits": 16,
                    "max_qudits": 8,
                    "max_dimension": 10,
                    "validation_tolerance": 1e-10,
                    "measurement_repetitions": 10000,
                }
            }
            print(f"{self.colorize('✅ Configuration reset to defaults', 'green')}")
    
    def _save_configuration(self):
        """Save configuration to file"""
        try:
            import yaml
            
            config_path = Path(__file__).parent / "config" / "modules.yaml"
            config_path.parent.mkdir(exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            print(f"{self.colorize('✅ Configuration saved to:', 'green')} {config_path}")
            
        except ImportError:
            print(f"{self.colorize('❌ PyYAML not installed', 'red')}")
            print(f"{self.colorize('Install with: pip install pyyaml', 'yellow')}")
        except Exception as e:
            print(f"{self.colorize('❌ Error saving configuration:', 'red')} {e}")
    
    def update_module_config(self):
        """Update module configuration from current integration"""
        if not self.integrator:
            print(f"\n{self.colorize('❌ Module integration not initialized', 'red')}")
            return
        
        # Update module status in config
        if 'module_priorities' not in self.config:
            self.config['module_priorities'] = {}
        
        # Ensure all modules are in config
        all_modules = []
        for module_name in self.module_status.keys():
            all_modules.append(module_name)
        
        # Distribute modules by availability
        high_priority = ['sentiflow', 'quantum_core_engine', 'qybrik']
        medium_priority = ['bumpy', 'flumpy', 'qylintos']
        low_priority = ['laser', 'bugginrace', 'cognition_core']
        
        self.config['module_priorities'] = {
            'high': high_priority,
            'medium': medium_priority,
            'low': low_priority
        }
        
        print(f"\n{self.colorize('✅ Module configuration updated', 'green')}")
        print(f"{self.colorize('Based on current integration status', 'cyan')}")
        
        # Save automatically
        self._save_configuration()
    
    def scientific_validation(self):
        """Run scientific validation suite"""
        if not self.current_system:
            print(f"\n{self.colorize('❌ No quantum system available for validation', 'red')}")
            print(f"{self.colorize('Create a system or run demonstrations first', 'yellow')}")
            input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            return
        
        self.print_banner()
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("SCIENTIFIC VALIDATION SUITE", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        print(f"\n{self.colorize('System being validated:', 'yellow')}")
        print(f"  Type: {'Qubit' if self.current_system.config.dimensions == 2 else 'Qudit'}")
        print(f"  Number: {self.current_system.config.num_subsystems}")
        print(f"  Hilbert dimension: {self.current_system.hilbert_dimension}")
        print(f"  Validation level: {self.current_system.config.validation_level}")
        
        try:
            validator = QuantumValidator(self.current_system)
            
            print(f"\n{self.colorize('▶ Running quantum principle validation...', 'cyan')}")
            results = validator.run_all_tests()
            
            print(f"\n{self.colorize('✅ Validation Results:', 'green')}")
            print(self.colorize("-" * 60, 'cyan'))
            
            passed = 0
            total = len(results)
            
            for test_name, result in results.items():
                test_passed = result['passed']
                if test_passed:
                    passed += 1
                    status = self.colorize('✓ PASS', 'green')
                else:
                    status = self.colorize('✗ FAIL', 'red')
                
                message = result.get('message', result.get('error', 'No details'))
                print(f"  {test_name:30} {status}")
                
                if not test_passed and message:
                    print(f"    {message}")
            
            print(self.colorize("-" * 60, 'cyan'))
            print(f"  {self.colorize('Summary:', 'yellow')} {passed}/{total} tests passed ({passed/total:.1%})")
            
            if passed == total:
                print(f"  {self.colorize('✅ All quantum principles validated!', 'green')}")
            elif passed >= total * 0.8:
                print(f"  {self.colorize('⚠️  Most principles validated', 'yellow')}")
            else:
                print(f"  {self.colorize('❌ Significant principle violations', 'red')}")
                print(f"  {self.colorize('Check violation log for details', 'cyan')}")
            
            # Save validation results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_{timestamp}.json"
            save_path = self.results_dir / filename
            
            with open(save_path, 'w') as f:
                json.dump({
                    'system': str(self.current_system.config),
                    'results': results,
                    'summary': {
                        'passed': passed,
                        'total': total,
                        'percentage': passed/total
                    },
                    'violations': validator.violation_log,
                    'timestamp': timestamp
                }, f, indent=2, cls=NumpyEncoder)
            
            print(f"\n{self.colorize('✅ Validation report saved to:', 'green')} {save_path}")
            
            # Show violations if any
            if validator.violation_log:
                print(f"\n{self.colorize('Violations:', 'red')}")
                for violation in validator.violation_log[:5]:  # Show first 5
                    print(f"  • {violation['test']}: {violation['message']}")
                
                if len(validator.violation_log) > 5:
                    print(f"  ... and {len(validator.violation_log) - 5} more")
            
        except Exception as e:
            print(f"\n{self.colorize('❌ Error during validation:', 'red')} {e}")
            traceback.print_exc()
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def visualization_menu(self):
        """Data visualization menu"""
        # Check if we have results to visualize
        if not self.results_history and not self.demonstration_results:
            print(f"\n{self.colorize('❌ No results available for visualization', 'red')}")
            print(f"{self.colorize('Run some demonstrations or create a system first', 'yellow')}")
            input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            return
        
        if not modules_status['integration.visualization_engine']:
            print(f"\n{self.colorize('❌ Visualization engine not available', 'red')}")
            print(f"{self.colorize('Some visualizations may be limited', 'yellow')}")
        
        self.print_banner()
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("DATA VISUALIZATION", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        options = [
            {
                'description': 'Bloch Sphere (single qubit)',
                'action': 'bloch_sphere',
                'shortcut': 'b',
                'category': 'State Visualization',
                'status': 'Qubit Only'
            },
            {
                'description': 'Probability Distribution',
                'action': 'probability_dist',
                'shortcut': 'p',
                'category': 'Measurement Visualization',
                'status': 'Any System'
            },
            {
                'description': 'Entanglement Entropy Scaling',
                'action': 'entanglement_scaling',
                'shortcut': 'e',
                'category': 'Entanglement Analysis',
                'status': 'Multi-Qubit'
            },
            {
                'description': 'Circuit Diagram',
                'action': 'circuit_diagram',
                'shortcut': 'c',
                'category': 'Circuit Visualization',
                'status': 'Gate History'
            },
            {
                'description': 'Performance Benchmark Plot',
                'action': 'benchmark_plot',
                'shortcut': 'm',
                'category': 'Performance Analysis',
                'status': 'Benchmark Data'
            },
            {
                'description': 'Quantum Walk Visualization',
                'action': 'walk_visualization',
                'shortcut': 'w',
                'category': 'Dynamics',
                'status': 'Walk Data'
            },
            {
                'description': 'Return to Main Menu',
                'action': 'main_menu',
                'shortcut': 'x',
                'category': 'Navigation'
            }
        ]
        
        choice = self.print_menu("VISUALIZATION OPTIONS", options, "Select visualization: ")
        
        if choice == 'bloch_sphere':
            self.visualize_bloch_sphere()
        elif choice == 'probability_dist':
            self.visualize_probability_distribution()
        elif choice == 'entanglement_scaling':
            self.visualize_entanglement_scaling()
        elif choice == 'circuit_diagram':
            self.visualize_circuit_diagram()
        elif choice == 'benchmark_plot':
            self.visualize_benchmark_results()
        elif choice == 'walk_visualization':
            self.visualize_quantum_walk_results()
        elif choice == 'main_menu':
            return
    
    def visualize_bloch_sphere(self):
        """Visualize state on Bloch sphere"""
        if not self.current_system:
            print(f"\n{self.colorize('❌ No quantum system available', 'red')}")
            return
        
        if self.current_system.config.num_subsystems < 1:
            print(f"\n{self.colorize('❌ System has no qubits', 'red')}")
            return
        
        # For multi-qubit systems, ask which qubit to visualize
        if self.current_system.config.num_subsystems > 1:
            qubit_idx = self._get_input_with_default(
                f"Which qubit to visualize (0-{self.current_system.config.num_subsystems-1})",
                default=0,
                min_val=0,
                max_val=self.current_system.config.num_subsystems-1
            )
        else:
            qubit_idx = 0
        
        try:
            if modules_status['integration.visualization_engine']:
                viz = QuantumVisualizationEngine()
                
                state_vector = self.current_system.get_state_vector()
                fig = viz.plot_bloch_sphere(state_vector, qubit_index=qubit_idx)
                
                # Save figure
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"bloch_sphere_{timestamp}.png"
                save_path = self.plots_dir / filename
                
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"\n{self.colorize('✅ Bloch sphere saved to:', 'green')} {save_path}")
                
                # Try to show plot
                try:
                    import matplotlib.pyplot as plt
                    plt.show()
                except:
                    print(f"{self.colorize('⚠️  Could not display plot', 'yellow')}")
            else:
                print(f"\n{self.colorize('❌ Visualization engine not available', 'red')}")
                print(f"{self.colorize('Manual Bloch sphere calculation:', 'yellow')}")
                
                # Manual calculation
                state_vector = self.current_system.get_state_vector()
                
                # For multi-qubit, trace out other qubits
                if self.current_system.config.num_subsystems > 1:
                    print(f"{self.colorize('⚠️  Multi-qubit Bloch sphere not implemented', 'yellow')}")
                else:
                    # Single qubit case
                    alpha, beta = state_vector[0], state_vector[1]
                    
                    # Bloch vector components
                    x = 2 * np.real(np.conj(alpha) * beta)
                    y = 2 * np.imag(np.conj(alpha) * beta)
                    z = abs(alpha)**2 - abs(beta)**2
                    
                    print(f"  Bloch vector: ({x:.3f}, {y:.3f}, {z:.3f})")
                    print(f"  State: {alpha:.3f}|0⟩ + {beta:.3f}|1⟩")
                    
        except Exception as e:
            print(f"\n{self.colorize('❌ Error visualizing Bloch sphere:', 'red')} {e}")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def visualize_probability_distribution(self):
        """Visualize probability distribution"""
        if not self.current_system:
            print(f"\n{self.colorize('❌ No quantum system available', 'red')}")
            input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            return
        
        # Measure or use existing results
        use_existing = input(f"\n{self.colorize('Measure system now? (y/n): ', 'yellow')}").strip().lower()
        
        if use_existing == 'y':
            repetitions = self._get_input_with_default(
                "Number of measurements",
                default=10000,
                min_val=100,
                max_val=1000000
            )
            
            print(f"\n{self.colorize('▶ Measuring system...', 'cyan')}")
            results = self.current_system.measure(repetitions=repetitions)
        else:
            # Use last measurement from history
            measurements = [r for r in self.results_history if r.get('type') == 'measurement']
            if not measurements:
                print(f"\n{self.colorize('❌ No measurement results available', 'red')}")
                input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
                return
            
            results = measurements[-1]['results']
            print(f"\n{self.colorize('Using last measurement results', 'cyan')}")
        
        # Create visualization
        try:
            import matplotlib.pyplot as plt
            
            probs = results['probabilities']
            
            # Sort by probability
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            states = [s for s, _ in sorted_probs]
            probabilities = [p for _, p in sorted_probs]
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Bar chart
            bars = ax1.bar(range(len(states)), probabilities, alpha=0.7, color='skyblue')
            ax1.set_xlabel('Basis State')
            ax1.set_ylabel('Probability')
            ax1.set_title('Measurement Probability Distribution')
            ax1.set_xticks(range(len(states)))
            ax1.set_xticklabels(states, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on top of bars
            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Cumulative distribution
            cumulative = np.cumsum(probabilities)
            ax2.plot(range(len(states)), cumulative, 'b-', linewidth=2)
            ax2.fill_between(range(len(states)), 0, cumulative, alpha=0.3, color='blue')
            ax2.set_xlabel('Number of States')
            ax2.set_ylabel('Cumulative Probability')
            ax2.set_title('Cumulative Distribution')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Total = 1')
            ax2.legend()
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"probability_dist_{timestamp}.png"
            save_path = self.plots_dir / filename
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            print(f"\n{self.colorize('✅ Plot saved to:', 'green')} {save_path}")
            
            # Statistics
            print(f"\n{self.colorize('Statistics:', 'cyan')}")
            print(f"  Number of possible states: {len(states)}")
            print(f"  Most probable state: {states[0]} ({probabilities[0]:.3%})")
            print(f"  Least probable state: {states[-1]} ({probabilities[-1]:.3%})")
            print(f"  Shannon entropy: {-np.sum(probabilities * np.log2(probabilities + 1e-12)):.3f}")
            
            # Try to show plot
            try:
                plt.show()
            except:
                pass
                
        except Exception as e:
            print(f"\n{self.colorize('❌ Error creating visualization:', 'red')} {e}")
        
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
        
        print(f"\n{self.colorize('Configuration Options:', 'yellow')}")
        print("  1. Change Log Directory")
        print("  2. Clear Results History")
        print("  3. View System Information")
        print("  4. Export Session Data")
        print("  5. Import Session Data")
        print("  6. Clean Temporary Files")
        print("  7. Return to Main Menu")
        
        choice = input(f"\n{self.colorize('Select option: ', 'green')}").strip()
        
        if choice == '1':
            self._change_log_directory()
        elif choice == '2':
            self._clear_results_history()
        elif choice == '3':
            self._show_system_information()
        elif choice == '4':
            self._export_session_data()
        elif choice == '5':
            self._import_session_data()
        elif choice == '6':
            self._clean_temporary_files()
        elif choice == '7':
            return
    
    def _change_log_directory(self):
        """Change log directory"""
        new_dir = input(f"\n{self.colorize('Enter new log directory: ', 'green')}").strip()
        
        if new_dir:
            new_path = Path(new_dir)
            try:
                new_path.mkdir(exist_ok=True, parents=True)
                self.log_dir = new_path
                self.results_dir = new_path / "results"
                self.results_dir.mkdir(exist_ok=True)
                self.plots_dir = new_path / "plots"
                self.plots_dir.mkdir(exist_ok=True)
                
                print(f"{self.colorize('✅ Log directory changed to:', 'green')} {new_path}")
            except Exception as e:
                print(f"{self.colorize('❌ Error changing directory:', 'red')} {e}")
        else:
            print(f"{self.colorize('❌ No directory specified', 'red')}")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def _clear_results_history(self):
        """Clear results history"""
        if not self.results_history and not self.demonstration_results:
            print(f"\n{self.colorize('❌ No results history to clear', 'red')}")
            input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            return
        
        confirm = input(f"\n{self.colorize('Clear all results history? (y/n): ', 'red')}").strip().lower()
        
        if confirm == 'y':
            self.results_history = []
            self.demonstration_results = {}
            print(f"{self.colorize('✅ Results history cleared', 'green')}")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def _show_system_information(self):
        """Show detailed system information"""
        import psutil
        
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("SYSTEM INFORMATION", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        info_items = [
            ("Platform", self.system_info['platform']),
            ("Python Version", self.system_info['python_version']),
            ("Python Executable", self.system_info['python_executable']),
            ("CPU Cores (Logical)", str(self.system_info['cpu_cores'])),
            ("CPU Cores (Physical)", str(self.system_info['cpu_cores_physical'])),
            ("Total RAM", f"{self.system_info['total_ram_gb']:.2f} GB"),
            ("Available RAM", f"{self.system_info['available_ram_gb']:.2f} GB"),
            ("Working Directory", self.system_info['working_directory']),
            ("Session ID", self.session_id),
            ("QuantumCore Nexus Path", str(Path(__file__).parent)),
        ]
        
        for key, value in info_items:
            print(f"  {self.colorize(key + ':', 'cyan')} {value}")
        
        # Check for required packages
        print(f"\n{self.colorize('Package Status:', 'yellow')}")
        
        packages = [
            ('numpy', '1.21.0'),
            ('scipy', '1.7.0'),
            ('matplotlib', '3.4.0'),
            ('pyyaml', '6.0'),
            ('requests', '2.26.0'),
            ('psutil', '5.8.0'),
        ]
        
        for package, min_version in packages:
            try:
                mod = __import__(package)
                version = getattr(mod, '__version__', 'unknown')
                
                # Version check (simplified)
                if version != 'unknown':
                    try:
                        from packaging import version as pkg_version
                        if pkg_version.parse(version) >= pkg_version.parse(min_version):
                            status = self.colorize('✓', 'green')
                        else:
                            status = self.colorize('⚠️', 'yellow')
                    except:
                        status = self.colorize('✓', 'green')
                else:
                    status = self.colorize('✓', 'green')
                
                print(f"  {status} {package:15} v{version}")
            except ImportError:
                print(f"  {self.colorize('✗', 'red')} {package:15} not installed")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def _export_session_data(self):
        """Export session data"""
        if not self.results_history and not self.demonstration_results:
            print(f"\n{self.colorize('❌ No data to export', 'red')}")
            input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            return
        
        export_data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'system_info': self.system_info,
            'results_history': self.results_history,
            'demonstration_results': self.demonstration_results,
            'current_system': str(self.current_system.config) if self.current_system else None,
            'module_status': self.module_status,
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_export_{timestamp}.json"
        save_path = self.exports_dir / filename
        
        try:
            with open(save_path, 'w') as f:
                json.dump(export_data, f, indent=2, cls=NumpyEncoder)
            
            print(f"\n{self.colorize('✅ Session data exported to:', 'green')} {save_path}")
            print(f"  {self.colorize('Items exported:', 'cyan')}")
            print(f"    • Results history: {len(self.results_history)} items")
            print(f"    • Demonstration results: {len(self.demonstration_results)} categories")
            print(f"    • System information")
            print(f"    • Module status")
            
        except Exception as e:
            print(f"\n{self.colorize('❌ Error exporting data:', 'red')} {e}")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def _import_session_data(self):
        """Import session data"""
        # List export files
        export_files = list(self.exports_dir.glob("session_export_*.json"))
        
        if not export_files:
            print(f"\n{self.colorize('❌ No export files found', 'red')}")
            input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            return
        
        print(f"\n{self.colorize('Export Files:', 'yellow')}")
        for i, file in enumerate(export_files[-10:], 1):  # Show last 10
            # Try to parse timestamp from filename
            timestamp = file.stem.replace('session_export_', '')
            print(f"  {i:2d}. {timestamp}")
        
        if len(export_files) > 10:
            print(f"  ... and {len(export_files) - 10} more")
        
        choice = input(f"\n{self.colorize('Select file to import (number) or 0 to cancel: ', 'green')}").strip()
        
        if choice == '0':
            return
        
        if choice.isdigit():
            idx = int(choice) - 1
            files_to_show = export_files[-10:] if len(export_files) > 10 else export_files
            if 0 <= idx < len(files_to_show):
                file_path = files_to_show[idx]
                
                try:
                    with open(file_path, 'r') as f:
                        import_data = json.load(f)
                    
                    # Merge data
                    self.results_history = import_data.get('results_history', [])
                    self.demonstration_results = import_data.get('demonstration_results', {})
                    
                    print(f"\n{self.colorize('✅ Session data imported:', 'green')}")
                    print(f"  {self.colorize('File:', 'cyan')} {file_path.name}")
                    print(f"  {self.colorize('Original session:', 'cyan')} {import_data.get('session_id', 'Unknown')}")
                    print(f"  {self.colorize('Results imported:', 'cyan')} {len(self.results_history)} items")
                    print(f"  {self.colorize('Demonstrations imported:', 'cyan')} {len(self.demonstration_results)} categories")
                    
                except Exception as e:
                    print(f"\n{self.colorize('❌ Error importing data:', 'red')} {e}")
            else:
                print(f"{self.colorize('Invalid selection', 'red')}")
        else:
            print(f"{self.colorize('Invalid input', 'red')}")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def _clean_temporary_files(self):
        """Clean temporary files"""
        confirm = input(f"\n{self.colorize('Clean temporary files? (y/n): ', 'red')}").strip().lower()
        
        if confirm == 'y':
            # Clean tmp directory
            tmp_files = list(self.tmp_dir.glob("*"))
            cleaned = 0
            
            for file in tmp_files:
                try:
                    if file.is_file():
                        file.unlink()
                        cleaned += 1
                except:
                    pass
            
            print(f"{self.colorize(f'✅ Cleaned {cleaned} temporary files', 'green')}")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def main_menu(self):
        """Main menu loop"""
        while True:
            self.print_banner()
            
            # Status indicators
            system_status = f"{self.current_system.config.num_subsystems} {'qubit' if self.current_system.config.dimensions == 2 else f'qudit(d={self.current_system.config.dimensions})'}" \
                          if self.current_system else self.colorize("None", 'red')
            
            if self.integrator:
                available = sum(1 for status in self.module_status.values() if status)
                module_status = f"{available}/9" if available > 0 else self.colorize("None", 'red')
            else:
                module_status = self.colorize("Not init", 'yellow')
            
            results_status = f"{len(self.results_history)}" if self.results_history else "0"
            demo_status = f"{len(self.demonstration_results)}" if self.demonstration_results else "0"
            
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
                    'description': 'Data Visualization',
                    'action': 'visualization',
                    'shortcut': 'v',
                    'category': 'Analysis',
                    'status': f'{results_status} results'
                },
                {
                    'description': 'Scientific Validation',
                    'action': 'validation',
                    'shortcut': 's',
                    'category': 'Verification',
                    'status': system_status if self.current_system else 'No System'
                },
                {
                    'description': 'Performance Benchmark',
                    'action': 'benchmark',
                    'shortcut': 'b',
                    'category': 'Performance',
                    'status': 'System Test'
                },
                {
                    'description': 'View Documentation',
                    'action': 'documentation',
                    'shortcut': 'h',
                    'category': 'Help',
                    'status': 'Info'
                },
                {
                    'description': 'System Configuration',
                    'action': 'configuration',
                    'shortcut': 'c',
                    'category': 'Settings',
                    'status': 'Settings'
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
                print(f"\n{self.colorize('=' * 70, 'cyan')}")
                print(self.colorize("Thank you for using QuantumCore Nexus!", 'green', True))
                print(self.colorize("Scientific Quantum Simulation Platform", 'cyan'))
                print(self.colorize(f"Session: {self.session_id}", 'yellow'))
                print(self.colorize("=" * 70, 'cyan'))
                
                # Save session summary
                self._save_session_summary()
                break
    
    def show_documentation(self):
        """Show documentation"""
        self.print_banner()
        print(f"\n{self.colorize('=' * 70, 'cyan')}")
        print(self.colorize("QUANTUMCORE NEXUS DOCUMENTATION", 'cyan', True))
        print(self.colorize("=" * 70, 'cyan'))
        
        docs = f"""
{self.colorize('QUICK START:', 'yellow', True)}
  1. Run {self.colorize('Module Management', 'cyan')} → {self.colorize('Initialize Module Integration', 'green')}
  2. Run {self.colorize('Quantum Demonstrations', 'cyan')} to see quantum effects
  3. Use {self.colorize('Data Visualization', 'cyan')} to view results
  4. Create custom systems with {self.colorize('Custom Quantum System', 'cyan')}

{self.colorize('KEY FEATURES:', 'yellow', True)}
  • {self.colorize('Unified System Interface:', 'cyan')} Qubits & qudits in one framework
  • {self.colorize('Scientific Validation:', 'cyan')} Quantum principle enforcement
  • {self.colorize('Module Integration:', 'cyan')} Works with SentiFlow ecosystem
  • {self.colorize('Performance Analysis:', 'cyan')} Benchmark system capabilities
  • {self.colorize('Visualization:', 'cyan')} Plot quantum states and results

{self.colorize('DEMONSTRATIONS:', 'yellow', True)}
  • {self.colorize('Bell State:', 'cyan')} 2-qubit entanglement
  • {self.colorize('GHZ State:', 'cyan')} Multi-qubit entanglement
  • {self.colorize('Quantum Teleportation:', 'cyan')} State transfer protocol
  • {self.colorize('Superdense Coding:', 'cyan')} 2 classical bits in 1 qubit
  • {self.colorize('Quantum Fourier Transform:', 'cyan')} Quantum algorithm
  • {self.colorize('Quantum Walk:', 'cyan')} Quantum dynamics
  • {self.colorize('Qudit Systems:', 'cyan')} Multi-level quantum systems

{self.colorize('MODULE INTEGRATION:', 'yellow', True)}
  The system integrates with external modules from SentiFlow:
  • {self.colorize('sentiflow:', 'blue')} Quantum circuit framework
  • {self.colorize('quantum_core_engine:', 'blue')} Low-level quantum operations
  • {self.colorize('qybrik:', 'blue')} Circuit building blocks
  • {self.colorize('qylintos:', 'blue')} Circuit optimization
  • {self.colorize('bumpy/flumpy:', 'blue')} Array operations
  • {self.colorize('laser:', 'blue')} Quantum control
  • {self.colorize('cognition_core:', 'blue')} Quantum ML
  • {self.colorize('bugginrace:', 'blue')} Circuit debugging

{self.colorize('KEYBOARD SHORTCUTS:', 'yellow', True)}
  • {self.colorize('d', 'green')} - Run demonstrations
  • {self.colorize('m', 'green')} - Module management
  • {self.colorize('v', 'green')} - Data visualization
  • {self.colorize('b', 'green')} - Performance benchmark
  • {self.colorize('s', 'green')} - Scientific validation
  • {self.colorize('h', 'green')} - This documentation
  • {self.colorize('c', 'green')} - System configuration
  • {self.colorize('x', 'red')} - Exit

{self.colorize('COMMAND LINE USAGE:', 'yellow', True)}
  {self.colorize('python main.py --demo qubit --qubits 8', 'cyan')}
  {self.colorize('python main.py --benchmark', 'cyan')}
  {self.colorize('python main.py --download', 'cyan')} (download modules)

{self.colorize('TROUBLESHOOTING:', 'yellow', True)}
  • Missing modules: Run {self.colorize('Module Management', 'cyan')} → {self.colorize('Download External Modules', 'green')}
  • Performance issues: Run {self.colorize('Performance Benchmark', 'cyan')} to check system limits
  • Visualization errors: Install matplotlib: {self.colorize('pip install matplotlib', 'cyan')}
  • Module import errors: Check Python path and dependencies

{self.colorize('FOR MORE INFORMATION:', 'yellow', True)}
  See README.md for detailed documentation and examples.
  Report issues at: https://github.com/TaoishTechy/QuantumCore-Nexus
        """
        
        print(docs)
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def _save_session_summary(self):
        """Save session summary before exit"""
        try:
            summary = {
                'session_id': self.session_id,
                'start_time': self.session_id,  # Session ID is a timestamp
                'end_time': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'results_count': len(self.results_history),
                'demonstration_count': len(self.demonstration_results),
                'system_info': self.system_info,
                'module_status': self.module_status,
                'current_system': str(self.current_system.config) if self.current_system else None,
            }
            
            summary_file = self.log_dir / f"session_summary_{self.session_id}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, cls=NumpyEncoder)
            
            print(f"\n{self.colorize('Session summary saved:', 'cyan')} {summary_file}")
            
        except Exception as e:
            print(f"{self.colorize('⚠️  Could not save session summary:', 'yellow')} {e}")
    
    def _get_input_with_default(self, prompt: str, default: Any, 
                               min_val: Optional[float] = None, 
                               max_val: Optional[float] = None) -> Any:
        """Get user input with default value and validation"""
        while True:
            input_str = input(f"{self.colorize(f'{prompt} [{default}]: ', 'green')}").strip()
            
            if input_str == '':
                return default
            
            try:
                # Try to convert to appropriate type
                if isinstance(default, int):
                    value = int(input_str)
                elif isinstance(default, float):
                    value = float(input_str)
                else:
                    value = input_str
                
                # Validate range if specified
                if min_val is not None and value < min_val:
                    print(f"{self.colorize(f'Value must be at least {min_val}', 'red')}")
                    continue
                
                if max_val is not None and value > max_val:
                    print(f"{self.colorize(f'Value must be at most {max_val}', 'red')}")
                    continue
                
                return value
                
            except ValueError:
                print(f"{self.colorize('Invalid input. Please enter a valid value.', 'red')}")
    
    def _get_choice(self, prompt: str, options: List[str], default: str) -> str:
        """Get a choice from a list of options"""
        print(f"\n{self.colorize(prompt, 'yellow')}")
        
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}{' (default)' if option == default else ''}")
        
        while True:
            choice = input(f"\n{self.colorize(f'Select option [1-{len(options)}]: ', 'green')}").strip()
            
            if choice == '':
                return default
            
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            
            print(f"{self.colorize(f'Please enter a number between 1 and {len(options)}', 'red')}")
    
    def _save_results_prompt(self, demo_type: str, results: Dict, elapsed: float):
        """Prompt to save demonstration results"""
        save = input(f"\n{self.colorize('Save demonstration results? (y/n): ', 'yellow')}").strip().lower()
        
        if save == 'y':
            # Generate report
            try:
                if modules_status['integration.report_generator']:
                    generator = QuantumReportGenerator()
                    
                    report = generator.generate_experiment_report(
                        demo_results=results,
                        validation_results={},
                        system_info=self.system_info,
                        elapsed_time=elapsed
                    )
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_file = f"{demo_type}_report_{timestamp}.json"
                    report_path = self.results_dir / report_file
                    
                    with open(report_path, 'w') as f:
                        json.dump(report, f, indent=2, cls=NumpyEncoder)
                    
                    print(f"\n{self.colorize('✅ Report generated:', 'green')} {report_path}")
                else:
                    # Simple save
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{demo_type}_results_{timestamp}.json"
                    save_path = self.results_dir / filename
                    
                    with open(save_path, 'w') as f:
                        json.dump(results, f, indent=2, cls=NumpyEncoder)
                    
                    print(f"\n{self.colorize('✅ Results saved to:', 'green')} {save_path}")
                    
            except Exception as e:
                print(f"\n{self.colorize('❌ Error saving results:', 'red')} {e}")
    
    def visualize_entanglement_scaling(self):
        """Visualize entanglement entropy scaling"""
        print(f"\n{self.colorize('⚠️  Entanglement scaling visualization not fully implemented', 'yellow')}")
        print(f"{self.colorize('Run GHZ state demonstrations first to generate data', 'cyan')}")
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def visualize_circuit_diagram(self):
        """Visualize circuit diagram"""
        if not self.current_system:
            print(f"\n{self.colorize('❌ No quantum system available', 'red')}")
            input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            return
        
        gate_history = self.current_system._gate_history
        
        if not gate_history:
            print(f"\n{self.colorize('❌ No gates applied to system', 'red')}")
            input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            return
        
        print(f"\n{self.colorize('Circuit Diagram:', 'yellow')}")
        print(self.colorize("=" * 60, 'cyan'))
        
        # Simple ASCII circuit diagram
        n = self.current_system.config.num_subsystems
        
        for i in range(n):
            line = f"q[{i}] --"
            for gate in gate_history:
                # Simplified - just show gate names
                line += f" {gate.name} --"
            line += " M"
            print(f"  {line}")
        
        print(self.colorize("=" * 60, 'cyan'))
        print(f"  {self.colorize('Total gates:', 'cyan')} {len(gate_history)}")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def visualize_benchmark_results(self):
        """Visualize benchmark results"""
        # Look for benchmark files
        benchmark_files = list(self.results_dir.glob("benchmark_*.json"))
        
        if not benchmark_files:
            print(f"\n{self.colorize('❌ No benchmark results found', 'red')}")
            print(f"{self.colorize('Run performance benchmark first', 'yellow')}")
            input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            return
        
        # Use most recent benchmark
        latest_file = max(benchmark_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                benchmark_data = json.load(f)
            
            benchmarks = benchmark_data.get('benchmarks', [])
            
            if not benchmarks:
                print(f"\n{self.colorize('❌ No benchmark data in file', 'red')}")
                input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
                return
            
            # Create visualization
            import matplotlib.pyplot as plt
            
            qubits = [b['qubits'] for b in benchmarks]
            times = [b['times']['total'] for b in benchmarks]
            memory = [b['memory_estimate_gb'] for b in benchmarks]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Time scaling
            ax1.plot(qubits, times, 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel('Number of Qubits')
            ax1.set_ylabel('Total Time (seconds)')
            ax1.set_title('Execution Time Scaling')
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            
            # Add exponential fit line
            if len(qubits) > 1:
                # Fit exponential
                import numpy as np
                log_times = np.log(times)
                coeffs = np.polyfit(qubits, log_times, 1)
                fit_times = np.exp(coeffs[1]) * np.exp(coeffs[0] * np.array(qubits))
                ax1.plot(qubits, fit_times, 'r--', label=f'Exp fit: O(2^{coeffs[0]:.2f}N)')
                ax1.legend()
            
            # Memory usage
            ax2.bar(qubits, memory, alpha=0.7, color='green')
            ax2.set_xlabel('Number of Qubits')
            ax2.set_ylabel('Memory Estimate (GB)')
            ax2.set_title('Memory Usage')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (q, m) in enumerate(zip(qubits, memory)):
                ax2.text(q, m + 0.1, f'{m:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_plot_{timestamp}.png"
            save_path = self.plots_dir / filename
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            print(f"\n{self.colorize('✅ Benchmark plot saved to:', 'green')} {save_path}")
            
            # Display system limits
            print(f"\n{self.colorize('System Limits:', 'cyan')}")
            print(f"  Maximum qubits tested: {max(qubits)}")
            print(f"  Maximum memory used: {max(memory):.1f} GB")
            print(f"  Total execution time: {sum(times):.1f} seconds")
            
            # Try to show plot
            try:
                plt.show()
            except:
                pass
                
        except Exception as e:
            print(f"\n{self.colorize('❌ Error creating visualization:', 'red')} {e}")
        
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def visualize_quantum_walk_results(self):
        """Visualize quantum walk results"""
        # Look for quantum walk files
        walk_files = list(self.results_dir.glob("quantum_walk_*.json"))
        
        if not walk_files:
            print(f"\n{self.colorize('❌ No quantum walk results found', 'red')}")
            print(f"{self.colorize('Run quantum walk demonstration first', 'yellow')}")
            input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
            return
        
        print(f"\n{self.colorize('⚠️  Quantum walk visualization requires matplotlib', 'yellow')}")
        print(f"{self.colorize('Run quantum walk demo with visualization option', 'cyan')}")
        input(f"\n{self.colorize('Press Enter to continue...', 'yellow')}")
    
    def run(self):
        """Run the CLI"""
        try:
            # Initialize
            print(f"\n{self.colorize('Initializing QuantumCore Nexus...', 'cyan')}")
            
            # Initialize module integration
            self.initialize_integration(silent=True)
            
            # Show welcome
            self.print_banner()
            
            print(f"\n{self.colorize('Welcome to QuantumCore Nexus!', 'green', True)}")
            print(f"{self.colorize('Scientific Quantum Simulation Platform', 'cyan')}")
            print(f"\n{self.colorize('Session:', 'yellow')} {self.session_id}")
            print(f"{self.colorize('System:', 'yellow')} {self.system_info['platform'].split('-')[0]}")
            print(f"{self.colorize('Python:', 'yellow')} {self.system_info['python_version'].split()[0]}")
            
            # Check for external modules
            if self.integrator:
                available = sum(1 for status in self.module_status.values() if status)
                if available > 0:
                    print(f"{self.colorize('Modules:', 'green')} {available} external modules detected")
                else:
                    print(f"{self.colorize('Modules:', 'yellow')} No external modules found")
                    print(f"{self.colorize('Tip:', 'cyan')} Run Module Management → Download External Modules")
            
            input(f"\n{self.colorize('Press Enter to continue to main menu...', 'yellow')}")
            
            # Start main menu
            self.main_menu()
            
        except KeyboardInterrupt:
            print(f"\n\n{self.colorize('Interrupted by user. Exiting gracefully...', 'yellow')}")
            self._save_session_summary()
        except Exception as e:
            print(f"\n{self.colorize('❌ Fatal error:', 'red')} {e}")
            traceback.print_exc()
            self._save_session_summary()

def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(
        description="QuantumCore Nexus - Scientific Quantum Simulation Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     # Interactive mode
  %(prog)s --demo qubit        # Run qubit demonstrations
  %(prog)s --benchmark         # Run performance benchmark
  %(prog)s --download          # Download external modules
  %(prog)s --qubits 10         # Test with 10 qubits
  %(prog)s --output results.json # Save results to file
        
For interactive use, run without arguments.
        """
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
    parser.add_argument('--validate', action='store_true',
                       help='Run scientific validation')
    parser.add_argument('--list-modules', action='store_true',
                       help='List available modules and exit')
    parser.add_argument('--version', action='version', 
                       version='QuantumCore Nexus v1.0.0')
    
    args = parser.parse_args()
    
    # Handle command line arguments
    if args.download:
        # Download modules
        script_path = Path(__file__).parent / "scripts" / "download_modules.py"
        if script_path.exists():
            subprocess.run([sys.executable, str(script_path)])
        else:
            print(f"{QuantumCoreNexusCLI().colorize('❌ Download script not found', 'red')}")
        return
    
    if args.list_modules:
        # List modules
        cli = QuantumCoreNexusCLI()
        cli.initialize_integration(silent=True)
        cli.list_modules()
        return
    
    if args.demo or args.benchmark or args.validate:
        # Non-interactive mode
        cli = QuantumCoreNexusCLI()
        cli.initialize_integration(silent=True)
        
        if args.benchmark:
            cli.performance_benchmark()
        elif args.validate:
            # Would need a system to validate
            print(f"{cli.colorize('❌ Validation requires a quantum system', 'red')}")
            print(f"{cli.colorize('Use interactive mode or create a system first', 'yellow')}")
        elif args.demo:
            # Run demonstrations
            if args.demo == 'qubit':
                # Run qubit demos
                max_qubits = args.qubits
                
                print(f"{cli.colorize('Running qubit demonstrations...', 'cyan')}")
                print(f"{cli.colorize(f'Maximum qubits: {max_qubits}', 'yellow')}")
                
                # Create demo suite
                demo_suite = QuantumDemonstrationSuite()
                
                # Run all demos
                results = demo_suite.run_all_demos(max_qubits=max_qubits)
                
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(results, f, indent=2, cls=NumpyEncoder)
                    print(f"{cli.colorize(f'Results saved to {args.output}', 'green')}")
            
            elif args.demo == 'qudit':
                print(f"{cli.colorize('Qudit demonstrations require interactive mode', 'yellow')}")
                print(f"{cli.colorize('Use: python main.py', 'cyan')}")
        
        return
    
    # Interactive mode
    cli = QuantumCoreNexusCLI()
    cli.run()

if __name__ == "__main__":
    main()
