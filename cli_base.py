#!/usr/bin/env python3
"""
QuantumCore Nexus - CLI Base Infrastructure
Contains core CLI classes, menu system, and basic utilities
"""

import os
import sys
import json
import time
import argparse
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import QuantumCore Nexus modules
def import_quantum_modules():
    """Import quantum modules with graceful fallback handling"""
    modules_status = {}
    
    # ... (same import logic as original, truncated for brevity)
    return modules_status

class QuantumCoreNexusCLIBase:
    """
    Base class for QuantumCore Nexus CLI
    Contains core functionality without demonstrations
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
            # This would initialize the integrator
            # For now, return True for demo purposes
            return True
            
        except Exception as e:
            if not silent:
                print(f"{self.colorize('❌', 'red')} Failed to initialize integration: {e}")
            return False
    
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{demo_type}_results_{timestamp}.json"
            save_path = self.results_dir / filename
            
            try:
                from quantum_core_nexus.utils.numpy_encoder import NumpyEncoder
                with open(save_path, 'w') as f:
                    json.dump(results, f, indent=2, cls=NumpyEncoder)
                
                print(f"\n{self.colorize('✅ Results saved to:', 'green')} {save_path}")
            except:
                with open(save_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"\n{self.colorize('✅ Results saved to:', 'green')} {save_path}")