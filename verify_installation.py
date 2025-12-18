# verify_installation.py
"""
Verify QuantumCore Nexus installation and dependencies
"""

import sys
import subprocess
import importlib

def check_package(package_name, min_version=None):
    """Check if a package is installed and meets version requirements"""
    try:
        module = importlib.import_module(package_name)
        if hasattr(module, '__version__'):
            version = module.__version__
            if min_version:
                from packaging import version as pkg_version
                if pkg_version.parse(version) < pkg_version.parse(min_version):
                    return False, f"Version {version} < {min_version}"
            return True, f"Version {version}"
        return True, "Installed (no version info)"
    except ImportError:
        return False, "Not installed"

def main():
    print("=" * 70)
    print("QuantumCore Nexus - Installation Verification")
    print("=" * 70)
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python Version: {python_version}")
    
    # Check essential packages
    packages = {
        'numpy': '1.21.0',
        'scipy': '1.7.0',
        'matplotlib': '3.4.0',
        'pyyaml': '6.0',
        'requests': '2.26.0',
        'psutil': '5.8.0',
    }
    
    all_ok = True
    for package, min_version in packages.items():
        ok, message = check_package(package, min_version)
        status = "✓" if ok else "✗"
        print(f"{status} {package:15} {message}")
        if not ok:
            all_ok = False
    
    # Check QuantumCore Nexus modules
    print("\nQuantumCore Nexus Modules:")
    try:
        import quantum_core_nexus
        print("✓ quantum_core_nexus package")
        
        # Try importing key components
        test_imports = [
            ('core.qubit_system', 'QubitSystem'),
            ('validation.scientific_validator', 'QuantumValidator'),
            ('demonstrations.qubit_demos', 'QuantumDemonstrationSuite'),
        ]
        
        for module_path, class_name in test_imports:
            try:
                module = __import__(f'quantum_core_nexus.{module_path}', fromlist=[class_name])
                getattr(module, class_name)
                print(f"✓ {module_path}.{class_name}")
            except ImportError as e:
                print(f"✗ {module_path}.{class_name}: {e}")
                all_ok = False
                
    except ImportError as e:
        print(f"✗ quantum_core_nexus: {e}")
        all_ok = False
    
    print("\n" + "=" * 70)
    if all_ok:
        print("✅ All checks passed! QuantumCore Nexus is ready to use.")
    else:
        print("⚠️  Some checks failed. Please install missing packages.")
    print("=" * 70)
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)