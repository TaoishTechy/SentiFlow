# setup.py
from setuptools import setup, find_packages
import os
import re

def read_requirements():
    """Read requirements from requirements.txt"""
    req_file = 'requirements.txt'
    if os.path.exists(req_file):
        with open(req_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    # Fallback to core dependencies
    return ['numpy>=1.21.0', 'scipy>=1.7.0']

def read_version():
    """Read version from __init__.py in the main package"""
    # Try multiple possible locations for the version
    version_files = [
        os.path.join('src', 'qnvm', '__init__.py'),
        os.path.join('src', 'sentiflow', '__init__.py'),
        os.path.join('src', '__init__.py')
    ]
    
    for version_file in version_files:
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                content = f.read()
                version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", content)
                if version_match:
                    return version_match.group(1)
    return '5.1.0'  # Default version based on recent test suite

def read_long_description():
    """Read long description from README.md"""
    readme_file = 'README.md'
    if os.path.exists(readme_file):
        try:
            with open(readme_file, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(readme_file, 'r', encoding='latin-1') as f:
                return f.read()
    return "SentiFlow: Experimental AGI Research Framework"

# Define package groups
quantum_core = [
    'numpy>=1.21.0',
    'scipy>=1.7.0',
    'psutil>=5.8.0',
    'h5py>=3.6.0',
]

visualization = [
    'matplotlib>=3.4.0',
    'plotly>=5.10.0',
    'seaborn>=0.11.0',
]

cli = [
    'click>=8.0.0',
    'rich>=12.0.0',
    'tqdm>=4.62.0',
]

data_processing = [
    'pandas>=1.3.0',
    'pyyaml>=6.0',
]

networking = [
    'requests>=2.26.0',
    'aiohttp>=3.8.0',
]

testing = [
    'pytest>=7.0.0',
    'pytest-cov>=4.0.0',
    'pytest-asyncio>=0.18.0',
    'pytest-benchmark>=3.4.0',
]

dev_tools = [
    'black>=22.0.0',
    'flake8>=5.0.0',
    'mypy>=0.991',
    'pre-commit>=2.20.0',
    'memory-profiler>=0.60.0',
]

documentation = [
    'sphinx>=4.0.0',
    'sphinx-rtd-theme>=0.5.0',
    'sphinx-autodoc-typehints>=1.18.0',
    'myst-parser>=0.18.0',
]

# Optional quantum frameworks (commented out by default)
quantum_frameworks = [
    # 'qiskit>=0.39.0',  # IBM Quantum integration
    # 'cirq>=0.14.0',    # Google Quantum integration
    # 'networkx>=2.6.0', # For quantum circuit graph analysis
]

setup(
    name="sentiflow",
    version=read_version(),
    description="Experimental AGI Research Framework for Modeling Emergent Agency",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="TaoishTechy",
    author_email="",  # Add contact email if desired
    url="https://github.com/TaoishTechy/SentiFlow",
    project_urls={
        "Bug Tracker": "https://github.com/TaoishTechy/SentiFlow/issues",
        "Documentation": "https://github.com/TaoishTechy/SentiFlow#readme",
        "Source Code": "https://github.com/TaoishTechy/SentiFlow",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src", include=['qnvm*', 'sentiflow*', 'quantum_core_nexus*']),
    include_package_data=True,
    install_requires=quantum_core,  # Core quantum dependencies only
    
    # Enhanced extras_require with logical grouping
    extras_require={
        'core': quantum_core,
        'visualization': visualization,
        'cli': cli,
        'data': data_processing,
        'networking': networking,
        'testing': testing,
        'dev': dev_tools,
        'docs': documentation,
        'quantum-extras': quantum_frameworks,
        
        # Composite extras
        'full': quantum_core + visualization + cli + data_processing + networking,
        'complete': quantum_core + visualization + cli + data_processing + 
                   networking + testing + dev_tools + documentation,
        'minimal': ['numpy>=1.21.0', 'scipy>=1.7.0'],
    },
    
    # Console scripts
    entry_points={
        'console_scripts': [
            'sentiflow=qnvm.cli_main:main',
            'qnvm=qnvm.cli_main:main',
            'sentiflow-test=qnvm.cli_demos:main',
            'sentiflow-benchmark=examples.qubit_test_32:main',
        ],
    },
    
    # Package classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: System :: Emulators",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    python_requires=">=3.8",
    keywords="AGI, quantum computing, simulation, emergent agency, "
             "cognitive architecture, quantum mechanics, artificial intelligence",
    license="MIT",
    
    # Additional metadata
    platforms=["Linux", "Mac OS-X", "Windows"],
    zip_safe=False,
)
