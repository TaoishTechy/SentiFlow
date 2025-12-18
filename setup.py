# setup.py
from setuptools import setup, find_packages
import os

def read_requirements():
    """Read requirements from requirements.txt"""
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def read_version():
    """Read version from __init__.py"""
    with open(os.path.join('src', 'quantum_core_nexus', '__init__.py'), 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    return '1.0.0'

setup(
    name="quantum_core_nexus",
    version=read_version(),
    description="Scientific-Grade Quantum Simulation Platform",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="QuantumCore Nexus Team",
    author_email="",
    url="https://github.com/TaoishTechy/QuantumCore-Nexus",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=read_requirements(),
    extras_require={
        'full': [
            'numpy>=1.21.0',
            'scipy>=1.7.0',
            'matplotlib>=3.4.0',
            'pyyaml>=6.0',
            'requests>=2.26.0',
            'psutil>=5.8.0',
            'tqdm>=4.62.0',
            'pandas>=1.3.0',
        ],
        'minimal': [
            'numpy>=1.21.0',
            'scipy>=1.7.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'quantumcore-nexus=quantum_core_nexus.cli:main',
            'qcn=quantum_core_nexus.cli:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    keywords="quantum computing, simulation, qubit, qudit, quantum mechanics",
)