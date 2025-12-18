from setuptools import setup, find_packages

setup(
    name="quantumneurovm",
    version="5.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "networkx>=3.1",
        "psutil>=5.9.0",
        # Optional: qiskit, cirq for backend integration
    ],
    python_requires=">=3.8",
)