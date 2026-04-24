"""Setup script for chest-xray-benchmark package."""

from setuptools import setup, find_packages

setup(
    name="chest-xray-benchmark",
    version="1.0.0",
    description=(
        "A Unified Benchmarking Multi-Disease Detection Framework for "
        "Thoracic Abnormalities Using Chest X-ray Images"
    ),
    author="Sister Nivedita University, Kolkata",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.2",
        "torchvision>=0.17",
        "timm>=0.9",
        "scikit-learn>=1.4",
        "xgboost>=2.0",
        "pandas>=2.2",
        "numpy>=1.26",
        "opencv-python-headless>=4.9",
        "Pillow>=10.2",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "fastapi>=0.110",
        "uvicorn>=0.28",
        "tqdm>=4.66",
        "PyYAML>=6.0",
        "joblib>=1.3",
        "optuna>=3.5",
        "onnx>=1.15",
        "onnxruntime>=1.17",
        "scipy>=1.12",
    ],
    extras_require={
        "dev": ["pytest>=8.0", "black>=24.2", "ruff>=0.3"],
    },
)
