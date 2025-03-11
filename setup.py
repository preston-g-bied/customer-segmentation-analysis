from setuptools import find_packages, setup

setup(
    name="online_retail_analysis",
    version="0.1.0",
    description="Analysis of online retail transaction data",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "dash>=2.0.0",
        "flask>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
            "isort>=5.9.0",
            "pre-commit>=2.13.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "preprocess=online_retail_analysis.data.preprocessing:main",
            "visualize=online_retail_analysis.visualization.visualize:main",
        ],
    },
)