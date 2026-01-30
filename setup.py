"""
Setup script for GradientDetachment package.
"""

from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read core requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() 
        for line in fh 
        if line.strip() and not line.startswith("#")
    ]

# Optional dependencies
extras_require = {
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.10.0',
        'pytest-xdist>=2.0.0',
        'black>=21.0',
        'flake8>=3.8.0',
        'mypy>=0.900',
        'isort>=5.0.0',
    ],
    'notebooks': [
        'jupyter>=1.0.0',
        'notebook>=6.0.0',
        'ipython>=7.0.0',
        'ipykernel>=5.0.0',
        'jupytext>=1.11.0',
    ],
    'docs': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0',
        'sphinx-autodoc-typehints>=1.11.0',
    ],
    'analysis': [
        'pandas>=1.0.0',
        'seaborn>=0.11.0',
        'plotly>=4.0.0',
    ]
}

# Add 'all' option that includes everything
extras_require['all'] = list(set(sum(extras_require.values(), [])))

setup(
    name="gradientdetachment",
    version="1.0.0",
    author="Trent Pierce",
    author_email="Pierce.trent@gmail.com",
    description="Neural ODE Cryptanalysis Framework - Demonstrates ARX cipher resistance to ML attacks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TrentPierce/gradientdetachment",
    project_urls={
        "Bug Tracker": "https://github.com/TrentPierce/gradientdetachment/issues",
        "Documentation": "https://github.com/TrentPierce/gradientdetachment/tree/main/docs",
        "Source Code": "https://github.com/TrentPierce/gradientdetachment",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Security :: Cryptography",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords="cryptography, neural-ode, gradient-descent, arx-ciphers, machine-learning, cryptanalysis, speck, security",
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "gradientdetachment-verify=reproduce_sawtooth:main",
            "gradientdetachment-diagnose=diagnose_inversion:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
