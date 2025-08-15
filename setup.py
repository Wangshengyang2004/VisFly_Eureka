#!/usr/bin/env python3
"""
Setup script for VisFly-Eureka Integration System
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# Read requirements
def read_requirements():
    requirements = []
    with open("requirements.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements

setup(
    name="eureka-visfly",
    version="1.0.0",
    author="Claude Code & Simon Wang",
    author_email="wangshengyang2004@gmail.com",
    description="VisFly-Eureka Integration: LLM-powered reward optimization for autonomous drone navigation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Wangshengyang2004/VisFly_Eureka",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "eureka-visfly=eureka_visfly.main:main",
            "visfly-train=eureka_visfly.visfly_training_wrapper:main",
        ],
    },
    include_package_data=True,
    package_data={
        "eureka_visfly": [
            "configs/*.yaml",
            "configs/*/*.yaml",
        ],
    },
    keywords=[
        "reinforcement learning",
        "drone navigation", 
        "reward optimization",
        "LLM",
        "autonomous systems",
        "eureka",
        "visfly",
        "simulation"
    ],
    project_urls={
        "Bug Reports": "https://github.com/Wangshengyang2004/VisFly_Eureka/issues",
        "Source": "https://github.com/Wangshengyang2004/VisFly_Eureka",
        "Documentation": "https://github.com/Wangshengyang2004/VisFly_Eureka/blob/main/README.md",
    },
)