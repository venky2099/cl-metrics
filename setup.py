from setuptools import setup, find_packages

setup(
    name="cl-metrics",
    version="0.1.0",
    author="Venkatesh Swaminathan",
    author_email="venkateshswaminathaniyer@gmail.com",
    description=(
        "A stateless, architecture-agnostic Python library for "
        "Continual Learning and Class-Incremental Learning evaluation metrics, "
        "including SNN energy-aware metrics."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/venky2099/cl-metrics",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["numpy>=1.21.0"],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    keywords=[
        "continual learning", "class-incremental learning",
        "spiking neural networks", "neuromorphic", "evaluation metrics",
        "catastrophic forgetting", "backward transfer"
    ],
)
