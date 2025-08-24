from setuptools import setup, find_packages

setup(
    name="phase4",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "gym>=0.17.0",
        "wandb>=0.12.0",
        "d4rl>=1.1"
    ]
) 