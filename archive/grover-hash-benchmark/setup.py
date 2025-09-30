from setuptools import setup, find_packages

setup(
    name="quantum-hash-benchmark",
    version="0.1.0",
    author="Samarth Bhadane",
    author_email="samarthbhadane23@gmail.com",
    description="A benchmarking framework for cryptographic hash functions with quantum resistance analysis.",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "numpy",
        "matplotlib",
        "tabulate",
        "blake3",
        "psutil"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)