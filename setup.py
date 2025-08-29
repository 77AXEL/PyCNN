from setuptools import setup, find_packages

setup(
    name="cnnfs",
    version="0.1.1",
    description="A Python from-scratch CNN framework",
    author="77AXEL",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pillow",
        "scipy",
        "matplotlib",
        "cupy"
    ],
    python_requires=">=3.6",
    include_package_data=True,
)


