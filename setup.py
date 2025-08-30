from setuptools import setup, find_packages

setup(
    name="pycnn",
    version="0.1.2",
    description="A Python from-scratch CNN framework",
    author="77AXEL",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pillow",
        "scipy",
        "matplotlib"
    ],
    python_requires=">=3.6",
    include_package_data=True,
)





