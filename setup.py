from setuptools import setup, find_packages

setup(
    name="cnnfs",
    version="0.1",
    description="A from-scratch CNN built with NumPy, SciPy, and Pillow",
    author="77AXEL",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pillow",
        "scipy"
    ],
    python_requires=">=3.6",
    include_package_data=True,
)
