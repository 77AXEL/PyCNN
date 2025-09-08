from setuptools import setup, find_packages

setup(
    name="pycnn",
    version="2.1",
    description="Python CNNs framework",
    author="77AXEL",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pillow",
        "scipy",
        "matplotlib",
        "numba"
    ],
    python_requires=">=3.6",
    include_package_data=True,
)







