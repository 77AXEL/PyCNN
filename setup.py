from setuptools import setup, find_packages

setup(
    name="pycnn",
    version="2.0",
    description="A Python CNN framework",
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






