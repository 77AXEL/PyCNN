from setuptools import setup, find_packages, Extension
import platform
import numpy

try:
    from Cython.Build import cythonize
except ImportError:
    def cythonize(extensions, **_ignore):
        return extensions

extra_args = ["-O3", "-march=native", "-ffast-math", "-funroll-loops"]

module_names = [
    "backward_pass", 
    "forward_pass", 
    "gradient_decent", 
    "max_pooling"
]

extensions = [
    Extension(
        f"pycnn.modules.{name}",
        [f"pycnn/modules/{name}.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_args,
        extra_link_args=extra_args if platform.system() != "Windows" else []
    ) for name in module_names
]

setup(
    name="pycnn",
    version="2.4",
    description="A Python library to easily build, train, and test your CNN AI models.",
    author="https://github.com/77axel",
    packages=find_packages(),
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    install_requires=[
        "numpy",
        "pillow",
        "scipy",
        "matplotlib",
        "Cython"
    ],
    python_requires=">=3.6",
    include_package_data=True,
    zip_safe=False,
)