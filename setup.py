import os
import subprocess
import platform
import numpy
from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py

class BuildLib(build_py):
    def run(self):
        lib_dir = os.path.join(os.getcwd(), 'pycnn', 'lib')
        
        make_cmd = "mingw32-make" if platform.system() == "Windows" else "make"
        
        print(f"--- Building optimized native library in {lib_dir} using {make_cmd} ---")
        
        try:
            subprocess.check_call([make_cmd], cwd=lib_dir, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: Native build failed. Ensure {make_cmd} is in your PATH.")
            raise e
        
        super().run()

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
    cmdclass={
        'build_py': BuildLib,
    },
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