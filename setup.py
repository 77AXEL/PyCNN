import os
import sys
import subprocess
import platform
import urllib.request
import json
import zipfile
import shutil
from pathlib import Path
import numpy
from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py
from setuptools.command.install import install

# GitHub repository information
GITHUB_REPO = "77axel/pycnn"  # Update with your actual repo
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

def get_platform_info():
    """Get current platform and Python version information"""
    system = platform.system()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    # Map platform names to GitHub Actions runner names
    platform_map = {
        'Linux': 'ubuntu-latest',
        'Darwin': 'macos-latest',
        'Windows': 'windows-latest'
    }
    
    os_name = platform_map.get(system)
    if not os_name:
        raise RuntimeError(f"Unsupported platform: {system}")
    
    return os_name, python_version, system

def get_library_extensions(system):
    """Get the library file extensions for the current platform"""
    ext_map = {
        'Linux': ['.so'],
        'Darwin': ['.dylib'],
        'Windows': ['.dll']
    }
    return ext_map.get(system, [])

def download_prebuilt_binaries():
    """Download pre-built binaries from GitHub releases"""
    os_name, py_version, system = get_platform_info()
    
    # Python tag format: 3.9 -> cp39
    py_tag = f"cp{py_version.replace('.', '')}"
    
    print(f"Detected platform: {os_name}, Python: {py_version} ({py_tag})")
    print(f"Attempting to download pre-built binaries from GitHub releases...")
    
    try:
        # Get latest release information
        # Use a User-Agent to avoid some basic blocks
        req = urllib.request.Request(GITHUB_API_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            release_data = json.loads(response.read().decode())
        
        assets = release_data.get('assets', [])
        tag_name = release_data.get('tag_name', 'unknown')
        
        print(f"Found release: {tag_name}")
        
        # Prepare directories
        lib_dir = Path('pycnn/lib')
        modules_dir = Path('pycnn/modules')
        lib_dir.mkdir(parents=True, exist_ok=True)
        modules_dir.mkdir(parents=True, exist_ok=True)
        
        # We look for the wheel that matches:
        # 1. Python version tag (e.g. cp39)
        # 2. Platform tag (e.g. win_amd64, manylinux, macosx)
        
        platform_keywords = {
            'Linux': 'linux',
            'Darwin': 'macos',
            'Windows': 'win'
        }
        kw = platform_keywords.get(system)
        
        matching_wheel = None
        for asset in assets:
            name = asset['name']
            if name.endswith('.whl') and py_tag in name and kw in name.lower():
                matching_wheel = asset
                break
        
        if not matching_wheel:
            print(f"Warning: No matching wheel found for {system} and Python {py_version}")
            return False
        
        print(f"Downloading wheel: {matching_wheel['name']}")
        whl_path = Path('temp_wheel.whl')
        urllib.request.urlretrieve(matching_wheel['browser_download_url'], whl_path)
        
        print("Extracting compiled modules from wheel...")
        with zipfile.ZipFile(whl_path, 'r') as whl_zip:
            for file in whl_zip.namelist():
                # Extract modules (*.pyd, *.so)
                if file.startswith('pycnn/modules/') and (file.endswith('.pyd') or file.endswith('.so')):
                    print(f"  Extracting: {file}")
                    whl_zip.extract(file, '.')
                
                # Extract native libs (*.dll, *.so, *.dylib)
                if file.startswith('pycnn/lib/') and any(file.endswith(ext) for ext in get_library_extensions(system)):
                    print(f"  Extracting: {file}")
                    whl_zip.extract(file, '.')
        
        os.remove(whl_path)
        print("Successfully downloaded and extracted pre-built binaries!")
        return True
        
    except Exception as e:
        print(f"Failed to download pre-built binaries: {e}")
        return False

class BuildLib(build_py):
    def run(self):
        # Try to download pre-built binaries first
        if download_prebuilt_binaries():
            print("Using pre-built binaries - skipping compilation")
            super().run()
            return
        
        # Fallback to building from source
        print("Building from source...")
        lib_dir = os.path.join(os.getcwd(), 'pycnn', 'lib')
        
        make_cmd = "mingw32-make" if platform.system() == "Windows" else "make"
        
        print(f"--- Building optimized native library in {lib_dir} using {make_cmd} ---")
        
        try:
            subprocess.check_call([make_cmd], cwd=lib_dir, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: Native build failed. Ensure {make_cmd} is in your PATH.")
            raise e
        
        super().run()

class InstallWithBinaries(install):
    def run(self):
        # Download binaries before installation
        download_prebuilt_binaries()
        super().run()

# Try to import Cython, but don't require it if binaries are available
try:
    from Cython.Build import cythonize
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    def cythonize(extensions, **_ignore):
        # Return empty list if Cython not available and we're using pre-built binaries
        return []

if os.environ.get('GITHUB_ACTIONS'):
    extra_args = ["-O3", "-funroll-loops"]
else:
    extra_args = ["-O3", "-march=native", "-ffast-math", "-funroll-loops"]

module_names = [
    "backward_pass", 
    "forward_pass", 
    "gradient_decent", 
    "max_pooling"
]

# Only define extensions if Cython is available (for source builds)
extensions = []
if CYTHON_AVAILABLE:
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
        'install': InstallWithBinaries,
    },
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}) if CYTHON_AVAILABLE else [],
    install_requires=[
        "numpy",
        "pillow",
        "scipy",
        "matplotlib",
    ],
    python_requires=">=3.6",
    include_package_data=True,
    zip_safe=False,
)