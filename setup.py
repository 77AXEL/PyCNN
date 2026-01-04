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
from setuptools.command.build_ext import build_ext

PREBUILT_DOWNLOADED = False

GITHUB_REPO = "77AXEL/PyCNN"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

def get_platform_info():
    """Get current platform and Python version information"""
    system = platform.system()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
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

def log(msg):
    """Log to stderr to ensure visibility in pip install -v"""
    print(msg, file=sys.stderr)
    sys.stderr.flush()

def ensure_binaries_ready():
    """Ensure binaries are present either by downloading or building from source"""
    global PREBUILT_DOWNLOADED
    if PREBUILT_DOWNLOADED:
        return True
        
    os_name, py_version, system = get_platform_info()
    lib_dir = Path('pycnn/lib').absolute()
    modules_dir = Path('pycnn/modules').absolute()
    lib_exts = get_library_extensions(system)
    
    module_names = ["backward_pass", "forward_pass", "gradient_decent", "max_pooling"]
    lib_exists = any(list(lib_dir.glob(f"optimized*{ext}")) for ext in lib_exts)
    modules_exist = all(any(modules_dir.glob(f"{name}.pyd")) or any(modules_dir.glob(f"{name}.so")) for name in module_names)
    
    if lib_exists and modules_exist:
        log(f"[PyCNN] Compiled binaries already found. Skipping preparation.")
        PREBUILT_DOWNLOADED = True
        return True

    if os.environ.get('GITHUB_ACTIONS'):
        log("[PyCNN] CI Environment detected. Building native library from source...")
        if not lib_exists:
            make_cmd = "make"
            try:
                lib_dir.mkdir(parents=True, exist_ok=True)
                subprocess.check_call([make_cmd], cwd=lib_dir, shell=True)
                if not any(list(lib_dir.glob(f"optimized*{ext}")) for ext in lib_exts):
                    raise RuntimeError("Native build finished but binary not found.")
                log("[PyCNN] Native library built successfully.")
                return True
            except Exception as e:
                log(f"[PyCNN] Critical Error: Native build failed in CI: {e}")
                raise e
        return False

    py_tag = f"cp{py_version.replace('.', '')}"
    current_version = "2.5"
    
    log(f"\n[PyCNN] Checking GitHub for pre-built binaries (Python {py_version}, {system})...")
    
    try:
        req = urllib.request.Request(GITHUB_API_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            release_data = json.loads(response.read().decode())
        
        assets = release_data.get('assets', [])
        tag_name = release_data.get('tag_name', 'unknown')
        
        log(f"[PyCNN] Latest release found: {tag_name}")
        
        if current_version not in tag_name:
            log(f"[PyCNN] Warning: Latest release {tag_name} does not match current version {current_version}")
        
        lib_dir.mkdir(parents=True, exist_ok=True)
        modules_dir.mkdir(parents=True, exist_ok=True)
        
        platform_keywords = {'Linux': 'linux', 'Darwin': 'macos', 'Windows': 'win'}
        kw = platform_keywords.get(system)
        
        matching_wheel = None
        for asset in assets:
            name = asset['name']
            if name.endswith('.whl') and py_tag in name and kw in name.lower():
                matching_wheel = asset
                break
        
        if not matching_wheel:
            log(f"[PyCNN] No matching pre-built wheel found in {tag_name}.")
            return False
        
        whl_path = Path('temp_wheel.whl').absolute()
        log(f"[PyCNN] Downloading from {matching_wheel['browser_download_url']}...")
        urllib.request.urlretrieve(matching_wheel['browser_download_url'], whl_path)
        
        with zipfile.ZipFile(whl_path, 'r') as whl_zip:
            extract_count = 0
            for file in whl_zip.namelist():
                if (file.endswith('.pyd') or file.endswith('.so')) and 'modules/' in file:
                    log(f"  -> Extracting module: {file}")
                    whl_zip.extract(file, '.')
                    extract_count += 1
                if any(file.endswith(ext) for ext in lib_exts) and 'lib/' in file:
                    log(f"  -> Extracting library: {file}")
                    whl_zip.extract(file, '.')
                    extract_count += 1
        
        os.remove(whl_path)
        if extract_count > 0:
            log(f"[PyCNN] Successfully installed {extract_count} pre-built components!\n")
            PREBUILT_DOWNLOADED = True
            return True
        else:
            log("[PyCNN] Warning: Matching wheel found but no binaries were extracted and they might be in a different path inside the wheel.")
            return False
        
    except Exception as e:
        log(f"[PyCNN] Binary download skipped: {e}")
        return False

class BuildLib(build_py):
    def run(self):
        ensure_binaries_ready()
        super().run()

class InstallWithBinaries(install):
    def run(self):
        ensure_binaries_ready()
        super().run()

class BuildExtMaybe(build_ext):
    def run(self):
        if PREBUILT_DOWNLOADED:
            log("[PyCNN] Skipping compilation: Pre-built binaries are in place.")
            return
            
        log("[PyCNN] Standard build: Compiling modules from source...")
        
        lib_dir = Path('pycnn/lib')
        system = platform.system()
        lib_exts = get_library_extensions(system)
        lib_exists = any(Path(lib_dir).glob(f"optimized*{ext}") for ext in lib_exts)
        
        if not lib_exists:
            make_cmd = "mingw32-make" if system == "Windows" else "make"
            log(f"--- Building optimized native library using {make_cmd} ---")
            try:
                subprocess.check_call([make_cmd], cwd=lib_dir, shell=True)
                lib_exists_now = any(list(lib_dir.glob(f"optimized*{ext}")) for ext in lib_exts)
                if not lib_exists_now:
                    raise RuntimeError(f"Native build finished but optimized binary was not found in {lib_dir}")
            except Exception as e:
                log(f"Error: Native build failed: {e}")
                if os.environ.get('GITHUB_ACTIONS'):
                    raise e
        
        super().run()

ensure_binaries_ready()

try:
    from Cython.Build import cythonize
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    def cythonize(extensions, **_ignore):
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

extensions = []
if CYTHON_AVAILABLE and not PREBUILT_DOWNLOADED:
    log("[PyCNN] Cython available and no pre-built binaries found. Preparing source extensions...")
    extensions = [
        Extension(
            f"pycnn.modules.{name}",
            [f"pycnn/modules/{name}.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=extra_args,
            extra_link_args=extra_args if platform.system() != "Windows" else []
        ) for name in module_names
    ]
elif PREBUILT_DOWNLOADED:
    log("[PyCNN] Using pre-built binaries. Python-level extensions will be skipped.")

setup(
    name="pycnn",
    version="2.5",
    description="A Python library to easily build, train, and test your CNN AI models.",
    author="https://github.com/77axel",
    packages=find_packages(),
    cmdclass={
        'build_py': BuildLib,
        'install': InstallWithBinaries,
        'build_ext': BuildExtMaybe,
    },
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}) if CYTHON_AVAILABLE else [],
    install_requires=[
        "numpy",
        "pillow",
        "scipy",
        "matplotlib",
    ],
    python_requires=">=3.8",
    package_data={
        'pycnn.lib': ['*.dll', '*.so', '*.dylib'],
        'pycnn.modules': ['*.pyd', '*.so'],
    },
    include_package_data=True,
    zip_safe=False,
)