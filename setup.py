import os
import sys
from pathlib import Path
from typing import List

from setuptools import setup
from setuptools import Extension
from setuptools.command.build import build
from setuptools.command.build_ext import build_ext


def get_build_type(is_debug: bool) -> str:
    """Determine build type based on debug flag."""
    return "Debug" if is_debug else "Release"


class _BasicExtension(Extension):
    """Base extension class handling path resolution."""

    def __init__(self, file_name: str, src_path: str, dst_path: str):
        super().__init__(name=file_name, sources=[])
        self.file_name = file_name
        self.src_path = src_path
        self.dst_path = dst_path

    def get_src_path(self, installer: "InstallBuiltExt") -> Path:
        """Resolve the source path using build directory from build command."""
        build_cmd = installer.get_finalized_command("build")
        build_base = Path(build_cmd.build_base)
        repo_root = Path.cwd()

        # Construct CMake build directory path
        cmake_cache_dir = repo_root / build_base / "cmake-out"

        build_type = get_build_type(installer.debug)
        src_pattern = (
            self.src_path.replace("%BUILD_TYPE%", build_type)
            if os.name == "nt"
            else self.src_path.replace("/%BUILD_TYPE%", "")
        )

        # Find source file in the CMake build directory
        src_files = list(cmake_cache_dir.glob(src_pattern))
        if len(src_files) != 1:
            raise ValueError(
                f"Expected one file matching '{src_pattern}'; found {src_files} in dir: {cmake_cache_dir}"
            )
        return src_files[0]


class BuiltExtension(_BasicExtension):
    """Extension for pre-built binaries via CMake."""

    def __init__(self, src: str, module_path: str):
        super().__init__(file_name=module_path, src_path=src, dst_path=module_path)

    def get_dst_path(self, installer: "InstallBuiltExt") -> Path:
        """Get destination path for the built extension."""
        return Path(installer.get_ext_fullpath(self.dst_path))


class InstallBuiltExt(build_ext):
    """Installs binaries built by CMake."""

    def build_extension(self, ext: _BasicExtension) -> None:
        src_path = ext.get_src_path(self)
        dst_path = ext.get_dst_path(self)

        self.mkpath(str(dst_path.parent))
        self.copy_file(str(src_path), str(dst_path))


class CustomBuild(build):
    """Custom build command to handle CMake compilation."""

    user_options = build.user_options + [
        ("cmake-args=", None, "Additional CMake arguments"),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.build_base = "py-build-out"
        self.cmake_args = ""
        self.parallel = max(1, int(os.cpu_count() / 2))  # Ensure at least 1 job

    def finalize_options(self):
        super().finalize_options()

    def run(self):
        """Execute CMake build."""
        build_type = get_build_type(self.debug)
        repo_root = os.path.abspath(os.getcwd())

        # Configure CMake arguments
        cmake_args = [
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            "-DMLLM_ENABLE_PY_MLLM=ON",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
            "-DHWY_ENABLE_TESTS=OFF",
            "-DHWY_ENABLE_EXAMPLES=OFF",
            "-DHWY_ENABLE_CONTRIB=OFF",
            '-DMLLM_X86_BACKEND_COMPILE_OPTIONS="-march=native"',
        ] + [arg for arg in self.cmake_args.split(" ") if arg]

        # Build arguments
        build_args = ["--config", build_type, f"-j{self.parallel}", "--target", "_C"]

        # Create and configure build directory
        cmake_cache_dir = os.path.join(self.build_base, "cmake-out")
        self.mkpath(cmake_cache_dir)

        # Run CMake configuration
        self.spawn(["cmake", "-S", repo_root, "-B", cmake_cache_dir] + cmake_args)
        # Compile the project
        self.spawn(["cmake", "--build", cmake_cache_dir] + build_args)

        self.spawn(["ls", f"{repo_root}/{cmake_cache_dir}"])

        super().run()


def get_ext_modules() -> List[Extension]:
    """Collect extension modules."""
    return [
        BuiltExtension(src="_C.*", module_path="pymllm._C"),
    ]


setup(
    version="0.0.1",
    package_dir={
        "pymllm": "pymllm",
        "pymllm._C": "pymllm/_C",
    },
    cmdclass={
        "build": CustomBuild,
        "build_ext": InstallBuiltExt,
    },
    ext_modules=get_ext_modules(),
)
