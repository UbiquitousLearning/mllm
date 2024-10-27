"""
The setup.py to install mllm package.
"""
import os
import sys
from pathlib import Path
from typing import List

from setuptools import setup
from setuptools import Extension
from setuptools.command.build import build
from setuptools.command.build_ext import build_ext


def get_build_type(is_debug: bool) -> bool:
    """to check debug options and get build types"""
    cfg = "Debug" if is_debug else "Release"
    return cfg


class _BasicExtention(Extension):
    """
    An override Extention, to recored where to copy files and store files
    """

    def __init__(self, file_name: str, src_path: str, dst_path: str):
        self.file_name = file_name
        self.src_path = src_path
        self.dst_path = dst_path

        super().__init__(name=self.file_name, sources=[])

    def get_src_path(self, installer: "InstallBuiltExt") -> Path:
        """resolve the path to the src file"""
        cmake_cache_dir = Path(
            installer.get_finalized_command("build").cmake_cache_dir)
        build_type = get_build_type(installer.debug)

        if os.name == "nt":
            self.src_path = self.src_path.replace("%BUILD_TYPE%", build_type)
        else:
            self.src_path = self.src_path.replace("/%BUILD_TYPE%", "")

        srcs = tuple(cmake_cache_dir.glob(self.src_path))
        if len(srcs) != 1:
            raise ValueError(
                f"Expected one file matching '{
                    self.src_path}'; found {repr(srcs)}"
            )
        return srcs[0]


class BuiltExtention(_BasicExtention):
    """install python bindings that was built by cmake"""

    def __init__(self, src: str, module_path: str):
        super().__init__(file_name=module_path,
                         src_path=src,
                         dst_path=module_path)

    def get_dst_path(self, installer: "InstallBuiltExt") -> Path:
        """return detination path"""
        return Path(installer.get_ext_fullpath(self.dst_path))


class InstallBuiltExt(build_ext):
    """
    install binary files that were built by cmake
    """

    def build_extension(self, ext: _BasicExtention) -> None:
        src_file: Path = ext.get_src_path(self)
        dst_file: Path = ext.get_dst_path(self)

        self.mkpath(os.fspath(dst_file.parent))

        # copy files to it
        self.copy_file(os.fspath(src_file), os.fspath(dst_file))


class CustomBuild(build):
    """
    customized builder to build cmake target
    """

    def initialize_options(self):
        """override initilizer"""
        super().initialize_options()
        self.build_base = "py-build-out"
        self.parallel = int(os.cpu_count() / 2)

    def run(self):
        """override runner"""
        self.dump_options()

        build_type = get_build_type(self.debug)
        repo_root = os.fspath(Path.cwd())

        cmake_args = [
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={build_type}",

            # add edgeinfer flag to build pybindings.
            "-DMLLM_ENABLE_PYTHON=ON",
            "-D_GLIBCXX_USE_CXX11_ABI=1"
        ]

        build_args = [f"-j{self.parallel}"]

        # add other targets
        build_args += ["--target", "_C"]

        if "CMAKE_BUILD_ARGS" in os.environ:
            build_args += [
                item for item in os.environ["CMAKE_BUILD_ARGS"].split(" ") if item  # noqa: E501
            ]

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [
                item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # prepare cmake cache dir
        cmake_cache_dir = os.path.join(
            repo_root, self.build_base, "cmake-out")
        self.mkpath(cmake_cache_dir)

        if not self.dry_run:
            (Path(cmake_cache_dir) / "CMakeCache.txt").unlink(missing_ok=True)
            self.spawn(["cmake", "-S", repo_root, "-B",
                       cmake_cache_dir, *cmake_args])

        # build what we need
        self.spawn(["cmake", "--build", cmake_cache_dir, *build_args])

        # place to store none-python files
        data_root = os.path.join(self.build_lib, "mllm", "data")

        # place to store binary files under ./data
        bin_dir = os.path.join(data_root, "bin")

        self.mkpath(bin_dir)
        self.cmake_cache_dir = cmake_cache_dir

        build.run(self)


def get_ext_modules() -> List[Extension]:
    """the set of modules that we need"""
    ext_modules = []

    # _C lib
    ext_modules.append(
        BuiltExtention(
            "_C.*", "mllm._C"
        )
    )
    return ext_modules


setup(
    version="0.0.1",
    package_dir={
        "mllm": "python/top",
        "mllm/_C": "python/src/_C",
    },
    cmdclass={
        "build": CustomBuild,
        "build_ext": InstallBuiltExt,
    },
    ext_modules=get_ext_modules()
)
