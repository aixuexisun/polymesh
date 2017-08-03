from distutils.core import setup
from distutils.extension import Extension

import numpy as np

meshModule = Extension("polymesh.mesh",
             sources=["polymesh/mesh.c"],
             include_dirs = [np.get_include()],
             extra_compile_args=["-Wno-unreachable-code"])

hydrostaticModule = Extension("polymesh.hydrostatic",
					sources=["polymesh/hydrostatic.c"],
					include_dirs = [np.get_include()],
					extra_compile_args=["-Wno-unreachable-code"])

setup(name="polymesh",
	  version="0.1",
	  description="A library for setting up simulations in OpenFOAM",
	  author="Jarle A. Kramer",
	  author_email="jarlekramer@gmail.com",
	  license="MIT",
	  packages=["polymesh"],
	  install_requires=["numpy",],
	  python_requires=">=3",
	  include_package_data=True,
	  ext_modules=[meshModule, hydrostaticModule],
)