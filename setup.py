from distutils.core import setup
from distutils.extension import Extension

import numpy as np

ext = [Extension("polymesh",
                sources=["polymesh/polymesh.c"],
                libraries=[],
                language="c", 
                include_dirs = [np.get_include()],
                extra_compile_args=["-Wno-unreachable-code"])]


setup(name="polymesh",
	  version="0.1",
	  description="polygonal mesh handling class, written in cython",
	  author="Jarle A. Kramer",
	  author_email="jarlekramer@gmail.com",
	  license="MIT",
	  packages=["polymesh"],
	  install_requires=["numpy",],
	  include_package_data=True,
	  ext_modules=ext,
)