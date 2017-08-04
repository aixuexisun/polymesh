# polymesh

This is a python package consisting of two main modules, described below. Both are written in cython, in order to speed the computation. The cython generated c-code is distributed with the source code, so you can install the package without having cython yourself. 

## polymesh.mesh
A simple mesh handling library. It can import mesh from .obj files and perform simple operations on it such as scaling, rotation and translation. The main purpose is to act as a mesh data structure for other applications. Export options are .obj, .stl and .vtk. Data can be added to faces and vertices, and exported along with the geometry information using the .vtk file format. This is useful for visualization (i.e. show pressure on the surface of the geometry). 

## polymesh.hydrostatic
A library of functions that can perform "hydrostatic" calculations on mesh objects. This includes extracting the part of the mesh that is below a specified plane, calculate volume, surface area, volume centroid and dimensions

## install instructions
It can be installed in the same way as most other python packages, by executing the following in the downloaded folder:
```
python setup.py install
```
It only works for python 3, so maybe you need to use ```python3``` rather that ```python```, depending on your setup.

## Usage
A simple example of how to use both modules are published as a jupyter notebook in the "Examples" folder. 
