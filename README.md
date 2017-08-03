# polymesh
A simple mesh handling library, written in Cython for speed. It can import mesh from .obj files and perform simple operations on it such as scaling, rotation and translation. The main purpose is to act as a mesh data structure for other applications. Export options are .obj, .stl and .vtk. Data can be added to faces and vertices, and exported along with the geometry information using the .vtk fileformat. This is useful for visualization (i.e. show pressure on the surface of the geometry). 
