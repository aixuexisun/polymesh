#!python
#cython: language_level=3, boundscheck=False, wraparound=False

import copy

import numpy as np
cimport numpy as np

from libc.stdlib cimport atof
from libc.math cimport sqrt, fabs

cimport polymesh.mesh

cimport cython

cdef class Mesh:
	def __cinit__(self, verts, face_verts, face_nrVerts, simple=False):
		''' Create a mesh object from verts data, and vert-face connectivity'''
		cdef int i, j
		# Vertices data
		self.nrVerts = len(verts)
		self.verts = verts

		# Face data
		self.nrFaces         = len(face_nrVerts)
		self.face_verts      = face_verts.astype(np.int)
		self.face_nrVerts    = face_nrVerts.astype(np.int)

		self.face_startIndex = np.zeros(self.nrFaces, dtype=int)
		for i in range(1, self.nrFaces):
			self.face_startIndex[i] = self.face_startIndex[i-1] + self.face_nrVerts[i-1]

		if not simple:
			self.updateMeshData()

		# Custom data
		self.face_dataNames = []
		self.vert_dataNames = []

	# --------- Public accessible variables -------------------------------
	property nrVerts:
		def __get__(self):
			return self.nrVerts
	property verts:
		def __get__(self):
			return np.asarray(self.verts)
		def __set__(self, np.ndarray[dtype=np.float_t, ndim=2] vert_inn):
			self.verts = vert_inn
	property vert_n:
		def __get__(self):
			return np.asarray(self.vert_n)
	property vert_dataNames:
		def __get__(self):
			return self.vert_dataNames
	property vert_data:
		def __get__(self):
			return np.asarray([self.vert_data])

	property nrFaces:
		def __get__(self):
			return self.nrFaces	
	property face_verts:
		def __get__(self):
			return np.asarray(self.face_verts)
	property face_nrVerts:
		def __get__(self):
			return np.asarray(self.face_nrVerts)
	property face_startIndex:
		def __get__(self):
			return np.asarray(self.face_startIndex)
	property face_edges:
		def __get__(self):
			return np.asarray(self.face_edges)
	property face_l:
		def __get__(self):
			return np.asarray(self.face_l)
	property face_m:
		def __get__(self):
			return np.asarray(self.face_m)
	property face_n:
		def __get__(self):
			return np.asarray(self.face_n)
	property face_area:
		def __get__(self):
			return np.asarray(self.face_area)
	property face_center:
		def __get__(self):
			return np.asarray(self.face_center)
	property face_dataNames:
		def __get__(self):
			return self.face_dataNames
	property face_data:
		def __get__(self):
			return np.asarray(self.face_data)
	
	property nrEdges:
		def __get__(self):
			return self.nrEdges
	property edge_verts:
		def __get__(self):
			return np.asarray(self.edge_verts)
	property edge_faces:
		def __get__(self):
			return np.asarray(self.edge_faces)
	property edge_nrFaces:
		def __get__(self):
			return np.asarray(self.edge_nrFaces)
	property edge_startIndex:
		def __get__(self):
			return np.asarray(self.edge_startIndex)

	# ---------- Methods --------------------------------------------------
	def deleteVertex(self, deleteVertIndex):
		''' This method deletes a single vertex, based in the given input index '''

		cdef int i, j, startIndex, vertIndex, nrFacesToDelete

		# Delete vertex from vertex list
		self.verts = np.delete(np.asarray(self.verts), deleteVertIndex, axis=0)
		self.nrVerts -= 1

		# Loop through all faces and update vertex indices based on new vertex array. delete faces that reference vertex
		cdef long[:] deleteFaces = np.zeros(self.nrFaces, dtype=np.int)
		nrFacesToDelete = 0
		for i in range(self.nrFaces):
			deleteFace = 0
			startIndex = self.face_startIndex[i]

			for j in range(self.face_nrVerts[i]):
				vertIndex = self.face_verts[startIndex + j]

				if vertIndex > deleteVertIndex:
					self.face_verts[startIndex+j] -= 1

	def calculateFaceData(self):
		''' This method calculates area, normals and geometric center for every face in the mesh '''
		cdef int i, j, startIndex, stopIndex, nrVerts
		cdef double l

		cdef long[:] indices

		cdef double[:] cross = np.zeros(3, dtype=np.double)
		cdef double[:] n     = np.zeros(3, dtype=np.double)
		cdef double[:] v1    = np.zeros(3, dtype=np.double)
		cdef double[:] v2    = np.zeros(3, dtype=np.double)

		self.face_n      = np.zeros((self.nrFaces, 3), dtype=np.double)
		self.face_area   = np.zeros( self.nrFaces,     dtype=np.double)
		self.face_center = np.zeros((self.nrFaces, 3), dtype=np.double)

		for i in range(self.nrFaces):
			cross = np.zeros(3, dtype=np.double)
			v1    = np.zeros(3, dtype=np.double)
			v2    = np.zeros(3, dtype=np.double)

			startIndex = self.face_startIndex[i]
			nrVerts    = self.face_nrVerts[i]
			stopIndex  = startIndex + nrVerts
			
			indices = self.face_verts[startIndex:stopIndex]

			# Calculate normal
			if nrVerts == 3:
				for j in range(3):
					v1[j] = self.verts[indices[1], j] - self.verts[indices[0], j]
					v2[j] = self.verts[indices[2], j] - self.verts[indices[0], j]
			else:
				for j in range(3):
					v1[j] = self.verts[indices[2], j] - self.verts[indices[0], j]
					v2[j] = self.verts[indices[3], j] - self.verts[indices[1], j]

			# Calculate cross product
			n[0] =  v1[1]*v2[2] - v2[1]*v1[2]
			n[1] = -v1[0]*v2[2] + v2[0]*v1[2]
			n[2] =  v1[0]*v2[1] - v2[0]*v1[1]

			# Normalize normal
			l = sqrt(n[0]**2 + n[1]**2 + n[2]**2)
			n[0] /= l
			n[1] /= l
			n[2] /= l

			self.face_n[i, 0] = n[0]
			self.face_n[i, 1] = n[1]
			self.face_n[i, 2] = n[2]

			# Calculate area
			for j in range(self.face_nrVerts[i]):
				v1 = self.verts[indices[j]]

				self.face_center[i, 0] += v1[0]
				self.face_center[i, 1] += v1[1]
				self.face_center[i, 2] += v1[2]

				if j == nrVerts-1:
					v2 = self.verts[indices[0]]
				else: 
					v2 = self.verts[indices[j+1]]

				cross[0] +=  v1[1]*v2[2] - v2[1]*v1[2]
				cross[1] += -v1[0]*v2[2] + v2[0]*v1[2]
				cross[2] +=  v1[0]*v2[1] - v2[0]*v1[1]

			self.face_area[i] = 0.5*(n[0]*cross[0] + n[1]*cross[1] + n[2]*cross[2])

			self.face_center[i, 0] /= nrVerts
			self.face_center[i, 1] /= nrVerts
			self.face_center[i, 2] /= nrVerts

	def calculateFaceCoordinateSystem(self):
		''' This methods calculates local coordinate system for every face. 
		The local x-axis, l, is pointing from the face center towards center of the edge that connects the second and the third vertex belonging to the face
		The local z-axis, n, is the (allready calculated) normal
		The local y-axis, m, is the cross-product n x l, giving a orthognoal coordinate system'''

		cdef int i, j, nrVerts, i1, i2

		self.face_l = np.zeros((self.nrFaces, 3), dtype=np.double)
		self.face_m = np.zeros((self.nrFaces, 3), dtype=np.double)

		cdef double[:] l  = np.zeros(3, dtype=np.double)
		cdef double[:] m  = np.zeros(3, dtype=np.double)
		cdef double[:] n  = np.zeros(3, dtype=np.double)
		cdef double[:] p0 = np.zeros(3, dtype=np.double)

		for i in range(self.nrFaces):
			p0 = self.face_center[i]

			i1 = self.face_verts[self.face_startIndex[i] + 1]
			i2 = self.face_verts[self.face_startIndex[i] + 2]
	
			for j in range(3):
				n[j]  = self.face_n[i, j]
				l[j]  = 0.5*(self.verts[i1, j] + self.verts[i2, j]) - self.face_center[i, j]

			l_length = sqrt(l[0]**2 + l[1]**2 + l[2]**2)
			l[0] /= l_length
			l[1] /= l_length
			l[2] /= l_length

			m[0] =  n[1]*l[2] - l[1]*n[2]
			m[1] = -n[0]*l[2] + l[0]*n[2]
			m[2] =  n[0]*l[1] - l[0]*n[1]

			m_length = sqrt(m[0]**2 + m[1]**2 + m[2]**2)

			for j in range(3):
				m[j] /= m_length

				self.face_l[i, j] = l[j]
				self.face_m[i, j] = m[j]

	def calculateEdgeConnectivity(self):
		''' This methid calculates the edge connectivity. That is, which vertices and faces each edge is connected to'''
		cdef int startIndex, edge_startIndex
		cdef int v1
		cdef int v2
		cdef int edgeExist
		cdef int i, j, k, offset, edgeIndex

		self.nrEdges = 0

		self.edge_verts      = np.array([[]], dtype=np.int)
		self.edge_faces      = np.array([],   dtype=np.int)
		self.edge_nrFaces    = np.array([],   dtype=np.int)
		self.edge_startIndex = np.array([],   dtype=np.int)

		self.face_edges = np.zeros(len(self.face_verts), dtype=np.int)

		# Go through all the faces
		for i in range(self.nrFaces):
			startIndex = self.face_startIndex[i]

			# Go through all the vertices in the face
			for j in range(self.face_nrVerts[i]):

				# Find vertices that make up an edge
				v1 = self.face_verts[startIndex+j]

				if j == self.face_nrVerts[i]-1:
					v2 = self.face_verts[startIndex]
				else:
					v2 = self.face_verts[startIndex+j+1]

				# Check if edge data allready exist in stored values
				edgeExist = 0

				checkEdge = np.zeros(self.nrEdges, dtype=np.int)
				for k in range(self.nrEdges):

					if (self.edge_verts[k, 0] == v1 or self.edge_verts[k, 0] == v2) and (self.edge_verts[k, 1] == v1 or self.edge_verts[k, 1] == v2):
						edgeExist = 1
						edgeIndex = k
						break

				# If edge does not exist, add it to edge list
				if edgeExist == 0:
					if self.nrEdges == 0:
						self.edge_verts = np.array([[v1, v2]], dtype=np.int)
					else:
						self.edge_verts = np.vstack([self.edge_verts, np.array([v1, v2])])

					edgeIndex = self.nrEdges
		
					self.nrEdges += 1

					self.edge_nrFaces = np.append(self.edge_nrFaces, 0)

				self.face_edges[startIndex + j] = edgeIndex

				# Add face to edge data
				self.edge_nrFaces[edgeIndex] += 1

		# Find start index for edge_face data
		self.edge_startIndex = np.zeros(self.nrEdges, dtype=np.int)
		for i in range(1, self.nrEdges):
			self.edge_startIndex[i] = self.edge_startIndex[i-1] + self.edge_nrFaces[i-1]

		# Find edge_faces
		self.edge_faces = np.zeros(np.sum(self.edge_nrFaces), dtype=np.int)
		offsetArray = np.zeros(self.nrEdges, dtype=np.int)

		# Go through each face
		for i in range(self.nrFaces):
			startIndex = self.face_startIndex[i]

			# GO through each edge (vertex) in that face
			for j in range(self.face_nrVerts[i]):
				# Find index of current edge
				edgeIndex = self.face_edges[startIndex+j]
				edge_startIndex = self.edge_startIndex[edgeIndex]

				# Add face index to right place in edge_faces array
				offset = offsetArray[edgeIndex]
				self.edge_faces[edge_startIndex + offset] = i
				offsetArray[edgeIndex] += 1

	def updateMeshData(self):
		''' Function that updates face data using other functions: calculateFaceData and calculateEdgeConnectivity '''
		self.calculateFaceData()
		self.calculateEdgeConnectivity()

	def addFaceData(self, face_dataName, face_data):
		''' Add arbitrary data that is connected to each face '''
		existingFaceData = False

		if len(self.face_dataNames) > 0:
			existingFaceData = True

		self.face_dataNames.append(face_dataName)

		if existingFaceData:
			self.face_data = np.vstack((self.face_data, face_data))
		else:
			self.face_data = np.zeros((1, self.nrFaces), dtype=np.double)

			for i in range(self.nrFaces):
				self.face_data[0, i] = face_data[i]

	def addVertData(self, vert_dataName, vert_data):
		''' Add arbitrary data that is connected to each vertex '''
		existingVertData = False

		if len(self.vert_dataNames) > 0:
			existingVertData = True

		self.vert_dataNames.append(vert_dataName)

		if existingVertData:
			self.vert_data = np.vstack((self.vert_data, vert_data))
		else:
			self.vert_data = np.zeros((1, self.nrVerts), dtype=np.double)

			for i in range(self.nrVerts):
				self.vert_data[0, i] = vert_data[i]

	def scale(self, double Sx, double Sy, double Sz, double x0 = 0, double y0 = 0, double z0 = 0):
		''' Scale the mesh. Sx is the scaling factor in x-direction, Sy in the y-direction and Sz in the z-direction
		The scaling is done around the point (x0, y0, z0) which defaults to (0, 0, 0) '''
		cdef int i
		for i in range(self.nrVerts):
			self.verts[i, 0] = (self.verts[i, 0] - x0)*Sx + x0
			self.verts[i, 1] = (self.verts[i, 1] - y0)*Sy + y0
			self.verts[i, 2] = (self.verts[i, 2] - z0)*Sz + z0

	def translate(self, double dx, double dy, double dz):
		''' Add the vector (dx, dy, dz) to all points in the mesh '''
		cdef int i
		for i in range(self.nrVerts):
			self.verts[i, 0] += dx
			self.verts[i, 1] += dy
			self.verts[i, 2] += dz

	def rotate(self, double rx, double ry, double rz, double x0 = 0, double y0 = 0, double z0 = 0):
		''' Rotate the mesh. rx is the rotation around the x-axis in radians, ry is around the y-axis and rz is around the z-axis
		The rotation is done around the point (x0, y0, z0) which defaults to (0, 0, 0)'''
		cdef int i
		cdef double angle
		cdef np.ndarray[dtype=np.double_t, ndim=2] Rx
		cdef np.ndarray[dtype=np.double_t, ndim=2] Ry
		cdef np.ndarray[dtype=np.double_t, ndim=2] Rz
		cdef np.ndarray[dtype=np.double_t, ndim=1] vert

		# Rotation matrix x axis
		Rx = np.array([[1, 0,                    0],
					   [0, np.cos(rx), -np.sin(rx)],
					   [0, np.sin(rx),  np.cos(rx)]])

		# Rotation matrix y axis
		Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
					   [0,              1,       0],
					   [-np.sin(ry), 0, np.cos(ry)]])

		# Rotation matrix z axis
		Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
					   [np.sin(rz),  np.cos(rz), 0],
					   [0,              0,       1]])

		for i in range(self.nrVerts):
			vert = np.asarray(self.verts[i])

			vert[0] -= x0
			vert[1] -= y0
			vert[2] -= z0

			vert = np.dot(Rx, vert)
			vert = np.dot(Ry, vert)
			vert = np.dot(Rz, vert)

			vert[0] += x0
			vert[1] += y0
			vert[2] += z0

			self.verts[i, 0] = vert[0]
			self.verts[i, 1] = vert[1]
			self.verts[i, 2] = vert[2]

	def flipNormals(self):
		''' Switch direction on the calculated normals'''
		cdef int i, j

		for i in range(self.nrFaces):
			for j in range(3):
				self.face_l[i, j] *= -1.0
				self.face_m[i, j] *= -1.0
				self.face_n[i, j] *= -1.0

	# ----- Export -------------------------
	def exportObj(self, filePath, exportNormals=False):
		''' Export the mesh as a WaveForm .obj file '''
		cdef int i, j, startIndex
		# create file
		f = open(filePath, 'w')

		# write header
		f.write('# Exported from pyMesh by Jarle Kramer\n')
		f.write("o object\n")

		# Write vertices
		for i in range(self.nrVerts):
			f.write('v {:.6f} {:.6f} {:.6f}\n'.format(self.verts[i][0], self.verts[i][1], self.verts[i][2]))

		# Write face normals
		if exportNormals:
			for i in range(self.nrFaces):
				f.write('vn {:.6f} {:.6f} {:.6f}\n'.format(self.face_n[i][0], self.face_n[i][1], self.face_n[i][2]))

		# Write faces
		for i in range(self.nrFaces):
			startIndex = self.face_startIndex[i]

			f.write('f ')
			for j in range(self.face_nrVerts[i]):
				if exportNormals:
					f.write('{:.0f}//{:.0f} '.format(self.face_verts[startIndex+j] + 1, i+1))
				else:
					f.write('{:.0f} '.format(self.face_verts[startIndex+j] + 1))

			f.write('\n')

		f.close()

	def exportVTK(self, filePath):
		''' Export the mesh, including added data, as VTK file format '''
		cdef int i, j

		f = open(filePath, 'w')

		# Write header
		f.write('<?xml version="1.0"?>\n')
		f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
		f.write('\t<PolyData>\n')
		f.write('\t\t<Piece NumberOfPoints="{:.0f}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{:.0f}">\n'.format(self.nrVerts, self.nrFaces))
		
		# Write vertices
		f.write('\t\t\t<Points>\n')
		f.write('\t\t\t\t<DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
		f.write('\t\t\t\t\t')
		for i in range(self.nrVerts):
			for j in range(3):
				f.write('{:.6f} '.format(self.verts[i][j]))

		f.write('\n')
		f.write('\t\t\t\t</DataArray>\n')
		f.write('\t\t\t</Points>\n')

		# Write faces
		f.write('\t\t\t<Polys>\n')
		f.write('\t\t\t\t<DataArray type="Int32" Name="connectivity" format="ascii">\n')
		f.write('\t\t\t\t\t')
		for i in range(len(self.face_verts)):
			f.write('{:.0f} '.format(self.face_verts[i]))
		f.write('\n')
		f.write('\t\t\t\t</DataArray>\n')
		f.write('\t\t\t\t<DataArray type="Int32" Name="offsets" format="ascii">\n')
		f.write('\t\t\t\t\t')
		for i in range(self.nrFaces):
			f.write('{:.0f} '.format(self.face_startIndex[i] + self.face_nrVerts[i]))
		f.write('\n')
		f.write('\t\t\t\t</DataArray>\n')
		f.write('\t\t\t</Polys>\n')

		# Write vert data
		if len(self.vert_dataNames) > 0:
			f.write('\t\t\t<PointData Scalars="')
			for i in range(len(self.vert_dataNames)):
				f.write(self.vert_dataNames[i])
				if i != len(self.vert_dataNames)-1:
					f.write(' ')

			f.write('">\n')

			for i in range(len(self.vert_dataNames)):
				f.write('\t\t\t\t<DataArray type="Float32" Name="{}" format="ascii">\n'.format(self.vert_dataNames[i]))
				f.write('\t\t\t\t\t')
				for j in range(self.nrVerts):
					f.write('{:.6f} '.format(self.vert_data[i, j]))
				f.write('\n')
				f.write('\t\t\t\t</DataArray>\n')
				
			f.write('\t\t\t</PointData>\n')

		# Write face data
		if len(self.face_dataNames) > 0:
			f.write('\t\t\t<CellData Scalars="')
			for i in range(len(self.face_dataNames)):
				f.write(self.face_dataNames[i])
				if i != len(self.face_dataNames)-1:
					f.write(' ')

			f.write('">\n')

			for i in range(len(self.face_dataNames)):
				f.write('\t\t\t\t<DataArray type="Float32" Name="{}" format="ascii">\n'.format(self.face_dataNames[i]))
				f.write('\t\t\t\t\t')
				for j in range(self.nrFaces):
					f.write('{:.6f} '.format(self.face_data[i, j]))
				f.write('\n')
				f.write('\t\t\t\t</DataArray>\n')
				
			f.write('\t\t\t</CellData>\n')



		# Write footer
		f.write('\t\t</Piece>\n')
		f.write('\t</PolyData>\n')
		f.write('</VTKFile>\n')

		f.close()

	def exportStl(self, filePath):
		''' Export the mesh as .stl file format'''
		cdef int i, j, startIndex, vertIndex, vertIndex2
		cdef double l1, l2
		cdef long vertIndices[3]
		cdef double faceCenter[3]

		f = open(filePath, 'w')

		f.write('solid mesh\n')

		for i in range(self.nrFaces):
			startIndex = self.face_startIndex[i]

			if self.face_nrVerts[i] == 3:
				f.write('facet normal {0} {1} {2}\n'.format(self.face_n[i, 0], self.face_n[i, 1], self.face_n[i, 2]))
				f.write('\touter loop\n')
				for j in range(3):
					vertIndex = self.face_verts[startIndex + j]

					f.write('\t\tvertex {0} {1} {2}\n'.format(self.verts[vertIndex, 0], self.verts[vertIndex, 1], self.verts[vertIndex, 2]))

				f.write('\tendloop\n')
				f.write('endfacet\n')

			elif self.face_nrVerts[i] == 4:
				# Calculate diagonal length
				l1 = 0
				l2 = 0
				for j in range(3):
					l1 += (self.verts[self.face_verts[startIndex + 2], j] - self.verts[self.face_verts[startIndex + 0], j])**2
					l2 += (self.verts[self.face_verts[startIndex + 3], j] - self.verts[self.face_verts[startIndex + 1], j])**2

				l1 = sqrt(l1)
				l2 = sqrt(l2)

				# Split shortest diagonal
				if l1 < l2:
					vertIndices[0] = self.face_verts[startIndex]
					vertIndices[1] = self.face_verts[startIndex + 1]
					vertIndices[2] = self.face_verts[startIndex + 2]

					f.write('facet normal {0} {1} {2}\n'.format(self.face_n[i, 0], self.face_n[i, 1], self.face_n[i, 2]))
					f.write('\touter loop\n')
					for j in range(3):
						f.write('\t\tvertex {0} {1} {2}\n'.format(self.verts[vertIndices[j], 0], self.verts[vertIndices[j], 1], self.verts[vertIndices[j], 2]))

					f.write('\tendloop\n')
					f.write('endfacet\n')

					vertIndices[0] = self.face_verts[startIndex]
					vertIndices[1] = self.face_verts[startIndex + 2]
					vertIndices[2] = self.face_verts[startIndex + 3]

					f.write('facet normal {0} {1} {2}\n'.format(self.face_n[i, 0], self.face_n[i, 1], self.face_n[i, 2]))
					f.write('\touter loop\n')
					for j in range(3):
						f.write('\t\tvertex {0} {1} {2}\n'.format(self.verts[vertIndices[j], 0], self.verts[vertIndices[j], 1], self.verts[vertIndices[j], 2]))

					f.write('\tendloop\n')
					f.write('endfacet\n')
				else:
					vertIndices[0] = self.face_verts[startIndex]
					vertIndices[1] = self.face_verts[startIndex + 1]
					vertIndices[2] = self.face_verts[startIndex + 3]

					f.write('facet normal {0} {1} {2}\n'.format(self.face_n[i, 0], self.face_n[i, 1], self.face_n[i, 2]))
					f.write('\touter loop\n')
					for j in range(3):
						f.write('\t\tvertex {0} {1} {2}\n'.format(self.verts[vertIndices[j], 0], self.verts[vertIndices[j], 1], self.verts[vertIndices[j], 2]))

					f.write('\tendloop\n')
					f.write('endfacet\n')

					vertIndices[0] = self.face_verts[startIndex + 1]
					vertIndices[1] = self.face_verts[startIndex + 2]
					vertIndices[2] = self.face_verts[startIndex + 3]

					f.write('facet normal {0} {1} {2}\n'.format(self.face_n[i, 0], self.face_n[i, 1], self.face_n[i, 2]))
					f.write('\touter loop\n')
					for j in range(3):
						f.write('\t\tvertex {0} {1} {2}\n'.format(self.verts[vertIndices[j], 0], self.verts[vertIndices[j], 1], self.verts[vertIndices[j], 2]))

					f.write('\tendloop\n')
					f.write('endfacet\n')

			else:
				for j in range(self.face_nrVerts[i]):
					vertIndex = self.face_verts[startIndex + j]
					faceCenter[0] += self.verts[vertIndex, 0]
					faceCenter[1] += self.verts[vertIndex, 1]
					faceCenter[2] += self.verts[vertIndex, 2]

				faceCenter[0] /= self.face_nrVerts[i]
				faceCenter[1] /= self.face_nrVerts[i]
				faceCenter[2] /= self.face_nrVerts[i]

				for j in range(self.face_nrVerts[i]):
					vertIndex = self.face_verts[startIndex + j]

					if j == self.face_nrVerts[i] - 1:
						vertIndex2 = self.face_verts[startIndex]
					else:
						vertIndex2 = self.face_verts[startIndex + j + 1]

					f.write('facet normal {0} {1} {2}\n'.format(self.face_n[i, 0], self.face_n[i, 1], self.face_n[i, 2]))
					f.write('\touter loop\n')

					f.write('\t\tvertex {0} {1} {2}\n'.format(self.verts[vertIndex, 0], self.verts[vertIndex, 1], self.verts[vertIndex, 2]))
					f.write('\t\tvertex {0} {1} {2}\n'.format(self.verts[vertIndex2, 0], self.verts[vertIndex2, 1], self.verts[vertIndex2, 2]))
					f.write('\t\tvertex {0} {1} {2}\n'.format(faceCenter[0], faceCenter[1], faceCenter[2]))

					f.write('\tendloop\n')
					f.write('endfacet\n')

		f.write('endsolid\n')

		f.close()

# ---------------- Functions to create mesh from file ---------------------------------
def importObj(filePath, simple=False):
	''' Import the mesh from a Wavefront .obj file as a Mesh class. Returns a Mesh object '''
	cdef int  i, j, nrLines, lineIndex, nrVerts, nrFaces, index
	cdef str  line
	cdef list lineList
	cdef list lines
	cdef double[:, :] verts
	cdef long[:]      face_verts   
	cdef long[:]      face_nrVerts

	cdef double x, y, z
	cdef double startTime, stopTime

	cdef long[:] vertLines = np.array([], dtype=np.int)
	cdef long[:] faceLines = np.array([], dtype=np.int)

	# Open file and read all lines
	f = open(filePath, 'r')
	lines = f.readlines()
	f.close()

	nrLines = len(lines)

	# Count number of vertices and faces
	vertLines = -np.ones(nrLines, dtype=np.int)
	faceLines = -np.ones(nrLines, dtype=np.int)
	nrVerts = 0
	nrFaces = 0

	for i in range(nrLines):
		line = lines[i]

		if line[0] == 'v':
			nrVerts += 1
			vertLines[nrVerts-1] = i
		elif line[0] == 'f':
			nrFaces += 1
			faceLines[nrFaces-1] = i


	# Initialize data arrays
	verts        = np.zeros((nrVerts, 3), dtype=np.double)
	face_nrVerts = np.zeros( nrFaces,     dtype=np.int)

	# Transfer verts data from text lines to vert array
	for i in range(nrVerts):
		lineIndex = vertLines[i]

		lineList = lines[lineIndex].strip().split()
		x = float(lineList[1])
		y = float(lineList[2])
		z = float(lineList[3])

		verts[i, 0] = x
		verts[i, 1] = y
		verts[i, 2] = z

	# Figure out number of verts in each face
	face_verts = np.array([], dtype=np.int)
	for i in range(nrFaces):
		lineIndex = faceLines[i]

		lineList = lines[lineIndex].strip().split()

		face_nrVerts[i] = len(lineList)-1


	# Transfer vert indices for each face to data array
	face_verts = np.zeros(np.sum(face_nrVerts), dtype=np.int)
	index = 0
	for i in range(nrFaces):
		lineIndex = faceLines[i]

		lineList = lines[lineIndex].strip().split()

		for j in range(face_nrVerts[i]):
			face_verts[index+j] = int(lineList[j+1].split('//')[0])-1

		index += face_nrVerts[i]

	# Create mesh instance
	mesh = Mesh(np.asarray(verts), np.asarray(face_verts), np.asarray(face_nrVerts), simple=simple)

	return mesh

def joinMeshes(Mesh mesh1, Mesh mesh2, simple=False):
	''' Join two meshes in to one mesh '''
	cdef int vertIndex, faceIndex, i, j

	cdef double[:, :] verts
	cdef long[:]      face_verts   
	cdef long[:]      face_nrVerts

	cdef int nrVerts = mesh1.nrVerts + mesh2.nrVerts
	cdef int nrFaces = mesh1.nrFaces + mesh2.nrFaces

	cdef int face_verts_length1 = len(mesh1.face_verts)
	cdef int face_verts_length2 = len(mesh2.face_verts)
	cdef int face_verts_length  = face_verts_length1 + face_verts_length2

	verts        = np.zeros((nrVerts, 3), dtype=np.double)
	face_verts   = np.zeros(face_verts_length, dtype=np.int) 
	face_nrVerts = np.zeros(nrFaces, dtype=np.int)

	# Transfer verts data
	for i in range(mesh1.nrVerts):
		verts[i, 0] = mesh1.verts[i, 0]
		verts[i, 1] = mesh1.verts[i, 1]
		verts[i, 2] = mesh1.verts[i, 2]

	for i in range(mesh2.nrVerts):
		verts[i+mesh1.nrVerts, 0] = mesh2.verts[i, 0]
		verts[i+mesh1.nrVerts, 1] = mesh2.verts[i, 1]
		verts[i+mesh1.nrVerts, 2] = mesh2.verts[i, 2]

	# Transfer face data
	for i in range(mesh1.nrFaces):
		face_nrVerts[i] = mesh1.face_nrVerts[i]

	for i in range(mesh2.nrFaces):
		face_nrVerts[i + mesh1.nrFaces] = mesh2.face_nrVerts[i]

	for i in range(face_verts_length1):
		face_verts[i] = mesh1.face_verts[i]

	for i in range(face_verts_length2):
		face_verts[face_verts_length1 + i] = mesh2.face_verts[i] + mesh1.nrVerts

	# Initialize new mesh and return
	mesh = Mesh(np.asarray(verts), np.asarray(face_verts), np.asarray(face_nrVerts), simple=simple)
	return mesh

def copyMesh(Mesh mesh, simple=True):
	''' Make a distinct copy of a mesh '''
	verts        = copy.deepcopy(np.asarray(mesh.verts))
	face_verts   = copy.deepcopy(np.asarray(mesh.face_verts))
	face_nrVerts = copy.deepcopy(np.asarray(mesh.face_nrVerts))

	mesh_copy = Mesh(np.asarray(verts), np.asarray(face_verts), np.asarray(face_nrVerts), simple=simple)

	return mesh_copy