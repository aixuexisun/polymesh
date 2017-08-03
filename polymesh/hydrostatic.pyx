#!python
#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import sys
import copy

import numpy as np
cimport numpy as np

import polymesh.mesh as Mesh
cimport polymesh.mesh as Mesh

def extractWetSurface(Mesh.Mesh mesh, double[:] p0 = np.array([0, 0, 0], dtype=np.double), double[:] up = np.array([0, 0, 1], dtype=np.double)):
	''' This function takes an input mesh, and removes everything above the plane which is defined by the point "p0", (defaults to [0, 0, 0]) and the "up" direction (defaults to [0, 0, 1])
	Faces that cross the cutting plane are modified so that only the portion below the plane is kept. This will sometime mean changing the topoly, as well as the coordinates of the face.'''

	cdef int i, j, k, i0, i1, i2, startIndex, nrVerts, nrFaces, vertIndex, vertIndex1, vertIndex2, face_nrVertsLength
	cdef double t1, t2, dotv0p0, dotv1p0, dotv2p0, dot1, dot2

	cdef double[:] v0 = np.zeros(3)
	cdef double[:] v1 = np.zeros(3)
	cdef double[:] v2 = np.zeros(3)
	cdef double[:] p1 = np.zeros(3)
	cdef double[:] p2 = np.zeros(3)
	cdef double[:] p  = np.zeros(3)
	cdef double[:] n  = np.zeros(3)

	cdef double up_length = np.sqrt(up[0]**2 + up[1]**2 + up[2]**2)
	n[0] = up[0]/up_length
	n[1] = up[1]/up_length
	n[2] = up[2]/up_length
  
	cdef double[:, :] verts = np.asarray(mesh.verts)

	# ----------- Count number of faces below the waterline ---------------------------------------
	cdef long[:] keepIndices = np.zeros(mesh.nrFaces, dtype=int)
	
	nrFaces          = 0
	face_vertsLength = 0

	cdef int keepFace

	for i in range(mesh.nrFaces):
		keepFace = 0
		startIndex = mesh.face_startIndex[i]

		for j in range(mesh.face_nrVerts[i]):
			vertIndex = mesh.face_verts[startIndex + j]

			dotv0p0 = 0
			for k in range(3):
				dotv0p0 += (mesh.verts[vertIndex, k] - p0[k])*n[k]

			if dotv0p0 < 0:
				keepFace = 1
				break
				
		if keepFace:
			keepIndices[i] = 1
			nrFaces += 1
			face_vertsLength += mesh.face_nrVerts[i]

	# ----------- Transfer faces below water line to new data arrays ------------------------------
	cdef long[:] face_verts   = np.zeros(face_vertsLength, dtype=int)
	cdef long[:] face_nrVerts = np.zeros(nrFaces, dtype=int)  

	cdef int face_index      = 0
	cdef int face_startIndex = 0

	for i in range(mesh.nrFaces):
		if keepIndices[i]:
			startIndex = mesh.face_startIndex[i]
			nrVerts    = mesh.face_nrVerts[i]

			face_nrVerts[face_index] = nrVerts
			
			for j in range(nrVerts):
				face_verts[face_startIndex + j] = mesh.face_verts[startIndex + j]

			face_index      += 1
			face_startIndex += nrVerts

	# ------------- Move vertices above waterline down to the waterline ------------------------------
	startIndex = 0
	for i in range(nrFaces):
		j = 0
		while j < face_nrVerts[i]:
			i0 = face_verts[startIndex + j]

			# Current vertex
			v0 = verts[i0]
				
			# Check if vertex is above plane
			dotv0p0 = 0
			for k in range(3):
				dotv0p0 += (v0[k] - p0[k])*n[k]

			if dotv0p0 > 0:
				# Find neighbours
				if j == 0:
					i1 = face_verts[startIndex + face_nrVerts[i] - 1]
					i2 = face_verts[startIndex + 1]
				elif j == face_nrVerts[i]-1:
					i1 = face_verts[startIndex + j - 1]
					i2 = face_verts[startIndex]
				else:
					i1 = face_verts[startIndex + j - 1]
					i2 = face_verts[startIndex + j + 1]
						
				v1 = verts[i1]
				v2 = verts[i2]

				# Calculate necessary dot products
				dotv1p0 = 0
				for k in range(3):
					dotv1p0 += (v1[k] - p0[k])*n[k]
				dotv2p0 = 0
				for k in range(3):
					dotv2p0 += (v2[k] - p0[k])*n[k]

				if dotv2p0 >= 0 and dotv1p0  >= 0:
					# Both neighbours are above the surface, remove point
					face_verts       = np.delete(face_verts, startIndex + j)
					face_nrVerts[i] -= 1
					j -= 1

				elif dotv1p0  < 0 and dotv2p0  < 0:
					# Both neighbours are below the surface, move first point towards v1, and generate new point and move it towards v2
					dot1 = 0
					for k in range(3):
						dot1 += (p0[k] - v0[k])*n[k]

					dot2 = 0
					for k in range(3):
						dot2 += (v1[k] - v0[k])*n[k]

					t1 = dot1/dot2

					dot2 = 0
					for k in range(3):
						dot2 += (v2[k] - v0[k])*n[k]

					t2 = dot1/dot2

					# Calculate points
					for k in range(3):
						p1[k] = v0[k] + (v1[k] - v0[k])*t1
						p2[k] = v0[k] + (v2[k] - v0[k])*t2

					verts = np.append(verts, [p1], axis = 0)
					verts = np.append(verts, [p2], axis = 0)

					vertIndex1 = len(verts) - 2
					vertIndex2 = len(verts) - 1

					# Generate new face
					face_verts[startIndex + j] = vertIndex1
					face_verts                 = np.insert(face_verts, startIndex + j + 1, vertIndex2)
					face_nrVerts[i]           += 1

				elif dotv1p0 < 0:
					# One point beneath the surface, try to move it towards v1
					dot1 = 0
					for k in range(3):
						dot1 += (p0[k] - v0[k])*n[k]

					dot2 = 0
					for k in range(3):
						dot2 += (v1[k] - v0[k])*n[k]

					t1 = dot1/dot2
					
					for k in range(3):
						p[k] = v0[k] + (v1[k] - v0[k])*t1

					verts = np.append(verts, [p], axis = 0)
					face_verts[startIndex + j] = len(verts) - 1

				elif dotv2p0 < 0:
					# One point beneath the surface, try to move it towards v2
					dot1 = 0
					for k in range(3):
						dot1 += (p0[k] - v0[k])*n[k]

					dot2 = 0
					for k in range(3):
						dot2 += (v2[k] - v0[k])*n[k]

					t2 = dot1/dot2

					for k in range(3):
						p[k] = v0[k] + (v2[k] - v0[k])*t2

					verts = np.append(verts, [p], axis = 0)

					face_verts[startIndex + j] = len(verts) - 1

			j += 1

		startIndex += face_nrVerts[i]

	mesh = Mesh.Mesh(np.asarray(verts), np.asarray(face_verts), np.asarray(face_nrVerts), simple=True)
	mesh.calculateFaceData()

	return mesh

def extractPlane(mesh, axis, pos):
	''' This functions extract the lines from the input mesh that lies in the input plane, defined by axis and pos '''
	cdef int i
	# Go through each edge and find crossing points
	verts = np.array([])
	lines = np.array([])

	edge_indices = np.array([])
	vert_indices = np.array([])

	index = 0

	for i in range(mesh.nrEdges):

		v1 = mesh.verts[mesh.edge_verts[i, 0]]
		v2 = mesh.verts[mesh.edge_verts[i, 1]]

		cross = False

		if (v1[axis] > pos and v2[axis] < pos):
			cross = True
		elif (v2[axis] > pos and v1[axis] < pos):
			cross = True
		elif (v1[axis] == pos and v2[axis] != pos):
			cross = True
		elif (v2[axis] == pos and v1[axis] != pos):
			cross = True

		if cross:
			# If edge crosses plane, store vertices at crossing point
			t = (pos - v1[axis])/(v2[axis]-v1[axis])

			v = v1 + t*(v2-v1)

			if len(verts) == 0:
				verts = np.array([v])
			else:
				verts = np.vstack([verts, v])

			edge_indices = np.append(edge_indices, i)
			vert_indices = np.append(vert_indices, index)
			index += 1

	# Go through all faces and find the ones that have edges crossing the plane
	for i in range(mesh.nrFaces):
		startIndex = mesh.face_startIndex[i]
		nrEdges    = mesh.face_nrVerts[i]

		stored_indices = np.array([])

		addedFace = False

		for j in range(nrEdges):
			edge_index = mesh.face_edges[startIndex + j]

			test = np.where(edge_indices == edge_index)[0]

			if len(test) > 0:
				stored_indices = np.append(stored_indices, vert_indices[test[0]])
				addedFace = True

		if addedFace:
			if len(stored_indices) == 2:
				if len(lines) == 0:
					lines = np.array([stored_indices])
				else:
					lines = np.vstack([lines, stored_indices])

	return verts, lines

def calculateVolume(mesh):
	''' This function calculates the volume of the input mesh. can be used with "wetSurfaces" generated by the "extractWetSurface" function '''
	volume = 0

	for i in range(mesh.nrFaces):
		z = mesh.face_center[i, 2]
		A = mesh.face_area[i]
		n = mesh.face_n[i]

		volume += z*A*n[2]

	return volume

def calculateSurface(mesh):
	''' This function adds upp all face areas, which results in the total surface area of the mesh '''
	s = 0
	for i in range(mesh.nrFaces):
		s += mesh.face_area[i]

	return s

def calculateDimensions(mesh):
	''' This function finds the dimensions of the mesh '''
	xMax = -999999
	yMax = -999999
	zMax = -999999

	xMin = 999999
	yMin = 999999
	zMin = 999999

	for i in range(mesh.nrFaces):
		startIndex = mesh.face_startIndex[i]
		nrVerts    = mesh.face_nrVerts[i]
		stopIndex  = startIndex + nrVerts

		indices = mesh.face_verts[startIndex:stopIndex]

		for j in range(nrVerts):
			v = mesh.verts[indices[j]]

			if v[0] > xMax:
				xMax = v[0]
			elif v[0] < xMin:
				xMin = v[0]

			if v[1] > yMax:
				yMax = v[1]
			elif v[1] < yMin:
				yMin = v[1]

			if v[2] > zMax:
				zMax = v[2]
			elif v[2] < xMin:
				zMin = v[2]

	return np.array([xMax - xMin, yMax - yMin, zMax - zMin])

def calculateVolumeCentroid(mesh):
	''' This function calculates the volume centroid of the mesh '''
	vol = 0
	T_x = 0
	T_y = 0
	T_z = 0 

	# Assuming that face is a triangle
	for i in range(mesh.nrFaces):
		# Volume
		z = mesh.face_center[i, 2]
		A = mesh.face_area[i]
		n = mesh.face_n[i]

		if not(np.isnan(A)):
			vol += z*A*n[2]

			# Volume moment
			J = 2*A

			startIndex = mesh.face_startIndex[i]
			vertIndex1 = mesh.face_verts[startIndex]
			vertIndex2 = mesh.face_verts[startIndex+1]
			vertIndex3 = mesh.face_verts[startIndex+2]

			# T_x
			x1 = mesh.verts[vertIndex1, 0]
			x2 = mesh.verts[vertIndex2, 0]
			x3 = mesh.verts[vertIndex3, 0]

			a = (x2 - x1)
			b = (x3 - x1)
			c = x1

			T_x += 0.5*n[0]*J*(a**2 + b**2 + 4*b*c + 6*c**2 + a*(b + 4*c))/12

			# T_y
			y1 = mesh.verts[vertIndex1, 1]
			y2 = mesh.verts[vertIndex2, 1]
			y3 = mesh.verts[vertIndex3, 1]

			a = (y2 - y1)
			b = (y3 - y1)
			c = y1

			T_y += 0.5*n[1]*J*(a**2 + b**2 + 4*b*c + 6*c**2 + a*(b + 4*c))/12

			# T_z
			z1 = mesh.verts[vertIndex1, 2]
			z2 = mesh.verts[vertIndex2, 2]
			z3 = mesh.verts[vertIndex3, 2]

			a = (z2 - z1)
			b = (z3 - z1)
			c = z1

			T_z += 0.5*n[2]*J*(a**2 + b**2 + 4*b*c + 6*c**2 + a*(b + 4*c))/12

	volume = vol
	volumeCentroid = np.array([T_x, T_y, T_z])/vol

	return volumeCentroid
