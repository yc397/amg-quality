from dolfin import *
import numpy as np
from petsc4py import PETSc
from petsc4py.PETSc import Mat


mesh4 = Mesh("./level1_bad.xml") 
V4 = FunctionSpace(mesh4, 'P', 1)
n4 = mesh4.num_vertices()
V3dm = V4.dofmap()
cq2 = MeshQuality.radius_ratios(mesh4)
cq = cq2.array()

indices = np.where(cq < 0.1)[0]

dof_set = []
cell_set = []
for i in indices:
 	cell = Cell(mesh4, i)
 	for v in vertices(cell):	
 		for c in cells(v):
 			cell_set += [c.index()]
 			dof_set.extend(V3dm.cell_dofs(c.index()))

bad_dofs = list(set(dof_set))
bad_cells = list(set(cell_set))


#print('BAD CELLS=', bad_cells)
print('BAD DOFS=', bad_dofs)
print(len(bad_dofs))