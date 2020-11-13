from dolfin import *
import numpy as np
from petsc4py import PETSc
from petsc4py.PETSc import Mat
import matplotlib.pyplot as plt

# import the mesh
mesh=Mesh("./level1_bad.xml")
V = FunctionSpace(mesh, 'P', 1)
n = mesh.num_vertices()

# Use FEniCS to formulate the FEM problem. A is the matrix, b is the rhs.
u_D = Expression('0.0', degree=0)


# Define boundary for DirichletBC
def boundary_L(x, on_boundary):
    tol = 1E-14
    return on_boundary 

bc = DirichletBC(V, u_D, boundary_L)
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0.0)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx 
A = PETScMatrix()
b = PETScVector()
assemble_system(a, L, bc, A_tensor=A, b_tensor=b)

Apt = A.mat()
bpt = b.vec()


# create the transfer operators by petsc gamg
def makepro(At,bt):
    '''the function gives the prolongation list and number of
    levels of amg, At is the matrix and bt is the rhs'''

    bg = bt.copy()
    Ah = At.copy()

    solver = PETScKrylovSolver()
    solver.set_operator(Ah)

    # Choose conjugate gradient method for the Krylov solver.
    PETScOptions.set("ksp_type", "cg")


    # Choose gamg for the preconditioner
    PETScOptions.set("pc_type", "gamg")

    # Set the preconditioner to be PETSc gamg with smoothed aggregation.
    PETScOptions.set("pc_gamg_type", "classical")

    # The number of smoothed aggregation steps. More smooths improve performance of
    # the preconditioner at the cost of memory.
    #PETScOptions.set("pc_gamg_agg_nsmooths", 1)

    # The maximum number of levels permitted in the MG preconditioner.
    PETScOptions.set("pc_mg_levels", 3)


    # a more powerful preconditioner, at greater construction and memory cost. A
    # lower threshold is cheaper to construct, but will increase the number of
    # iterations required by the Krylov solver. The PETSc manual suggests a
    # threshold of 0.08 for 3D problems. In Nate's experience, this is far too
    # expensive for large problems beyond 15M DoF.
    PETScOptions.set("pc_gamg_threshold", 0.15)

    ksp = solver.ksp()
    pc = ksp.getPC()
    pc.setFromOptions()

    ug = Function(V)
    solver.set_from_options()
    solver.solve(ug.vector(), bg)

    # get number of levels and prolongations by getMG
    nlevel = pc.getMGLevels() 
    print(nlevel)
    prouse = []
    for ih in range(nlevel-1, 0, -1):
      mat = pc.getMGInterpolation(ih)
      prouse.append(mat.copy())
      print(mat.size)

    pc.destroy()
    ksp.destroy()
    del ksp

    return nlevel,  prouse


nl, prolongation = makepro(A,b)

restriction = [None] * (nl - 1)
Alist = [None] * nl
restriction[0] = Mat()
prolongation[0].transpose(restriction[0])
Alist[0] = Apt

for i_level in range(1, nl - 1):
    restriction[i_level] = Mat()
    prolongation[i_level].transpose(restriction[i_level])
    Alist[i_level] = Mat()
    Alist[i_level - 1].PtAP(prolongation[i_level - 1], Alist[i_level])

Bfinemore=[960, 840, 841, 842, 843, 844, 784, 1013, 919, 920, 921, 922, 959]

ptt=prolongation[0]

bdd=[]
for bt in Bfinemore:
    for jj in range(Alist[1].size[0]):
        ct=ptt.getValue(bt,jj)
        if ct!=0:
            bdd.append(jj)

bdd=list(set(bdd))
print(bdd)
print(len(bdd))

quit()


