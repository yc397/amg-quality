from dolfin import *
import numpy as np
from petsc4py import PETSc
from petsc4py.PETSc import Mat
from petsc4py.PETSc import Vec
import matplotlib.pyplot as plt

# import the mesh
#mesh = UnitCubeMesh(50, 50, 50)
mesh=Mesh("./level1_bad.xml")
V = FunctionSpace(mesh, 'P', 2)
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

# Set initial guess
fe = Expression('sin(pi*k*x[0])*sin(pi*k*x[1])', degree=6, k=10.0)
fp = interpolate(fe, V)
fph = fp.vector().vec()

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
    PETScOptions.set("pc_gamg_type", "agg")

    # The number of smoothed aggregation steps. More smooths improve performance of
    # the preconditioner at the cost of memory.
    PETScOptions.set("pc_gamg_agg_nsmooths", 0)

    # The maximum number of levels permitted in the MG preconditioner.
    PETScOptions.set("pc_mg_levels", 4)


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
      #mat = mat
      prouse.append(mat.copy())
      print(mat.size)

    pc.destroy()
    ksp.destroy()
    del ksp

    return nlevel,  prouse

# find the number of levels and prolongation list of AMG
nl, puse = makepro(A,b)
# eigfind.eigfind(Apt)
# quit()

#===============================================================================================
ruse = [None] * (nl-1)
Ause = [None] * nl
ruse[0] = Mat()
puse[0].transpose(ruse[0])
Ause[0] = Apt

for il in range(1, nl-1):
    ruse[il] = Mat()
    puse[il].transpose(ruse[il])
    Ause[il] = Mat()
    Ause[il - 1].PtAP(puse[il - 1], Ause[il])
    

# find the coarsest grid matrix
Ause[nl-1] = Mat()
Ause[nl-2].PtAP(puse[nl-2], Ause[nl-1])


prolongation = puse
Bfinemore=[3241, 3748, 3751, 3496, 3753, 3242, 3243, 3244, 3245, 3246, 3500, 3501, 3502, 3503, 3504, 3505, 3506, 3763, 3764, 3511, 3762, 3761, 3509, 3253, 3508, 3255, 3752, 3497, 3498, 3247, 3239, 3757, 3758, 3251, 3252, 3507, 3059, 3318, 3510, 3765]

ptt=prolongation[0]
Alist=Ause
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
