from dolfin import *
import numpy as np
from petsc4py import PETSc
from petsc4py.PETSc import Mat
import matplotlib.pyplot as plt

# import the mesh
#mesh = UnitCubeMesh(50, 50, 50)
mesh=Mesh("./level1_bad.xml")
V = FunctionSpace(mesh, 'P', 1)
n = mesh.num_vertices()

# Use FEniCS to formulate the FEM problem. A is the matrix, b is the rhs.
u_D = Expression('0.0', degree=0)


# Define boundary for DirichletBC
def boundary_L(x, on_boundary):
    tol = 1E-14
    return on_boundary and (near(x[0], 0, tol) or near(x[0], 1, tol) or near(x[2], 0, tol) or near(x[2], 1, tol))

bc = DirichletBC(V, u_D, boundary_L)
u = TrialFunction(V)
v = TestFunction(V)
f = Expression('3*pi*pi*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])',degree=6)
g = Expression('pi*sin(pi*x[0])*sin(pi*x[2])',degree=6)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx - g * v * ds
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
    PETScOptions.set("pc_mg_levels", 4)


    # a more powerful preconditioner, at greater construction and memory cost. A
    # lower threshold is cheaper to construct, but will increase the number of
    # iterations required by the Krylov solver. The PETSc manual suggests a
    # threshold of 0.08 for 3D problems. In Nate's experience, this is far too
    # expensive for large problems beyond 15M DoF.
    PETScOptions.set("pc_gamg_threshold", 0.08)
    PETScOptions.set("ksp_rtol", 70)
    PETScOptions.set("ksp_max_it", 1)

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

Bfinemore=[23041, 23042, 514, 31238, 31239, 31240, 31241, 41482, 31243, 36876, 36877, 41484, 36879, 41485, 38434, 38435, 38436, 38437, 557, 26161, 26162, 26163, 26164, 26165, 26166, 26167, 26168, 26169, 36930, 36931, 36933, 36934, 36935, 36424, 36425, 36426, 36427, 36936, 36428, 36423, 36429, 36938, 593, 594, 595, 592, 596, 598, 599, 597, 600, 36443, 36444, 603, 1119, 1120, 610, 34427, 34429, 34432, 34433, 34438, 34440, 34441, 37007, 37018, 37019, 668, 669, 672, 200, 201, 38612, 38616, 38617, 38618, 38619, 38620, 38621, 38622, 38623, 733, 736, 40162, 734, 735, 732, 226, 40167, 40166, 40169, 40168, 40173, 40174, 40175, 40176, 40177, 26362, 26365, 26366, 26368, 26369, 1292, 30495, 30496, 38697, 38699, 26425, 34618, 26426, 26428, 26427, 28152, 40161, 28153, 28154, 342, 343, 344, 38745, 38746, 38747, 38748, 38749, 347, 346, 345, 880, 34686, 908, 28047, 28049, 28050, 28051, 28052, 28053, 28054, 28055, 920, 921, 922, 923, 919, 19883, 950, 951, 440, 441, 442, 22982, 22983, 22984, 22985, 22986, 999, 1000, 1001, 28142, 28143, 28144, 28145, 28146, 28147, 31220, 31221, 31222, 28151, 28150, 28149, 28148, 28155]

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

#quit()
bdd2=[]
for bag in bdd:
    for kl in range(Alist[2].size[0]):
        ckg=prolongation[1].getValue(bag,kl)
        if ckg!=0:
            bdd2.append(kl)

bdd2=list(set(bdd2))
print(bdd2)
print(len(bdd2))

