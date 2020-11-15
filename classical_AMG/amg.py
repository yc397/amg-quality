from dolfin import *
import numpy as np
from petsc4py import PETSc
from petsc4py.PETSc import Mat
import pybind11

# set petsc options at beginning
petsc_options = PETSc.Options()


# Use petsc4py to define the smoothers
def direct(Ah, bh):
    '''LU factorisation. Ah is the matrix, bh is the rhs'''
    ksp = PETSc.KSP().create()
    ksp.setOptionsPrefix('coarse_')
    yh = bh.duplicate()
    ksp.setNormType(PETSc.KSP.NormType.NONE)
    pc = ksp.getPC()
    pc.setOptionsPrefix('coarse_pc_')
    petsc_options['coarse_pc_type'] = 'lu'
    ksp.setOperators(Ah)
    ksp.setFromOptions()
    ksp.solve(bh, yh)
    return yh


def smoother(Ag, bg, Ng, igg, ksptype, pctype):
    '''Smoother for multigrid. Ag, and bg are the LHS and RHS respectively.
    Ng is the number of iterations (usually 1), igg is the initial guess
    for the solution.
    ksptype and pctype can be ('richardson', 'jacobi'), ('richardson', 'sor')
    or ('chebyshev', 'jacobi') for example '''

    ksp = PETSc.KSP().create()
    ksp.setOptionsPrefix('smoother_')
    petsc_options['smoother_ksp_type'] = ksptype
    ksp.setNormType(PETSc.KSP.NormType.NONE)
    pc = ksp.getPC()
    pc.setOptionsPrefix('smpc_')
    petsc_options['smpc_pc_type']=pctype
    petsc_options['smoother_ksp_initial_guess_nonzero']=True
    # ksp.setInitialGuessNonzero(True)
    ksp.setTolerances(max_it=Ng)
    ksp.setOperators(Ag)
    ksp.setFromOptions()
    ksp.solve(bg, igg)


def residual(Ah, bh, xh):
    '''a function to calculate the residual
    Ah is the matrix, bh is the rhs, xh is the approximation'''
    resh = bh - Ah * xh
    normr = PETSc.Vec.norm(resh, 2)
    return normr
#=================================================================

# import the mesh
mesh = UnitCubeMesh(50, 50, 50)
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
f = Constant(1.0)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx
A = PETScMatrix()
b = PETScVector()
assemble_system(a, L, bc, A_tensor=A, b_tensor=b)

Apt = A.mat()
bpt = b.vec()

# Set initial guess
fe = Constant(0.0)
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
    PETScOptions.set("ksp_type", "richardson")


    # Choose gamg for the preconditioner
    PETScOptions.set("pc_type", "gamg")

    # Set the preconditioner to be PETSc gamg with smoothed aggregation.
    PETScOptions.set("pc_gamg_type", "classical")

    # The number of smoothed aggregation steps. More smooths improve performance of
    # the preconditioner at the cost of memory.
    #PETScOptions.set("pc_gamg_agg_nsmooths", 1)

    # The maximum number of levels permitted in the MG preconditioner.
    #PETScOptions.set("pc_mg_levels", 5)


    # a more powerful preconditioner, at greater construction and memory cost. A
    # lower threshold is cheaper to construct, but will increase the number of
    # iterations required by the Krylov solver. The PETSc manual suggests a
    # threshold of 0.08 for 3D problems. In Nate's experience, this is far too
    # expensive for large problems beyond 15M DoF.
    PETScOptions.set("pc_gamg_threshold", 0.05)

    ksp = solver.ksp()
    pc = ksp.getPC()
    pc.setFromOptions()

    ug = Function(V)
    solver.set_from_options()
    solver.solve(ug.vector(), bg)

    # get number of levels and prolongations by getMG
    nlevel = pc.getMGLevels() 
    prouse = []
    for ih in range(nlevel-1, 0, -1):
      mat = pc.getMGInterpolation(ih)
      prouse.append(mat.copy())
      #print(mat.size)

    pc.destroy()
    ksp.destroy()
    del ksp

    return nlevel,  prouse

# find the number of levels and prolongation list of AMG
nl, puse = makepro(A,b)
#=========================================================

# multigrid part, same as GMG

def mg(Ah, bh, uh, prolongation, N_cycles, N_levels, ksptype, pctype):
    '''multigrid for N level mesh
    Ah is the matrix, bh is rhs on the finest grid,
    uh is the initial guess for finest grid
    prolongation is a list containing all of operators from fine-to-coarse
    N_cycles is number of cycles and N_levels is number of levels'''

    r0 = residual(Ah, bh, uh)

    # make a restriction list and gird operator list and rhs list
    # and initial guess list
    # initialize the first entity
    restriction = [None] * (N_levels - 1)
    Alist = [None] * N_levels
    blist = [None] * N_levels
    restriction[0] = Mat()
    prolongation[0].transpose(restriction[0])
    uhlist = [None] * N_levels
    Alist[0] = Ah
    blist[0] = bh
    uhlist[0] = uh

    # calculate the restriction, matrix, and initial guess lists
    # except coarsest grid, since transfer operator and initial guess
    # is not defined on that level
    for i_level in range(1, N_levels - 1):
        restriction[i_level] = Mat()
        prolongation[i_level].transpose(restriction[i_level])
        Alist[i_level] = Mat()
        Alist[i_level - 1].PtAP(prolongation[i_level - 1], Alist[i_level])
        uhlist[i_level] = restriction[i_level - 1] * uhlist[i_level - 1]

    # find the coarsest grid matrix
    Alist[N_levels - 1] = Mat()
    Alist[N_levels - 2].PtAP(prolongation[N_levels - 2], Alist[N_levels - 1])

    for num_cycle in range(N_cycles):

        # restriction to coarse grids
        for i in range(N_levels - 1):

            # apply smoother to every level except the coarsest level
            smoother(Alist[i], blist[i], 2, uhlist[i], ksptype, pctype)

            # obtain the rhs for next level
            res = blist[i] - Alist[i] * uhlist[i]
            blist[i + 1] = restriction[i] * res

        # on the coarsest grid, apply direct lu
        uhlist[N_levels - 1] = direct(Alist[N_levels - 1], blist[N_levels - 1])

        # prolongation back to fine grids
        for j in range(N_levels - 2, -1, -1):

            uhlist[j] += prolongation[j] * uhlist[j + 1]
            smoother(Alist[j], blist[j], 2, uhlist[j], ksptype, pctype)
            
        # calculate the relative residual
        res4 = residual(Ah, bh, uhlist[0]) / r0
        #print('relative residual after', num_cycle + 1, 'cycles is:', res4)

    return uhlist[0]

# multigrid preconditioned conjugate gradient
def mgcg(Ah, bh, igh, Ncg, Nmg):
    '''multigrid preconditioned conjugate gradient
    Ah is the matrix, bh is the rhs, igh is initial guess
    Ncg is the number of iterations of cg, 
    and Nmg is number of cycles for multigrid'''

    # initialize the problem
    r0 = residual(Ah, bh, igh)
    rk = bh - Ah * igh
    wk = igh.copy()
    zk = mg(Ah, rk, wk, puse, Nmg, nl, 'richardson', 'sor')
    pk = zk.copy()
    xk = igh.copy()

    #conjugate gradient
    for ite in range(Ncg):
        alpha = (zk.dot(rk)) / ((Ah * pk).dot(pk))
        w1 = alpha * pk
        xk = (xk+w1).copy()
        rtest = Ah * pk
        rs = alpha * rtest
        rk2 = (rk - rs).copy()
        rt = igh.copy()
        zk2 = mg(Ah, rk2, rt, puse, Nmg, nl, 'richardson', 'sor')
        beta = (rk2.dot(zk2)) / (rk.dot(zk))
        y1 = beta * pk
        pk = (zk2 + y1).copy()
        rk = rk2.copy()
        zk = zk2.copy()

        res4 = residual(Ah, bh, xk)
        print('relative residual after', ite + 1, 'iterations is:', res4/r0)
        
    return xk


# implement multigrid
print('Initial residual is:', residual(Apt, bpt, fph))
wh = mgcg(Apt, bpt, fph, 10, 1)
#wh = mg(Av, bv, fph, puse, 10, nl, 'richardson', 'sor')