from dolfin import *
import numpy as np
from petsc4py import PETSc
from petsc4py.PETSc import Mat
from petsc4py.PETSc import Vec
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

def smoother_ch1(Ag, bg, Ng, igg, ksptype, pctype):
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
    petsc_options['smoother_ksp_chebyshev_eigenvalues']='0.2549419,2.549419'
    # ksp.setInitialGuessNonzero(True)
    ksp.setTolerances(max_it=Ng)
    ksp.setOperators(Ag)
    ksp.setFromOptions()
    ksp.solve(bg, igg)

def smoother_ch2(Ag, bg, Ng, igg, ksptype, pctype):
    '''Smoother for multigrid. Ag, and bg are the LHS and RHS respectively.
    Ng is the number of iterations (usually 1), igg is the initial guess
    for the solution.
    ksptype and pctype can be ('richardson', 'jacobi'), ('richardson', 'sor')
    or ('chebyshev', 'jacobi') for example '''

    ksp = PETSc.KSP().create()
    ksp.setOptionsPrefix('smoother_')
    petsc_options['smoother_ksp_type'] = ksptype
    ksp.setNormType(PETSc.KSP.NormType.NONE)
    ksp.setComputeEigenvalues(1)
    pc = ksp.getPC()
    pc.setOptionsPrefix('smpc_')
    petsc_options['smpc_pc_type']=pctype
    petsc_options['smoother_ksp_initial_guess_nonzero']=True
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

def build_nullspace(V, x):
    """Function to build null space for 3D elasticity"""

    # Create list of vectors for null space
    nullspace_basis = [x.copy() for i in range(6)]

    # Build translational null space basis
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0);
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0);
    V.sub(2).dofmap().set(nullspace_basis[2], 1.0);

    # Build rotational null space basis
    V.sub(0).set_x(nullspace_basis[3], -1.0, 1);
    V.sub(1).set_x(nullspace_basis[3],  1.0, 0);
    V.sub(0).set_x(nullspace_basis[4],  1.0, 2);
    V.sub(2).set_x(nullspace_basis[4], -1.0, 0);
    V.sub(2).set_x(nullspace_basis[5],  1.0, 1);
    V.sub(1).set_x(nullspace_basis[5], -1.0, 2);

    for x in nullspace_basis:
        x.apply("insert")

    # Create vector space basis and orthogonalize
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    return basis


# Load mesh from file
#mesh = Mesh()
#XDMFFile(MPI.comm_world, "./pulley.xdmf").read(mesh)
#mesh=refine(mesh)
#mesh=refine(mesh)
mesh = Mesh("../level1.xml")
# Function to mark inner surface of pulley
def inner_surface(x, on_boundary):
    r = 3.75 - x[2]*0.17
    return (x[0]*x[0] + x[1]*x[1]) < r*r and on_boundary

# Rotation rate and mass density
omega = 300.0
rho = 10.0

# Loading due to centripetal acceleration (rho*omega^2*x_i)
f = Expression(("rho*omega*omega*x[0]", "rho*omega*omega*x[1]", "0.0"),
               omega=omega, rho=rho, degree=2)

# Elasticity parameters
E = 1.0e9
nu = 0.3
mu = E/(2.0*(1.0 + nu))
lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

# Stress computation
def sigma(v):
    return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(len(v))

# Create function space
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = inner(sigma(u), grad(v))*dx
L = inner(f, v)*dx

# Set up boundary condition on inner surface
c = Constant((0.0, 0.0, 0.0))
bc = DirichletBC(V, c, inner_surface)

# Assemble system, applying boundary conditions and preserving
# symmetry)

A = PETScMatrix()
b = PETScVector()
A, b = assemble_system(a, L, bc, A_tensor=A, b_tensor=b)


# Create solution function
#u = Function(V)

# Create near null space basis (required for smoothed aggregation
# AMG). The solution vector is passed so that it can be copied to
# generate compatible vectors for the nullspace.
#null_space = build_nullspace(V, u.vector())

# Attach near nullspace to matrix
#as_backend_type(A).set_near_nullspace(null_space)

Apt = A.mat()
bpt = b.vec()
#========================================================
# create the transfer operators by petsc gamg
def makepro(At,bt):
    '''the function gives the prolongation list and number of
    levels of amg, At is the matrix and bt is the rhs'''

    bg = bt.copy()
    Ah = At.copy()

    solver = PETScKrylovSolver()
    

    # Choose conjugate gradient method for the Krylov solver.
    PETScOptions.set("ksp_type", "cg")


    # Choose gamg for the preconditioner
    PETScOptions.set("pc_type", "gamg")

    # Set the preconditioner to be PETSc gamg with smoothed aggregation.
    PETScOptions.set("pc_gamg_type", "agg")

    # The number of smoothed aggregation steps. More smooths improve performance of
    # the preconditioner at the cost of memory.
    PETScOptions.set("pc_gamg_agg_nsmooths", 1)

    # The maximum number of levels permitted in the MG preconditioner.
    PETScOptions.set("pc_mg_levels", 5)


    # a more powerful preconditioner, at greater construction and memory cost. A
    # lower threshold is cheaper to construct, but will increase the number of
    # iterations required by the Krylov solver. The PETSc manual suggests a
    # threshold of 0.08 for 3D problems. In Nate's experience, this is far too
    # expensive for large problems beyond 15M DoF.
    PETScOptions.set("pc_gamg_threshold", 0.08)

    PETScOptions.set("ksp_rtol", 70)

    ksp = solver.ksp()
    pc = ksp.getPC()
    pc.setFromOptions()

    ug = Function(V)
    null_space = build_nullspace(V, ug.vector())
    Ah.set_near_nullspace(null_space)
    solver.set_operator(Ah)
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

# find the number of levels and prolongation list of AMG
nl, puse = makepro(A,b)
#eigfind.eigfind(Apt)
#quit()

#================================================================================

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

#============================================================================
def mg(Alist, bh, uh, prolongation, restriction, N_cycles, N_levels, ksptype, pctype):
    '''multigrid for N level mesh
    Ah is the matrix, bh is rhs on the finest grid,
    uh is the initial guess for finest grid
    prolongation is a list containing all of operators from fine-to-coarse
    N_cycles is number of cycles and N_levels is number of levels'''
    Ah = Alist[0]
    r0 = residual(Ah, bh, uh)

    # make a restriction list and gird operator list and rhs list
    # and initial guess list
    # initialize the first entity
    blist = [None] * N_levels
    uhlist = [None] * N_levels
    blist[0] = bh
    uhlist[0] = uh

    # calculate the restriction, matrix, and initial guess lists
    # except coarsest grid, since transfer operator and initial guess
    # is not defined on that level
    for i_level in range(1, N_levels - 1):
        uhlist[i_level] = restriction[i_level - 1] * uhlist[i_level - 1]

    for num_cycle in range(N_cycles):

        # restriction to coarse grids
        for i in range(N_levels - 1):

            smoother(Alist[i], blist[i], 2, uhlist[i], ksptype, pctype)
            #print(blist[i].getSize())
            # obtain the rhs for next level
            res = blist[i] - Alist[i] * uhlist[i]
            blist[i + 1] = restriction[i] * res

        # on the coarsest grid, apply direct lu
        uhlist[N_levels - 1] = direct(Alist[N_levels - 1], blist[N_levels - 1])
        #print(blist[N_levels - 1].getSize())
        # prolongation back to fine grids
        for j in range(N_levels - 2, -1, -1):

            uhlist[j] += prolongation[j] * uhlist[j + 1]
            
            smoother(Alist[j], blist[j], 2, uhlist[j], ksptype, pctype)
                #if j==1:
            #    print(residual(Alist[j],blist[j],uhlist[j]))
        # calculate the relative residual
        res4 = residual(Ah, bh, uhlist[0]) / r0
        #print(res4)

    return uhlist[0]
#================================================================

def mgcg(Alist, bh, igh, Ncg, Nmg):
    '''multigrid preconditioned conjugate gradient
    Ah is the matrix, bh is the rhs, igh is initial guess
    Ncg is the number of iterations of cg, 
    and Nmg is number of cycles for multigrid'''

    # initialize the problem
    Ah = Alist[0]
    r0 = residual(Ah, bh, igh)
    rk = bh - Ah * igh
    wk = igh.copy()
    zk = mg(Alist, rk, wk, puse, ruse, Nmg, nl, 'chebyshev', 'jacobi')
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
        zk2 = mg(Alist, rk2, rt, puse, ruse, Nmg, nl, 'chebyshev', 'jacobi')
        beta = (rk2.dot(zk2)) / (rk.dot(zk))
        y1 = beta * pk
        pk = (zk2 + y1).copy()
        rk = rk2.copy()
        zk = zk2.copy()

        res4 = residual(Ah, bh, xk)
        #print(PETSc.Vec.norm(rk, 2))
        #print('residual after', ite + 1, 'iterations is:', res4/r0)
        print(res4/r0)
    return xk
#=========================================================
igh = as_backend_type(Function(V).vector())
igh[:] = 0.0
fph = igh.vec() 

print('Initial residual is:', residual(Apt, bpt, fph))
#wh = mg(Alst, b, fph, puse, ruse, 30, 4, 'richardson', 'sor')
#print('Final residual is:', residual(A, b, fph))
wh = mgcg(Ause, bpt, fph, 25, 1)
