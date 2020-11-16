from dolfin import *
import numpy as np
from petsc4py import PETSc
from petsc4py.PETSc import Mat
from petsc4py.PETSc import Vec
import pybind11
import eigfind

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
mesh = Mesh("./fine_bad.xml")
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
    #PETScOptions.set("pc_mg_levels", 5)
    PETScOptions.set("pc_gamg_coarse_eq_limit", 1000)
    PETScOptions.set("pc_gamg_sym_graph", True)
    PETScOptions.set("pc_gamg_square_graph", 1)


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
#========================================================


B1=[106496, 53346, 53347, 53348, 53349, 53350, 53351, 53352, 53353, 53354, 53358, 53359, 53360, 53361, 53362, 53363, 53364, 53365, 53366, 53367, 53368, 53369, 53370, 53371, 53372, 53376, 53377, 53378, 53379, 53380, 53381, 53382, 53383, 53384, 53385, 53386, 53387, 53388, 53389, 53390, 84273, 84274, 84275, 106872, 106873, 106874, 106875, 106876, 106877, 106878, 106879, 106880, 211752, 211753, 211754, 106965, 106966, 106967, 195060, 195061, 195062, 195066, 195067, 195068, 195069, 195070, 195071, 59898, 59899, 59900, 195090, 195091, 195092, 195093, 195094, 195095, 186924, 186925, 186926, 186927, 186928, 186929, 186930, 186931, 186932, 186945, 186946, 186947, 186948, 186949, 186950, 186951, 186952, 186953, 59991, 59992, 59993, 59994, 59995, 59996, 59997, 59998, 59999, 187005, 187006, 187007, 187008, 187009, 187010, 187011, 187012, 187013, 187014, 187015, 187016, 187017, 187018, 187019, 195219, 195220, 195221, 178839, 178840, 195225, 195226, 195227, 178842, 178843, 195228, 195229, 178845, 195230, 178846, 195232, 195233, 195234, 195235, 195239, 195240, 195241, 195242, 195243, 195244, 195245, 195237, 195238, 178855, 178856, 178872, 178873, 178874, 178875, 178876, 178877, 170688, 170689, 170690, 170691, 170692, 170693, 195270, 195271, 195272, 178841, 178844, 178847, 203505, 203506, 203507, 178848, 203514, 203515, 203516, 203517, 203518, 203519, 203520, 203521, 203522, 203523, 203524, 203525, 203526, 203527, 203528, 203529, 203530, 203531, 203532, 203533, 203534, 203535, 203536, 203537, 178852, 178853, 178854, 195357, 195358, 195359, 195360, 195361, 187170, 187171, 187172, 187167, 187168, 187169, 107298, 107299, 187178, 107300, 107301, 107302, 107303, 107304, 107305, 107306, 107307, 107308, 107309, 187180, 187181, 195381, 195382, 195383, 195384, 195385, 195386, 178998, 178999, 179000, 179001, 179002, 179003, 60234, 60235, 60236, 60237, 60238, 60239, 60240, 60241, 60242, 60243, 60244, 60245, 60246, 60247, 60248, 60249, 60250, 60251, 60252, 60253, 60254, 60255, 60256, 60257, 60258, 60259, 60260, 60261, 60262, 60263, 60264, 60265, 60266, 60267, 60268, 60269, 60270, 60271, 60272, 60273, 60274, 60275, 60276, 60277, 60278, 203649, 203650, 203651, 203652, 203653, 203654, 203655, 203656, 203657, 203658, 203659, 203660, 60303, 60304, 60305, 60306, 60307, 60308, 60309, 60310, 60311, 93081, 93082, 93083, 99105, 99106, 99107, 93135, 93136, 93137, 170961, 170962, 170964, 170965, 170966, 170963, 170967, 170968, 170969, 93153, 93154, 93155, 93156, 93157, 93158, 93159, 93160, 93161, 93162, 93163, 93164, 211941, 211942, 211943, 93177, 93178, 195579, 195580, 195581, 195582, 195583, 195584, 93179, 93180, 93181, 93182, 195591, 195592, 195593, 60426, 60427, 60428, 46188, 46189, 46190, 46191, 46192, 46193, 46194, 46195, 46196, 187581, 187582, 187583, 187587, 187588, 187589, 68826, 68827, 68828, 68832, 68833, 68834, 195231, 179448, 179449, 179450, 179451, 179452, 179453, 195236, 179466, 179467, 179468, 107799, 107800, 107801, 107802, 107803, 107804, 46365, 46366, 46367, 107862, 107863, 107864, 107877, 107878, 107879, 107880, 107881, 107882, 107883, 107884, 107885, 52599, 52600, 52601, 52602, 52603, 52604, 52605, 52606, 52607, 52608, 52609, 52610, 52611, 52612, 52613, 52614, 52615, 52616, 178992, 178993, 178994, 178995, 178996, 178997, 52743, 52744, 52745, 52746, 52747, 52748, 52749, 52750, 52751, 99942, 99943, 99944, 99945, 99946, 99947, 99948, 99949, 99950, 99951, 99952, 99953, 99954, 99955, 99956, 99993, 99994, 99995, 99996, 99997, 99998, 99999, 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008, 100009, 100010, 187176, 187177, 100011, 100012, 100013, 187179, 91830, 91831, 91832, 91833, 91834, 91835, 91836, 91837, 91838, 91839, 91840, 91841, 91842, 91843, 91844, 91845, 91846, 91847, 100041, 100042, 100043, 100044, 100045, 100046, 100047, 100048, 100049, 100050, 100051, 100052, 100056, 100057, 100058, 100065, 100066, 100067, 100077, 100078, 100079, 100080, 100081, 100082, 100083, 100084, 100085, 100086, 100087, 100088, 46944, 46945, 46946, 195362, 178849, 178850, 188352, 188353, 188354, 178851, 106488, 106489, 106490, 106491, 106492, 106493, 106494, 106495]
ptt=puse[0]

bdd=[]
for bt in B1:
    for jj in range(Apt.size[0]):
        ct=ptt.getValue(bt,jj)
        if ct!=0:
            bdd.append(jj)

bdd=list(set(bdd))
print(bdd)
print(len(bdd))