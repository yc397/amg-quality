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
mesh = Mesh("../level1_bad.xml")
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
print(mu)
print(lmbda)
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
print(bpt.getSize())
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
    #PETScOptions.set("pc_gamg_coarse_eq_limit", 1000)
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

#==============================================================================================

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
#=====================================================
Bfinemore=[[106496, 53346, 53347, 53348, 53349, 53350, 53351, 53352, 53353, 53354, 53358, 53359, 53360, 53361, 53362, 53363, 53364, 53365, 53366, 53367, 53368, 53369, 53370, 53371, 53372, 53376, 53377, 53378, 53379, 53380, 53381, 53382, 53383, 53384, 53385, 53386, 53387, 53388, 53389, 53390, 84273, 84274, 84275, 106872, 106873, 106874, 106875, 106876, 106877, 106878, 106879, 106880, 211752, 211753, 211754, 106965, 106966, 106967, 195060, 195061, 195062, 195066, 195067, 195068, 195069, 195070, 195071, 59898, 59899, 59900, 195090, 195091, 195092, 195093, 195094, 195095, 186924, 186925, 186926, 186927, 186928, 186929, 186930, 186931, 186932, 186945, 186946, 186947, 186948, 186949, 186950, 186951, 186952, 186953, 59991, 59992, 59993, 59994, 59995, 59996, 59997, 59998, 59999, 187005, 187006, 187007, 187008, 187009, 187010, 187011, 187012, 187013, 187014, 187015, 187016, 187017, 187018, 187019, 195219, 195220, 195221, 178839, 178840, 195225, 195226, 195227, 178842, 178843, 195228, 195229, 178845, 195230, 178846, 195232, 195233, 195234, 195235, 195239, 195240, 195241, 195242, 195243, 195244, 195245, 195237, 195238, 178855, 178856, 178872, 178873, 178874, 178875, 178876, 178877, 170688, 170689, 170690, 170691, 170692, 170693, 195270, 195271, 195272, 178841, 178844, 178847, 203505, 203506, 203507, 178848, 203514, 203515, 203516, 203517, 203518, 203519, 203520, 203521, 203522, 203523, 203524, 203525, 203526, 203527, 203528, 203529, 203530, 203531, 203532, 203533, 203534, 203535, 203536, 203537, 178852, 178853, 178854, 195357, 195358, 195359, 195360, 195361, 187170, 187171, 187172, 187167, 187168, 187169, 107298, 107299, 187178, 107300, 107301, 107302, 107303, 107304, 107305, 107306, 107307, 107308, 107309, 187180, 187181, 195381, 195382, 195383, 195384, 195385, 195386, 178998, 178999, 179000, 179001, 179002, 179003, 60234, 60235, 60236, 60237, 60238, 60239, 60240, 60241, 60242, 60243, 60244, 60245, 60246, 60247, 60248, 60249, 60250, 60251, 60252, 60253, 60254, 60255, 60256, 60257, 60258, 60259, 60260, 60261, 60262, 60263, 60264, 60265, 60266, 60267, 60268, 60269, 60270, 60271, 60272, 60273, 60274, 60275, 60276, 60277, 60278, 203649, 203650, 203651, 203652, 203653, 203654, 203655, 203656, 203657, 203658, 203659, 203660, 60303, 60304, 60305, 60306, 60307, 60308, 60309, 60310, 60311, 93081, 93082, 93083, 99105, 99106, 99107, 93135, 93136, 93137, 170961, 170962, 170964, 170965, 170966, 170963, 170967, 170968, 170969, 93153, 93154, 93155, 93156, 93157, 93158, 93159, 93160, 93161, 93162, 93163, 93164, 211941, 211942, 211943, 93177, 93178, 195579, 195580, 195581, 195582, 195583, 195584, 93179, 93180, 93181, 93182, 195591, 195592, 195593, 60426, 60427, 60428, 46188, 46189, 46190, 46191, 46192, 46193, 46194, 46195, 46196, 187581, 187582, 187583, 187587, 187588, 187589, 68826, 68827, 68828, 68832, 68833, 68834, 195231, 179448, 179449, 179450, 179451, 179452, 179453, 195236, 179466, 179467, 179468, 107799, 107800, 107801, 107802, 107803, 107804, 46365, 46366, 46367, 107862, 107863, 107864, 107877, 107878, 107879, 107880, 107881, 107882, 107883, 107884, 107885, 52599, 52600, 52601, 52602, 52603, 52604, 52605, 52606, 52607, 52608, 52609, 52610, 52611, 52612, 52613, 52614, 52615, 52616, 178992, 178993, 178994, 178995, 178996, 178997, 52743, 52744, 52745, 52746, 52747, 52748, 52749, 52750, 52751, 99942, 99943, 99944, 99945, 99946, 99947, 99948, 99949, 99950, 99951, 99952, 99953, 99954, 99955, 99956, 99993, 99994, 99995, 99996, 99997, 99998, 99999, 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008, 100009, 100010, 187176, 187177, 100011, 100012, 100013, 187179, 91830, 91831, 91832, 91833, 91834, 91835, 91836, 91837, 91838, 91839, 91840, 91841, 91842, 91843, 91844, 91845, 91846, 91847, 100041, 100042, 100043, 100044, 100045, 100046, 100047, 100048, 100049, 100050, 100051, 100052, 100056, 100057, 100058, 100065, 100066, 100067, 100077, 100078, 100079, 100080, 100081, 100082, 100083, 100084, 100085, 100086, 100087, 100088, 46944, 46945, 46946, 195362, 178849, 178850, 188352, 188353, 188354, 178851, 106488, 106489, 106490, 106491, 106492, 106493, 106494, 106495]]
#b1=PETSc.IS()
b1.createGeneral(Bfinemore[0])


B2=[[10240, 10241, 4110, 4111, 4112, 4113, 4114, 4115, 4116, 4117, 4118, 4119, 4120, 4121, 20508, 20509, 20510, 20511, 20512, 20513, 20514, 20515, 20516, 20517, 20518, 20519, 20520, 20521, 20522, 20523, 20524, 20525, 4140, 4141, 4142, 4143, 4144, 4145, 20544, 20545, 20546, 20547, 20548, 20549, 8316, 8317, 8318, 8319, 8320, 8321, 8322, 8323, 8324, 8325, 8326, 8327, 4248, 4249, 4250, 4251, 4252, 4253, 8184, 8185, 8186, 8187, 8188, 8189, 14652, 14653, 14654, 14655, 14656, 14657, 14664, 14665, 14666, 14667, 14668, 14669, 16740, 16741, 16742, 16743, 16744, 16745, 16746, 16747, 16748, 16749, 16750, 16751, 16752, 16753, 16754, 16755, 16756, 16757, 16758, 16759, 16760, 16761, 16762, 16763, 16764, 16765, 16766, 16767, 16768, 16769, 16776, 16777, 16778, 16779, 16780, 16781, 16782, 16783, 16784, 16785, 16786, 16787, 16788, 16789, 16790, 16791, 16792, 16793, 10698, 10699, 10700, 10701, 10702, 10703, 4608, 4609, 4610, 4611, 4612, 4613, 4626, 4627, 4628, 4629, 4630, 4631, 6732, 6733, 4686, 4687, 4688, 4689, 4690, 4691, 4692, 4693, 4694, 4695, 19032, 19033, 19034, 19035, 4696, 4697, 6750, 19036, 6751, 6752, 6753, 6754, 19044, 19045, 19046, 19047, 19048, 19049, 6755, 6756, 6757, 6758, 6759, 6760, 6761, 4722, 4723, 4724, 4725, 4726, 4727, 8874, 8875, 8876, 8877, 8878, 8879, 6846, 6847, 6848, 6849, 6850, 6851, 8934, 8935, 8936, 8937, 8938, 8939, 8940, 8941, 8942, 8943, 8944, 8945, 8946, 8947, 8948, 8949, 8950, 8951, 8952, 8953, 8954, 8955, 8956, 8957, 8958, 8959, 8960, 8961, 8962, 8963, 7482, 7483, 7484, 9522, 9523, 9524, 9525, 9526, 9527, 3036, 3037, 3038, 3039, 3040, 3041, 15336, 15337, 15338, 15339, 15340, 15341, 15354, 15355, 15356, 15357, 15358, 15359, 17472, 17473, 17474, 17475, 17476, 17477, 17478, 17479, 17480, 17481, 17482, 17483, 17484, 17485, 17486, 17487, 17488, 17489, 17490, 17491, 17492, 17493, 17494, 17495, 17496, 17497, 17498, 6734, 17499, 17500, 6735, 17502, 17504, 17505, 17506, 17507, 17501, 6736, 17503, 6737, 17514, 17515, 15468, 15469, 15470, 15471, 15472, 15473, 17516, 17517, 17518, 17519, 17580, 17581, 17582, 17583, 17584, 17585, 5334, 5335, 5336, 5337, 5338, 5339, 5340, 5341, 5342, 5343, 5344, 5345, 5346, 5347, 5348, 5349, 5350, 5351, 5376, 5377, 5378, 5379, 5380, 5381, 7470, 7471, 7472, 7473, 7474, 7475, 7476, 7477, 7478, 7479, 9528, 9529, 9530, 9531, 9532, 9533, 7480, 7481, 19776, 19777, 19778, 19779, 19780, 19781, 7485, 7486, 7487, 9546, 9547, 9548, 9549, 9550, 9551, 19830, 19831, 19832, 19833, 19834, 19835, 9594, 9595, 9596, 9597, 9598, 9599, 5982, 5983, 5984, 9636, 9637, 9638, 9639, 9640, 9641, 3516, 3517, 3518, 3519, 3520, 3521, 19037, 3636, 3637, 3638, 3639, 3640, 3641, 13950, 13951, 13952, 13953, 13954, 13955, 16074, 16075, 16076, 16077, 16078, 16079, 16092, 16093, 16094, 16095, 16096, 16097, 16098, 16099, 16100, 16101, 16102, 16103, 5922, 5923, 5924, 5925, 5926, 5927, 18258, 18259, 18260, 18261, 18262, 18263, 18264, 18265, 18266, 18267, 18268, 18269, 18270, 18271, 18272, 18273, 18274, 18275, 18276, 18277, 18278, 18279, 18280, 18281, 5985, 5986, 5987, 18288, 18289, 18290, 18291, 18292, 18293, 18294, 18295, 18296, 18297, 18298, 18299, 18324, 18325, 18326, 18327, 18328, 18329, 18330, 18331, 10140, 10141, 10142, 10143, 10144, 10145, 18332, 18333, 18334, 18335, 10158, 10159, 10160, 10161, 10162, 10163, 18366, 18367, 18368, 18369, 18370, 18371, 10188, 10189, 10190, 10191, 10192, 10193, 10206, 10207, 10208, 10209, 10210, 10211, 10212, 10213, 10214, 10215, 10216, 10217, 10224, 10225, 10226, 10227, 10228, 10229, 10230, 10231, 10232, 10233, 10234, 10235, 10236, 10237, 10238, 10239]]
b2=PETSc.IS()
b2.createGeneral(B2[0])

B3=[[]]
b3=PETSc.IS()
b3.createGeneral(B3[0])
B4=[[]]
b4=PETSc.IS()
b4.createGeneral(B4[0])
B5=[[]]
b5=PETSc.IS()
b5.createGeneral(B5[0])
B6=[[]]
b6=PETSc.IS()
b6.createGeneral(B6[0])
Bpt=[b1,b2,b3,b4,b5,b6]
Blist=[Bfinemore,B2,B3,B4,B5,B6]

def makecorrection(Ahlist, Bpet):
    nl = len(Ahlist) - 1
    Acorlist = [None] * nl
    for k in range(nl):
        Ah = Ahlist[k]
        btt=Bpet[k]
        Abad=Mat()
        Ah.createSubMatrix(btt,btt,Abad)
        Abad.assemblyBegin()
        Abad.assemblyEnd()
        Acorlist[k] = Abad.copy()

    return Acorlist

def local_correction(Ah, bh, ugh, Acc, btt, bset):
    '''
    Local correction smoother. Ah is the whole matrix,
    bh is the whole rhs, ugh is the initial guess or input.
    Also need a vector B which contains the indices of all bad nodes.
    fixa is the submatrix, fixr is the corresponding residual part and
    fixu is the error obtained.
    '''
    rh = bh - Ah * ugh
    bcc = Vec()
    #ucc = Vec()
    rh.getSubVector(btt,bcc)
    nb = bcc.getSize()
    #ugh.getSubVector(btt,ucc)
    #ucc = cg(Acc,bcc,10)
    ucc = direct(Acc,bcc)
    for i in range(nb):
        row = bset[i]
        ugh[row] += ucc.getValue(i)

    return ugh

Aclist = makecorrection(Ause, Bpt)

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

            # apply smoother to every level except the coarsest level
            local_correction(Alist[i], blist[i], uhlist[i], Aclist[i],Bpt[i],Blist[i][0])
            smoother(Alist[i], blist[i], 1, uhlist[i], ksptype, pctype)
            local_correction(Alist[i], blist[i], uhlist[i], Aclist[i],Bpt[i],Blist[i][0])
            smoother(Alist[i], blist[i], 1, uhlist[i], ksptype, pctype)
            local_correction(Alist[i], blist[i], uhlist[i], Aclist[i],Bpt[i],Blist[i][0])
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
            local_correction(Alist[j], blist[j], uhlist[j], Aclist[j],Bpt[j],Blist[j][0])
            smoother(Alist[j], blist[j], 1, uhlist[j], ksptype, pctype)
            local_correction(Alist[j], blist[j], uhlist[j], Aclist[j],Bpt[j],Blist[j][0])
            smoother(Alist[j], blist[j], 1, uhlist[j], ksptype, pctype)
            local_correction(Alist[j], blist[j], uhlist[j], Aclist[j],Bpt[j],Blist[j][0])
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
    zk = mg(Alist, rk, wk, puse, ruse, Nmg, nl, 'richardson', 'sor')
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
        zk2 = mg(Alist, rk2, rt, puse, ruse, Nmg, nl, 'richardson', 'sor')
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
