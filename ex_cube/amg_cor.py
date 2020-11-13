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
#mesh = UnitCubeMesh(50, 50, 50)
mesh=Mesh("./level1_bad.xml")
V = FunctionSpace(mesh, 'P', 1)
#n = mesh.num_vertices()

# Use FEniCS to formulate the FEM problem. A is the matrix, b is the rhs.
u_D = Expression('0.0', degree=0)


# Define boundary for DirichletBC
def boundary_L(x, on_boundary):
    tol = 1E-14
    return on_boundary and (near(x[0], 0, tol) or near(x[0], 1, tol) or near(x[2], 0, tol) or near(x[2], 1, tol))

bc = DirichletBC(V, u_D, boundary_L)
u = TrialFunction(V)
v = TestFunction(V)
#f = Constant(1.0)
f = Expression('3*pi*pi*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])',degree=6)
g = Expression('pi*sin(pi*x[0])*sin(pi*x[2])',degree=6)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx - g * v * ds
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

# find the number of levels and prolongation list of AMG
nl, puse = makepro(A,b)
#=========================================================
Bfinemore=[[23041, 23042, 514, 31238, 31239, 31240, 31241, 41482, 31243, 36876, 36877, 41484, 36879, 41485, 38434, 38435, 38436, 38437, 557, 26161, 26162, 26163, 26164, 26165, 26166, 26167, 26168, 26169, 36930, 36931, 36933, 36934, 36935, 36424, 36425, 36426, 36427, 36936, 36428, 36423, 36429, 36938, 593, 594, 595, 592, 596, 598, 599, 597, 600, 36443, 36444, 603, 1119, 1120, 610, 34427, 34429, 34432, 34433, 34438, 34440, 34441, 37007, 37018, 37019, 668, 669, 672, 200, 201, 38612, 38616, 38617, 38618, 38619, 38620, 38621, 38622, 38623, 733, 736, 40162, 734, 735, 732, 226, 40167, 40166, 40169, 40168, 40173, 40174, 40175, 40176, 40177, 26362, 26365, 26366, 26368, 26369, 1292, 30495, 30496, 38697, 38699, 26425, 34618, 26426, 26428, 26427, 28152, 40161, 28153, 28154, 342, 343, 344, 38745, 38746, 38747, 38748, 38749, 347, 346, 345, 880, 34686, 908, 28047, 28049, 28050, 28051, 28052, 28053, 28054, 28055, 920, 921, 922, 923, 919, 19883, 950, 951, 440, 441, 442, 22982, 22983, 22984, 22985, 22986, 999, 1000, 1001, 28142, 28143, 28144, 28145, 28146, 28147, 31220, 31221, 31222, 28151, 28150, 28149, 28148, 28155]]
#finemore has 171
Bfine=[[20, 21, 5654, 23, 31, 32, 33, 34, 39, 6702, 6703, 6704, 6705, 6706, 50, 6709, 6198, 54, 56, 59, 60, 71, 72, 73, 83, 7254, 7255, 86, 87, 92, 7261, 94, 7260, 98, 100, 101, 102, 105, 106, 107, 117, 118, 119, 120, 130, 6797, 6798, 6799, 144, 146, 147, 5268, 149, 5270, 5271, 152, 6809, 6810, 6811, 153, 150, 7329, 5281, 7331, 7332, 7333, 5286, 5287, 7335, 7334, 5285, 164, 6827, 6829, 6830, 6831, 5288, 181, 182, 189, 5826, 3788, 3789, 3790, 205, 3791, 3793, 6355, 6356, 6357, 6358, 6359, 6360, 6868, 6366, 226, 227, 148, 233, 6394, 253, 6399, 6400, 5911, 5912, 5913, 5915, 5917, 4385, 3877, 4390, 4391, 4392, 4393, 4394, 4389, 4403, 165, 4942, 4943, 4945, 4946, 4947, 7506, 7507, 4950, 4951, 7508, 4953, 6490, 7523, 7524, 7525, 7526, 7527, 7542, 7031, 6016, 7041, 7042, 7043, 5267, 7059, 7060, 7079, 7080, 7081, 6056, 7082, 7093, 7094, 7103, 3556, 3557, 3559, 7145, 7147]]
#fine has 157
Bmedium=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 537, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 1061, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 518, 552, 50, 51, 519, 53, 54, 49, 568, 57, 569, 59, 58, 61, 520, 63, 64, 65, 66, 578, 68, 69, 71, 72, 73, 75, 78, 595, 596, 87, 601, 603, 92, 605, 604, 95, 607, 1036, 93, 100, 614, 103, 530, 109, 628, 121, 536, 650, 649, 654, 655, 661, 665, 669, 1049, 1050, 685, 687, 688, 1054, 1055, 719, 724, 727, 730, 731, 737, 738, 743, 757, 772, 773, 783, 784, 785, 786, 787, 788, 791, 794, 796, 803, 804, 295, 296, 811, 812, 813, 301, 60, 816, 305, 819, 330, 333, 855, 856, 857, 858, 859, 350, 351, 864, 863, 873, 877, 878, 882, 378, 379, 381, 382, 384, 896, 898, 899, 897, 904, 906, 907, 910, 399, 400, 401, 914, 915, 916, 917, 912, 403, 921, 937, 426, 425, 938, 944, 945, 947, 436, 949, 438, 441, 956, 446, 970, 971, 972, 973, 470, 983, 984, 985, 986, 987, 989, 990, 992, 1007, 496, 497, 1010, 1011, 1013, 504, 1022, 1023]]
#medium has 214
Blist=[Bfinemore,Bfine,Bmedium]

def local_correction(Ah, bh, ugh, Buse):
    '''
    Local correction smoother. Ah is the whole matrix,
    bh is the whole rhs, ugh is the initial guess or input.
    Also need a vector B which contains the indices of all bad nodes.
    fixa is the submatrix, fixr is the corresponding residual part and
    fixu is the error obtained.
    '''
    for s in range(len(Buse)):
        B=Buse[s]
        nt = len(B)
        fixa = np.zeros((nt, nt))
        fixr = np.zeros(nt)
        rh = bh - Ah * ugh
        for i in range(nt):
            row = B[i]
            fixr[i] = rh[row]
            for j in range(nt):
                col = B[j]
                fixa[i, j] = Ah[row, col]

        fixu = np.linalg.solve(fixa, fixr)
        for i in range(nt):
            row = B[i]
            ugh[row] += fixu[i]

    return ugh
#==============================================================

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
            local_correction(Alist[i], blist[i], uhlist[i], Blist[i])
            smoother(Alist[i], blist[i], 1, uhlist[i], ksptype, pctype)
            local_correction(Alist[i], blist[i], uhlist[i], Blist[i])
            smoother(Alist[i], blist[i], 1, uhlist[i], ksptype, pctype)
            local_correction(Alist[i], blist[i], uhlist[i], Blist[i])
            
            # obtain the rhs for next level
            res = blist[i] - Alist[i] * uhlist[i]
            blist[i + 1] = restriction[i] * res

        # on the coarsest grid, apply direct lu
        uhlist[N_levels - 1] = direct(Alist[N_levels - 1], blist[N_levels - 1])

        # prolongation back to fine grids
        for j in range(N_levels - 2, -1, -1):

            uhlist[j] += prolongation[j] * uhlist[j + 1]
            local_correction(Alist[i], blist[i], uhlist[i], Blist[i])
            smoother(Alist[j], blist[j], 1, uhlist[j], ksptype, pctype)
            local_correction(Alist[j], blist[j], uhlist[j], Blist[j])
            smoother(Alist[j], blist[j], 1, uhlist[j], ksptype, pctype)
            local_correction(Alist[j], blist[j], uhlist[j], Blist[j])
            
        # calculate the relative residual
        res4 = residual(Ah, bh, uhlist[0]) / r0
        #print('relative residual after', num_cycle + 1, 'cycles is:', res4)
        print(res4)
    return uhlist[0]



# implement multigrid
print('Initial residual is:', residual(Apt, bpt, fph))
#wh = mgcg(Apt, bpt, fph, 10, 1)
wh = mg(Apt, bpt, fph, puse, 20, nl, 'richardson', 'sor')