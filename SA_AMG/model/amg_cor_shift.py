from dolfin import *
import numpy as np
from petsc4py import PETSc
from petsc4py.PETSc import Mat
from petsc4py.PETSc import Vec
import pybind11
import eigfind
import matrix_split

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
nl, pnone = makepro(A,b)
# eigfind.eigfind(Apt)
# quit()

#======================================
def makenew(Att,phere,emaxhere):
    nhere=Att.getSize()
    Dt=Vec()
    Att.getDiagonal(Dt)
    Dt.reciprocal()
    Dpt=PETSc.Mat().createAIJ([nhere, nhere])
    Dpt.setUp()
    Dpt.setDiagonal(Dt)
    Dpt.assemblyBegin()
    Dpt.assemblyEnd()
    omega=-4.0/3.0/emaxhere
    second=omega*Dpt*Att*phere
    tmat=phere+second
    return tmat

Ahere=Apt.copy()
bhere=bpt.copy()
fguess=bpt.copy()

puse=[]
#first one
#emax1=1.973297
#emax1=2.968916 
#calc
# B1=[3241, 3748, 3751, 3496, 3753, 3242, 3243, 3244, 3245, 3246, 3500, 3501, 3502, 3503, 3504, 3505, 3506, 3763, 3764, 3511, 3762, 3761, 3509, 3253, 3508, 3255, 3752, 3497, 3498, 3247, 3239, 3757, 3758, 3251, 3252, 3507, 3059, 3318, 3510, 3765]
# B1.sort()
# matrix_split.matrix_split(Ahere,B1)
# quit()
emax1 = 1.973094
ptt=pnone[0].copy()
pnew=makenew(Ahere,ptt,emax1)
puse.append(pnew)

#second one
A2=Mat()
Ahere.PtAP(pnew, A2)
#eigfind.eigfind(A2)
# B2 = [471, 408, 582, 583, 584, 585, 631, 455, 431, 498, 434, 500, 469, 470, 501, 499, 474, 539, 540, 541]
# B2.sort()
# matrix_split.matrix_split(A2,B2)
# quit()
ptt2=pnone[1].copy()
#emax2=1.505117
#emax2=1.559350
# clac
emax2 = 1.504936
pnew2=makenew(A2,ptt2,emax2)
puse.append(pnew2)

#third one
A3=Mat()
A2.PtAP(pnew2,A3)
#eigfind.eigfind(A3)
# B3=[52, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 79, 80, 81, 82, 84, 85, 86, 87, 91, 96]
# B3.sort()
# matrix_split.matrix_split(A3,B3)
#quit()
ptt3=pnone[2].copy()
#geometry
#emax3= 2.239345
#emax3=2.240673
#calc
#emax3=2.240648
#default
emax3 = 2.236823
pnew3=makenew(A3,ptt3,emax3)
puse.append(pnew3)

#fourth one
A4=Mat()
A3.PtAP(pnew3,A4)
#eigfind.eigfind(A4)
#quit()
#==============================================
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
#===========================================================
Bfinemore=[[3241, 3748, 3751, 3496, 3753, 3242, 3243, 3244, 3245, 3246, 3500, 3501, 3502, 3503, 3504, 3505, 3506, 3763, 3764, 3511, 3762, 3761, 3509, 3253, 3508, 3255, 3752, 3497, 3498, 3247, 3239, 3757, 3758, 3251, 3252, 3507, 3059, 3318, 3510, 3765]]
#Bfinemore=[[]]
Bfine=[[583, 455, 585, 431, 434, 500, 501, 470, 540, 541]]
#=[[]]
#Bmedium=[[52, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 79, 80, 81, 82, 84, 85, 86, 87, 91, 96]]
Bmedium=[[]]
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
            local_correction(Alist[j], blist[j], uhlist[j], Blist[j])
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
quit()
rei=Function(V)
#reh=bpt-Apt*wh

rei.vector()[:]=(wh).getArray()

file_p=File('./bad_all_it10.pvd')
file_p << rei