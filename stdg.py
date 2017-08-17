from firedrake import * 
# import ufl 
import numpy as np 
from triprism.expandFunctionFrom2dTo3d import ExpandFunctionTo3d , ConstantInTime3d
from triprism.extractFunctionFrom3dTo2d import ExtractFunctionTo2d


# FIRST DO :
# source ~/firedrake/bin/activate

# space-time dG formulation 
# based on the extruded meshes implemented in Firedrake 


n = 4
T = 1.0 
dt= 1.0
tLeft = 0.0 # starting point of the current time interval 
tRight = dt # end point of the current time interval 

mesh2d = UnitSquareMesh(n,n)
mesh = ExtrudedMesh(mesh2d, 1, layer_height=dt, extrusion_type='uniform')

x, y, t = SpatialCoordinate(mesh)

p = 1
q = 1
horiz_elt = FiniteElement("CG", triangle, p)
vert_elt  = FiniteElement("DG", interval, q)
elt = TensorProductElement(horiz_elt, vert_elt)
# space and space-time discrete space 
V_s = FunctionSpace( mesh2d, horiz_elt )
V = FunctionSpace(mesh, elt)

u = TrialFunction(V)
v = TestFunction(V)

f = Function(V)
uExact = Function(V)
# d/dt u - \lapl u 
# TODO: should be f an expression - how can it be updated in time stepping?
k = 10.0
f.interpolate( k*x*(1.-x)*y*(1.-y) + \
               2.*(1.+k*(t + tLeft))*( x*(1.-x) + y*(1.-y) ) )
uExact.interpolate( (1 + k*(t+tLeft)) * x*(1.-x) * y*(1.-y)  )

# f = Constant( 1.0 )
# uExact.interpolate( t  )



# initial condition - has to be updated after each time step
ic = Function(V) 
ic.assign(  uExact )

# Dirichlet boundary conditions have to be defined on the sideways only
bcval = uExact 
bc1 = DirichletBC(V, bcval, 1)
bc2 = DirichletBC(V, bcval, 2) 
bc3 = DirichletBC(V, bcval, 3) 
bc4 = DirichletBC(V, bcval, 4) 


#bcval.assign(sin(2*pi*5*t))
# FEM in space and DG in time
# i dont want grad but only dx(0:1)
a =  u.dx(2)*v*dx + ( u.dx(0)*v.dx(0)+u.dx(1)*v.dx(1) )*dx + u*v*ds_b
L = f*v*dx + ic*v*ds_b


## Finally we solve the equation. We redefine `u` to be a function
## holding the solution:: 
u = Function(V)
i = 0

while tRight <= T: 
    i =i+1
    solve(a == L, u, bcs = [bc1,bc2,bc3,bc4], \
         solver_parameters={'ksp_type': 'gmres'})


    print('L2 error: ' , sqrt(assemble(dot(u - uExact, u - uExact) * dx)))
    print('Max error: {}'.format(max(abs(u.dat.data - uExact.dat.data))))
    tLeft += dt; tRight += dt 
    output = "sol_" + str(i) + ".pvd"
    print(output)
    File(str(output)).write(u) 

    
    # now it depends on time: 
    #we need extrapolate u(t_{m-1}) to next interval constantly in time
#     ex = ConstantInTime3d( u, ic, surface = "top" )
#     ex.solve()
    initialCond = "ic_" + str(i) + ".pvd"
    File(initialCond).write(ic)
    
    
    
    
    
    
    
    