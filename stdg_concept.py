from firedrake import *
# FIRST TO DO :
# source ~/firedrake/bin/activate

# space-time dG formulation 
# based on the extruded meshes implemented in Firedrake 

n = 2
fineN = n*4
T = 0.1
dt= 0.1
tLeft = 0.0 # starting point of the current time interval 
tRight = dt # end point of the current time interval 

mesh2d = UnitSquareMesh(n,n)
mesh = ExtrudedMesh(mesh2d, 1, layer_height=dt, extrusion_type='uniform')

mesh2dFine = UnitSquareMesh(fineN,fineN)
meshFine = ExtrudedMesh(mesh2dFine, 4, layer_height=dt/4.0, extrusion_type='uniform')

x, y, t = SpatialCoordinate(mesh)

p = 1
q = 1
horiz_elt = FiniteElement("CG", triangle, p)
vert_elt  = FiniteElement("DG", interval, q)
elt = TensorProductElement(horiz_elt, vert_elt)
V = FunctionSpace(mesh, elt)
fineV = FunctionSpace(meshFine, elt)

u = TrialFunction(V)
v = TestFunction(V)

## We declare a function over our function space and give it the
## value of our right hand side function::
f = Function(V)
k = 100.0
# d/dt u - \lapl u 
# TODO: should be f an expression - how can it be updated in time stepping?
f.interpolate( k*x*(1.-x)*y*(1.-y) + \
               2.*(1.+k*(t + tLeft))*( x*(1.-x) + y*(1.-y) ) )


# exact solution
uExact = Function(fineV) 
fineU = Function(fineV)

uExact.interpolate( (1 + k*(t+tLeft)) * x*(1.-x) * y*(1.-y)  )

#uExact.interpolate( x*(1.-x) * y*(1.-y)  )

# initial condition - has to be updated after each time step
ic = Function(V) 
ic.interpolate(  uExact )

# Dirichlet boundary conditions have to be defined on the sideways only
bcval = Constant(0.0)
bc1 = DirichletBC(V, bcval, 1)
bc2 = DirichletBC(V, bcval, 2) 
bc3 = DirichletBC(V, bcval, 3) 
bc4 = DirichletBC(V, bcval, 4) 

#bcval.assign(sin(2*pi*5*t))
# FEM in space and DG in time
a = ( u.dx(2)*v + dot(grad(v), grad(u)) ) * dx + u*v*ds_b
L = f*v*dx + ic*v*ds_b

## Finally we solve the equation. We redefine `u` to be a function
## holding the solution:: 
u = Function(V)


## Since we know that the Helmholtz equation is
## symmetric, we instruct PETSc to employ the conjugate gradient method::
while tRight <= T: 
   solve(a == L, u, bcs = [bc1,bc2,bc3,bc4], \
         solver_parameters={'ksp_type': 'gmres'})

   fineU = project( u, fineV) 
   print 'L2 error: ' , sqrt(assemble(dot(fineU - uExact, fineU - uExact) * dx))
   tLeft += dt; tRight += dt 
   File("stdgConcept.pvd").write(u)
   quit()

   # we need to assign u(t_{m-1}^-) -> ic(t_{m-1}^+
   # TODO: This is wrong - just testing time stepping
   #ic.assign( u ) 
   
   # now it depends on time: 
   #we need extrapolate u(t_{m-1}) to next interval constantly in time




