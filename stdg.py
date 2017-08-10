from firedrake import *

# FIRST DO :
# source ~/firedrake/bin/activate

# space-time dG formulation 
# based on the extruded meshes implemented in Firedrake 


n = 8
T = 1.0 
dt= 0.5
tLeft = 0.0 # starting point of the current time interval 
tRight = dt # end point of the current time interval 

mesh2d = UnitSquareMesh(n,n)
mesh = ExtrudedMesh(mesh2d, 1, layer_height=dt, extrusion_type='uniform')
# extruded mesh -> how to move in time from (0,dt) to (t_m-1 , t_m)

x, y, t = SpatialCoordinate(mesh)

p = 1
q = 1
horiz_elt = FiniteElement("CG", triangle, p)
vert_elt  = FiniteElement("DG", interval, q)
elt = TensorProductElement(horiz_elt, vert_elt)
V = FunctionSpace(mesh, elt)

u = TrialFunction(V)
v = TestFunction(V)

## We declare a function over our function space and give it the
## value of our right hand side function::

f = Function(V)
# d/dt u - \lapl u 
# TODO: should be f an expression - how can it be updated in time stepping?
f.interpolate( 100.*x*(1.-x)*y*(1.-y) + \
               2.*(1.+100*(t + tLeft))*( x*(1.-x) + y*(1.-y) ) )

gg = Expression(V)

#print gg

#inflow = conditional(And(z < 0.02, x > 0.5), 1.0, -1.0)
#q_in = Function(V)
#q_in.interpolate(inflow)
## We can now define the bilinear and linear forms for the left and right
## hand sides of our equation respectively::

# exact solution
uExact = Function(V) 
uExact.interpolate( (1 + 100*(t+tLeft)) * x*(1.-x) * y*(1.-y)  )

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
   
#   print 'L2 error: ' , sqrt(assemble(dot(u - uExact, u - uExact) * dx))
   tLeft += dt; tRight += dt 

   # we need to assign u(t_{m-1}^-) -> ic(t_{m-1}^+
   # TODO: This is wrong - just testing time stepping
   ic.assign( u ) 
   # now it depends on time: 
   #we need extrapolate u(t_{m-1}) to next interval constantly in time
   
## For more details on how to specify solver parameters, see the section
## of the manual on :doc:`solving PDEs <../solving-interface>`.
##
## Next, we might want to look at the result, so we output our solution
## to a file::
File("stdg.pvd").write(u)


## Alternatively, since we have an analytic solution, we can check the
## :math:`L_2` norm of the error in the solution::

#print 'L2 error: ' , sqrt(assemble(dot(u - uExact, u - uExact) * dx))




#xx = Point( 0.5,0.5, 1.0) 

#print u( (0.5,0.5,0.1) ) , u( (0.5,0.5,0.0) )
