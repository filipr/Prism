# FIRST TO DO :
# source ~/firedrake/bin/activate

# space-time dG formulation 
# based on the extruded meshes implemented in Firedrake 

from firedrake import * 
import numpy as np 
from expandFunctionFrom2dTo3d import ExpandFunctionTo3d 
from extractFunctionFrom3dTo2d import ExtractFunctionTo2d 

n = 2 # number of elements 2*n^2 in 2d mesh
dt = 0.1 # diameter in vertical (time) direction

mesh2d = UnitSquareMesh(n,n)
mesh = ExtrudedMesh(mesh2d, 1, layer_height=dt, extrusion_type='uniform')

p = 1 # polynomial degree with respect to space
q = 1 # polynomial degree with respect to time

x_s, y_s = SpatialCoordinate(mesh2d)
x, y, t = SpatialCoordinate(mesh) 

horiz_elt = FiniteElement("CG", triangle, p) # space discretization
vert_elt  = FiniteElement("DG", interval, q) # time discretization
# space-time element (triangular prism)
elt = TensorProductElement(horiz_elt, vert_elt) 
# space-time discrete space 
V_s = FunctionSpace( mesh2d, horiz_elt )
V = FunctionSpace(mesh, elt) 


f = Function(V) 
g_s = Function( V_s ) 

g_s.interpolate( 1.5 + x_s*(1.-x_s) * y_s*(1.-y_s)  )
f.interpolate( 10. * t * x*(1-x)*y*(1-y) ) 

g = Function(V)

# from 2d to 3d 
#ex = ExpandFunctionTo3d( g_s, g )
#ex.solve()
#File("g.pvd").write(g)

# from 3d to 2d 
f_s = Function( V_s )
ex = ExtractFunctionTo2d(f, f_s, boundary='bottom', elem_facet='top')
ex.solve()
File("f_s.pvd").write(f_s)





