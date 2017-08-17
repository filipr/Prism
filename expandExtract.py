# FIRST TO DO :
# source ~/firedrake/bin/activate

# space-time dG formulation 
# based on the extruded meshes implemented in Firedrake 

from firedrake import * 
# import ufl 
import numpy as np 
from triprism.expandFunctionFrom2dTo3d import ExpandFunctionTo3d , ConstantInTime3d
from triprism.extractFunctionFrom3dTo2d import ExtractFunctionTo2d


n = 2 # number of elements 2*n^2 in 2d mesh
dt = 1.0 # diameter in vertical (time) direction

mesh2d = UnitSquareMesh(n,n)
mesh = ExtrudedMesh(mesh2d, 1, layer_height=dt, extrusion_type='uniform')

p = 4 # polynomial degree with respect to space
q = 4 # polynomial degree with respect to time

x_s, y_s = SpatialCoordinate(mesh2d)
x, y, t = SpatialCoordinate(mesh) 

horiz_elt = FiniteElement("DG", triangle, p) # space discretization
vert_elt  = FiniteElement("DG", interval, q) # time discretization
# space-time element (triangular prism)
elt = TensorProductElement(horiz_elt, vert_elt) 
# space-time discrete space 
V_s = FunctionSpace( mesh2d, horiz_elt )
V = FunctionSpace(mesh, elt) 


f = Function(V) 
g_s = Function( V_s ) 

g_s.interpolate( 1.5 + x_s*(1.-x_s) * y_s*(1.-y_s)  )
f.interpolate( t * sin(np.pi * x) * cos( np.pi * y ) ) 

g = Function(V)

# from 2d to 3d 
#ex = ExpandFunctionTo3d( g_s, g )
#ex.solve()
#File("g.pvd").write(g)

# from 3d to 2d 
# f_s = Function( V_s )
# ex = ExtractFunctionTo2d(f, f_s, boundary='bottom', elem_facet='top')
# ex.solve()
# File("f_s.pvd").write(f_s)
# 
h = Function(V)


# ex = ConstantInTime3d( f, h, surface = 'bottom' )
# ex.solve()
# # File("f.pvd").write(f)
# File("hBottom.pvd").write(h)

ex = ConstantInTime3d( f, h, surface = 'top' )
ex.solve()
File("hTop.pvd").write(h)




# ex = ConstantFunctionInTime(f, h, boundary='top', elem_facet='top')
# ex.solve()
# File("f.pvd").write(f)
# File("hTop.pvd").write(h)
# 
# ex = ConstantFunctionInTime(f, h, boundary='bottom', elem_facet='bottom')
# ex.solve()
# File("hBottom.pvd").write(h)

# print( 'Space dim f = {}'.format(V.finat_element.space_dimension()) )
# print( 'Space dim g_s= {}'.format(V_s.finat_element.space_dimension()) )
# 
# print( 'Bt mask V:{}'.format( V.bt_masks['geometric'][0] ))
# print( 'Bt mask V top:{}'.format( V.bt_masks['geometric'][1] ))
# print( 'Bt mask V aver:{}'.format( (V.bt_masks['geometric'][0] + V.bt_masks['geometric'][1]) ))
# 
# print('f arity: {}'.format(f.cell_node_map().arity))
# print('g_s arity: {}'.format(g_s.cell_node_map().arity))




















