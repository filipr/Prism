from __future__ import absolute_import
from firedrake import *
import os
import numpy as np
import sys
#from .physical_constants import physical_constants
from pyop2.profiling import timed_region, timed_function, timed_stage  # NOQA
#from mpi4py import MPI  # NOQA
import ufl  # NOQA
#import coffee.base as ast  # NOQA
#from collections import OrderedDict, namedtuple  # NOQA
#from .field_defs import field_metadata
#from .log import *
from firedrake import Function as FiredrakeFunction
from firedrake import Constant as FiredrakeConstant
#from abc import ABCMeta, abstractmethod

__all__ = ['ConstantInTime3d','ExpandFunctionTo3d']

class ExpandFunctionTo3d(object):
    """
    Copy a 2D field to 3D
    Copies a field from 2D mesh to 3D mesh, assigning the same value over the
    vertical dimension. Horizontal function spaces must be the same.
    .. code-block:: python
        U = FunctionSpace(mesh, 'DG', 1)
        U_2d = FunctionSpace(mesh2d, 'DG', 1)
        func2d = Function(U_2d)
        func3d = Function(U)
        ex = ExpandFunctionTo3d(func2d, func3d)
        ex.solve()
    """
    def __init__(self, input_2d, output_3d, elem_height=None):
        """
        :arg input_2d: 2D source field
        :type input_2d: :class:`Function`
        :arg output_3d: 3D target field
        :type output_3d: :class:`Function`
        :kwarg elem_height: scalar :class:`Function` in 3D mesh that defines
            the vertical element size. Needed only in the case of HDiv function
            spaces.
        """
        self.input_2d = input_2d
        self.output_3d = output_3d
        self.fs_2d = self.input_2d.function_space()
        self.fs_3d = self.output_3d.function_space()

        family_2d = self.fs_2d.ufl_element().family()
        ufl_elem = self.fs_3d.ufl_element()
        if isinstance(ufl_elem, ufl.VectorElement):
            # Unwind vector
            ufl_elem = ufl_elem.sub_elements()[0]
        if isinstance(ufl_elem, ufl.HDivElement):
            # RT case
            ufl_elem = ufl_elem._element
        if ufl_elem.family() == 'TensorProductElement':
            # a normal tensorproduct element
            family_3dh = ufl_elem.sub_elements()[0].family()
            if family_2d != family_3dh:
                raise Exception('2D and 3D spaces do not match: {0:s} {1:s}'.format(family_2d, family_3dh))
        if family_2d == 'Raviart-Thomas' and elem_height is None:
            raise Exception('elem_height must be provided for Raviart-Thomas spaces')
        self.do_rt_scaling = family_2d == 'Raviart-Thomas'

        self.iter_domain = op2.ALL

        # number of nodes in vertical direction
        n_vert_nodes = self.fs_3d.finat_element.space_dimension() / self.fs_2d.finat_element.space_dimension()

        nodes = self.fs_3d.bt_masks['geometric'][0]
        self.idx = op2.Global(len(nodes), nodes, dtype=np.int32, name='node_idx')
        self.kernel = op2.Kernel("""
            void my_kernel(double **func, double **func2d, int *idx) {
                for ( int d = 0; d < %(nodes)d; d++ ) {
                    for ( int c = 0; c < %(func_dim)d; c++ ) {
                        for ( int e = 0; e < %(v_nodes)d; e++ ) {
                            func[idx[d]+e][c] = func2d[d][c];
                        }
                    }
                }
            }""" % {'nodes': self.fs_2d.finat_element.space_dimension(),
                    'func_dim': self.input_2d.function_space().value_size,
                    'v_nodes': n_vert_nodes},
            'my_kernel')

        if self.do_rt_scaling:
            solver_parameters = {}
            solver_parameters.setdefault('ksp_atol', 1e-12)
            solver_parameters.setdefault('ksp_rtol', 1e-16)
            test = TestFunction(self.fs_3d)
            tri = TrialFunction(self.fs_3d)
            a = inner(tri, test)*dx
            l = inner(self.output_3d, test)*elem_height*dx
            prob = LinearVariationalProblem(a, l, self.output_3d)
            self.rt_scale_solver = LinearVariationalSolver(
                prob, solver_parameters=solver_parameters)

    def solve(self):
        with timed_stage('copy_2d_to_3d'):
            # execute par loop
            op2.par_loop(
                self.kernel, self.fs_3d.mesh().cell_set,
                self.output_3d.dat(op2.WRITE, self.fs_3d.cell_node_map()),
                self.input_2d.dat(op2.READ, self.fs_2d.cell_node_map()),
                self.idx(op2.READ),
                iterate=self.iter_domain)

            if self.do_rt_scaling:
                self.rt_scale_solver.solve()

# works only for q <= 2
class ConstantInTime3d(object):
    """
    Copy a 3D field to 3D constant in time 
    Copies a field from 3D mesh to 3D mesh, assigning the same value over the
    vertical dimension
    .. code-block:: python
        U = FunctionSpace(mesh, 'DG', 1) where the mesh is ExtrudedMesh 
        func2d = Function(U)
        func3d = Function(U)
        ex = ExpandFunctionTo3d(func3d, funcConst, surface = "top")
        ex.solve()
    """
    def __init__(self, input_3d, output_3d, surface = "bottom" ):
        """
        :arg input_2d: 2D source field
        :type input_2d: :class:`Function`
        :arg output_3d: 3D target field
        :type output_3d: :class:`Function`
        :kwarg elem_height: scalar :class:`Function` in 3D mesh that defines
            the vertical element size. Needed only in the case of HDiv function
            spaces.
        """
        self.input_3d = input_3d
        self.output_3d = output_3d
#         self.fs_2d = self.input_2d.function_space()
        self.fs_3d = self.output_3d.function_space()

#         self.iter_domain = op2.ALL
        
        # get the value from the bottom or from top surface of the mesh
        if surface == 'bottom':
            nodes = self.fs_3d.bt_masks['geometric'][0]
            sign = 1 
            self.iter_domain = op2.ON_BOTTOM
        elif surface == 'top': 
#             print('geometric {}'.format( self.fs_3d.bt_masks['geometric'][1] ))
            nodes = self.fs_3d.bt_masks['geometric'][1]
            sign = -1
            self.iter_domain = op2.ON_TOP
        else:
            raise Exception('Unsupported surface: {:}'.format(surface))
           
        fs_2d_space_dimension = len(nodes) 
        # number of nodes in vertical direction
        n_vert_nodes = self.fs_3d.finat_element.space_dimension() / fs_2d_space_dimension
#         print( fs_2d_space_dimension, n_vert_nodes )
        # mapping from 2d points to 3d DOFs
        self.idx = op2.Global(len(nodes), nodes, dtype=np.int32, name='node_idx')
        
        # on top we have idx mapping the TOP nodes
        # hence we cannot use idx[d]+e but -e in the op2.Kernel 
        self.kernel = op2.Kernel("""
            void my_kernel(double **func, double **func3d, int *idx) {
                for ( int d = 0; d < %(nodes)d; d++ ) {
                    for ( int c = 0; c < %(func_dim)d; c++ ) {
                        for ( int e = 0; e < %(v_nodes)d; e++ ) {
                            func[idx[d]+%(signVal)d *e][c] = func3d[idx[d]][c];
                        }
                    }
                }
            }""" % {'nodes': fs_2d_space_dimension,
                    'func_dim': self.input_3d.function_space().value_size,
                    'v_nodes': n_vert_nodes, 
                    'signVal': sign},
            'my_kernel')


    def solve(self):
        with timed_stage('constant_3d'):
            # execute par loop
            op2.par_loop(
                self.kernel, self.fs_3d.mesh().cell_set,
                self.output_3d.dat(op2.WRITE, self.fs_3d.cell_node_map()),
                self.input_3d.dat(op2.READ, self.fs_3d.cell_node_map()),
                self.idx(op2.READ),
                iterate=self.iter_domain)
