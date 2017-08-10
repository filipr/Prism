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


class ExtractFunctionTo2d(object):
    """
    Extract a 2D sub-function from a 3D function in an extruded mesh
    Given 2D and 3D functions,
    .. code-block:: python
        U = FunctionSpace(mesh, 'DG', 1)
        U_2d = FunctionSpace(mesh2d, 'DG', 1)
        func2d = Function(U_2d)
        func3d = Function(U)
    Get surface value:
    .. code-block:: python
        ex = ExtractFunctionTo2d(func3d, func2d,
            boundary='top', elem_facet='top')
        ex.solve()
    Get bottom value:
    .. code-block:: python
        ex = ExtractFunctionTo2d(func3d, func2d,
            boundary='bottom', elem_facet='bottom')
        ex.solve()
    Get value at the top of bottom element:
    .. code-block:: python
        ex = ExtractFunctionTo2d(func3d, func2d,
            boundary='bottom', elem_facet='top')
        ex.solve()
    """
    def __init__(self, input_3d, output_2d,
                 boundary='top', elem_facet=None,
                 elem_height=None):
        """
        :arg input_3d: 3D source field
        :type input_3d: :class:`Function`
        :arg output_2d: 2D target field
        :type output_2d: :class:`Function`
        :kwarg str boundary: 'top'|'bottom'
            Defines whether to extract from the surface or bottom 3D elements
        :kwarg str elem_facet: 'top'|'bottom'|'average'
            Defines which facet of the 3D element is extracted. The 'average'
            computes mean of the top and bottom facets of the 3D element.
        :kwarg elem_height: scalar :class:`Function` in 2D mesh that defines
            the vertical element size. Needed only in the case of HDiv function
            spaces.
        """
        self.input_3d = input_3d
        self.output_2d = output_2d
        self.fs_3d = self.input_3d.function_space()
        self.fs_2d = self.output_2d.function_space()

        if elem_facet is None:
            # extract surface/bottom face by default
            elem_facet = boundary

        family_2d = self.fs_2d.ufl_element().family()
        elem = self.fs_3d.ufl_element()
        if isinstance(elem, ufl.VectorElement):
            elem = elem.sub_elements()[0]
        if isinstance(elem, ufl.HDivElement):
            elem = elem._element
        if isinstance(elem, ufl.TensorProductElement):
            # a normal tensorproduct element
            family_3dh = elem.sub_elements()[0].family()
            if family_2d != family_3dh:
                raise Exception('2D and 3D spaces do not match: {0:s} {1:s}'.format(family_2d, family_3dh))
        if family_2d == 'Raviart-Thomas' and elem_height is None:
            raise Exception('elem_height must be provided for Raviart-Thomas spaces')
        self.do_rt_scaling = family_2d == 'Raviart-Thomas'

        if elem_facet == 'bottom':
            nodes = self.fs_3d.bt_masks['geometric'][0]
        elif elem_facet == 'top':
            nodes = self.fs_3d.bt_masks['geometric'][1]
        elif elem_facet == 'average':
            nodes = (self.fs_3d.bt_masks['geometric'][0] +
                     self.fs_3d.bt_masks['geometric'][1])
        else:
            raise Exception('Unsupported elem_facet: {:}'.format(elem_facet))
        if boundary == 'top':
            self.iter_domain = op2.ON_TOP
        elif boundary == 'bottom':
            self.iter_domain = op2.ON_BOTTOM

        out_nodes = self.fs_2d.finat_element.space_dimension()

        if elem_facet == 'average':
            assert (len(nodes) == 2*out_nodes)
        else:
            assert (len(nodes) == out_nodes)

        self.idx = op2.Global(len(nodes), nodes, dtype=np.int32, name='node_idx')
        if elem_facet == 'average':
            # compute average of top and bottom elem nodes
            self.kernel = op2.Kernel("""
                void my_kernel(double **func, double **func3d, int *idx) {
                    int nnodes = %(nodes)d;
                    for ( int d = 0; d < nnodes; d++ ) {
                        for ( int c = 0; c < %(func_dim)d; c++ ) {
                            func[d][c] = 0.5*(func3d[idx[d]][c] +
                                              func3d[idx[d + nnodes]][c]);
                        }
                    }
                }""" % {'nodes': self.output_2d.cell_node_map().arity,
                        'func_dim': self.output_2d.function_space().value_size},
                'my_kernel')
        else:
            self.kernel = op2.Kernel("""
                void my_kernel(double **func, double **func3d, int *idx) {
                    for ( int d = 0; d < %(nodes)d; d++ ) {
                        for ( int c = 0; c < %(func_dim)d; c++ ) {
                            func[d][c] = func3d[idx[d]][c];
                        }
                    }
                }""" % {'nodes': self.output_2d.cell_node_map().arity,
                        'func_dim': self.output_2d.function_space().value_size},
                'my_kernel')

        if self.do_rt_scaling:
            solver_parameters = {}
            solver_parameters.setdefault('ksp_atol', 1e-12)
            solver_parameters.setdefault('ksp_rtol', 1e-16)
            test = TestFunction(self.fs_2d)
            tri = TrialFunction(self.fs_2d)
            a = inner(tri, test)*dx
            l = inner(self.output_2d, test)/elem_height*dx
            prob = LinearVariationalProblem(a, l, self.output_2d)
            self.rt_scale_solver = LinearVariationalSolver(
                prob, solver_parameters=solver_parameters)

    def solve(self):
        with timed_stage('copy_3d_to_2d'):
            # execute par loop
            op2.par_loop(self.kernel, self.fs_3d.mesh().cell_set,
                         self.output_2d.dat(op2.WRITE, self.fs_2d.cell_node_map()),
                         self.input_3d.dat(op2.READ, self.fs_3d.cell_node_map()),
                         self.idx(op2.READ),
                         iterate=self.iter_domain)

            if self.do_rt_scaling:
                self.rt_scale_solver.solve()

class ConstantFunctionInTime(object):
    """
        ex = ExtractFunctionTo2d(func3d, func2d,
            boundary='bottom', elem_facet='top')
        ex.solve()
    """
    def __init__(self, input_3d, output_3d,
                 boundary='top', elem_facet=None,
                 elem_height=None):
        """
        :arg input_3d: 3D source field
        :type input_3d: :class:`Function`
        :arg output_2d: 2D target field
        :type output_2d: :class:`Function`
        :kwarg str boundary: 'top'|'bottom'
            Defines whether to extract from the surface or bottom 3D elements
        :kwarg str elem_facet: 'top'|'bottom'|'average'
            Defines which facet of the 3D element is extracted. The 'average'
            computes mean of the top and bottom facets of the 3D element.
        :kwarg elem_height: scalar :class:`Function` in 2D mesh that defines
            the vertical element size. Needed only in the case of HDiv function
            spaces.
        """       
        self.input_3d = input_3d
        self.output_3d = output_3d

        self.fs_3d = self.input_3d.function_space()
        self.mesh3d = self.fs_3d.mesh 
        self.mesh2d = self.mesh3d._base_mesh 
        elem_horizontal = self.fs_3d.ufl_element()).vertical 
                
        self.fs_2d = FunctionSpace( self.mesh2d, elem_horizontal )

        if elem_facet is None:
            # extract surface/bottom face by default
            elem_facet = boundary

        family_2d = self.fs_2d.ufl_element().family()
        elem = self.fs_3d.ufl_element()

        if isinstance(elem, ufl.TensorProductElement):
            # a normal tensorproduct element
            family_3dh = elem.sub_elements()[0].family()
            if family_2d != family_3dh:
                raise Exception('2D and 3D spaces do not match: {0:s} {1:s}'.format(family_2d, family_3dh))
       

        if elem_facet == 'bottom':
            nodes = self.fs_3d.bt_masks['geometric'][0]
        elif elem_facet == 'top':
            nodes = self.fs_3d.bt_masks['geometric'][1]
        elif elem_facet == 'average':
            nodes = (self.fs_3d.bt_masks['geometric'][0] +
                     self.fs_3d.bt_masks['geometric'][1])
        else:
            raise Exception('Unsupported elem_facet: {:}'.format(elem_facet))
        
        # botom or top of the whole 3d domain
        if boundary == 'top':
            self.iter_domain = op2.ON_TOP
        elif boundary == 'bottom':
            self.iter_domain = op2.ON_BOTTOM

        out_nodes = self.fs_2d.finat_element.space_dimension()

        if elem_facet == 'average':
            assert (len(nodes) == 2*out_nodes)
        else:
            assert (len(nodes) == out_nodes)

        self.idx = op2.Global(len(nodes), nodes, dtype=np.int32, name='node_idx')
        if elem_facet == 'average':
            # compute average of top and bottom elem nodes
            self.kernel = op2.Kernel("""
                void my_kernel(double **func, double **func3d, int *idx) {
                    int nnodes = %(nodes)d;
                    for ( int d = 0; d < nnodes; d++ ) {
                        for ( int c = 0; c < %(func_dim)d; c++ ) {
                            func[d][c] = 0.5*(func3d[idx[d]][c] +
                                              func3d[idx[d + nnodes]][c]);
                        }
                    }
                }""" % {'nodes': self.output_2d.cell_node_map().arity,
                        'func_dim': self.output_2d.function_space().value_size},
                'my_kernel')
        else:
            self.kernel = op2.Kernel("""
                void my_kernel(double **func, double **func3d, int *idx) {
                    for ( int d = 0; d < %(nodes)d; d++ ) {
                        for ( int c = 0; c < %(func_dim)d; c++ ) {
                            func[d][c] = func3d[idx[d]][c];
                        }
                    }
                }""" % {'nodes': self.output_2d.cell_node_map().arity,
                        'func_dim': self.output_2d.function_space().value_size},
                'my_kernel')



    def solve(self):
        with timed_stage('copy_3d_to_2d'):
            # execute par loop
            op2.par_loop(self.kernel, self.fs_3d.mesh().cell_set,
                         self.output_2d.dat(op2.WRITE, self.fs_2d.cell_node_map()),
                         self.input_3d.dat(op2.READ, self.fs_3d.cell_node_map()),
                         self.idx(op2.READ),
                         iterate=self.iter_domain)

            if self.do_rt_scaling:
                self.rt_scale_solver.solve()



