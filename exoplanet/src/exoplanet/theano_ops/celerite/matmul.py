# -*- coding: utf-8 -*-

__all__ = ["MatmulOp"]

import theano.tensor as tt
from theano import gof

from .base_op import CeleriteBaseOp


class MatmulOp(CeleriteBaseOp):

    func_file = "./matmul.cc"
    func_name = "APPLY_SPECIFIC(matmul)"
    num_input = 5
    output_ndim = (2, 2, 2)

    def make_node(self, *args):
        in_args = [tt.as_tensor_variable(a) for a in args]
        out_args = [in_args[-1].type(), in_args[-1].type(), 
                    in_args[-1].type()]
        return gof.Apply(self, in_args, out_args)

    #def infer_shape(self, node, shapes):
    #    return (shapes[-1],)