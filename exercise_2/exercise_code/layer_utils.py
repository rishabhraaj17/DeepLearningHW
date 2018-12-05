from exercise_code.layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU
  
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
  
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU and a batch norm
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    bout, b_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bout)
    cache = (fc_cache, b_cache, relu_cache)
    return out, cache


def affine_batchnorm_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu and a batch norm convenience layer
    """
    fc_cache, b_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dbatch, dgamma, dbeta = batchnorm_backward(da, b_cache)
    dx, dw, db = affine_backward(dbatch, fc_cache)
    return dx, dw, db, dgamma, dbeta

def affine_relu_dropout_forward(x, w, b, dropout_params):
    """
    Convenience layer that perorms an affine transform followed by a ReLU and a dropout
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    relu, relu_cache = relu_forward(a)
    dropout_out, dropout_cache = dropout_forward(relu, dropout_params)
    cache = (fc_cache, relu_cache, dropout_cache)
    return dropout_out, cache


def affine_relu_dropout_backward(dout, cache):
    """
    Backward pass for the affine-relu and a dropout convenience layer
    """
    fc_cache, relu_cache, dropout_cache = cache
    ddropout = dropout_backward(dout, dropout_cache)
    da = relu_backward(ddropout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db