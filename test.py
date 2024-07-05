import numpy as np
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def Exp(x, v):
    """
    Exponential map on the n-sphere.
    
    Args:
        x (ndarray): Base point on the n-sphere (n-dimensional).
        v (ndarray): Tangent vector at x.
    
    Returns:
        y (ndarray): Point on the n-sphere.
    """

    # Ensure the input vectors are numpy arrays
    x = np.asarray(x)
    v = np.asarray(v)
    #ensure x is on the circle
    x = x/np.linalg.norm(x)
    #ensure v is in tangent space
    v = v- np.dot(x,v)*x
    # Compute the norm of the tangent vector v
    v_norm = np.linalg.norm(v)
    
    # Compute the exponential map
    if v_norm > 0:
        y = np.cos(v_norm) * x + np.sin(v_norm) * (v / v_norm)
    else:
        y = x  # This handles the case where v is zero
    
    return y
x = [1,0]
v = [1,1]
print(Exp(x,v))
x = [1,0]
v = [1,2]
print(Exp(x,v))
x = [1,0]
v = [2,1]
print(Exp(x,v))
x = [1,0]
v = [0,1]
print(Exp(x,v))
x = [0.707,0.707]
v = [0,1]
print(Exp(x,v))
#经过测试，该exp需要的正切空间，实际是要求以x为原点