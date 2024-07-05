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
def Log(x, y):
    """
    Logarithm map on the n-sphere.
    
    Args:
        x (ndarray): Base point on the n-sphere (n-dimensional).
        y (ndarray): Point on the n-sphere to map to the tangent space at x.
    
    Returns:
        v (ndarray): Tangent vector at x.
    """
    # Ensure the input vectors are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Compute the inner product <x, y>
    inner_product = np.dot(x, y)
    
    # Ensure the inner product is within valid range for arccos
    inner_product = np.clip(inner_product, -1.0, 1.0)
    
    # Compute u = y - (inner_product) * x
    u = y - inner_product * x
    
    # Compute the norm of u
    u_norm = np.linalg.norm(u)
    
    # Compute the logarithm map
    if u_norm > 0:
        v = np.arccos(inner_product) * (u / u_norm)
        # add this to turn back to global representation
        v=v+x
        #you need add 
    else:
        v = np.zeros_like(x)  # This handles the case where x == y
        v = v+x
    
    return v
x = [1,0]
v = [1,1]
print(Exp(x,v))
print(Log(x,Exp(x,v)))
x = [1,0]
v = [1,2]
print(Exp(x,v))
print(Log(x,Exp(x,v)))
x = [1,0]
v = [2,1]
print(Exp(x,v))
print(Log(x,Exp(x,v)))
x = [1,0]
v = [0,1]
print(Exp(x,v))
print(Log(x,Exp(x,v)))
x = [0.707,0.707]
v = [0,1.414]
print(Exp(x,v))
print(Log(x,Exp(x,v)))
#经过测试，该exp需要的正切空间，实际是要求以x为原点