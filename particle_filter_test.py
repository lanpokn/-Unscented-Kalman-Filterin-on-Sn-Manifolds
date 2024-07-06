import numpy as np
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
def process_model(state, gamma_state):
    """
    Process model as described in the image.
    Propagate the state using the exponential map with Gaussian noise.
    
    Args:
        state (ndarray): Current state on the n-sphere (n-dimensional).
        gamma_state (float): Standard deviation of the state noise.
    
    Returns:
        new_state (ndarray): New state on the n-sphere.
    """
    # noise = tangent_noise(state, gamma_state)
    noise = np.random.multivariate_normal(np.zeros_like(state), gamma_state**2 * np.eye(len(state)))
    return Exp(state, noise)

def observation_model(state, gamma_obs):
    # Observation model as described in the image
    # Sample observation from a Gaussian distribution centered at the state
    noise = np.random.multivariate_normal(np.zeros_like(state), gamma_obs**2 * np.eye(len(state)))
    return state + noise
def Particle_filter(mean, cov, observations, gamma_state, gamma_obs, M=10):
    """
    Particle filter implementation for state estimation.
    
    Args:
        mean (ndarray): Mean of the initial state (N-dimensional).
        cov (ndarray): Covariance matrix of the initial state.
        observations (ndarray): Observed data.
        gamma_state (float): Standard deviation of the state noise.
        gamma_obs (float): Standard deviation of the observation noise.
        M (int): Number of particles.
    
    Returns:
        estimated_state (ndarray): Estimated state.
    """
    # Initialize particles randomly on N-dimensional unit sphere
    particles = np.random.multivariate_normal(mean, cov, M)
    weights = np.ones(M) / M  # Uniform weights initially
    
    for obs in observations:
        # Prediction step: propagate particles using process model
        for i in range(M):
            particles[i] = process_model(particles[i], gamma_state)
        
        # Weight update step: compute likelihood for each particle
        likelihoods = np.exp(-0.5 * np.sum((particles - obs)**2 / gamma_obs**2, axis=1))
        weights *= likelihoods
        weights /= np.sum(weights)  # Normalize weights
        
        # Resampling step: select particles based on weights
        #TODO, do I need resample?
        # particles, _ = simple_resample(particles, weights)
    
    # Estimate the state using weighted average of particles
    # estimated_state = np.average(particles, weights=weights, axis=0)
    # Compute the mean on the spherical manifold using the log and exp maps
    tangent_vectors = np.array([Log(mean, pt) for pt in particles])
    weighted_tangent_mean = np.sum(weights[:, np.newaxis] * tangent_vectors, axis=0)
    estimated_state = Exp(mean, weighted_tangent_mean)
    return estimated_state

# Example usage
N=3

observations = np.array([1.01,0,0])  # Your observed data
mean = np.zeros(N)  # Initial mean
mean[0] = 1
cov = np.eye(N)*0.01  # Initial covariance
gamma_state = 0.2 # State noise
gamma_obs = 0.1  # Observation noise

estimated_state = Particle_filter(mean, cov, observations, gamma_state, gamma_obs)
print("Estimated state:", estimated_state)