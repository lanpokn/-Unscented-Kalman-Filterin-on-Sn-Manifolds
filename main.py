import numpy as np
from scipy.linalg import cholesky

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
    else:
        v = np.zeros_like(x)  # This handles the case where x == y
    
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
    
    # Compute the norm of the tangent vector v
    v_norm = np.linalg.norm(v)
    
    # Compute the exponential map
    if v_norm > 0:
        y = np.cos(v_norm) * x + np.sin(v_norm) * (v / v_norm)
    else:
        y = x  # This handles the case where v is zero
    
    return y

def sigma_points(mean, cov, kappa):
    n = mean.shape[0]
    sigma_points = np.zeros((2 * n + 1, n))
    sigma_points[0] = mean
    
    sqrt_cov = cholesky((n + kappa) * cov, lower=True)
    for i in range(n):
        sigma_points[i + 1] = mean + sqrt_cov[:, i]
        sigma_points[n + i + 1] = mean - sqrt_cov[:, i]
    
    return sigma_points

def unscented_transform(sigma_points, weights_mean, weights_cov, mean):
    n = sigma_points.shape[1]
    mean_out = np.zeros(n)
    cov_out = np.zeros((n, n))

    for i in range(sigma_points.shape[0]):
        mean_out += weights_mean[i] * Exp(mean, sigma_points[i] - mean)
    
    for i in range(sigma_points.shape[0]):
        diff = Log(mean_out, sigma_points[i])
        cov_out += weights_cov[i] * np.outer(diff, diff)
    
    return mean_out, cov_out

def ukf_predict(mean, cov, process_model, process_noise_cov, kappa, gamma_state):
    n = mean.shape[0]
    sigma_pts = sigma_points(mean, cov, kappa)
    
    sigma_pts_prop = np.zeros_like(sigma_pts)
    for i in range(sigma_pts.shape[0]):
        sigma_pts_prop[i] = process_model(sigma_pts[i], gamma_state)
    
    weights_mean = np.full(2 * n + 1, 1 / (2 * (n + kappa)))
    weights_mean[0] = kappa / (n + kappa)
    weights_cov = weights_mean.copy()
    
    mean_pred, cov_pred = unscented_transform(sigma_pts_prop, weights_mean, weights_cov, mean)
    cov_pred += process_noise_cov
    
    return mean_pred, cov_pred

def ukf_update(mean_pred, cov_pred, observation, observation_model, obs_noise_cov, kappa, gamma_obs):
    n = mean_pred.shape[0]
    sigma_pts = sigma_points(mean_pred, cov_pred, kappa)
    
    sigma_pts_obs = np.zeros((2 * n + 1, observation.shape[0]))
    for i in range(sigma_pts.shape[0]):
        sigma_pts_obs[i] = observation_model(sigma_pts[i], gamma_obs)
    
    weights_mean = np.full(2 * n + 1, 1 / (2 * (n + kappa)))
    weights_mean[0] = kappa / (n + kappa)
    weights_cov = weights_mean.copy()
    
    mean_obs, cov_obs = unscented_transform(sigma_pts_obs, weights_mean, weights_cov, observation)
    cov_obs += obs_noise_cov
    
    cross_cov = np.zeros((n, observation.shape[0]))
    for i in range(sigma_pts.shape[0]):
        diff_state = Log(mean_pred, sigma_pts[i])
        diff_obs = Log(mean_obs, sigma_pts_obs[i])
        cross_cov += weights_cov[i] * np.outer(diff_state, diff_obs)
    
    kalman_gain = cross_cov @ np.linalg.inv(cov_obs)
    mean_upd = Exp(mean_pred, kalman_gain @ Log(mean_obs, observation))
    cov_upd = cov_pred - kalman_gain @ cov_obs @ kalman_gain.T
    
    return mean_upd, cov_upd

# Example usage with specific gamma values
gamma_state = 0.1
gamma_obs = 0.1

def process_model(state, gamma_state):
    # Process model as described in the image
    # Propagate the state using the exponential map with Gaussian noise
    noise = np.random.multivariate_normal(np.zeros_like(state), gamma_state**2 * np.eye(len(state)))
    return Exp(state, noise)

def observation_model(state, gamma_obs):
    # Observation model as described in the image
    # Sample observation from a Gaussian distribution centered at the state
    noise = np.random.multivariate_normal(np.zeros_like(state), gamma_obs**2 * np.eye(len(state)))
    return state + noise

# Example usage
n = 3  # Dimensionality of the state space (n-sphere embedded in R^(n+1))
mean = np.array([1.0, 0.0, 0.0])
cov = np.eye(n) * 0.1
process_noise_cov = np.eye(n) * 0.01
obs_noise_cov = np.eye(n) * 0.1
observation = np.array([0.9, 0.1, 0.0])
kappa = 0

# Prediction step
mean_pred, cov_pred = ukf_predict(mean, cov, process_model, process_noise_cov, kappa, gamma_state)

# Update step
mean_upd, cov_upd = ukf_update(mean_pred, cov_pred, observation, observation_model, obs_noise_cov, kappa, gamma_obs)

print("Updated Mean:", mean_upd)
print("Updated Covariance:", cov_upd)
