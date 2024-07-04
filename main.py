import numpy as np
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
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
    sigma_pts = np.zeros((2 * n + 1, n))
    sigma_pts[0] = mean
    # Eq.(12): sigma points calculation using Cholesky decomposition
    U = np.linalg.cholesky((n + kappa) * cov)
    for i in range(n):
        sigma_pts[i + 1] = mean + U[i]
        sigma_pts[n + i + 1] = mean - U[i]
    return sigma_pts

#sigma_pts =h(σ (m) M )or
# this is used for step1, eq20 amd 21
# for Pxx and Pxy, mean_pred is yt
def unscented_transform(sigma_pts_L,sigma_pts_R,weights_mean, weights_cov):
    n = sigma_pts_R.shape[1]
    # Eq.(13): mean prediction using weighted sum of sigma points
    mean_pred = np.average(sigma_pts_R, axis=0)
    cov_pred = np.zeros((n, n))
    for i in range(2 * n + 1):
        diff_L = sigma_pts_L[i] - mean_pred
        diff_R = sigma_pts_R[i] - mean_pred
        # Eq.(14): covariance prediction using weighted outer product of differences
        cov_pred += weights_cov[i] * np.outer(diff_L, diff_R)
    return mean_pred, cov_pred
def unscented_transform_xy(sigma_pts_L,sigma_pts_R,weights_mean, weights_cov):
    n = sigma_pts_R.shape[1]
    # Eq.(13): mean prediction using weighted sum of sigma points
    mean_pred = np.average(sigma_pts_R, axis=0)
    cov_pred = np.zeros((n, n))
    for i in range(2 * n + 1):
        diff_L = sigma_pts_L[i]
        diff_R = sigma_pts_R[i] - mean_pred
        # Eq.(14): covariance prediction using weighted outer product of differences
        cov_pred += weights_cov[i] * np.outer(diff_L, diff_R)
    return mean_pred, cov_pred
# sigma_pts = σ (m) M, h or f should be added later
# Use the Riemannian generalisation of the unscented transform to estimate the predicted state mean
def ukf_predict(mean, cov, process_model, process_noise_cov, kappa, gamma_state):
    n = mean.shape[0]
    # Generate sigma points
    sigma_pts = sigma_points(mean, cov, kappa)
    
    sigma_pts_pro = np.zeros_like(sigma_pts)
    for i in range(sigma_pts.shape[0]):
        sigma_pts_pro[i] = process_model(sigma_pts[i], gamma_state)
    
    # Eq.(15): calculate weights for mean
    weights_mean = np.full(2 * n + 1, 1 / (2 * (n + kappa)))
    weights_mean[0] = kappa / (n + kappa)
    weights_cov = weights_mean.copy()
    
    # Predict mean and covariance
    # this is eq20 and 21, sigma_pts is h(x), get e(fx), h(fx)
    mean_pred, cov_pred = unscented_transform(sigma_pts_pro,sigma_pts_pro,weights_mean, weights_cov)
    # Add process noise covariance
    # cov_pred += process_noise_cov*100
    
    return mean_pred, cov_pred
def parallel_transport(Mx, alpha, t):
    M = Mx.shape[0]
    eigvals, eigvecs = np.linalg.eigh(Mx)
    vms = [eigvecs[:, i] for i in range(M)]
    
    def Pt(vm):
        # Implement the parallel transport function, assuming alpha(t) is the curve
        # Parallel transport a vector vm along the curve alpha on the n-dimensional sphere
        def exp_map(v, theta):
            # Exponential map for n-dimensional sphere
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                return v
            return np.cos(theta) * v + np.sin(theta) * (v / norm_v)

        # Assume alpha(t) = [cos(t), sin(t), 0, ..., 0] on the sphere
        # Here we need to compute the transport at t along the great circle
        theta = t
        vm_t = exp_map(vm, theta)
        
        return vm_t
        
    vms_t = [Pt(vm) for vm in vms]
    Mt = sum(eigvals[i] * np.outer(vms_t[i], vms_t[i]) for i in range(M))
    
    return Mt

def ukf_CovCompute(mean,cov,kappa,gamma_obs,gamma_state):
    n = mean.shape[0]
    # Generate sigma points
    sigma_pts = sigma_points(mean, cov, kappa)
    
    # Eq.(15): calculate weights for mean
    weights_mean = np.full(2 * n + 1, 1 / (2 * (n + kappa)))
    weights_mean[0] = kappa / (n + kappa)
    weights_cov = weights_mean.copy()
    
    # step2: Predict observation mean and covariance
    #TODO they all use unscented_transform and use Pyy,Pxy rather than below name!
    #eq 33,32 and 31(h(sigma))
    #you need verify mean_pred is sigma's corresponding value
    sigma_pts_obs = np.zeros((2 * n + 1, mean.shape[0]))
    for i in range(sigma_pts.shape[0]):
        # Propagate each sigma point through the observation model
        sigma_pts_obs[i] = observation_model(sigma_pts[i], gamma_obs)
    sigma_pts_pro = np.zeros_like(sigma_pts)
    for i in range(sigma_pts.shape[0]):
        sigma_pts_pro[i] = process_model(sigma_pts[i], gamma_state)
    _, Pyy = unscented_transform(sigma_pts_obs,sigma_pts_obs,weights_mean, weights_cov)
    _, Pxy = unscented_transform_xy(sigma_pts,sigma_pts_obs,weights_mean, weights_cov)
    yt_hat = np.average(sigma_pts_obs)

    return Pxy, Pyy,yt_hat
# step2:Compute the Riemannian generalisation of the unscented transform
# step3:Compute state updates
#observation is yt(true)
def ukf_update(mean_pred, cov_pred, observation, yt_hat, Pxy,Pyy):
    # n = mean_pred.shape[0]
    # # Generate sigma points
    # sigma_pts = sigma_points(mean_pred, cov_pred, kappa)
    
    # # Eq.(15): calculate weights for mean
    # weights_mean = np.full(2 * n + 1, 1 / (2 * (n + kappa)))
    # weights_mean[0] = kappa / (n + kappa)
    # weights_cov = weights_mean.copy()
    
    # # step2: Predict observation mean and covariance
    # #TODO they all use unscented_transform and use Pyy,Pxy rather than below name!
    # #eq 33,32 and 31(h(sigma))
    # #you need verify mean_pred is sigma's corresponding value
    # sigma_pts_obs = np.zeros((2 * n + 1, observation.shape[0]))
    # for i in range(sigma_pts.shape[0]):
    #     # Propagate each sigma point through the observation model
    #     sigma_pts_obs[i] = observation_model(sigma_pts[i], gamma_obs)
    # sigma_pts_pro = np.zeros_like(sigma_pts)
    # for i in range(sigma_pts.shape[0]):
    #     sigma_pts_pro[i] = process_model(sigma_pts[i], gamma_state)
    # _, Pyy = unscented_transform(sigma_pts_obs,sigma_pts_obs,weights_mean, weights_cov)
    # _, Pxy = unscented_transform_xy(sigma_pts,sigma_pts_obs,weights_mean, weights_cov)
    # yt_hat = np.average(sigma_pts_obs)
    #step2 END

    #below is step3, 
    kalman_gain = Pxy @ np.linalg.inv(Pyy)
    mean_upd_tangent = mean_pred+kalman_gain@Log(yt_hat,observation)
    # Update covariance using the Kalman gain
    #TODO below Pyy,Pxy,cov_pred is positive, but cov_upd is not
    #mean_upd is on sphere, so it's not totally wrong
    cov_upd = cov_pred - kalman_gain @ Pyy @ kalman_gain.T
    # Update mean using the Kalman gain
    mean_upd = Exp(mean_pred,mean_upd_tangent)
    #TODO I'm not sure what this step is doing....
    cov_upd =  parallel_transport(cov_upd, lambda t: mean_upd, 1.0)  # Assuming alpha(t) = mean_upd, t = 1.0
    return mean_upd, cov_upd
def ukf(mean, cov, observations, process_model, observation_model, process_noise_cov, obs_noise_cov, kappa, gamma_state, gamma_obs):
    n = mean.shape[0]
    num_obs = len(observations)
    filtered_means = np.zeros((num_obs, n))
    filtered_covs = np.zeros((num_obs, n, n))
    
    #get a trajectory of a filtered result, 
    # TODO len(observations) is not dimension, it's (should be) number of observations
    for i in range(num_obs):
        #based on x_t-1 to predict xt, no observation here
        mean_pred, cov_pred = ukf_predict(mean, cov, process_model, process_noise_cov, kappa, gamma_state)
        Pxy,Pyy,yt_hat = ukf_CovCompute(mean,cov,kappa,gamma_obs,gamma_state)
        #add an observation here and get new x 
        mean, cov = ukf_update(mean_pred, cov_pred, observations[i], yt_hat, Pxy,Pyy)
        filtered_means[i] = mean
        filtered_covs[i] = cov
    
    return filtered_means, filtered_covs

def tangent_noise(state, gamma_state):
    """
    Generate a noise vector in the tangent space of the sphere at the given state.
    
    Args:
        state (ndarray): Base point on the n-sphere (n-dimensional).
        gamma_state (float): Standard deviation of the Gaussian noise.
    
    Returns:
        tangent_noise (ndarray): Noise vector in the tangent space at the given state.
    """
    noise = np.random.multivariate_normal(np.zeros_like(state), gamma_state**2 * np.eye(len(state)))
    proj = np.dot(noise, state) * state
    tangent_noise = noise - proj
    return tangent_noise

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
    noise = tangent_noise(state, gamma_state)
    return Exp(state, noise)

def observation_model(state, gamma_obs):
    # Observation model as described in the image
    # Sample observation from a Gaussian distribution centered at the state
    noise = np.random.multivariate_normal(np.zeros_like(state), gamma_obs**2 * np.eye(len(state)))
    return state + noise


def generate_synthetic_data(num_points, gamma_state, gamma_obs,dim=3):
    true_states = []
    observations = []
    
    # state = np.array([0.557, 0.557, -0.557])
    # Initialize state on the n-sphere (random point on unit sphere in `dim` dimensions)
    state = np.random.randn(dim)
    state /= np.linalg.norm(state)
    for _ in range(num_points):
        # noise = np.random.multivariate_normal(np.zeros_like(state), (gamma_state**2 / 3) * np.eye(len(state)))
        # state = Exp(state, noise)
        state = process_model(state,gamma_state)
        true_states.append(state)
        
        observation = observation_model(state, gamma_obs)
        observations.append(observation)
    
    return np.array(true_states), np.array(observations)
def plot_synthetic_data(true_states, observations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Sphere for reference
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='b', alpha=0.1)
    
    # Plot true states and connect them to show the trajectory
    ax.scatter(true_states[:, 0], true_states[:, 1], true_states[:, 2], color='orange', label='True States')
    ax.plot(true_states[:, 0], true_states[:, 1], true_states[:, 2], color='orange', linestyle='-', marker='o', markersize=2)
    
    # Plot observations and connect them to corresponding true states
    ax.scatter(observations[:, 0], observations[:, 1], observations[:, 2], color='purple', marker='s', label='Observations')
    for i in range(len(true_states)):
        ax.plot([true_states[i, 0], observations[i, 0]], [true_states[i, 1], observations[i, 1]], [true_states[i, 2], observations[i, 2]], color='purple', linestyle='-')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])  # Set equal scaling for all axes
    ax.legend(loc='upper right')
    plt.title('Fig. 3 Example of synthetic data\nTrue states and observations on the sphere')
    plt.show()


def evaluate_ukf(dimensions, num_points=10, kappa=3):
    errors = []
    times = []
    
    for dim in dimensions:
        gamma_state = 0.5 / np.sqrt(dim)
        gamma_obs = 0.1
        process_noise_cov = (gamma_state**2 / dim) * np.eye(dim)
        obs_noise_cov = (gamma_obs**2) * np.eye(dim)
        
        true_states, observations = generate_synthetic_data(num_points, gamma_state, gamma_obs, dim)
        
        # Initial state for UKF
        mean = true_states[0]
        cov = np.eye(dim) * 1
        
        start_time = time.time()
        filtered_means, filtered_covs = ukf(mean, cov, observations, process_model, observation_model, process_noise_cov, obs_noise_cov, kappa, gamma_state, gamma_obs)
        end_time = time.time()
        
        mse = np.mean(np.linalg.norm(filtered_means - true_states, axis=1)**2)
        
        errors.append(mse)
        times.append(end_time - start_time)
    
    return np.array(errors), np.array(times)

def plot_ukf_results(dimensions, errors, times):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Error plot
    axs[0].plot(dimensions, errors, color='red', label='Riemannian UKF')
    axs[0].set_xlabel('State Space Dimensionality')
    axs[0].set_ylabel('Error')
    axs[0].set_title('(a)')
    axs[0].legend()

    # Time plot
    axs[1].plot(dimensions, times, color='red', label='Riemannian UKF')
    axs[1].set_xlabel('State Space Dimensionality')
    axs[1].set_ylabel('Running Time (sec)')
    axs[1].set_yscale('log')
    axs[1].set_title('(b)')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# Define the range of dimensions to test
dimensions = np.arange(2, 202, 100)

# Evaluate UKF for different dimensions
errors, times = evaluate_ukf(dimensions)

# Plot results
plot_ukf_results(dimensions, errors, times)