from msilib import Feature
import numpy as np
from pr3_utils import *
from scipy.linalg import expm
# import matplotlib.pyplot as plt

### Functions ###

def skew_symmetric(v):
    """Convert a 3D vector to a skew-symmetric matrix."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def hat_map(u):
    """Convert 6D velocity vector [v, w] to 4x4 se(3) matrix."""
    v = u[:3]
    w = u[3:]
    w_hat = np.array([[0, -w[2], w[1]],
                      [w[2], 0, -w[0]],
                      [-w[1], w[0], 0]])
    u_hat = np.zeros((4, 4))
    u_hat[:3, :3] = w_hat
    u_hat[:3, 3] = v
    return u_hat

def adjoint(u):
    """Convert 6D velocity vector [v, w] to 6x6 adjoint matrix."""
    v = u[:3]
    w = u[3:]
    w_hat = np.array([[0, -w[2], w[1]],
                      [w[2], 0, -w[0]],
                      [-w[1], w[0], 0]])
    v_hat = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
    adj = np.zeros((6, 6))
    adj[:3, :3] = w_hat
    adj[:3, 3:6] = v_hat
    adj[3:6, 3:6] = w_hat
    return adj

def EKF_prediction(mu_t, Sigma_t, u_t, tau_t, W):
    """EKF prediction step for IMU pose estimation."""
    u_hat = hat_map(u_t)
    mu_update = mu_t @ expm(tau_t * u_hat)  # Update pose
    adj_u = adjoint(u_t)
    exp_adj = expm(-tau_t * adj_u)
    Sigma_update = exp_adj @ Sigma_t @ exp_adj.T + W  # Update covariance
    return mu_update, Sigma_update

def filter_and_downsample_features(features, downsample_factor=20):
    """Filter invalid features and downsample along time axis."""
    features_down = features[:, :, ::downsample_factor]
    filtered_features = []
    num_time_steps = features_down.shape[2]
    
    for t in range(num_time_steps):
        feats_t = features_down[:, :, t]  # Shape: (4, n)
        valid_mask = ~np.all(feats_t == -1, axis=0)
        valid_feats_t = feats_t[:, valid_mask]
        filtered_features.append(valid_feats_t)
    
    ret = np.empty(len(filtered_features), dtype=object)
    for i, feat in enumerate(filtered_features):
        ret[i] = feat
    return ret

def cal_baseline(extL_T_imu, extR_T_imu):
    """Calculate stereo baseline from extrinsic matrices."""
    T_imu_left = np.linalg.inv(extL_T_imu)
    T_imu_right = np.linalg.inv(extR_T_imu)
    center_left = T_imu_left[:3, 3]
    center_right = T_imu_right[:3, 3]
    return np.linalg.norm(center_left - center_right)

def initialize_landmark(uL, uR, vL, K_l, b, T_t, extL_T_imu):
    """Initialize 3D landmark position using stereo triangulation."""
    fsu, fsv = K_l[0, 0], K_l[1, 1]
    cu, cv = K_l[0, 2], K_l[1, 2]
    d = uL - uR  
    z = -(fsu * b) / d  
    x = (uL - cu) * z / fsu
    y = (vL - cv) * z / fsv
    landmark_camera = np.array([x, y, z, 1])
    T_imu_to_camera = np.linalg.inv(extL_T_imu)
    landmark_world = T_imu_to_camera @ T_t @ landmark_camera  # IMU pose is world_T_imu
    return landmark_world[:3]

def ekf_update(mu, Sigma, z_t, landmarks, T_t, K_l, extL_T_imu, V):
    """Perform EKF update step to refine landmark positions."""
    N_t = len(landmarks)
    if N_t == 0:
        return mu, Sigma
    
    z_pred = np.zeros(2 * N_t)
    H_t = np.zeros((2 * N_t, mu.shape[0]))
    T_imu_to_camera = np.linalg.inv(extL_T_imu)
    
    for i, idx in enumerate(landmarks):
        mj = mu[3 * idx:3 * idx + 3]
        mj_camera = T_imu_to_camera @ np.linalg.inv(T_t) @ np.append(mj, 1)
        proj = K_l @ mj_camera[:3]
        z_pred[2 * i:2 * i + 2] = proj[:2] / proj[2]
        
        q = mj_camera[:3]
        z = q[2]
        d_pi_dq = np.array([[1/z, 0, -q[0]/(z**2)],
                           [0, 1/z, -q[1]/(z**2)]])  # 2×3
        T_world_to_camera = T_imu_to_camera @ np.linalg.inv(T_t)
        P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        T_3x3 = (T_world_to_camera @ P.T)[:3, :3]
        Ks = K_l[:2, :2]  # Corrected to 2×2
        H_j = Ks @ d_pi_dq @ T_3x3
        H_t[2 * i:2 * i + 2, 3 * idx:3 * idx + 3] = H_j
    
    S = H_t @ Sigma @ H_t.T + np.kron(np.eye(N_t), V)
    K_t = Sigma @ H_t.T @ np.linalg.inv(S)
    mu_t_plus_1 = mu + K_t @ (z_t - z_pred)
    Sigma_t_plus_1 = (np.eye(mu.shape[0]) - K_t @ H_t) @ Sigma
    
    return mu_t_plus_1, Sigma_t_plus_1
def ekf_update_joint(mu_t, Sigma_t, z_t, landmark_ids, T_t, K_l, extL_T_imu, V):
    """Joint update step for VI-SLAM: Update robot pose and landmarks."""
    N_t = len(landmark_ids)
    if N_t == 0:
        return mu_t, Sigma_t
    
    M = mu_t['landmarks'].size // 3  # Number of landmarks
    state_size = 6 + 3 * M  # 6 for pose, 3M for landmarks
    
    # Predicted observations (4D per landmark: uL, vL, uR, vR)
    z_pred = np.zeros(4 * N_t)
    H_t = np.zeros((4 * N_t, state_size))  # 4 observations per landmark
    
    T_imu_to_camera = np.linalg.inv(extL_T_imu)
    T_world_to_camera = T_imu_to_camera @ np.linalg.inv(T_t)
    
    for i, idx in enumerate(landmark_ids):
        mj = mu_t['landmarks'][3 * idx:3 * idx + 3]
        mj_camera = T_world_to_camera @ np.append(mj, 1)
        
        # Predicted observations for left camera
        proj_left = K_l @ mj_camera[:3]
        z_pred[4 * i:4 * i + 2] = proj_left[:2] / proj_left[2]
        
        # Right camera projection (assume T_r = T_l @ baseline translation)
        T_left_to_right = np.eye(4)
        T_left_to_right[0, 3] = -baseline  # Baseline along x-axis
        mj_camera_right = T_left_to_right @ mj_camera
        proj_right = K_l @ mj_camera_right[:3]  # Using K_l for simplicity
        z_pred[4 * i + 2:4 * i + 4] = proj_right[:2] / proj_right[2]
        
        # Jacobian computation
        q = mj_camera[:3]
        z = q[2]
        d_pi_dq = np.array([[1/z, 0, -q[0]/(z**2)],
                           [0, 1/z, -q[1]/(z**2)]])  # 2×3
        
        P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # 3×4
        T_3x3 = (T_world_to_camera @ P.T)[:3, :3]  # 3×3
        
        Ks = K_l[:2, :2]
        H_landmark_left = Ks @ d_pi_dq @ T_3x3  # 2×3 for landmark
        
        # Robot pose Jacobian (simplified, linearized around T_t)
        H_robot_left = -Ks @ d_pi_dq @ skew_symmetric(q)  # 2×3 for rotation
        H_robot_left_full = np.hstack((H_robot_left, np.zeros((2, 3))))  # 2×6
        
        # Right camera Jacobians (similar process)
        q_right = mj_camera_right[:3]
        z_right = q_right[2]
        d_pi_dq_right = np.array([[1/z_right, 0, -q_right[0]/(z_right**2)],
                                 [0, 1/z_right, -q_right[1]/(z_right**2)]])
        H_landmark_right = Ks @ d_pi_dq_right @ T_left_to_right[:3, :3] @ T_3x3
        H_robot_right = -Ks @ d_pi_dq_right @ skew_symmetric(q_right)
        H_robot_right_full = np.hstack((H_robot_right, np.zeros((2, 3))))
        
        # Fill H_t
        H_t[4 * i:4 * i + 2, :6] = H_robot_left_full
        H_t[4 * i:4 * i + 2, 6 + 3 * idx:6 + 3 * idx + 3] = H_landmark_left
        H_t[4 * i + 2:4 * i + 4, :6] = H_robot_right_full
        H_t[4 * i + 2:4 * i + 4, 6 + 3 * idx:6 + 3 * idx + 3] = H_landmark_right
    
    # Innovation covariance and Kalman gain
    S = H_t @ Sigma_t @ H_t.T + np.kron(np.eye(N_t), V)
    K_t = Sigma_t @ H_t.T @ np.linalg.inv(S)
    
    # Update state
    innovation = z_t - z_pred
    delta = K_t @ innovation
    
    # Update robot pose
    delta_pose = delta[:6]
    T_t_next = T_t @ expm(hat_map(delta_pose))
    
    # Update landmarks
    delta_landmarks = delta[6:]
    landmarks_next = mu_t['landmarks'] + delta_landmarks
    
    # Update covariance
    Sigma_t_next = (np.eye(state_size) - K_t @ H_t) @ Sigma_t
    
    return {'pose': T_t_next, 'landmarks': landmarks_next}, Sigma_t_next
def EKF_prediction_joint(mu_t, Sigma_t, u_t, dt, W):
    """Joint prediction step for VI-SLAM: Predict robot pose and propagate covariance."""
    # Extract robot pose (4x4) and landmarks (3M)
    T_t = mu_t['pose']
    landmarks = mu_t['landmarks']  # Shape: (3M,)
    
    # Predict new robot pose
    u_hat = hat_map(u_t)
    T_t_next = T_t @ expm(dt * u_hat)
    
    # Jacobian F for robot pose update (6x6 identity for landmarks, no motion)
    F = np.eye(Sigma_t.shape[0])
    F[:6, :6] = expm(-dt * adjoint(u_t))  # Linearized motion model for pose
    
    # Update joint covariance
    Sigma_t_next = F @ Sigma_t @ F.T + np.block([
        [W, np.zeros((6, Sigma_t.shape[1] - 6))],
        [np.zeros((Sigma_t.shape[0] - 6, 6)), np.zeros((Sigma_t.shape[0] - 6, Sigma_t.shape[1] - 6))]
    ])
    
    return {'pose': T_t_next, 'landmarks': landmarks}, Sigma_t_next
if __name__ == '__main__':
    # Load the measurements
    data_set = "00"
    filename = "../data/dataset" + data_set + "/dataset" + data_set + ".npy"
    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(filename)
    
    ### Variables ###
    T = len(timestamps)
    epsilon = 1e-3
    W = epsilon * np.eye(6)  # Process noise covariance
    V = epsilon * np.eye(2)  # Observation noise covariance
    
    # Part (a): IMU Localization via EKF Prediction
    mu_t = np.eye(4)  # Initial pose
    Sigma_t = epsilon * np.eye(6)  # Initial pose covariance
    trajectory = [mu_t]
    covariances = [Sigma_t]
    
    for t in range(T - 1):
        dt = timestamps[t + 1] - timestamps[t]
        u_t = np.hstack((v_t[t], w_t[t]))
        mu_t_next, Sigma_t_next = EKF_prediction(mu_t, Sigma_t, u_t, dt, W)
        mu_t = mu_t_next
        Sigma_t = Sigma_t_next
        trajectory.append(mu_t)
        covariances.append(Sigma_t)
    
    trajectory = np.array(trajectory)  # Shape: (T, 4, 4)
    print("---- Finished Part (a): IMU Localization ----")
    # Plotting a
    # visualize_trajectory_2d(trajectory, show_ori=True)

    # Part (b): Landmark Mapping via EKF Update
    features_downsampled = filter_and_downsample_features(features, downsample_factor=20)
    T_downsampled = len(features_downsampled)  # Number of downsampled timesteps
    print(T_downsampled )
    baseline = cal_baseline(extL_T_imu, extR_T_imu)
    # print(f"Baseline: {baseline}")
    # print(f"Downsampled features length: {T_downsampled}")
    
    max_landmarks = 1000
    landmark_indices = {}  # {feature_idx: state_idx}
    seen_indices = set()
    mu_landmarks = np.array([])  # Landmark positions
    Sigma_landmarks = np.array([])  # Landmark covariance
    i=0
    for t in range(T_downsampled):
        T_t = trajectory[t * 20]  # Match downsampled time to original trajectory
        current_features = features_downsampled[t]  # Shape: (4, n_valid)
        
        observed_ids = []
        z_t = []
        for idx in range(current_features.shape[1]):
            uL, uR, vL, vR = current_features[:, idx]
            disparity = uL - uR
            if disparity <= 0 or disparity > 100:  # Basic disparity check
                continue
            print(i)
            i+=1
            if idx not in seen_indices and len(landmark_indices) < max_landmarks:
                mj = initialize_landmark(uL, uR, vL, K_l, baseline, T_t, extL_T_imu)
                landmark_indices[idx] = len(landmark_indices)
                seen_indices.add(idx)
                
                if mu_landmarks.size == 0:
                    mu_landmarks = mj
                    Sigma_landmarks = epsilon * np.eye(3)
                else:
                    mu_landmarks = np.hstack((mu_landmarks, mj))
                    Sigma_landmarks = np.block([
                        [Sigma_landmarks, np.zeros((Sigma_landmarks.shape[0], 3))],
                        [np.zeros((3, Sigma_landmarks.shape[1])), epsilon * np.eye(3)]
                    ])
                # print(f"Initialized landmark {idx}: {mj}")
            
            if idx in landmark_indices:
                observed_ids.append(landmark_indices[idx])
                z_t.extend([uL, vL])
        if observed_ids:
            z_t = np.array(z_t)
            mu_landmarks, Sigma_landmarks = ekf_update(
                mu_landmarks, Sigma_landmarks, z_t, observed_ids, T_t, K_l, extL_T_imu, V
            )
            # print(f"Time {t} (original {t*5}): Updated {len(observed_ids)} landmarks")
    
    traj_x = trajectory[:, 0, 3]
    traj_y = trajectory[:, 1, 3]
    print(len(seen_indices))
    # Extract x, y coordinates from landmarks
    M = len(landmark_indices)
    landmark_x = mu_landmarks[0::3]  # Every 3rd element starting at 0 (x)
    landmark_y = mu_landmarks[1::3]  # Every 3rd element starting at 1 (y)
    print(len(landmark_x))
    # Plotting b
    fig, ax = plt.subplots()
    ax.plot(traj_x, traj_y, color='red', label='Trajectory')  # Plot trajectory
    # visualize_trajectory_2d(trajectory, show_ori=True)  # Add orientation (from pr3_utils)
    ax.scatter(landmark_x, landmark_y, s=3, color='blue', label='Landmarks')  # Scatter landmarks
    ax.legend()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Visual-Inertial SLAM: Trajectory and Landmarks')
    plt.grid(True)
    plt.show()
    print("---- Finished Part (b): Landmark Mapping ----")

    # Part (c): Visual-Inertial SLAM
    T = len(timestamps)
    epsilon = 1e-3
    W = np.eye(6) * 1e-4  # Tuned process noise (smaller for IMU trust)
    V = np.eye(4) * 1e-2  # Tuned observation noise (larger for stereo uncertainty)
    baseline = cal_baseline(extL_T_imu, extR_T_imu)

    features_downsampled = filter_and_downsample_features(features, downsample_factor=20)
    T_downsampled = len(features_downsampled)
    
    # Initialize joint state
    mu_t = {'pose': np.eye(4), 'landmarks': np.array([])}
    Sigma_t = np.zeros((6, 6))  # Start with pose covariance only
    Sigma_t[:6, :6] = epsilon * np.eye(6)
    trajectory = [mu_t['pose']]
    
    max_landmarks = 1000
    landmark_indices = {}
    seen_indices = set()
    
    for t in range(T - 1):
        dt = timestamps[t + 1] - timestamps[t]
        u_t = np.hstack((v_t[t], w_t[t]))
        
        # Prediction step
        mu_t, Sigma_t = EKF_prediction_joint(mu_t, Sigma_t, u_t, dt, W)
        
        # Update step (every 20th frame to match downsampled features)
        if t % 20 == 0 and t // 20 < T_downsampled:
            current_features = features_downsampled[t // 20]
            observed_ids = []
            z_t = []
            
            for idx in range(current_features.shape[1]):
                uL, uR, vL, vR = current_features[:, idx]
                disparity = uL - uR
                if disparity <= 0 or disparity > 100:
                    continue
                    
                if idx not in seen_indices and len(landmark_indices) < max_landmarks:
                    mj = initialize_landmark(uL, uR, vL, K_l, baseline, mu_t['pose'], extL_T_imu)
                    landmark_indices[idx] = len(landmark_indices)
                    seen_indices.add(idx)
                    
                    if mu_t['landmarks'].size == 0:
                        mu_t['landmarks'] = mj
                        Sigma_t = np.block([
                            [Sigma_t, np.zeros((6, 3))],
                            [np.zeros((3, 6)), epsilon * np.eye(3)]
                        ])
                    else:
                        mu_t['landmarks'] = np.hstack((mu_t['landmarks'], mj))
                        Sigma_t = np.block([
                            [Sigma_t, np.zeros((Sigma_t.shape[0], 3))],
                            [np.zeros((3, Sigma_t.shape[1])), epsilon * np.eye(3)]
                        ])
                    print(f"Initialized landmark {idx}: {mj}")
                
                if idx in landmark_indices:
                    observed_ids.append(landmark_indices[idx])
                    z_t.extend([uL, vL, uR, vR])  # 4D observation
            
            if observed_ids:
                z_t = np.array(z_t)
                mu_t, Sigma_t = ekf_update_joint(mu_t, Sigma_t, z_t, observed_ids, mu_t['pose'], K_l, extL_T_imu, V)
                print(f"Time {t}: Updated {len(observed_ids)} landmarks and pose")
        
        trajectory.append(mu_t['pose'])
    
    trajectory = np.array(trajectory)
    
    # Visualization
    traj_x = trajectory[:, 0, 3]
    traj_y = trajectory[:, 1, 3]
    landmark_x = mu_t['landmarks'][0::3]
    landmark_y = mu_t['landmarks'][1::3]
    # # Plotting c
    # fig, ax = plt.subplots()
    # ax.plot(traj_x, traj_y, color='red', label='Trajectory')
    # # visualize_trajectory_2d(trajectory, show_ori=True, ax=ax)
    # ax.scatter(landmark_x, landmark_y, s=3, color='blue', label='Landmarks')
    # ax.legend()
    # ax.set_xlabel('X (m)')
    # ax.set_ylabel('Y (m)')
    # ax.set_title('Visual-Inertial SLAM: Trajectory and Landmarks')
    # plt.grid(True)
    # plt.show()
    print("---- Finished Part (c): Visual-Inertial SLAM ----")