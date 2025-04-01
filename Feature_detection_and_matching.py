import numpy as np
import cv2
from scipy.linalg import expm
from pr3_utils import * 
import matplotlib.pyplot as plt
from main import *
### Functions ###

def detect_initial_features(frame, max_corners=1000, quality_level=0.01, min_distance=10):
    # Convert to grayscale if needed
    if len(frame.shape) == 2:
        gray = frame
    elif len(frame.shape) == 3 and frame.shape[2] in [3, 4]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {frame.shape}")
    
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=quality_level, 
                                     minDistance=min_distance)
    if corners is None or len(corners) == 0:
        # Return an empty array with the proper shape if no features are found.
        return np.array([]).reshape(0, 1, 2)
    return corners

def stereo_match(left_frame, right_frame, features_left):
    # Convert frames to grayscale if needed
    if len(left_frame.shape) == 2:
        gray_left = left_frame
    else:
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    
    if len(right_frame.shape) == 2:
        gray_right = right_frame
    else:
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, 
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Calculate optical flow from left to right image
    features_right, status, err = cv2.calcOpticalFlowPyrLK(gray_left, gray_right, features_left, None, **lk_params)
    
    # If optical flow fails, return empty arrays with the appropriate shape
    if status is None:
        print("stereo_match: calcOpticalFlowPyrLK failed, returning empty features")
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
    
    # Flatten the status array to match the number of feature points
    status = status.ravel()
    if status.shape[0] != features_left.shape[0]:
        raise ValueError(f"Status shape {status.shape} does not match features_left shape {features_left.shape}")
    
    # Use boolean indexing to select only the successfully tracked features
    good_left = features_left[status == 1].reshape(-1, 2)
    good_right = features_right[status == 1].reshape(-1, 2)
    return good_left, good_right

def temporal_track(prev_frame, curr_frame, prev_features):

    # Ensure prev_features has the required shape for optical flow (N, 1, 2)
    if prev_features.ndim == 2:
        prev_features = prev_features.reshape(-1, 1, 2)
    
    # Convert frames to grayscale if needed
    if len(prev_frame.shape) == 2:
        gray_prev = prev_frame
    else:
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    if len(curr_frame.shape) == 2:
        gray_curr = curr_frame
    else:
        gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Lucas-Kanade parameters for temporal tracking
    lk_params = dict(winSize=(15, 15), maxLevel=2, 
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Compute optical flow from prev_frame to curr_frame
    curr_features, status, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, prev_features, None, **lk_params)
    
    if status is None:
        print("temporal_track: calcOpticalFlowPyrLK failed, returning empty features")
        return np.array([]).reshape(0, 1, 2), np.array([]).reshape(0, 1, 2)
    
    # Flatten status to one dimension for correct boolean indexing
    status = status.ravel()
    good_prev = prev_features[status == 1].reshape(-1, 2)
    good_curr = curr_features[status == 1].reshape(-1, 2)
    return good_prev, good_curr

def build_feature_tracks_from_images(data_file, timestamps):

    data = np.load(data_file, allow_pickle=True).item()
    left_images = data['cam_imgs_L']
    right_images = data['cam_imgs_R']
    T = len(timestamps)
    
    if len(left_images) != T or len(right_images) != T:
        raise ValueError("Number of images does not match timestamps.")
    
    all_features = {}  # Dictionary to store all features by unique id
    feature_counter = 0  # Counter for unique feature IDs
    z_t_list = []  # List of measurement matrices (each column corresponds to a feature measurement)
    
    # Detect initial features in the first left image
    prev_frame_left = left_images[0]
    prev_features = detect_initial_features(prev_frame_left)
    if prev_features is None or len(prev_features) == 0:
        raise ValueError("No initial features detected in the first frame.")
    
    for t in range(T):
        curr_frame_left = left_images[t]
        curr_frame_right = right_images[t]
        
        # Perform stereo matching between current left and right images
        features_left, features_right = stereo_match(curr_frame_left, curr_frame_right, prev_features)
        
        if len(features_left) == 0:
            print(f"Warning: No features tracked at time {t}, skipping frame.")
            # Create a dummy measurement matrix with -1 for missing features
            z_t_list.append(np.full((4, len(all_features)), -1, dtype=np.float32))
            if t < T - 1:
                next_frame_left = left_images[t + 1]
                # Attempt to track features temporally even if stereo tracking failed
                _, prev_features = temporal_track(curr_frame_left, next_frame_left, prev_features)
            continue
        
        current_features = {}
        # Process each tracked feature from stereo matching
        for i in range(len(features_left)):
            # Unpack the left image feature (already reshaped to (2,))
            lx, ly = features_left[i]
            # Get corresponding right image feature if available, otherwise assign default values
            rx, ry = features_right[i] if i < len(features_right) else (-1, -1)
            feature_id = feature_counter + i
            # Save the measurements for the feature
            all_features[feature_id] = [lx, ly, rx, ry]
            current_features[feature_id] = [lx, ly, rx, ry]
        feature_counter += len(features_left)
        
        # Build a measurement matrix for the current time step.
        # The matrix has 4 rows (uL, vL, uR, vR) and columns for each feature.
        M = len(all_features)
        z_t = np.full((4, M), -1, dtype=np.float32)
        for j, fid in enumerate(all_features.keys()):
            if fid in current_features:
                z_t[:, j] = current_features[fid]
        z_t_list.append(z_t)
        
        # For the next time step, track features temporally from the current left image.
        if t < T - 1:
            next_frame_left = left_images[t + 1]
            _, prev_features = temporal_track(curr_frame_left, next_frame_left, features_left)
            if len(prev_features) == 0:
                print(f"Warning: Temporal tracking failed at time {t}, reinitializing features.")
                # If tracking fails, re-detect features in the next frame.
                prev_features = detect_initial_features(next_frame_left)
    
    return z_t_list, all_features



if __name__ == '__main__':
    # Load dataset and calibration parameters
    data_set = "02"
    filename = f"../data/dataset{data_set}/dataset{data_set}.npy"
    v_t, w_t, timestamps, _, K_l, K_r, extL_T_imu, extR_T_imu = load_data(filename)
    
    # Load image data
    image_file = f"../data/dataset{data_set}/dataset{data_set}_imgs.npy"
    
    # Build feature tracks from images
    z_t_list, all_features = build_feature_tracks_from_images(image_file, timestamps)
    print(f"Total unique features detected: {len(all_features)}")
    
    
    # Setup variables for SLAM
    T = len(timestamps)
    epsilon = 1e-3
    W = np.eye(6) * 1e-4
    V = np.eye(4) * 1e-2
    baseline = cal_baseline(extL_T_imu, extR_T_imu)
    
    # Initialize the state and covariance for the EKF (robot pose and landmarks)
    T_downsampled = len(z_t_list)
    mu_t = {'pose': np.eye(4), 'landmarks': np.array([])}
    Sigma_t = np.zeros((6, 6))
    Sigma_t[:6, :6] = epsilon * np.eye(6)
    trajectory = [mu_t['pose']]
    
    max_landmarks = 50
    landmark_indices = {}
    seen_indices = set()
    
    # Main loop for Visual-Inertial SLAM
    for t in range(T - 1):
        dt = timestamps[t + 1] - timestamps[t]
        u_t = np.hstack((v_t[t], w_t[t]))
        
        # Prediction step of EKF
        mu_t, Sigma_t = EKF_prediction_joint(mu_t, Sigma_t, u_t, dt, W)
        
        # Every 20th frame, perform an EKF update with the stereo measurements
        if t % 20 == 0 and t // 20 < T_downsampled:
            z_t = z_t_list[t // 20]
            observed_ids = []
            z_t_filtered = []
            
            # Filter out invalid measurements and initialize new landmarks if needed
            for j in range(z_t.shape[1]):
                uL, vL, uR, vR = z_t[:, j]
                if np.all(z_t[:, j] == -1):
                    continue
                disparity = uL - uR
                if disparity <= 0 or disparity > 100:
                    continue
                
                if j not in seen_indices and len(landmark_indices) < max_landmarks:
                    # Initialize a new landmark from stereo observations
                    mj = initialize_landmark(uL, uR, vL, K_l, baseline, mu_t['pose'], extL_T_imu)
                    landmark_indices[j] = len(landmark_indices)
                    seen_indices.add(j)
                    
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
                    print(f"Initialized landmark {j}: {mj}")
                
                if j in landmark_indices:
                    observed_ids.append(landmark_indices[j])
                    z_t_filtered.extend([uL, vL, uR, vR])
            
            if observed_ids:
                z_t_filtered = np.array(z_t_filtered)
                mu_t, Sigma_t = ekf_update_joint(mu_t, Sigma_t, z_t_filtered, observed_ids, mu_t['pose'], K_l, extL_T_imu, V)
                print(f"Time {t}: Updated {len(observed_ids)} landmarks and pose")
        
        trajectory.append(mu_t['pose'])
    
    trajectory = np.array(trajectory)
    print("---- Finished Visual-Inertial SLAM with Custom Image Features ----")
    
    # Visualization of the trajectory and landmarks
    traj_x = trajectory[:, 0, 3]
    traj_y = trajectory[:, 1, 3]
    landmark_x = mu_t['landmarks'][0::3]
    landmark_y = mu_t['landmarks'][1::3]
    
    fig, ax = plt.subplots()
    ax.plot(traj_x, traj_y, color='red', label='Trajectory')
    visualize_trajectory_2d(trajectory, show_ori=True, ax=ax)
    ax.scatter(landmark_x, landmark_y, s=3, color='blue', label='Landmarks')
    ax.legend()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('VI-SLAM with Image Feature Detection (Dataset02)')
    plt.grid(True)
    plt.show()
