
# Visual-Inertial SLAM

## Overview
This repository contains my implementation of Project 3 for ECE 276A: Sensing & Estimation in Robotics at UCSD, due on March 21, 2025. The project focuses on implementing a Visual-Inertial Simultaneous Localization and Mapping (SLAM) system using an Extended Kalman Filter (EKF). The system integrates IMU measurements and stereo camera observations to estimate the robot's trajectory and map landmark positions. The dataset is collected from Clearpath Jackal robots navigating MIT's campus.

For more details, refer to the full assignment description [here](#) (replace with a link if available) and my project report in `report.pdf`.

## Repository Structure
- **`/src`**: Contains all the source code for the project.
  - `main.py`: The main script to run the Visual-Inertial SLAM algorithm.
  - `ekf_prediction.py`: Implements the EKF prediction step using IMU data.
  - `ekf_update.py`: Implements the EKF update step for landmark mapping and IMU pose correction.
  - `feature_tracking.py` (optional): Implements feature detection and matching for dataset02 (extra credit).
  - `utils.py`: Utility functions for data processing, calibration, and visualization.
- **`/results`**: Stores output plots, trajectories, and optional videos.
  - `imu_trajectory.png`: Plot of the estimated IMU trajectory.
  - `landmark_map.png`: Plot of the estimated landmark positions.
- **`report.pdf`**: The project report detailing the problem, approach, and results.
- **`README.md`**: This file.

## Dependencies
To run the code, ensure you have the following Python libraries installed:
- `numpy`
- `scipy` (for sparse matrix operations like `csr_matrix`)
- `opencv-python` (for feature detection and tracking)
- `matplotlib` (for visualization)

Install them using:
```bash
pip install numpy scipy opencv-python matplotlib
```

## How to Run the Code
1. **Download the Dataset**: The required datasets (IMU measurements, stereo images, etc.) are available at the [UCSD Cloud link](https://ucsdcloud-ny.sharepoint.com/:f:/g/personal/natanasov_ucsd_edu/E1KB_H_Rfu9GuA38wucC_UEBNBOHKK4Bs-339Mq-p-F5Ew). Place them in a folder named `/data` within this repository (not included here due to size).
2. **Configure the Code**: Update the file paths in `main.py` to point to your local dataset directory if necessary.
3. **Run the Main Script**:
   ```bash
   python src/main.py
   ```
   This will execute the full Visual-Inertial SLAM pipeline, including:
   - EKF prediction for IMU localization.
   - (Optional) Feature detection and tracking for dataset02.
   - EKF update for landmark mapping and IMU pose correction.
   - Visualization of the trajectory and landmark map saved in `/results`.

## Main File Description
- **`main.py`**: The entry point of the project. It loads the dataset, runs the EKF-based SLAM algorithm, and generates visualizations. The script is structured as follows:
  - Data loading and preprocessing.
  - EKF prediction step using IMU measurements.
  - (Optional) Feature detection and tracking for dataset02.
  - EKF update step integrating stereo camera observations.
  - Plotting and saving results.



---
