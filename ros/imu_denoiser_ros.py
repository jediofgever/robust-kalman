"""
Using Robust Kalmans to denoise IMU data
========================================
Author: Fetullah Atas (github: jediofgever)
"""
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '..')

from robust_kalman import RobustKalman
from robust_kalman.utils import HuberScore, VariablesHistory, WindowStatisticsEstimator

import rclpy
from sensor_msgs.msg import Imu
 
rclpy.init()
node = rclpy.create_node('kalman_imu_denoiser')

# Define a linear state space model
dt = 0.01
sigma_process = 10.0
sigma_measure = 0.1
param_list = []

# Angular velocity x idx -> 0
# Angular velocity y idx -> 1
# Angular velocity z idx -> 2
# Linear acceleration x idx -> 3
# Linear acceleration y idx -> 4
# Linear acceleration z idx -> 5
# Create a map of parameters for each of the 6 states

for i in range(0, 6):
    F = np.array([[1, dt], [0, 1]], np.float32)
    G = np.array([[0.5 * dt**2, dt]], np.float32).T
    H = np.array([[1, 0]], np.float32)
    x0 = np.array([[0.01, 0.01]], np.float32).T
    P0 = np.ones((2, 2), np.float32) * 0.001
    x0_kalman = np.array([[0, 0]], np.float32).T
    if i == 5:
        x0_kalman = np.array([[9.81, 0]], np.float32).T
        
    Q0 = np.matmul(G, G.T) * sigma_process**2
    R0 = np.eye(1, dtype=np.float32) * sigma_measure**2
    # Create instance of the robust Kalman filter filter
    kalman_robust = RobustKalman(F, None, H, x0_kalman, P0, Q0, R0, use_robust_estimation=True)
    # Initializ.en
    x = x0
    z = np.matmul(H, x0)
    # Create a dictionary of parameters to be passed to the robust estimator
    
    estimator_params = {
        "F" : F,
        "G" : G,
        "H" : H, 
        "x0" : x0_kalman, 
        "P0" : P0, 
        "Q0" : Q0,
        "R0" : R0, 
        "kalman_filter" : kalman_robust,
        "x": x,
        "z": z}
    param_list.append(estimator_params)
            

def imu_callback(msg: Imu):
    
    denoised_imu = Imu()
    denoised_imu = msg
    var = msg.angular_velocity.x
    
    for i in range(0, 6):
        if i == 0:
            var = msg.angular_velocity.x
        elif i == 1:
            var = msg.angular_velocity.y
        elif i == 2:
            var = msg.angular_velocity.z
        elif i == 3:
            var = msg.linear_acceleration.x                
        elif i == 4:
            var = msg.linear_acceleration.y
        else:
            var = msg.linear_acceleration.z       
        estimator_params = param_list[i]
            
        q =  np.random.normal(0.0, sigma_process, size=(1, 1))
        x = np.array([[var, var]], np.float32).T

        x = np.matmul(estimator_params['F'], x) + np.matmul(estimator_params['G'], q)
        z = np.matmul(estimator_params['H'], x) 
        
  
        estimator_params['kalman_filter'].time_update()
        estimator_params['kalman_filter'].measurement_update(z)
    
        denoise_var = estimator_params['kalman_filter'].current_estimate
        if i == 0:
            denoised_imu.angular_velocity.x = float(denoise_var[0][0])
        elif i == 1:
            denoised_imu.angular_velocity.y = float(denoise_var[0][0])
        elif i == 2:
            denoised_imu.angular_velocity.z = float(denoise_var[0][0])
        elif i == 3:
            denoised_imu.linear_acceleration.x = float(denoise_var[0][0])
        elif i == 4:
            denoised_imu.linear_acceleration.y = float(denoise_var[0][0])
        else:
            denoised_imu.linear_acceleration.z = float(denoise_var[0][0])                
                
    corr_sub.publish(msg)

imu_sub = node.create_subscription(Imu, '/dobbie/sensing/imu/tamagawa/imu_raw', imu_callback, 10)
corr_sub = node.create_publisher(Imu,'/dobbie/sensing/imu/tamagawa/imu_denoised', 10)

rclpy.spin(node)
 
