{\rtf1\ansi\ansicpg1252\cocoartf2757
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;\f1\fmodern\fcharset0 Courier-Bold;\f2\froman\fcharset0 Times-Roman;
\f3\fmodern\fcharset0 Courier-BoldOblique;\f4\fmodern\fcharset0 Courier-Oblique;}
{\colortbl;\red255\green255\blue255;\red19\green19\blue19;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c9804\c9804\c9804;\cssrgb\c0\c0\c0;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs24\fsmilli12333 \cf2 \expnd0\expndtw0\kerning0
from matplotlib import pyplot 
\f1\b as
\f0\b0  plt
\f2\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f3\i\b\fs24\fsmilli12333 \cf2 # Plot robot position from odometry
\f2\i0\b0\fs22 \cf3 \

\f3\i\b\fs24\fsmilli12333 \cf2 # Use blue cross marker to designate the robot position
\f2\i0\b0\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f0\fs24\fsmilli12333 \cf2 pose 
\f1\b =
\f0\b0  robot
\f1\b .
\f0\b0 last_odom_msg
\f1\b .
\f0\b0 pose
\f1\b .
\f0\b0 pose
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 plt
\f1\b .
\f0\b0 plot([pose
\f1\b .
\f0\b0 position
\f1\b .
\f0\b0 x],[pose
\f1\b .
\f0\b0 position
\f1\b .
\f0\b0 y],'bx') 
\f4\i # TODO: show robot front
\f2\i0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f3\i\b\fs24\fsmilli12333 \cf2 # Plot lidar points from last message
\f2\i0\b0\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f0\fs24\fsmilli12333 \cf2 def plot_lidar_scan_points(msg,pose):
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 
\f3\i\b # Convert received lidar points into cartesian coordinates considering the current position from odometry
\f2\i0\b0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 import numpy
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 points 
\f1\b =
\f0\b0  [(numpy
\f1\b .
\f0\b0 cos(angle)
\f1\b *
\f0\b0 radius,numpy
\f1\b .
\f0\b0 sin(angle)
\f1\b *
\f0\b0 radius) 
\f1\b for
\f0\b0  angle, radius \\
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 
\f1\b in
\f0\b0  zip(numpy
\f1\b .
\f0\b0 linspace(msg
\f1\b .
\f0\b0 angle_min,msg
\f1\b .
\f0\b0 angle_max,len(msg
\f1\b .
\f0\b0 ranges)),msg
\f1\b .
\f0\b0 ranges)]
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 T 
\f1\b =
\f0\b0  robot
\f1\b .
\f0\b0 reduce_transform_to_2D(robot
\f1\b .
\f0\b0 convert_odom_to_transform(pose))
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 T_base_lidar 
\f1\b =
\f0\b0  numpy
\f1\b .
\f0\b0 array([[ 2.22044605e-16, 
\f1\b -
\f0\b0 1.00000000e+00, 
\f1\b -
\f0\b0 4.00000000e-02],\'a0\'a0\'a0\'a0\'a0\'a0 [ 1.00000000e+00,\'a0 2.22044605e-16,\'a0 0.00000000e+00],\'a0\'a0\'a0\'a0\'a0\'a0 [ 0.00000000e+00,\'a0 0.00000000e+00,\'a0 1.00000000e+00]])
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 
\f3\i\b # remove points which cannot be measured
\f2\i0\b0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 filtered_points 
\f1\b =
\f0\b0  filter(lambda x: numpy
\f1\b .
\f0\b0 isfinite(x)
\f1\b .
\f0\b0 all(),points)
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 
\f3\i\b # display points in world frame
\f2\i0\b0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 transformed_points 
\f1\b =
\f0\b0  [numpy
\f1\b .
\f0\b0 matmul(numpy
\f1\b .
\f0\b0 dot(T,T_base_lidar), numpy
\f1\b .
\f0\b0 vstack([ numpy
\f1\b .
\f0\b0 atleast_2d(x)
\f1\b .
\f0\b0 T,numpy
\f1\b .
\f0\b0 ones((1,1)) ]) ) for x in filtered_points]
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 plt
\f1\b .
\f0\b0 plot(
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 [x[0] for x in transformed_points],
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 [x[1] for x in transformed_points],'r.')
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 plot_lidar_scan_points(robot
\f1\b .
\f0\b0 last_scan_msg,pose)
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 import numpy as np
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 from sklearn.linear_model import RANSACRegressor
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 from matplotlib import pyplot as plt
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 from sklearn.cluster import DBSCAN
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 from geometry_msgs.msg import Twist
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 from sklearn.linear_model import LinearRegression
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 from sklearn.preprocessing import PolynomialFeatures
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 from scipy.optimize import minimize
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 import time
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f3\i\b\fs24\fsmilli12333 \cf2 # Function to reduce 3D transformation to 2D
\f2\i0\b0\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f0\fs24\fsmilli12333 \cf2 def reduce_transform_to_2D(transform):
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 
\f3\i\b # Assuming transform is a 4x4 matrix
\f2\i0\b0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 return transform[:2, :2]
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f3\i\b\fs24\fsmilli12333 \cf2 # Function to convert odometry to transformation
\f2\i0\b0\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f0\fs24\fsmilli12333 \cf2 def convert_odom_to_transform(pose):
\f2\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f3\i\b\fs24\fsmilli12333 \cf2 # Convert the 2D pose (position and orientation) to a 3x3 homogeneous transformation matrix
\f2\i0\b0\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 return np.array([
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 [np.cos(pose.orientation.z), -np.sin(pose.orientation.z), pose.position.x],
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 [np.sin(pose.orientation.z), np.cos(pose.orientation.z), pose.position.y],
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 [0, 0, 1]
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 ])
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 def rotate(angle):
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 
\f3\i\b # Create a Twist message
\f2\i0\b0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 twist = Twist()
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 
\f3\i\b # Set the angular velocity in the z-axis to achieve the desired rotation
\f2\i0\b0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 twist.angular.z = angle / abs(angle) * 0.5\'a0 
\f3\i\b # Adjust the value
\f2\i0\b0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 
\f3\i\b # Publish the message for a certain duration to achieve the desired rotation
\f2\i0\b0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 start_time = robot.get_clock().now()
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 while robot.get_clock().now() - start_time < rclpy.time.Duration(seconds=abs(angle) / 0.5):
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 robot.cmd_vel_publisher.publish(twist)
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 time.sleep(0.1)\'a0 
\f3\i\b # Sleep for 100 ms
\f2\i0\b0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 
\f3\i\b # Stop the rotation
\f2\i0\b0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 twist.angular.z = 0.0
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 robot.cmd_vel_publisher.publish(twist)
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 robot.set_cmd_vel(0.1, 0.0, 0.2)
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 def plot_lidar_scan_points(msg, pose):
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 points = [(np.cos(angle) * radius, np.sin(angle) * radius, 1) for angle, radius \\
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 in zip(np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges)), msg.ranges)]
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 T = convert_odom_to_transform(pose)
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 
\f3\i\b # Remove points which cannot be measured
\f2\i0\b0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 filtered_points = filter(lambda x: np.isfinite(x).all(), points)
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 transformed_points = [np.matmul(T, x) for x in filtered_points]
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 transformed_points = [x[:2] for x in transformed_points] 
\f4\i # Reduce the homogeneous coordinates back to 2D
\f2\i0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0 
\f3\i\b # Convert to array for clustering
\f2\i0\b0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 transformed_points_array = np.array(transformed_points)
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 
\f3\i\b # Apply DBSCAN algorithm to cluster the points
\f2\i0\b0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 clustering = DBSCAN(eps=0.5, min_samples=5).fit(transformed_points_array)
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 
\f3\i\b # Find the cluster with the most points (above threshold)
\f2\i0\b0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 threshold_value = 50 
\f3\i\b # Example threshold
\f2\i0\b0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 largest_cluster = []
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 for cluster_id in set(clustering.labels_):
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 if cluster_id == -1:
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 continue 
\f3\i\b # Ignore noise (-1 label)
\f2\i0\b0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 cluster_points = transformed_points_array[clustering.labels_ == cluster_id]
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0 if len(cluster_points) > threshold_value and len(cluster_points) > len(largest_cluster):
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 largest_cluster = cluster_points
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 plt.plot([x[0] for x in largest_cluster], [x[1] for x in largest_cluster], 'r.')
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 return largest_cluster
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 T_base_lidar = np.array([[ 2.22044605e-16, -1.00000000e+00, -4.00000000e-02],\'a0\'a0\'a0\'a0\'a0\'a0 [ 1.00000000e+00,\'a0 2.22044605e-16,\'a0 0.00000000e+00],\'a0\'a0\'a0\'a0\'a0\'a0 [ 0.00000000e+00,\'a0 0.00000000e+00,\'a0 1.00000000e+00]])
\f2\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f3\i\b\fs24\fsmilli12333 \cf2 # Assuming robot.last_odom_msg and robot.last_scan_msg contain the necessary information
\f2\i0\b0\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f0\fs24\fsmilli12333 \cf2 pose = robot.last_odom_msg.pose.pose
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 plt.plot([pose.position.x], [pose.position.y], 'bx') 
\f3\i\b # Robot position
\f2\i0\b0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 transformed_points = plot_lidar_scan_points(robot.last_scan_msg, pose)
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 transformed_points_array = np.array(transformed_points)
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f3\i\b\fs24\fsmilli12333 \cf2 # Extract X and Y coordinates
\f2\i0\b0\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f0\fs24\fsmilli12333 \cf2 X = transformed_points_array[:, 0].reshape(-1, 1)
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 y = transformed_points_array[:, 1]
\f2\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f3\i\b\fs24\fsmilli12333 \cf2 # Apply RANSAC algorithm to fit a line to the points
\f2\i0\b0\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f0\fs24\fsmilli12333 \cf2 ransac = RANSACRegressor()
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 ransac.fit(X, y)
\f2\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f3\i\b\fs24\fsmilli12333 \cf2 # Plot the line
\f2\i0\b0\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f0\fs24\fsmilli12333 \cf2 line_X = np.array([np.min(X), np.max(X)])
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 line_y = ransac.predict(line_X.reshape(-1, 1))
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 plt.plot(line_X, line_y, 'g-') 
\f3\i\b # Line in green color
\f2\i0\b0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f3\i\b\fs24\fsmilli12333 \cf2 # Determine the slope of the fitted line
\f2\i0\b0\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f0\fs24\fsmilli12333 \cf2 m = (line_y[1] - line_y[0]) / (line_X[1] - line_X[0])
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f3\i\b\fs24\fsmilli12333 \cf2 # Compute the angle of inclination to the x-axis in radians
\f2\i0\b0\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f0\fs24\fsmilli12333 \cf2 theta_rad = np.arctan(m)
\f2\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f3\i\b\fs24\fsmilli12333 \cf2 #theta_deg = theta_rad * (180 / np.pi)
\f2\i0\b0\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 print(f"Angle of the fitted line with respect to the x-axis: \{theta_rad:.2f\} radians")
\f2\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f3\i\b\fs24\fsmilli12333 \cf2 # Calculate the distance between the robot and the wall
\f2\i0\b0\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f0\fs24\fsmilli12333 \cf2 distance_to_wall = np.abs(line_y[0] - pose.position.y) / np.sqrt(1 + m ** 2)
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 print(f"Distance to the wall: \{distance_to_wall:.2f\} meters")
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f3\i\b\fs24\fsmilli12333 \cf2 # Define target distance to the wall
\f2\i0\b0\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f0\fs24\fsmilli12333 \cf2 target_distance = 0.5 
\f4\i # meters
\f2\i0\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f3\i\b\fs24\fsmilli12333 \cf2 # Calculate rotation angle based on the distance to the wall
\f2\i0\b0\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f0\fs24\fsmilli12333 \cf2 if distance_to_wall > target_distance:
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 rotation_angle = theta_rad
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 else:
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0\'a0\'a0 rotation_angle = np.pi - theta_rad
\f2\fs22 \cf3 \

\f0\fs24\fsmilli12333 \cf2 \'a0
\f2\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f3\i\b\fs24\fsmilli12333 \cf2 # Apply rotation to the robot
\f2\i0\b0\fs22 \cf3 \
\pard\pardeftab720\partightenfactor0

\f0\fs24\fsmilli12333 \cf2 rotate(rotation_angle)
\f2\fs22 \cf3 \
}