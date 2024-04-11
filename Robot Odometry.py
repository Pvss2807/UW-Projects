Drive robot and aquire odometry

# Use preprogrammed behavior: Drive off the dock in a controlled way
robot.undock()

# Display raw data: Odometry based position
robot.odom_future = rclpy.Future()
pose1 = robot.spin_until_future_completed(robot.odom_future).pose.pose
print(pose1.position)
print(pose1.orientation)

# Drive with desired velocity command:
velocity_x = 0.1 # in m/s
velocity_phi = 0.0 # for rotation test use 0.5 rad/s
duration = 2.0 # in s
robot.set_cmd_vel(velocity_x, velocity_phi, duration)

# Display raw data: Odometry based position
robot.odom_future = rclpy.Future()
pose2 = robot.spin_until_future_completed(robot.odom_future).pose.pose
print(pose2.position)
print(pose2.orientation)

# translation
print("Desired distance: {}".format(velocity_x*duration))
import numpy
print("Measured distance: {}".format(
    numpy.sqrt( (pose1.position.x - pose2.position.x)**2 + (pose1.position.y - pose2.position.y)**2 )))

# rotation
import eigenpy # makes use of the Eigen C++ library ()
def make_quaternion(q):
    return eigenpy.Quaternion(q.w,q.x,q.y,q.z)
print("Desired rotation: {}".format(
    velocity_phi * duration))
print("Measured rotation: {}".format(
    make_quaternion(pose1.orientation).angularDistance(make_quaternion(pose2.orientation))))


------------------------------------------------------------------------------------------------

Visual Representation of the desired and measured calculation with somewhat different approach


import numpy as np
import eigenpy
import matplotlib.pyplot as plt

# Given data (replace these values with your actual data)
velocity_x = 0.1
velocity_phi = 0.0
duration = 2.0

# Calculate desired distance and rotation
desired_distance = velocity_x * duration

# Define a function to calculate the angular distance between two quaternions
def angular_distance(q1, q2):
    return eigenpy.Quaternion(q1).angularDistance(eigenpy.Quaternion(q2))

# Calculate desired rotation
desired_rotation = velocity_phi * duration

# Sample data (replace these values with your actual data)
pose1_position = np.array([1.1809054613113403, -1.800580620765686])  
pose2_position = np.array([1.1232298612594604, -1.9839327335357666])  
pose1_orientation = np.array([-0.001508902758359909, 2.1100277081131935e-05, -0.8044115900993347, 0.594070553779602])  # Replace with actual pose1 orientation (w, x, y, z)
pose2_orientation = np.array([-0.0030553662218153477, -0.0009766248986124992, -0.8045011162757874, 0.5939425230026245])  # Replace with actual pose2 orientation (w, x, y, z)

# Calculate measured distance and rotation
measured_distance = np.sqrt((pose1_position[0] - pose2_position[0]) ** 2 + (pose1_position[1] - pose2_position[1]) ** 2)
measured_rotation = angular_distance(pose1_orientation, pose2_orientation)

# Calculate differences
distance_difference = desired_distance - measured_distance
rotation_difference = desired_rotation - measured_rotation

# Plot the differences
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(['Desired', 'Measured'], [desired_distance, measured_distance])
plt.ylabel('Distance')
plt.title('Desired vs. Measured Distance')

plt.subplot(1, 2, 2)
plt.bar(['Desired', 'Measured'], [desired_rotation, measured_rotation])
plt.ylabel('Rotation')
plt.title('Desired vs. Measured Rotation')

plt.tight_layout()
plt.show()