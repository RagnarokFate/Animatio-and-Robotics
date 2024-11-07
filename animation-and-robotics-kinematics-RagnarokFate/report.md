# Kinematics Assignment 

Student & ID: Bashar Beshoti - 207370248

## Task 1: Extend the implementation of the FK function to any number of joints.

### Task 1.1 : Change SimpleArm such that the constructor can accept different link lengths.

<image src="images\DifferentArmJoints.png" width="720" height="405"/>

<em> Figure 1 : An arm with 5 joints that the Link Lengths are [0.25,0.5,0.75,1,1.25] </em>

<image src="images\5JointsPose-1.png" width="720" height="405"/>

<em> Figure 2 : The arm with the 5 joints from Figure 1 with a robot pose. </em>

<image src="images\5JointsPose-2.png" width="720" height="405"/>

<em> Figure 3 : Another robot pose. </em>


### Task 1.2 : The FK method computes the position of each joint in world coordinates explicitly.

<image src="images\FK-1.png" width="720" height="405"/>

<em> Figure 4 : Forward Kinematics pose with angles degree of `[0,30,60,90,120]`. </em>


```py
# TEMP
angles = [0,30,60,90,120]
angles_rad = np.deg2rad(angles)
print("Initial angles: ",angles) # Initial angles:  [0, 30, 60, 90, 120]
print("Initial angles in radians: ",angles_rad) # Initial angles in radians:  [0 0.52359878 1.04719755 1.57079633 2.0943951 ]

arm.FK(angles_rad)
```


## Task 2: Gradient descent based IK (the Jacobian transpose method)

### Implement the VelocityJacobian method in SimpleArm.

The `VelocityJacobian` function computes the Jacobian matrix, which relates the joint velocities to the end effector's linear and angular velocities. Each column of the Jacobian matrix corresponds to a joint, and each row corresponds to a degree of freedom (DOF) of the end effector

```
Velocity Jacobian :
 [[0.   0.   0.   0.   0.  ]
 [3.75 3.5  3.   2.25 1.25]
 [0.   0.   0.   0.   0.  ]]
 ```

<image src="images\VelocityJacobian-1.png" width="720" height="405"/>

<em> Figure 5 : Inital Velocity Jacobian </em>

Explanation : The `VisualizeVelocityJacobian` function visualizes the directions in which the end effector will move when each joint is actuated. This is typically done by plotting vectors originating from the end effector's current position, pointing in the direction of motion caused by each joint's velocity.

```
Velocity Jacobian :
 [[-2.05779181 -2.13774595 -2.57858538 -2.2154428  -1.24481088]
 [ 1.2558066   1.01893674  0.78301175  0.12678964 -0.11377988]
 [ 0.          0.          0.          0.          0.        ]]
 ```

<image src="images\VelocityJacobian-2.png" width="720" height="405"/>

<em> Figure 5 : Modified Arm Pose - Velocity Jacobian </em>
 
By analyzing both the Jacobian matrix and its visualization, we can gain insights into the robot's behavior and optimize its movements for tasks like reaching a target position.

### Implement the IK method in SimpleArm using gradient descent and adapt the GUI accordingly.

Explanation : The `IK_GradientDescent` function is an implementation of inverse kinematics (IK) using the gradient descent algorithm. This method iteratively adjusts the joint angles of a robotic arm to minimize the error between the current end effector position and a target position.

<image src="images\IK-GradientDescent-1.png" width="720" height="405"/>

<em> Figure 6 : One Gradient Descent IK Iteration </em>



<image src="images\IK-GradientDescent-2.gif" width="720" height="405"/>

<em> Figure 7 : IK Gradient Descent Full Run Example  </em>


### Several images demonstrating IK with placing the IK target beyond what the arm can reach.

Explanation : When putting the target beyond what the robotic arm can reach and apply IK_GradientDescent, the algorithm will attempt to minimize the error norm between the current position of the arm's end effector and the target position. Since the target is unreachable, the algorithm will converge to a set of joint angles that bring the end effector as close as possible to the target, but it will not be able to reach the exact target position. The error norm will be minimized but not zero, indicating that the target is beyond the arm's reach.

<image src="images\IK-GradientDescentOff-1.gif" width="720" height="405"/>

<em> Figure 8 : Gradient Descent - Example A </em>


<image src="images\IK-GradientDescentOff-2.gif" width="720" height="405"/>

<em> Figure 9 : Gradient Descent - Example B </em>


<image src="images\IK-GradientDescentOff-3.gif" width="720" height="405"/>

<em> Figure 10 : Gradient Descent - Example C </em>


## Task 3: Gauss-Newton based IK (the Jacobian inverse method)

### Gauss-Newton in IK and adapt the GUI accordingly. Pick a target pose and take screenshot of several consequtive iterations and add them to the report. 

Explanation : 
1. *Convergence:* While Gradient slowly converge a set of joint angles that bring the end effector as close as possible to the target. Newton converges faster because it uses second-order derivative information (Hessian matrix) to adjust the step size, direction and precision.
2. *Adaptive Step Size:* On the one hand, Gradient Descent requires manual tuning of the learning rate, which can be challenging and time-consuming. The Newton-Gauss generally achieves higher precision in fewer iterations due to its use of second-order information.

<image src="images\IK-GaussNewton.gif" width="720" height="405"/>

<em> Figure 11 : Gauss-Newton GIF </em>

<image src="images\IK-GaussNewton-1.png" width="720" height="405"/>

<em> Figure 12 : Gauss-Newton - Example A </em>

<image src="images\IK-GaussNewton-2.png" width="720" height="405"/>

<em> Figure 13 : Gauss-Newton - Example B </em>


### Cases where the robot posseses more than 2 joints are said to be redundant, meaning there are extra degrees of freedom, and that the IK solution is not unique.

Regularizing the joint movements : It uses the Damped Least Squares method to compute the pseudo-inverse of the Jacobian, which helps in handling redundancy and avoiding large joint movements. This approach ensures a unique solution by regularizing the joint movements, making it suitable for robots with redundant degrees of freedom.

```py
    def IK_GaussNewton(self, target, tolerance=1e-6, max_iterations=100, lambda_=0.01):
        for iteration in range(max_iterations):
            self.FK()
            current_position = self.Jw[-1, :]
            error = target - current_position
            if np.linalg.norm(error) < tolerance:
                break

            J = self.VelocityJacobian()
            JTJ = J.T @ J
            damping_matrix = lambda_ * np.eye(JTJ.shape[0])
            pseudo_inverse = np.linalg.inv(JTJ + damping_matrix) @ J.T
            delta_angles = pseudo_inverse @ error
            self.angles += delta_angles

        return iteration
```

