#%%
import vedo as vd
vd.settings.default_backend= 'vtk'
import numpy as np


# ==================================================================== Rotation Function ====================================================================
#%% class for a robot arm
def Rot(angle, axis):
    # calculate the rotation matrix for a given angle and axis using Rodrigues' formula
    # return a 3x3 numpy array
    # also see scipy.spatial.transform.Rotation.from_rotvec
    axis = np.array(axis)
    axis = axis/np.linalg.norm(axis)
    I = np.eye(3)
    K = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])
    R = I + np.sin(angle)*K + (1-np.cos(angle))*np.dot(K,K)
    return R


# ==================================================================== SimpleArm Class ====================================================================
class SimpleArm:
    # Initalizer
    def __init__(self, n=3,link_lengths=None):
        self.n = n
        if link_lengths is None:
            self.link_lengths = [1] * self.n
        else:
            self.link_lengths = link_lengths

        self.angles = [0] * self.n
        self.Jl = np.zeros((self.n + 1, 3))
        for i in range(1, n + 1):
            self.Jl[i, :] = np.array([self.link_lengths[i - 1], 0, 0])

        self.Jw = np.zeros((self.n + 1, 3))
        self.FK()
    
    # Forward Kinematics
    def FK(self, angles=None):
        if angles is not None:
            self.angles = angles

        Ri = np.eye(3)
        for i in range(1, self.n + 1):
            Ri = Rot(self.angles[i - 1], [0, 0, 1]) @ Ri
            self.Jw[i, :] = Ri @ self.Jl[i, :] + self.Jw[i - 1, :]

        return self.Jw[-1, :]

    
    # Inverse Kinematics - NOTE : [NEW CODE]
    def IK_GradientDescent(self, target, learning_rate=0.01, tolerance=1e-6, max_iterations=1000):
        iteration = 0
        while iteration < max_iterations:
            current_position = self.FK()
            error = target - current_position
            error_norm = np.linalg.norm(error)
            if error_norm < tolerance:
                break
            J = self.VelocityJacobian()
            gradient = J.T @ error
            # adaptive_learning_rate = self.adaptive_learning_rate(learning_rate, iteration, decay_factor, error_norm)
            self.angles += learning_rate * gradient
            iteration += 1
        return iteration

    def adaptive_learning_rate(self, learning_rate, iteration, decay_factor, error_norm):
        # Adjust learning rate based on error magnitude and iteration
        return learning_rate * (decay_factor ** iteration) * (1 + error_norm)

    # Inverse Kinematics - NOTE : [NEW CODE]
    # def IK_GaussNewton(self, target, tolerance=1e-6, max_iterations=1000):
    #     iteration = 0
    #     while iteration < max_iterations:
    #         current_position = self.FK()
    #         error = target - current_position
    #         if np.linalg.norm(error) < tolerance:
    #             break
    #         J = self.VelocityJacobian()
    #         J_pseudo_inv = np.linalg.pinv(J)
    #         delta_theta = J_pseudo_inv @ error
    #         self.angles += delta_theta
    #         iteration += 1
    #     return iteration

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


    # Velocity Jacobian - NOTE : [NEW CODE]
    def VelocityJacobian(self, angles=None):
        if angles is not None:
            self.angles = angles

        J = np.zeros((3, self.n))
        Ri = np.eye(3)
        for i in range(self.n):
            Ri = Rot(self.angles[i], [0, 0, 1]) @ Ri
            J[:, i] = np.cross([0, 0, 1], self.Jw[-1, :] - self.Jw[i, :])
        return J

    def draw(self):
        vd_arm = vd.Assembly()
        vd_arm += vd.Sphere(pos = self.Jw[0,:], r=0.05)
        for i in range(1,self.n+1):
            vd_arm += vd.Cylinder(pos = [self.Jw[i-1,:], self.Jw[i,:]], r=0.02)
            vd_arm += vd.Sphere(pos = self.Jw[i,:], r=0.05)
        return vd_arm
    
    # Visualize Velocity Jacobian - NOTE : [NEW CODE]
    def VisualizeVelocityJacobian(self):
        global plt, VelocityJacobian
        J = self.VelocityJacobian()
        VelocityJacobian = J
        print("Velocity Jacobian :\n",J)
        origin = self.Jw[-1, :]
        for i in range(self.n):
            plt += vd.Line([self.Jw[i],origin], lw=2, c='r').pattern('- -', repeats=10)
            plt += vd.Line([origin, origin + J[:, i]], lw=2, c='b').pattern('- -', repeats=10)

#%%
# ==================================================================== GLOBAL VARIABLES ====================================================================
activeJoint = 0
IK_target = [1,1,0]
link_lengths = [0.25, 0.5, 0.75, 1.0, 1.25]
currentMode = "Gradient-Descent"
iteration_number = 0
VelocityJacobian = None

# ==================================================================== GUI Interactions ====================================================================
def OnSliderAngle(widget, event):
    global activeJoint
    arm.angles[activeJoint] = widget.value
    arm.FK()
    plt.remove("Assembly")
    plt.add(arm.draw())
    plt.remove("Line")
    if currentMode == "Velocity-Jacobian":
        arm.VisualizeVelocityJacobian()
    plt.render()

def OnCurrentJoint(widget, event):
    global activeJoint
    activeJoint = round(widget.value)
    sliderAngle.value = arm.angles[activeJoint]

def LeftButtonPress(evt,iterations=1000):
    global IK_target, arm, plt, currentMode, iteration_number,iteration_text
    IK_target = evt.picked3d
    plt.remove("Sphere")
    plt.add(vd.Sphere(pos = IK_target, r=0.05, c='b'))
    for iteration in range(iterations):
        if currentMode == "Gradient-Descent":
            iteration_number = arm.IK_GradientDescent(IK_target, learning_rate=0.01, tolerance=1e-6, max_iterations=1)
        elif currentMode == "Gauss-Newton":
            iteration_number = arm.IK_GaussNewton(IK_target, tolerance=1e-6, max_iterations=1)
        else:
            return # do nothing    
        plt.remove("Assembly")
        plt.add(arm.draw())
        plt.remove("Line")
        iteration_text.text("Iterations number for the curent mode: \n"+str(iteration))

        plt.render()
        distance = np.linalg.norm(arm.Jw[-1] - IK_target)
        if distance < 1e-2:
            break
    

def switchMode()->str:
    global currentMode,arm,plt
    plt.remove("Line")
    if currentMode == "Gradient-Descent":
        currentMode = "Gauss-Newton"
    elif currentMode == "Gauss-Newton":
        currentMode = "Velocity Jacobian"
        arm.VisualizeVelocityJacobian()
    else:
        currentMode = "Gradient-Descent"
    return currentMode

def OnKeyPress(evt):
    global currentMode, currentMode_text,plt,arm
    if evt.keypress in ['m', 'M']:
        currentMode = switchMode()
        currentMode_text.text("Current Mode: "+currentMode) 
    elif evt.keypress in ['d', 'D']:
        currentMode = "Gradient-Descent"     
        currentMode_text.text("Current Mode: "+currentMode)
    elif evt.keypress in ['n', 'N']:
        currentMode = "Gauss-Newton"
        currentMode_text.text("Current Mode: "+currentMode)
    elif evt.keypress in ['v', 'V']:
        currentMode = "Velocity-Jacobian"     
        arm.VisualizeVelocityJacobian()
        currentMode_text.text("Current Mode: "+currentMode)
        print("Velocity Jacobian :\n",VelocityJacobian)
    elif evt.keypress in ['z', 'Z']: # sets circle with radius of total arm length
        plt.remove("Circle")
        circle = vd.Circle(pos=arm.Jw[0], r=np.sum(arm.link_lengths), c='g').wireframe(True).linewidth(2)
        plt.add(circle)
    elif evt.keypress in ['c', 'C']:
        plt.remove("Sphere")
        plt.remove("Line")
        plt.remove("Circle")
    plt.render()


# ==================================================================== MAIN ====================================================================
arm = SimpleArm(5, link_lengths)

plt = vd.Plotter(title="Assignment 3",screensize="auto")
plt += arm.draw()
plt += vd.Sphere(pos = IK_target, r=0.05, c='b').draggable(True)
plane = vd.Plane(s=[2*arm.n,2*arm.n]) # a plane to catch mouse events
sliderCurrentJoint = plt.add_slider(OnCurrentJoint, 0, arm.n-1, 0, title="Current joint", pos=3, delayed=True)
sliderAngle =  plt.add_slider(OnSliderAngle,-np.pi,np.pi,0., title="Joint Angle", pos=4)
plt.add_callback('LeftButtonPress', LeftButtonPress) # add Mouse callback





# NOTE : [NEW CODE]
plt.add_callback('key press', OnKeyPress) # add Keyboard callback

instructions_string = "Press 'm' to switch between Gradient Descent and Gauss-Newton\n press 'd' for Gradient Descent\n press 'n' for Gauss-Newton\n press 'v' to toggle Velocity Jacobian mode\n press 'c' to clear the target\n press 'z' to draw a circle with radius of the total arm length"
instructions_text = vd.Text2D(instructions_string, pos="top-center", s=0.8)

currentMode_text = vd.Text2D("Current Mode: "+currentMode, pos=(0.4,0.85))
iteration_text= vd.Text2D("Iterations number for the curent mode: \n"+str(iteration_number), pos="top-left")

plt.add(iteration_text)
plt.add(plane)
plt.add(instructions_text)
plt.add(currentMode_text)

plt.user_mode('2d').show(zoom="tightest",size=[1920,1000]).close()
# %%