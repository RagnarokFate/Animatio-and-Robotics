import vedo as vd
import numpy as np
import time

from vedo.pyplot import plot
from vedo import Latex, Plotter, Text2D, precision, Arrow, Cylinder, Line, settings


# KEYWORDS: gradient descent, Newton's method, optimization, gradient, Hessian, line search, finite difference
# NOTE - DON'T CHANGE THE CODE
# TODO - ADDED CODE
# FIXME - MODIFY THE CODE TO FULLFILL THE REQUIREMENTS

vd.settings.default_backend= 'vtk'

msg = vd.Text2D(pos='top-left', font="VictorMono") # an empty text
# Global objects
plt = Plotter(bg2='lightblue')
Xi = np.empty((0, 3))
Xi_temp = np.empty((0, 3))

Xi_gradient = np.empty((0, 3))
Xi_newton = np.empty((0, 3))

selected_point = None
# GLOBAL GRAPH VARIABLES
# Initialize a list to store the function values
objective_values = []

# Initialize the function graph variables
function_graph  = None 
old_graph = None
BothGraph = None


# Initialize paths for gradient descent and Newton's method together
gradient_descent_array = []
newton_array = []
DoubleGraphVisiblie = False
#%% Callbacks

# NOTE - When the mouse is moved, the path is drawn on the surface and the cylinder is drawn to show the path
# def OnMouseMove(evt):
#     global Xi_temp
    
#     if evt.actor and evt.picked3d is None:
#         return
#     pt  = evt.picked3d               # 3d coords of point under mouse
#     X   = np.array([pt[0],pt[1],objective([pt[0],pt[1]])])  # X = (x,y,e(x,y))
#     Xi_temp = np.append(Xi_temp,[X],axis=0)             # append to the list of points


#     if arrow:
#         plt.remove(arrow)

#     if len(Xi_temp) >= 1:               # need at least two points to compute a distance
#         txt =(
#             f"X:  {vd.precision(X,2)}\n"
#             f"dX: {vd.precision(Xi_temp[-1,0:2] - Xi_temp[-2,0:2],2)}\n"
#             f"dE: {vd.precision(Xi_temp[-1,2] - Xi_temp[-2,2],2)}\n"
#         )
#         # ar = vd.Arrow(Xi[-2,:], Xi[-1,:], s=0.001, c='orange5')
#         # plt.add(ar) # add the arrow
#     else:
#         txt = f"X: {vd.precision(X,2)}"
#     # msg.text(txt)                    # update text message

#     c = vd.Cylinder([np.append(Xi_temp[-1,0:2], 0.0), Xi_temp[-1,:]], r=0.01, c='orange5')
#     plt.remove("Cylinder")    
#     fp = fplt3d[0].flagpole(txt, point=X,s=0.08, c='k', font="Quikhand")
#     fp.follow_camera()                 # make it always face the camera
#     plt.remove("FlagPole") # remove the old flagpole

#     plt.add(fp, c) # add the new flagpole and new cylinder
#     plt.render()

# NOTE - When the 'c' key is pressed, the path is cleared   
def OnKeyPress(evt):               ### called every time a key is pressed
    global Xi, objective_values,old_graph,function_graph, DoubleGraphVisiblie, Xi_gradient, Xi_newton
    if evt.keypress in ['c', 'C']: # reset Xi and the function graph
        Xi = np.empty((0, 3))
        Xi_gradient = np.empty((0, 3))
        Xi_newton = np.empty((0, 3))

        objective_values = []
        function_graph = None
        old_graph = None
        DoubleGraphVisiblie = False

        plt.remove("Arrow")
        plt.remove("Point")
        plt.remove("plot")
        plt.render()

# NOTE - When the alpha slider is moved, the transparency of the surface and isolines is updated
def OnSliderAlpha(widget, event): ### called every time the slider is moved
    val = widget.value         # get the slider value
    fplt3d[0].alpha(val)       # set the alpha (transparency) value of the surface
    fplt3d[1].alpha(val)       # set the alpha (transparency) value of the isolines

# TODO - TASK1 A. Draw the X, Y, and Z axes in red, blue, and green respectively upon right-clicking
def OnMouseRight(evt):
    if evt.object is not None:
        # Get the clicked 3D coordinates
        pt = evt.picked3d
        
        # Define the axes endpoints
        axis_length = 1.0
        x_axis_end = pt + [axis_length, 0, 0]
        y_axis_end = pt + [0, axis_length, 0]
        z_axis_end = pt + [0, 0, axis_length]
        
        # Create the X, Y, Z axes lines
        x_axis = vd.Line(pt, x_axis_end, c='red')
        y_axis = vd.Line(pt, y_axis_end, c='blue')
        z_axis = vd.Line(pt, z_axis_end, c='green')
        
        # Add the axes to the plot
        plt.add(x_axis)
        plt.add(y_axis)
        plt.add(z_axis)

        plt.render()
        time.sleep(1.0)

        plt.remove(x_axis)
        plt.remove(y_axis)
        plt.remove(z_axis)   

# TODO - when clicked on the surface or the plane under it, clears the path and sets the current optimization candidate to that position
def OnMouseLeft(evt):
    global Xi, objective_values, selected_point, Xi_gradient, Xi_newton, DoubleGraphVisiblie,old_graph, function_graph
    if evt.actor and evt.picked3d is None:
        return

    reset()
    pt = evt.picked3d
    X = np.array([pt[0], pt[1], objective([pt[0], pt[1]])])
    Xi = np.append(Xi, [X], axis=0)
    Xi_gradient = np.append(Xi_gradient, [X], axis=0)
    Xi_newton = np.append(Xi_newton, [X], axis=0)
    if len(Xi) > 1:               # need at least two points to compute a distance
        txt = (
        f"X:  {vd.precision(X,2)}\n"
        f"dX: {vd.precision(Xi[-1,0:2] - Xi[-2,0:2],2)}\n"
        f"dE: {vd.precision(Xi[-1,2] - Xi[-2,2],2)}\n"
            )
    else:
        txt = f"X: {vd.precision(X,2)}"    
    msg.text(txt)
    selected_point = np.array([pt[0], pt[1]])
    objective_values.append(objective(selected_point))
    newton_array.append(objective(selected_point))
    gradient_descent_array.append(objective(selected_point))
    print(f"Selected point: {selected_point}")
    plt.add(vd.Point(selected_point,r=8, c="blue"))
    plt.add(vd.Point(pt,r=8, c="blue"))
    plt.render()   

# TODO - TASK1 B. Update the function graph with the new path
def update_function_graph():
    global function_graph,old_graph

    if(DoubleGraphVisiblie is True and len(Xi) > 1):
        print("Double Graph is Visiblie")
        print("Xi is EMPTY")
        return 
    
    if len(Xi) > 1:
        if old_graph:
            plt.remove(old_graph).render()
            plt.remove("plot").render()
        # x_values = np.arange(len(Xi))
        function_graph = plot(objective_values, title="Function Values on Path", xlabel="Points", ylabel="Gradient Step Value", c='red').clone2d() 
        old_graph = function_graph
        plt.add(old_graph).render()


# 3. Add a button to run a single gradient descent step and update the path
def GradientDescentStep(obj, ename):
    global Xi, objective_values, DoubleGraphVisiblie, selected_point
    temp = selected_point
    # If no point is selected, return
    if len(Xi) < 1 or selected_point is None:
        print("Please select a point on the surface first")
        return
    
    # If a point is selected, run a single gradient descent step
    selected_point = Xi[-1][:2]
    X = optimize(objective, selected_point, gradient_direction, tol=1e-6, iter_max=1)
    value = objective([X[0], X[1]])
    Xi = np.append(Xi, [[X[0], X[1], value]], axis=0)
    objective_values.append(value)
    # adding a visual arrow to show the step
    plt.add(vd.Arrow(Xi[-2,:], Xi[-1,:], s = 0.001, c='green'))
    # selected_point = Xi[-1][:2]
    update_function_graph()
    DoubleGraphVisiblie = False
    selected_point = temp

# 4. Add a button to run a single Newton's step
def NewtonsStep(obj, ename):
    global Xi, objective_values, DoubleGraphVisiblie, selected_point
    temp = selected_point
    # If no point is selected, return
    if len(Xi) < 1 or selected_point is None:
        print("Please select a point on the surface first")
        return
    
    # If a point is selected, run a single gradient descent step
    selected_point = Xi[-1][:2]
    X = optimize(objective, selected_point, Newton_direction, tol=1e-6, iter_max=1)
    value = objective([X[0], X[1]])
    Xi = np.append(Xi, [[X[0], X[1], value]], axis=0)
    objective_values.append(value)
    print(objective_values)
    # adding a visual arrow to show the step
    # create if statment that checks if the new point is not outside the surface
    if value < objective(selected_point):
        plt.add(vd.Arrow(Xi[-2,:], Xi[-1,:], s = 0.001, c='green')).render()
    else:
        plt.remove(vd.Arrow(Xi[-2,:], Xi[-1,:], s = 0.001, c='green'))
    plt.add(vd.Arrow(Xi[-2,:], Xi[-1,:], s = 0.001, c='green')).render()
    # selected_point = Xi[-1][:2]
    update_function_graph()
    DoubleGraphVisiblie = False
    selected_point = temp

# 5. Add a button to run a single Gradient Descet & Newton's step
def BothStep(obj, ename):
    global Xi_gradient,Xi_newton, gradient_descent_array,newton, DoubleGraphVisiblie, selected_point
    # If no point is selected, return
    if len(Xi) < 1 or selected_point is None:
        print("Please select a point on the surface first")
        return
    DoubleGraphVisiblie = True
    # If a point is selected, run a single gradient descent step    
    X_gradient = optimize(objective, Xi_gradient[-1][:2], gradient_direction, tol=1e-6, iter_max=1)
    gradient_value = objective([X_gradient[0], X_gradient[1]])
    gradient_descent_array.append(gradient_value)
    # if a point is selected, run a single Newton's step
    X_newton = optimize(objective, Xi_newton[-1][:2], Newton_direction, tol=1e-6, iter_max=1)
    Newton_value = objective([X_newton[0], X_newton[1]])
    newton_array.append(Newton_value)
    print(f"Gradient Descent: {gradient_value}, Newton's Step: {Newton_value}")

    Xi_gradient = np.append(Xi_gradient, [[X_gradient[0], X_gradient[1], gradient_value]], axis=0)
    Xi_newton = np.append(Xi_newton, [[X_newton[0], X_newton[1], Newton_value]], axis=0)

    # adding a visual arrow to show the step
    plt.add(vd.Arrow(Xi_gradient[-2,:], Xi_gradient[-1,:], s = 0.001, c='red'))
    plt.add(vd.Arrow(Xi_newton[-2,:], Xi_newton[-1,:], s = 0.001, c='blue'))
    update_Double_graph()

def update_Double_graph():
    
    global function_graph, old_graph, BothGraph, newton_array, gradient_descent_array
    if DoubleGraphVisiblie is False:
        return
    
    if len(Xi) >= 1:
        if old_graph:
            plt.remove(old_graph)

        BothGraph = plot(gradient_descent_array, xlabel="Points", ylabel="Objective Value", c='red')
        BothGraph += plot(newton_array, xlabel="Points", ylabel="Objective Value", c='blue', like=BothGraph)


        BothGraph = BothGraph.clone2d()
        old_graph = BothGraph
        
        plt.add(old_graph).render()
    else:
        BothGraph = None


def reset():
    global Xi, objective_values, function_graph, old_graph, arrow, BothGraph, Xi_gradient, Xi_newton,gradient_descent_array,newton_array, DoubleGraphVisiblie
    Xi = np.empty((0, 3))
    Xi_gradient = np.empty((0, 3))
    Xi_newton = np.empty((0, 3))

    objective_values = []
    gradient_descent_array = []
    newton_array = []
    try:
        plt.remove("Arrow")
        plt.remove("Point")
        plt.remove("Plot")
        print("Arrow, Point and Plot removed")
        plt.remove(function_graph)
        plt.remove(old_graph)
        plt.remove(BothGraph)
    except:
        print("No Arrow, Point or Plot to remove")
        pass
    
    function_graph = None
    old_graph = None
    BothGraph = None
    plt.render()

#%% Optimization functions
def objective(X):
    x, y = X[0], X[1]
    return np.sin(2*x*y) * np.cos(3*y)/2+1/2

def gradient_fd(func, X, h=0.001): # finite difference gradient
    x, y = X[0], X[1]
    gx = (func([x+h, y]) - func([x-h, y])) / (2*h)
    gy = (func([x, y+h]) - func([x, y-h])) / (2*h)
    return gx, gy

def Hessian_fd(func, X, h=0.001): # finite difference Hessian
    x, y = X[0], X[1]
    gxx = (func([x+h, y]) - 2*func([x, y]) + func([x-h, y])) / h**2
    gyy = (func([x, y+h]) - 2*func([x, y]) + func([x, y-h])) / h**2
    gxy = (func([x+h, y+h]) - func([x+h, y-h]) - func([x-h, y+h]) + func([x-h, y-h])) / (4*h**2)
    H = np.array([[gxx, gxy], [gxy, gyy]])
    return H

def gradient_direction(func, X): # compute gradient step direction
    g = gradient_fd(func, X)
    return -np.array(g)

def Newton_direction(func, X):   # compute Newton step direction
    g = gradient_fd(func, X)
    H = Hessian_fd(func, X)
    d = -np.linalg.solve(H, np.array(g))
    return np.array(d[0],d[1])

def line_search(func, X, d): 
    alpha = 1.0
    while func(X + d*alpha) > func(X):  # If the function value increases, reduce alpha
        alpha *= 0.5                                # by half and try again
    return alpha

def step(func, X, search_direction_function):
    d = search_direction_function(func, X)
    alpha = line_search(func, X, d)
    return X + d*alpha

def optimize(func, X, search_direction_function, tol=1e-6, iter_max=10):
    for i in range(iter_max):
        X = step(func, X, search_direction_function)
        if np.linalg.norm(gradient_fd(func, X)) < tol:
            break
    return X

Xi = np.empty((0, 3))
# test the optimization functions
print("Gradient Descent Test")
X = optimize(objective, [0.6, 0.6], gradient_direction, tol=1e-6, iter_max=1)
print(X)
X = optimize(objective, X, gradient_direction, tol=1e-6, iter_max=1)
print(X)
X = optimize(objective, X, gradient_direction, tol=1e-6, iter_max=1)
print(X)
print("Newton Test")
X = optimize(objective, [0.4, 2], Newton_direction, tol=1e-6, iter_max=1)
print(X)
X = optimize(objective, X, Newton_direction, tol=1e-6, iter_max=1)
print(X)
X = optimize(objective, X, Newton_direction, tol=1e-6, iter_max=1)
print(X)

#%% Plotting
plt = vd.Plotter(bg2='lightblue')  # Create the plotter
fplt3d = plot(lambda x,y: objective([x,y]), c='terrain')      # create a plot from the function e. fplt3d is a list containing surface mesh, isolines, and axis
fplt2d = fplt3d.clone()            # clone the plot to create a 2D plot

fplt2d[0].lighting('off')          # turn off lighting for the 2D plot
fplt2d[0].vertices[:,2] = 0        # set the z-coordinate of the mesh to 0
fplt2d[1].vertices[:,2] = 0        # set the z-coordinate of the isolines to 0

# TODO
plt.add_callback('right click mouse', OnMouseRight)
plt.add_callback('left click mouse', OnMouseLeft)

# plt.add_callback('mouse move', OnMouseMove) # add Mouse move callback
plt.add_callback('key press', OnKeyPress) # add Keyboard callback
plt.add_slider(OnSliderAlpha,0.,1.,1., title="Alpha") # add a slider for the alpha value of the surface

# TODO - Adding a Gradient Descent Step button
GradientDescentButton = plt.add_button(GradientDescentStep,
    pos=(0.1, 0.5),   # x,y fraction from bottom left corner
    states=["Gradient Descent"],
    c="w",     # font color for each state
    bc="dg"  # background color for each state
)

# TODO - Adding a Newton Step button
NewtonButton = plt.add_button(NewtonsStep,
    pos=(0.1, 0.57),   # x,y fraction from bottom left corner
    states=["Newton's Step"],
    c="w",     # font color for each state
    bc="dg"  # background color for each state
)

# TODO - Adding a Gradient Descent and Newton's Step button
BothButton = plt.add_button(BothStep,
    pos=(0.5, 0.9),   # x,y fraction from bottom left corner
    states=["Activate Gradient Descent and Newton's Step"],
    c="w",     # font color for each state
    bc="dg"  # background color for each state
)


plt.show([fplt3d, fplt2d], msg, __doc__, viewup='z',size=[1800,960],title = "Assignment 1")
plt.close()