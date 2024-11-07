#%%
import vedo as vd
vd.settings.default_backend = 'vtk'

from vedo import show
import numpy as np

from abc import ABC, abstractmethod
import numdifftools as nd
from scipy.sparse import coo_matrix
import triangle as tr # pip install triangle  

# EXTRA
import random  
from vedo.pyplot import plot

class Stencil(ABC):
    @abstractmethod
    def ExtractElementsFromMesh(F):
        return 0

    @abstractmethod
    def ExtractVariblesFromVectors(x):
        return 0

class EdgeStencil(Stencil):
    # Extract the edges from a mesh
    @staticmethod
    def ExtractElementsFromMesh(F):
        edges = {tuple(sorted((F[i, j], F[i, (j+1) % 3]))) for i in range(F.shape[0]) for j in range(3)}
        return list(edges)
    
    # Extract x1, x2, the two vertices that define the edge, from the vector x, assuming that the variables are stored in the order x1, x2, x3, y1, y2, y3, z1, z2, z3
    # or as a 3x2 matrix, where the columns are the vertices
    @staticmethod
    def ExtractVariblesFromVectors(x):
        # return x.flat[0:3], x.flat[3:6]
        return x.flat[0:2], x.flat[2:4]

#%% Energy functions
# Abstract element energy class that implements finite differences for the gradient and hessian
# of the energy function. The energy function is defined in the derived classes, which must implement
# the energy method. The gradient and hessian methods should override the finite differences implementation.
# X is the undeformed configuration, x is the deformed configuration in a nx3 matrix, where n is the number
# of vertices in the element.
# the variable ordering for the gradient and hessian methods be in the x1, x2, x3, ..., y1, y2, y3, ... z1, z2, z3 format


class ElementEnergy(ABC):    
    @abstractmethod
    def energy(X, x):
        return 0

    # should be overridden by the derived class, otherwise the finite difference implementation will be used
    def gradient(self, X, x):
        return self.gradient_fd(X, x)

    def hessian(self, X, x):
        return self.hessian_fd(X, x)
    
    # finite difference gradient and hessian
    def gradient_fd(self, X, x):
        return nd.Gradient(lambda X, x: self.energy(X, x.flatten()))

    def hessian_fd(self, X, x):
        return nd.Hessian(lambda X, x: self.energy(X, x.flatten()))
    
    # check that the gradient is correct by comparing it to the finite difference gradient
    def check_gradient(self, X, x):
        grad = self.gradient(X, x)
        grad_fd = self.gradient_fd(X, x)
        return np.linalg.norm(grad - grad_fd)


# Spring energy function for a zero-length spring, defined as E = 0.5*||x1-x2||^2, regardless of the undeformed configuration
class ZeroLengthSpringEnergy(ElementEnergy):
    def __init__(self):
        self.stencil = EdgeStencil()
    
    def energy(self, X, x):
        x1,x2 = self.stencil.ExtractVariblesFromVectors(x)
        return 0.5*np.linalg.norm(x1 - x2)**2

    def gradient(self, X, x):
        x1,x2 = self.stencil.ExtractVariblesFromVectors(x)
        return np.array([x1 - x2, x2 - x1], dtype=float)

    def hessian(self, X, x):
        # The hessian is constant and is shapes like [I -I; -I I], where I is the identity matrix
        I = np.eye(3)
        return np.block([[I, -I], [-I, I]])
    
# Spring energy function for a spring with a rest length, defined as E = 0.5*(||x1-x2|| - l)^2, where l is the rest length
class SpringEnergy(ElementEnergy):
    def __init__(self):
        self.stencil = EdgeStencil()
    
    def energy(self, X, x):
        x1, x2 = self.stencil.ExtractVariblesFromVectors(x)
        X1, X2 = self.stencil.ExtractVariblesFromVectors(X)
        return 0.5 * (np.linalg.norm(x1 - x2) - np.linalg.norm(X1 - X2))**2
    
    def gradient(self, X, x):
        x1, x2 = self.stencil.ExtractVariblesFromVectors(x)
        X1, X2 = self.stencil.ExtractVariblesFromVectors(X)
        rest_length = np.linalg.norm(X1 - X2)
        current_length = np.linalg.norm(x1 - x2)
        
        if current_length == 0:
            return np.array([x1 - x2, x2 - x1], dtype=float)

        direction = (x1 - x2) / current_length
        grad_x1 = (current_length - rest_length) * direction
        grad_x2 = -grad_x1
        
        return np.array([grad_x1, grad_x2], dtype=float)
    
    def hessian(self, X, x):
        x1, x2 = self.stencil.ExtractVariblesFromVectors(x)
        X1, X2 = self.stencil.ExtractVariblesFromVectors(X)
        rest_length = np.linalg.norm(X1 - X2)
        current_length = np.linalg.norm(x1 - x2)

        I = np.eye(2)
        direction = (x1 - x2) / current_length

        if current_length == 0:
            return np.block([[I, -I], [-I, I]])
        
        # Compute the components of the Hessian matrix
        term1 = I * (1 - rest_length / current_length)
        term2 = np.outer(direction, direction) * rest_length / current_length

        hessian_x1x1 = term1 + term2
        hessian_x1x2 = -term1 - term2
        hessian_x2x2 = term1 + term2
        
        hessian_matrix = np.block([
            [hessian_x1x1, hessian_x1x2],
            [hessian_x1x2.T, hessian_x2x2]
        ])
        reg_factor = 1e-8  # Adjust as needed
        hessian_matrix += reg_factor * np.eye(hessian_matrix.shape[0])
        return hessian_matrix


#%% Mesh class
class FEMMesh:
    def __init__(self, V, F, energy, stencil,pinned_vertices):
        self.V = V
        self.F = F
        self.energy = energy
        self.stencil = stencil
        self.elements = self.stencil.ExtractElementsFromMesh(F)
        self.X = self.V.copy()
        self.nV = self.V.shape[0]
        self.pinned_vertices = pinned_vertices

    def compute_energy(self,x):
        energy = 0
        for element in self.elements:
            Xi = self.X[element,:]
            xi = x[element,:]
            energy += self.energy.energy(Xi, xi)
        return energy
    
    def compute_gradient(self,x):
        grad = np.zeros((self.X.shape[0], 2))
        for element in self.elements:
            Xi = self.X[element,:]
            xi = x[element,:]
            gi = self.energy.gradient(Xi, xi)

            grad[element[0]] += gi[0]
            grad[element[1]] += gi[1]

        return grad
    
    def compute_hessian(self,x):
        # create arrays to store the sparse hessian matrix
        I = []
        J = []
        S = []
        for element in self.elements:
            Xi = self.X[element,:]
            xi = x[element,:]
            hess = self.energy.hessian(Xi,xi) # The hessian is a 6x6 matrix
            for i in range(4):
                for j in range(4):
                    I.append(element[i%2]+self.nV*(i//2))
                    J.append(element[j%2]+self.nV*(j//2))
                    S.append(hess[i,j])
        H = coo_matrix((S, (I, J)), shape=(2*self.nV, 2*self.nV))
        return H

            
#%% Optimization
class MeshOptimizer:
    def __init__(self, femMesh,method = "GD"):
        self.femMesh = femMesh
        if method == "Newton":
            self.SearchDirection = self.Newton
        else:
            self.SearchDirection = self.GradientDescent
        self.LineSearch = self.BacktrackingLineSearch

    def BacktrackingLineSearch(self, x, d, alpha=1,epsilon=1e-4):
        x0 = x.copy()
        f0 = self.femMesh.compute_energy(x0)
        grad = self.femMesh.compute_gradient(x0).flatten()
        while self.femMesh.compute_energy(x0 + alpha*d) > f0 + alpha * epsilon * np.dot(grad, d.flatten()):
            alpha *= 0.5
        return x0 + alpha*d, alpha
    

    def GradientDescent(self, x):
        d = self.femMesh.compute_gradient(x)
        return -d

    def Newton(self, x):
        grad = self.femMesh.compute_gradient(x)
        hess = self.femMesh.compute_hessian(x)
        hess = hess.toarray()  # Convert hess to a dense array
        grad = grad.reshape((2*self.femMesh.nV, 1))  # Reshape grad to have shape (2*self.nV, 1)
        epsilon = 1e-6  # Small regularization term
        d = np.linalg.solve(hess.T + epsilon * np.eye(hess.shape[0]), grad).T
        d = d.reshape(-1,2)
        return -d
    
    def step(self, x):
        d = self.SearchDirection(x)
        new_x, alpha = self.LineSearch(x,d)
        # enable pinned vertices by setting their position to the initial position
        for i in self.femMesh.pinned_vertices:
            new_x[i] = self.femMesh.X[i]
        return new_x,alpha

    def optimize(self, x, max_iter=100, tol=1e-6):
        for i in range(max_iter):
            x,alpha = self.step(x)
            if np.linalg.norm(self.femMesh.compute_gradient(x)) < tol:
                break
        return x,alpha

#%% Assignment 2 Solution
# Original mesh
def DrawMesh()->None:
    global plt, mesh, V, F, pinned_vertices, V_initial
    # Main program
    vertices = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) # square
    # convert vertices from 2d to 3d as homogeneous coordinates
    vertices = np.hstack((vertices, np.zeros((vertices.shape[0], 1))))

    tris = tr.triangulate({"vertices":vertices[:,0:2]}, f'qa0.01') # triangulate the square
    V = tris['vertices'] # get the vertices of the triangles
    V_initial = V.copy()  # Save the initial state
    F = tris['triangles'] # get the triangles
    mesh = vd.Mesh([V,F]).linecolor('black')
    plt += mesh
    plt += vd.Points(V[pinned_vertices,:])

# COMPLETED
def DrawTask1()->None:
    # TODO - Task 1.1: Implement the DrawObject function that create a creative shape
    def DrawObject()->None:
        global plt, mesh, V, F, pinned_vertices, V_initial
        # vertices = np.array([[0, 1.0], [-2.0, -4.0], [0, -2], [2, -4]]) # arrow
        vertices = np.array([[2,0], [3.0, 2.0], [4, 0], [4, -2],[10,2],[8,-6],[0,-10],[-8,-6],[-10,2],[-4,-2],[-4,0],[-3,2],[-2,0]]) # batman logo
        vertices /= 10
        vertices_dct = {'vertices': vertices}
        # vertices += [10,10]
        segments = np.array([[i,(i+1)%len(vertices)] for i in range(len(vertices))])
        vertices_dct ['segments'] = segments
        tris = tr.triangulate(vertices_dct, 'pqa0.01') # triangulate the square
        V = tris['vertices'] # get the vertices of the triangles
        V_initial = V.copy()  # Save the initial state
        F = tris['triangles'] # get the triangles
        mesh = vd.Mesh([V,F]).linecolor('black')
        plt.add(mesh)
        plt.add(vd.Points(V[pinned_vertices,:],r=10))


    # TODO - Task 1.2: Implement of triangulation with approximately 100 vertices
    def DrawTask1_with_Interior_vertices()->None:
        global plt, mesh, V, F, pinned_vertices, V_initial
        # Main program
        # Define your creative shape here
        vertices = np.array([
            [0, 0], [1, 0], [0.5, 0.5], 
            [0, 1], [1, 1], [1.5, 0.5]
        ])  # Example: hexagon-like shape

        # Set parameters for triangulation
        max_area = 0.005  # Adjust this to control the number of vertices
        tris = tr.triangulate({"vertices": vertices[:, 0:2]}, f'qa{max_area}Y')  # 'Y' for no interior points

        V = tris['vertices']  # Get the vertices of the triangles
        V_initial = V.copy()  # Save the initial state
        print(len(V))
        F = tris['triangles']  # Get the triangles
        mesh = vd.Mesh([V, F]).linecolor('black')
        plt += mesh
        plt += vd.Points(V)


    # TODO - Task 1.2: Implement of triangulation with no interior vertices
    def DrawTask1_without_Interior_vertices() -> None:
        global plt, mesh, V, F, pinned_vertices
        right_wing = np.array([[0,0], [2, 2], [4, 0], [5, 0],[5,-3],[4,-3],[4,-1],[2,1],[0,-1]]) # right wing
        left_wing = np.array([[0,0], [-2, 2], [-4, 0], [-5, 0],[-5,-3],[-4,-3],[-4,-1],[-2,1],[0,-1]]) # left wing
                    
        vertices = np.concatenate((right_wing,left_wing),axis=0)

        vertices_dct = {'vertices': vertices}
        V_initial = V.copy()  # Save the initial state
        segments = np.array([[i,(i+1)%len(vertices)] for i in range(len(vertices))])
        vertices_dct ['segments'] = segments
        tris = tr.triangulate(vertices_dct, 'pqa0.9') # triangulate the square
        V = tris['vertices'] # get the vertices of the triangles
        F = tris['triangles'] # get the triangles
        mesh = vd.Mesh([V,F]).linecolor('black')
        plt.add(mesh)
        plt.add(vd.Points(V[pinned_vertices,:],r=10))

    # main DrawTask1 implementation
    DrawObject()
    # DrawTask1_with_Interior_vertices()
    # DrawTask1_without_Interior_vertices()

    pass

# TODO - Task 2.1: Instead of printing to Jupyter (IPython), change the code such that it prints to the Vedo Window
def UpdateTerminalMessage(mousePos,Vi)->None:
    global terminalMessage
    mousePos = np.round(mousePos, 3)  # Round to three decimal places
    message = (f'Mouse hits the mesh\n'
               f'Coordinates: {mousePos}\n'
               f'Point ID: {Vi}')
    terminalMessage.text(message)
    plt.render()


# TODO - Task 3.1 : Run x = optimizer.step(x) and show the result after each iteration. What do you expect to see? Compare gradient descent and Newton's method and report on your findings.
# Gradient Descent Optimizer
def OptimizeMeshGD_ZeroEnergy(obj, ename, iterations = 10):
    global V, F, mesh, plt, pinned_vertices
    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}")
        femMesh = FEMMesh(V, F, ZeroLengthSpringEnergy(), EdgeStencil,pinned_vertices=pinned_vertices)
        
        inital_energy = femMesh.compute_energy(V)
        EnergyPlotUpdate(inital_energy)
        # Initialize the optimizer
        optimizer = MeshOptimizer(femMesh,method = "GD")
        V,alpha = optimizer.optimize(V, max_iter=1, tol=1e-6)
        print(f"Learning Rate: {alpha}") 
        redraw()
        plt.render()  

# Newton's Method Optimizer

def OptimizeMeshNewton_ZeroEnergy(obj, ename, iterations = 10):
    global V, F, mesh, plt, pinned_vertices
    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}")
        femMesh = FEMMesh(V, F, ZeroLengthSpringEnergy(), EdgeStencil,pinned_vertices=pinned_vertices)
        inital_energy = femMesh.compute_energy(V)
        EnergyPlotUpdate(inital_energy)
        # Initialize the optimizer
        optimizer = MeshOptimizer(femMesh,method = "Newton")
        V,alpha = optimizer.optimize(V, max_iter=1, tol=1e-6)
        print(f"Learning Rate: {alpha}")
        redraw()
        plt.render()  

# TODO - Task 3.2 : Implement the optimization of the spring energy function using gradient descent and Newton's method. Compare the results of the two methods and report on your findings.

def OptimizeMeshGD_SpringEnergy(obj, ename, iterations=10):
    global V, F, mesh, plt, pinned_vertices, V_initial    
    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}")
        femMesh = FEMMesh(V_initial - V, F, SpringEnergy(), EdgeStencil,pinned_vertices=pinned_vertices)
        
        initial_energy = femMesh.compute_energy(V_initial)  # Compute initial energy
        EnergyPlotUpdate(initial_energy)

        optimizer = MeshOptimizer(femMesh, method="GD")
        V_optimized, alpha = optimizer.optimize(V_initial, max_iter=2, tol=1e-6)
        # V += V_initial  # Add the initial state to the optimized state
        optimized_energy = femMesh.compute_energy(V)  # Compute optimized energy
        V = V_optimized # Subtract the initial state from the optimized state

        print(f"Learning Rate: {alpha}")
        print(f"Initial Energy: {initial_energy}")
        print(f"Optimized Energy: {optimized_energy}")
        print(f"Energy Difference: {optimized_energy - initial_energy}")
        
        redraw()
        plt.render()

def OptimizeMeshNewton_SpringEnergy(obj, ename, iterations=10):
    global V, F, mesh, plt, pinned_vertices, V_initial
    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}")
        femMesh = FEMMesh(V_initial - V, F, SpringEnergy(), EdgeStencil,pinned_vertices=pinned_vertices)
        
        initial_energy = femMesh.compute_energy(V_initial)  # Compute initial energy
        EnergyPlotUpdate(initial_energy)
        optimizer = MeshOptimizer(femMesh, method="Newton")
        V_optimized, alpha = optimizer.optimize(V_initial, max_iter=1, tol=1e-6)
        
        optimized_energy = femMesh.compute_energy(V)  # Compute optimized energy
        V = V_optimized # Subtract the initial state from the optimized state

        print(f"Learning Rate: {alpha}")
        print(f"Initial Energy: {initial_energy}")
        print(f"Optimized Energy: {optimized_energy}")
        print(f"Energy Difference: {optimized_energy - initial_energy}")
        
        redraw()
        plt.render()


def EnergyPlotUpdate(EnergyValue):
    global EnergysArray, EnergyGraph,plt
    if EnergyGraph is not None:
        plt.remove(EnergyGraph).render()
    EnergysArray.append(EnergyValue)
    EnergyGraph = plot(EnergysArray, title="Energy Estimation Graph", xlabel="Step", ylabel="Energy Value", c='red').clone2d()
     
    plt.add(EnergyGraph).render()

    

# =============================================================================

#%% Main program & UI functions (Task 2.1 & 2.2)
# Static VARIABLES
mesh = None
V = None
F = None
terminalMessage = None
pinned_vertices = []

# dragging variables
dragging = False
dragged_vertex = None

# EXTRA
EnergysArray = []
EnergyGraph = None

V_initial = None

def redraw():
    plt.remove("Mesh")
    mesh = vd.Mesh([V,F]).linecolor('black')
    plt.add(mesh)
    plt.remove("Points")
    plt.add(vd.Points(V[pinned_vertices,:],r=10))
    plt.render()


def OnLeftButtonPress(event):
    global mesh, terminalMessage, pinned_vertices, dragging, dragged_vertex
    if event.object is None:  # mouse hits nothing, return.
        print('Mouse hits nothing')
        return
    
    if isinstance(event.object, vd.mesh.Mesh):  # mouse hits the mesh
        Vi = mesh.closest_point(event.picked3d, return_point_id=True)
        print('Mouse hits the mesh')
        print('Coordinates:', event.picked3d)
        print('Point ID:', Vi)
        # Update the terminal message 2.1
        UpdateTerminalMessage(mousePos=event.picked3d, Vi=Vi)
        # modify the pin vertices 2.2
        if Vi in pinned_vertices:
            # Deselect the vertex
            pinned_vertices.remove(Vi)
            dragging = False
            dragged_vertex = None
        else:
            # Select the vertex and start dragging
            pinned_vertices.append(Vi)
            dragging = True
            dragged_vertex = Vi
    redraw()

# Mouse Move Event Handler 2.2
def OnMouseMove(event):
    global mesh, pinned_vertices, V, dragging, dragged_vertex  
    if event.object is None:  # mouse hits nothing, return.
        return  
    if dragging and dragged_vertex is not None:
        mouse_pos = np.array(event.picked3d[:2])  # Get the mouse position in 2D
        V[dragged_vertex][:2] = mouse_pos  # Update the position of the dragged vertex
        redraw()

# Right Button Press Event Handler 2.2
def OnRightButtonPress(event):
    global dragging, dragged_vertex
    if dragging and dragged_vertex is not None:
        mouse_pos = np.array(event.picked3d[:2])  # Get the mouse position in 2D
        V[dragged_vertex][:2] = mouse_pos  # Update the position of the dragged vertex
        dragging = False
        dragged_vertex = None
        redraw()

        
def OnKeyPress(evt):
    global pinned_vertices, dragging, dragged_vertex, V, V_initial,EnergysArray,EnergyGraph,plt
    if evt.keypress in ['c', 'C']: # reset Xi and the function graph
        plt.remove("Points")
        pinned_vertices = []
        V = V_initial.copy()
        redraw()
        plt.remove("EnergyGraph")
        plt.remove("plot").render()
        EnergysArray = []
        if dragging and dragged_vertex is not None:
            dragging = False
            dragged_vertex = None
            redraw()
        plt.render()

    # if x is pressed, scale up random vertices to test the optimization
    if evt.keypress in ['x', 'X']:
        for i in range(24):
            V[random.randint(0, len(V)-1)] *= 1.5
        redraw()
        plt.render()

    if evt.keypress in ['1']:
        OptimizeMeshGD_ZeroEnergy(None, None, iterations=10)
    if evt.keypress in ['2']:
        OptimizeMeshNewton_ZeroEnergy(None, None, iterations=10)    
    if evt.keypress in ['3']:
        OptimizeMeshGD_SpringEnergy(None, None, iterations=10)
    if evt.keypress in ['4']:
        OptimizeMeshNewton_SpringEnergy(None, None, iterations=10)



plt = vd.Plotter(title="Assignment 2",screensize="auto" )

plt.add_callback('LeftButtonPress', OnLeftButtonPress) # add Keyboard callback
plt.add_callback('KeyPress', OnKeyPress)  # add Key Press callback



#%%  Main Program Modification
# NOTE - Restore the Origin Mesh - RECTANGLE
DrawMesh()

# NOTE - Task 1: 
# DrawTask1()

# NOTE - Task 2.1: Instead of printing to Jupyter (IPython), change the code such that it prints to the Vedo Window
# NOTE : I ADDED IT AS TerminalMessage
terminalMessage = vd.Text2D("", pos=(0.5,0.05), s=0.7, c='black',justify='center')
plt.add(terminalMessage)

# NOTE - Task 2.2: Implement the OnLeftButtonPress function to pin/unpin vertices And move the pinned vertices to the new location (use a creative method to move the vertices along with keyboard modifer keys)
# plt.add_callback('MouseMove', OnMouseMove) # add Mouse Move callback
plt.add_callback('RightButtonPress', OnRightButtonPress)  # add Right button press callback

# NOTE - 3.1 
OptimizerStepGD_ZeroEnergy = plt.add_button(OptimizeMeshGD_ZeroEnergy,
    pos=(0.2, 0.95),   # x,y fraction from bottom left corner
    states=["Optimize Mesh GD"],
    c="w",     # font color for each state
    bc="dg"  # background color for each state
)

OptimizerStepNewton_ZeroEnergy = plt.add_button(OptimizeMeshNewton_ZeroEnergy,
    pos=(0.2, 0.90),   # x,y fraction from bottom left corner
    states=["Optimize Mesh Newton"],
    c="w",     # font color for each state
    bc="dg"  # background color for each state
)

# NOTE - 3.2 
OptimizerStepGD_SpringEnergy = plt.add_button(OptimizeMeshGD_SpringEnergy,
    pos=(0.8, 0.95),   # x,y fraction from bottom left corner
    states=["Optimize Mesh GD"],
    c="w",     # font color for each state
    bc="dg"  # background color for each state
)

OptimizerStepNewton_SpringEnergy = plt.add_button(OptimizeMeshNewton_SpringEnergy,
    pos=(0.8, 0.90),   # x,y fraction from bottom left corner
    states=["Optimize Mesh Newton"],
    c="w",     # font color for each state
    bc="dg"  # background color for each state
)

ZeroLengthEnergySpringMessage = vd.Text2D("Zero Length Energy Spring", pos=(0.1, 0.99))
EnergySpringMessage = vd.Text2D("Energy Spring", pos=(0.75, 0.99))

instructions_string = "On The Left ZeroLegnthSpringEnergy | On The Right - SpringEnergy"
introduction_text = vd.Text2D(instructions_string, pos="bottom-center", s=0.8)
 
instructions_string = "Press 'c' to clear the pinned vertices\nPress 'x' to scale up random vertices \n Press '1' to OptimizeMeshGD_ZeroEnergy\n Press '2' to OptimizeMeshNewton_ZeroEnergy\n Press '3' to OptimizeMeshGD_SpringEnergy\n Press '4' to OptimizeMeshNewton_SpringEnergy"
instructions_text = vd.Text2D(instructions_string, pos="top-center", s=0.5)

plt.add(introduction_text)
plt.add(instructions_text)
plt.add(ZeroLengthEnergySpringMessage)
plt.add(EnergySpringMessage)


plt.user_mode('2d').show(size=[1920,1000]).close()

# %%
