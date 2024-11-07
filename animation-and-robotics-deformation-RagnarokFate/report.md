# Deformations Assignment (Mass-Spring systems)

Student & ID: Bashar Beshoti - 207370248

**IMPORTANT NOTE** : If you witness spikes in mesh, it is a shortcut for me that scales 25 random vertices by 1.5x. I am too lazy to apply deformations through mouse drags for each execution :P

### Task 1: Create a mesh for the simulation

1. I've created batman logo shape as the new shape for the program and a convex Dalton . Here it is an image;

<div style="            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;">
<p>
    <img src="images\CreativeObject-1.png" width='480' height='405'> </br>
    <em>Figure 1 : Convex Dalton Mesh</em>
</p>


<p>
    <img src="images\CreativeObject-2.png" width='480' height='405'> </br>
    <em>Figure 2 : Batman Logo Mesh</em>
</p>

</div>

The logo has been drawn on a draft paper and used the points coordinates to create my desired shape.

2. Based on Task's requirement and discord clarification; I'm ought to create two different shapes; a shape with no interior vertices, and a shape of 100 vertices including interior verticies.

<div style="            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;">
<p>
    <img src="images\NoInteriorVertices.png" width='480' height='405'>  </br>
    <em>Figure 3 : shape with no interior vertices</em>
</p>


<p>
    <img src="images\Object96Vertices.png" width='480' height='405'>  </br>
    <em>Figure 4 : shape of approximate 100 vertices (96) including interior verticies.</em>
</p>

</div>


### Task 2: Implement Basic UI for Moving Points

1. Making the Terminal Text content to appear in Vedo Window instead of IDE terminal.

In order to achieve the requirement, I had to add a new global variable `terminalMessage` such that it would contain the nessecary data and print it out on vedo along with IDE's terminal.

<div style="            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;">
<p>
    <img src="images\TerminalMessage.png" width='480' height='405'>  </br>
    <em>Figure 5 : Terminal Text content in Vedo Window</em>
</p>


</div>



2. Moving pinned vertices that allows the user to set vertices into new positions.


<div style="            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;">
<p>
    <img src="images\OriginalVertices.png" width='480' height='405'>  </br>
    <em>Figure 5 : Original Vertices</em>
</p>

<p>
    <img src="images\ModifiedVertices.png" width='480' height='405'>  </br>
    <em>Figure 5 : Modified Vertices</em>
</p>

</div>


### Task 3: Test the optimization pipeline

1. Optimizing The Mesh through `MeshOptimizer` and `FEMesh` on different two approaches; Gradient Descent and Newton when Energy is set to `ZeroLengthSpringEnergy`. 

##### Gradient Descent on `ZeroLengthSpringEnergy`

<div style="            display: flex; 
            flex-direction: row;
            align-items: center;
            text-align: center;">

<p>
    <img src="images\OriginalMesh.png" width='360' height='360' style = "margin-right:16px;">  </br>
    <em>Figure 6 : Original Mesh </em>
</p>
<p>
    <img src="images\GradientDescent-3.1.png" width='360' height='360' style = "margin-left:16px;">  </br>
    <em>Figure 7 : Gradient Descent Applied (10 Iterations)</em>
</p>



</div>

##### Newton on `ZeroLengthSpringEnergy`

<div style="            display: flex; 
            flex-direction: row;
            align-items: center;
            text-align: center;">

<p>
    <img src="images\OriginalMesh.png" width='360' height='360' style = "margin-right:16px;">  </br>
    <em>Figure 8 : Original Mesh </em>
</p>
<p>
    <img src="images\Newton-3.1.png" width='360' height='360' style = "margin-left:16px;">  </br>
    <em>Figure 9 : Newton Applied (10 Iterations) </em>
</p>

</div>

**VIDEO** : 
<div style="            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;">
<p>
    <img src="images\videos\Gradient_vs_Newton-3.1.gif" width='480' height='405'>  </br>
    <em>Figure 10 : Gradient vs Newton on <code>ZeroLengthSpringEnergy</code> </em>
</p>


</div>

Explanation : In the context of mesh optimization, ZeroSpringLength refers to a scenario where the rest length of the springs in the mesh is zero. This means that the springs are always under tension or compression, and the goal is to minimize the energy of the system.

- Gradient Descent: Simpler to implement and requires only first-order derivatives (gradients). However, it may converge slowly, especially near the minimum.
- Newton's Method: More complex as it requires second-order derivatives (Hessian) and solving a linear system. It generally converges faster and more accurately, especially near the minimum, but can be computationally expensive.

Both methods aim to minimize the energy of the mesh by adjusting the positions of the vertices, but they differ in their approach and computational requirements.


2. Optimizing The Mesh through `MeshOptimizer` and `FEMesh` on different two approaches; Gradient Descent and Newton when Energy is set to `SpringEnergy`. 

##### Gradient Descent on `SpringEnergy`

<div style="            display: flex; 
            flex-direction: row;
            align-items: center;
            text-align: center;">

<p>
    <img src="images\OriginalMesh.png" width='360' height='360' style = "margin-right:16px;">  </br>
    <em>Figure 11 : Original Mesh </em>
</p>
<p>
    <img src="images\ModifiedMesh-3.2_1.png" width='360' height='360' style = "margin-left:16px;margin-right:16px;">  </br>
    <em>Figure 12 : Modified Mesh Vertices (Randomly) </em>
</p>

<p>
    <img src="images\GradientDescent-3.2.png" width='360' height='360' style = "margin-left:16px;">  </br>
    <em>Figure 13 : Gradient Descent Applied (10 Iterations)</em>
</p>

</div>

##### Newton on `SpringEnergy`

<div style="            display: flex; 
            flex-direction: row;
            align-items: center;
            text-align: center;">


<p>
    <img src="images\OriginalMesh.png" width='360' height='360' style = "margin-right:16px;">  </br>
    <em>Figure 14 : Original Mesh </em>
</p>
<p>
    <img src="images\ModifiedMesh-3.2_2.png" width='360' height='360' style = "margin-left:16px;margin-right:16px;">  </br>
    <em>Figure 15 : Modified Mesh Vertices (Randomly) </em>
</p>

<p>
    <img src="images\Newton-3.2.png" width='360' height='360' style = "margin-left:16px;">  </br>
    <em>Figure 16 : Newton Vertices Applied (10 Iterations) </em>
</p>

</div>

**VIDEO** : 
<div style="            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;">
<p>
    <img src="images\videos\Gradient_vs_Newton-3.2.gif" width='480' height='405'>  </br>
    <em>Figure 17 : Gradient vs Newton on <code>SpringEnergy</code> </em>
</p>


</div>

Explanation : In the context of mesh optimization, SpringEnergy refers to the energy stored in the springs of the mesh due to their deformation. The goal is to minimize this energy to achieve an optimal configuration of the mesh.

- Gradient Descent: Simpler to implement and requires only first-order derivatives (gradients). However, it may converge slowly, especially near the minimum.
- Newton's Method: More complex as it requires second-order derivatives (Hessian) and solving a linear system. It generally converges faster and more accurately,


3. Enabling pinning vertices; 
The suggested approach is to restore the coordinates values of pinned_vertices with each optimization process. Otherwords, `new_v[pinned_vertices] = old_v[pinned_vertices]` such that the elements in V that correspond to the fixed vertex match its coordinates.


<div style="            display: flex; 
            flex-direction: row;
            align-items: center;
            text-align: center;">


<p>
    <img src="images\PinnedVertices-1.png" width='360' height='360' style = "margin-right:16px;">  </br>
    <em>Figure 14 : Original Mesh with pinned vertex </em>
</p>
<p>
    <img src="images\PinnedVertices-2.png" width='360' height='360' style = "margin-left:16px;margin-right:16px;">  </br>
    <em>Figure 15 : Modified Mesh with pinned vertex </em>
</p>

<p>
    <img src="images\PinnedVertices-3.png" width='360' height='360' style = "margin-left:16px;">  </br>
    <em>Figure 16 : Mesh after Optimization </em>
</p>

</div>

**VIDEO** : 
<div style="            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;">
<p>
    <img src="images\videos\PinnedVertices-3.3.gif" width='480' height='405'>  </br>
    <em>Figure 17 : Gradient Descent With Pinned Vertex</em>
</p>


</div>

```py
# enable pinned vertices by setting their position to the initial position
        for i in self.femMesh.pinned_vertices:
            new_x[i] = self.femMesh.X[i]
```

OPTIONAL : The current approach has a flaw, propose a better approach:

This approach is flawed as it updates vertex positions after each optimization step, causing inconsistencies. The gradient and Hessian still account for pinned vertices, which are later reset, leading to inaccuracies. A better solution is to exclude pinned vertices entirely from these computations, improving efficiency and consistency.In other words, The flaw with this approach that both gradient and hessian values in pinned vertices are not suited based on their corresponding coordinates after retrieving their previous values. 



4. Optional: Implement the missing gradient and Hessian of `SpringEnergy` and compare the performance of the analytical derivatives.
```py
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
```