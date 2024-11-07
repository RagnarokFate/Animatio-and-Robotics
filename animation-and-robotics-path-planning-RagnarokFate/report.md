# Assignment 4 - Path Planning

Student & ID: Bashar Beshoti - 207370248

**IMPORTANT NOTE** :

## Task 1 : Complete the GUI
### 1. Visualize Source and destination points, add reset button and step slider.

- <font color ="red"> Red point represents Source. </font>
- <font color ="green"/> Green point represent target/destination. </font>

<img src="images\SourceDestinationStepSlider.png" alt="Source Destination Step Slider" width="480" height="405" />
</br>
<em>Figure 1 : Source & Destination, Step Slider and Reset Button Added! </em>
</br>

<img src="images\SourceDestinationStepSlider-2.png" alt="Source Destination Step Slider 2" width="480" height="405" />
</br>
<em>Figure 2 : different Source and Destination point </em>
</br>

### 2. Method to change the obstacle map

So, I've added couple of modes; "Source & Destination", "Draw", "Erase" where the mouse apply different process depending on the selected mode. The mode can be switched through pressing 'm' or 'M' key on the keyboard.

<img src="images\SourceDestinationMode.png" alt="Source Destination Mode" width="480" height="405" />
</br>
<em>Figure 3 : Source & Destination Mode </em>
</br>


<img src="images\DrawMode.png" alt="Draw Mode" width="480" height="405" />
</br>
<em>Figure 4 : Draw Mode </em>
</br>

<img src="images\EraseMode.png" alt="Erase Mode" width="480" height="405" />
</br>
<em>Figure 5 : Erase Mode </em>
</br>

How it operates?
Explanation : based on a new variable called 'BrushSize' which determine the size of the brush for both Draw and Erase. it locates the mouse position and acts corresponding to the current selected mode.

On Left Click Mouse Down, it changes the values of the osbtacle_map that represent black and white map. if the selected mode is draw, it change the pixels in BrushSize x BrushSize square to false representing a black wall/obstacle.

otherwise, if the selected mode is erase. it modify the square of pixels of size BrushSize x BrushSize away from the selected pixel to true resulting in a free space.

Note: Since the coordinates between Numpy Libarary and Vedo don't handle origin placement. `vedo` often uses a lower-left origin (like OpenGL), whereas `numpy` could have a different orientation in matrices, especially if it’s using an image grid


<img src="images\DrawEraseMode.gif" alt="Draw Erase Mode" width="480" height="405" />
</br>
<em>Figure 6 : Draw and Erase Mode </em>
</br>



### 3. Change Visualization
I've added path listing and total distance after reaching destination. it shows on plot the nodes and their depth corresponding to the `ns`.


## Task 2 : Implement RRT
### 1. Complete the implementation of `getNextSample`.
```py
def getNextSample(dim, samples, stepsize):
    # Generate a random point within the dimensions
    x = np.array([random.randint(0, d) for d in dim])
    sid = NearestSample(samples, x)
    nearest_sample = samples[sid].x
    
    # Calculate the direction vector from nearest sample to x
    direction = x - nearest_sample
    norm = np.linalg.norm(direction)
    if norm == 0: 
        return None
    
    # Normalize the direction and move stepSize distance
    direction = direction / norm
    nx = nearest_sample + direction * stepsize
    nx = np.clip(nx, [0, 0], dim)  # Ensure the sample stays within bounds
    
    ns = Sample(nx, sid)
    return ns
```

Explaination : The `getNextSample` function generates a new sample point within specified dimensions. It starts by creating a random point `x` within the given dimensions. It then finds the nearest existing sample to `x` and calculates the direction vector from this nearest sample to `x`. If the direction vector's norm is zero, it returns `None`. Otherwise, it normalizes the direction vector, scales it by the `stepsize`, and adds it to the nearest sample's coordinates to get the new sample point `nx`. The new sample point is clipped to stay within bounds, and a new Sample object is created and returned.

<img src="images\getNextSample.png" alt="get Next Sample" width="480" height="405" />
</br>
<em>Figure 7 : `getNextSample` without collision </em>
</br>


<img src="images\getNextSample-RUN.gif" alt="get Next Sample RUN" width="480" height="405" />
</br>
<em>Figure 8 : `getNextSample` without collision - RUN </em>
</br>


### 2. Complete the implementation of `Collision`.

NOTE : There was a small bug where in `doRRTIteration`:
```py
    # test for collision going from p to nn
    if Collision(img, ns.x, dest): # there is a collision
        return

```
But it should be :
```py
    # test for collision going from p to nn or ns is null
    if ns is None or Collision(img, samples[ns.parent].x, ns.x):
        return
```

Collision Function : 

```py

# check if the line between x1 and x2 intersects with the obstacles
def Collision(img, x1, x2):
    x1, x2 = np.array(x1, dtype=float), np.array(x2, dtype=float)
    num_points = int(np.linalg.norm(x2 - x1))
    for i in range(num_points + 1):
        point = x1 + (x2 - x1) * i / num_points
        x, y = int(point[0]), int(point[1])
        # Check if the point is within the image boundaries
        if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
            return True
        if img[383-y, x] == False:  # Check if the pixel is black
            return True
    return False
```

<img src="images\CollisionFunction.png" alt="Collision Function" width="480" height="405" />
</br>
<em>Figure 7 : `Collision` Iteration Picture </em>
</br>

<img src="images\CollisionFunction-2.png" alt="Collision Function 2" width="480" height="405" />
</br>
<em>Figure 8 : `Collision` Iteration Picture [More iterations] </em>
</br>



### 3. Add a stopping condition for `doRRTIteration`.

```py
    # Check if destination is reached within a threshold
    if np.linalg.norm(ns.x - dest) <= stepSize:
        print("Goal reached!")
        plt.timer_callback("destroy", timer_id)
```

Explanation : The stop condition checks if the newly generated sample `ns` is within a certain distance (threshold) from the destination `dest`. This is done by calculating the Euclidean distance between `ns.x` and `dest` using `np.linalg.norm`. If this distance is less than or equal to the `stepSize`, it means the destination is reached or very close to being reached. Therefore, it ensure that the algorithm stops when the destination is sufficiently close within `stepsize` as a threshold.


### 4. Setup an experiment and run it until the destination point is found.

<img src="images\RRT-RUN.gif" alt="RRT RUN" width="480" height="405" />
</br>
<em>Figure 9 : RRT-RUN </em>
</br>


### 5. Highlight the path

<img src="images\Full-RRT-RUN.gif" alt="RRT RUN" width="480" height="405" />
</br>
<em>Figure 10 : Full RRT-RUN with Highlight </em>
</br>

## Task 3 : Different parameters and different maps, and reason about the strengths and weaknesses of the algorithm.
Strength of the algorithm is the rapid exploring though the image and the weakness of it is that the found path is not optimal. that's can be improved if we modify the algorithm by adding two key additions to the algorithm result in significantly different results; Cost Optimization and Tree rewiring.


<img src="images\RandomMapGenerated-RUN.png" alt="RRT RUN" width="480" height="405" />
</br>
<em>Figure 11 : Full RRT-RUN on Random Generated Map  </em>
</br>


<img src="images\RandomMapGenerated-RUN-2.png" alt="RRT RUN" width="480" height="405" />
</br>
<em>Figure 12 : Full RRT-RUN on Random Generated Map 2 </em>
</br>