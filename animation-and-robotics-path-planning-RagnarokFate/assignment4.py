#%%
from scipy.spatial import KDTree
import numpy as np
import math
import random
import vedo as vd
from vedo.pyplot import DirectedGraph

vd.settings.default_backend = 'vtk'
#%%
class Sample:
    def __init__(self, x, parent = None):
        self.x = x
        self.parent = parent

# find the nearest sample using KDTree
def NearestSample(samples, x):
    tree = KDTree([sample.x for sample in samples])
    dist, sid = tree.query(x)
    return sid


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
        

# # get the next sample point
# def getNextSample(dim, samples, stepsize):
#     x = np.array([random.randint(0, d) for d in dim])
#     sid = NearestSample(samples,x)
#     nx = np.round(x) # TODO
#     ns = Sample(nx, sid)
#     return ns

# get the next sample point
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

    
#%%
def bfunc(obj, ename):
    global timer_id,path_info_text_instance
    plt.timer_callback("destroy", timer_id)
    if "Run" in runButton.status():
        path_info_text_instance.text("Path length: 0\n")
        timer_id = plt.timer_callback("create", dt=10)
    runButton.switch()

def doRRTIteration(event):
    y, x = img.shape
    ns = getNextSample([x, y], samples, stepSize)
    
    if ns is None or Collision(img, samples[ns.parent].x, ns.x):
        return

    samples.append(ns)
    points.vertices = np.vstack([points.vertices, np.hstack([ns.x, 0])])
    edges.append(vd.Line(ns.x, samples[ns.parent].x, lw=2))
    plt.add(points)
    plt.remove('Line')
    plt.add(edges)
    plt.render()

    distance = np.linalg.norm(ns.x - dest)
    # Check if destination is reached within a threshold
    if distance <= stepSize:
        print("Goal reached!")
        plt.timer_callback("destroy", timer_id)
        runButton.switch()
        processtPath(ns)


# ============================================= Assistant Function =============================================
def visualizeSourceDestination():
    global plt, source, dest, points, edges, samples, tree
    plt.remove('Points')
    plt.add(vd.Points([source], c='r', r=10))
    plt.add(vd.Points([dest], c='g', r=10))
    plt.render()

    tree = [Sample(source)]
    samples = [Sample(source)] # list to store all the node points
    points = vd.Points([source]) # list to store all the node points
    edges = []
    plt.remove('Line')
    plt.render()



def changeMode():
    global plt, currentMode, currentModeText,instruction_text_instance
    if currentMode == "Source & Destination":
        currentMode = "Draw"
        instruction_text_instance.text("Left click to draw obstacles")
    elif currentMode == "Draw":
        currentMode = "Erase"
        instruction_text_instance.text("Left click to erase obstacles")
    elif currentMode == "Erase":
        currentMode = "None"
        instruction_text_instance.text("no mode selected")
    elif currentMode == "None":
        currentMode = "Source & Destination"
        instruction_text_instance.text("Left click to set source \nRight click to set destination")
    currentModeText.text(currentMode)
    plt.render()


def processtPath(ns):
    global path_info_text_instance, path_info_text
    path = []
    node = ns
    while node is not None:
        path.append(node.x)
        node = samples[node.parent] if node.parent is not None else None
    
    path = path[::-1]  # Reverse to get from start to goal
    path_lines = [vd.Line(path[i], path[i+1], c='blue', lw=4) for i in range(len(path) - 1)]
    plt.add(path_lines)
    plt.render()


     # Display path information
    plt.remove(path_info_text_instance)
    path_info_text = f"Path length: {len(path)}\n"
    total_distance = 0.0
    for i, p in enumerate(path):
        prev_p = path[i - 1]
        distance = np.linalg.norm(p - prev_p)
        total_distance += distance
        path_info_text += f"Step/Depth {i} - Position: ({p[0]:.2f}, {p[1]:.2f})\n"
    path_info_text += f"Total distance: {total_distance:.2f}\n"
    print(path_info_text)
    
    path_info_text_instance.text(path_info_text)
    plt.add(path_info_text_instance)
    plt.render()



# ============================================= UI INTERACTIONS =============================================

def OnLeftButtonPress(event):
    global source, currentMode, vd_img, img, drawing
    if currentMode == "Source & Destination" and not isButtonClicked(event) and isImageClicked(event):
        mouse_pos = np.array(event.picked3d[:2])  # Get the mouse position in 2D
        source = np.array([mouse_pos[0], mouse_pos[1]])
        visualizeSourceDestination()
    elif currentMode == "Draw" and not isButtonClicked(event):
        drawing = True
        drawObstacle(event)
    elif currentMode == "Erase" and not isButtonClicked(event):
        drawing = True
        eraseObstacle(event)
    elif currentMode == "None" and not isButtonClicked(event):
        mouse_pos = np.array(event.picked3d[:2])
        x, y = int(mouse_pos[0]), int(mouse_pos[1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            plt.remove("Circle")
            plt.add(vd.Circle([x, y], r=1, c='purple'))
            pixel_value = img[383-y, x]
            selected_pixel_text = f"Coordinates: ({x}, {y}), Pixel Value: {pixel_value}"
            selected_pixel_text_instance.text(selected_pixel_text)
            plt.add(selected_pixel_text_instance)
            plt.render()

def OnLeftButtonRelease(event):
    global drawing
    drawing = False

def OnMouseMove(event):
    global drawing, currentMode
    if drawing:
        if currentMode == "Draw":
            drawObstacle(event)
        elif currentMode == "Erase":
            eraseObstacle(event)

def drawObstacle(event):
    global img, vd_img, BrushSize
    if event.picked3d is not None:
        mouse_pos = np.array(event.picked3d[:2])
        x, y = int(mouse_pos[0]), int(mouse_pos[1])
        y = img.shape[0] - y
        brush_size = int(BrushSize)
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            for i in range(-brush_size, brush_size + 1):
                for j in range(-brush_size, brush_size + 1):
                    if 0 <= x + i < img.shape[1] and 0 <= y + j < img.shape[0]:
                        img[y + j, x + i] = False
            plt.remove(vd_img)
            img_display = np.flipud(img)  # Flip vertically if necessary
            vd_img = vd.Image(img_display.astype(np.uint8) * 255).bw().binarize()
            plt.add(vd_img)
            plt.render()

def eraseObstacle(event):
    global img, vd_img, BrushSize
    if event.picked3d is not None:
        mouse_pos = np.array(event.picked3d[:2])
        x, y = int(mouse_pos[0]), int(mouse_pos[1])
        y = img.shape[0] - y
        brush_size = int(BrushSize)
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            for i in range(-brush_size, brush_size + 1):
                for j in range(-brush_size, brush_size + 1):
                    if 0 <= x + i < img.shape[1] and 0 <= y + j < img.shape[0]:
                        img[y + j, x + i] = True
            plt.remove(vd_img)
            img_display = np.flipud(img)  # Flip vertically if necessary
            vd_img = vd.Image(img_display.astype(np.uint8) * 255).bw().binarize()
            plt.add(vd_img)
            plt.render()


    # global source,currentMode, vd_img, img
    # if currentMode == "Source & Destination" and isButtonClicked(event) == False and isImageClicked(event):
    #     mouse_pos = np.array(event.picked3d[:2])  # Get the mouse position in 2D
    #     source = np.array([mouse_pos[0], mouse_pos[1]])
    #     visualizeSourceDestination()
    #     pass
    # elif currentMode == "Draw" and isButtonClicked(event) == False:
    #     mouse_pos = np.array(event.picked3d[:2])
    #     x, y = int(mouse_pos[0]), int(mouse_pos[1])
    #     # bounds check
    #     if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
    #         print(f"draw obstacle at {x}, {y}")
    # elif currentMode == "Erase" and isButtonClicked(event) == False:
    #     mouse_pos = np.array(event.picked3d[:2])
    #     x, y = int(mouse_pos[0]), int(mouse_pos[1])
    #     # bounds check
    #     if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
    #         print(f"erase obstacle at {x}, {y}")
    # elif currentMode == "None" and isButtonClicked(event) == False:
    #     mouse_pos = np.array(event.picked3d[:2])
    #     x, y = int(mouse_pos[0]), int(mouse_pos[1])
    #     if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
    #         plt.remove("Circle")
    #         plt.add(vd.Circle([x, y], r=1, c='purple'))
    #         pixel_value = img[383-y, x]
    #         selected_pixel_text = f"Coordinates: ({x}, {y}), Pixel Value: {pixel_value}"
    #         selected_pixel_text_instance.text(selected_pixel_text)
    #         plt.add(selected_pixel_text_instance)
    #         plt.render() 


def OnRightButtonPress(event):
    # TODO - set destination
    global dest , currentMode , vd_img, img
    if currentMode == "Source & Destination" and isButtonClicked(event) == False and isImageClicked(event):
        mouse_pos = np.array(event.picked3d[:2])  # Get the mouse position in 2D
        dest = np.array([mouse_pos[0], mouse_pos[1]])
        visualizeSourceDestination()



def resetTree(obj, ename):
    global plt, source, dest, points, edges, samples, tree
    tree = [Sample(source)]
    samples = [Sample(source)] # list to store all the node points
    points = vd.Points([source]) # list to store all the node points
    edges = []
    plt.remove('Line')
    plt.render()


# def generateRandomMap(obj, ename):
#     global img, vd_img
#     img = np.random.choice([False, True], size=img.shape, p=[0.1, 0.9])
#     plt.remove(vd_img)
#     img_display = np.flipud(img)  # Flip vertically if necessary
#     vd_img = vd.Image(img_display.astype(np.uint8) * 255).bw().binarize()
#     plt.add(vd_img)
#     plt.render()
def generateRandomMap(obj, ename,num_obstacles = 20,num_circles = 10):
        global img, vd_img
        img = np.ones((383, 683), dtype=bool)  # Start with an empty map

        # Add some rectangular obstacles
        for _ in range(num_obstacles):
            x_start = random.randint(0, img.shape[1] - 50)
            y_start = random.randint(0, img.shape[0] - 50)
            width = random.randint(10, 100)
            height = random.randint(10, 100)
            img[y_start:y_start + height, x_start:x_start + width] = False

        # Add some circular obstacles
        num_circles = 5
        for _ in range(num_circles):
            x_center = random.randint(50, img.shape[1] - 50)
            y_center = random.randint(50, img.shape[0] - 50)
            radius = random.randint(10, 75)
            for y in range(y_center - radius, y_center + radius):
                for x in range(x_center - radius, x_center + radius):
                    if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                            img[y, x] = False

        plt.remove(vd_img)
        img_display = np.flipud(img)  # Flip vertically if necessary
        vd_img = vd.Image(img_display.astype(np.uint8) * 255).bw().binarize()
        plt.add(vd_img)
        plt.render()

def setSliderStepSize(widget, event):
    global stepSize
    stepSize = widget.value
    resetTree(None, None)


def setBrushSize(widget, event):
    global BrushSize
    BrushSize = widget.value

def OnKeyPress(evt):
    if evt.keypress in ['m', 'M']:
        changeMode()

def isButtonClicked(event):
    global runButton, resetTreeButton, stepSizeSlider
    if event.object == runButton or event.object == resetTreeButton or event.object == stepSizeSlider:
        return True
    return False

def isImageClicked(event):
    global vd_img
    if event.object == vd_img:
        return True
    return False

imagePath = 'obstacle_map.png'
stepSize = 10.0
source = np.array([0,0])
dest = np.array([50,50])
tree = [Sample(source)]
drawing = False

vd_img = vd.Image(imagePath).bw().binarize()
img = vd_img.tonumpy().astype(bool)


currentMode = "Source & Destination"
samples = [Sample(source)] # list to store all the node points
points = vd.Points([source]) # list to store all the node points
edges = []
plt = vd.Plotter()
plt += vd_img
plt.user_mode('2d')
runButton = plt.add_button(bfunc, pos=(0.61, 0.2), states=[" Run ","Pause"])
resetTreeButton = plt.add_button(resetTree, pos=(0.65, 0.15),states=["Reset Tree"])
generateRandomMapButton = plt.add_button(generateRandomMap, pos=(0.2, 0.15),states=["Generate Random Map"])
stepSizeSlider = plt.add_slider(setSliderStepSize, 1.0,100.0, value=stepSize, title="Step Size")
# MODE TEXT
currentModeText = vd.Text2D(currentMode, pos='top-center', font="Normografo", s=1.0, c='black')
plt.add(currentModeText)
# Instruction Text
instruction_text = "Left click to set source \nRight click to set destination"
instruction_text_instance = vd.Text2D(instruction_text, pos='top-left', font="Normografo", s=1.0, c='black')

path_info_text = "Path length: 0\n"
path_info_text_instance = vd.Text2D(path_info_text, pos=(0.005,0.935), font="Normografo", s=0.4, c='black')
selected_pixel_text_instance = vd.Text2D("", pos=(0.05,0.15), font="Normografo", s=1.0, c='black')
# Callbacks
evntid = plt.add_callback("timer", doRRTIteration, enable_picking=False)
plt.add_callback('key press', OnKeyPress) # add Keyboard callback
plt.add_callback('LeftButtonPress', OnLeftButtonPress) 
plt.add_callback('RightButtonPress', OnRightButtonPress) 
plt.add_callback('LeftButtonRelease', OnLeftButtonRelease)
plt.add_callback('MouseMove', OnMouseMove,enable_picking= False) 
plt.add(instruction_text_instance)
plt.add(path_info_text_instance)
BrushSize = 1.0  # Default size
sizeSlider = plt.add_slider(setBrushSize, 1, 10, value=BrushSize,pos= 'bottom-left', title="Draw/Eraser Size")

timer_id = -1
visualizeSourceDestination()
plt.show(zoom="tightest").close()

# %%