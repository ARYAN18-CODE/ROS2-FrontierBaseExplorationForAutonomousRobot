import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid , Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np
import heapq , math , random , yaml
import scipy.interpolate as si
import sys , threading , time


with open("src/autonomous_exploration/config/params.yaml", 'r') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

lookahead_distance = params["lookahead_distance"]
speed = params["speed"]
expansion_size = params["expansion_size"]
target_error = params["target_error"]
robot_r = params["robot_r"]

pathGlobal = 0

def euler_from_quaternion(x,y,z,w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return yaw_z

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

class BiRRTNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0
        self.children = []

def distance(node1, node2):
    """Calculate Euclidean distance between two nodes"""
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

def is_collision_free(array, node1, node2):
    """Check if path between two nodes is collision-free using line traversal"""
    x1, y1 = int(node1.x), int(node1.y)
    x2, y2 = int(node2.x), int(node2.y)
    
    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    x, y = x1, y1
    
    while True:
        # Check bounds and collision
        if x < 0 or x >= array.shape[0] or y < 0 or y >= array.shape[1]:
            return False
        if array[x, y] == 1:  # Obstacle
            return False
            
        if x == x2 and y == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    return True

def get_nearest_node(tree, random_node):
    """Find the nearest node in the tree to the random node"""
    min_dist = float('inf')
    nearest_node = None
    
    for node in tree:
        dist = distance(node, random_node)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    
    return nearest_node

def steer(from_node, to_node, extend_length=3.0):
    """Steer from one node towards another with limited extension"""
    dist = distance(from_node, to_node)
    if dist <= extend_length:
        return BiRRTNode(to_node.x, to_node.y)
    
    # Calculate unit vector and extend
    theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
    new_x = from_node.x + extend_length * math.cos(theta)
    new_y = from_node.y + extend_length * math.sin(theta)
    
    return BiRRTNode(new_x, new_y)

def get_near_nodes(tree, node, radius):
    """Get all nodes within a radius for RRT* optimization"""
    near_nodes = []
    for tree_node in tree:
        if distance(tree_node, node) <= radius:
            near_nodes.append(tree_node)
    return near_nodes

def choose_parent(near_nodes, new_node, array):
    """Choose the best parent for new node (RRT* optimization)"""
    if not near_nodes:
        return None
    
    best_parent = None
    best_cost = float('inf')
    
    for node in near_nodes:
        if is_collision_free(array, node, new_node):
            cost = node.cost + distance(node, new_node)
            if cost < best_cost:
                best_cost = cost
                best_parent = node
    
    return best_parent

def rewire(tree, new_node, near_nodes, array):
    """Rewire tree to optimize paths (RRT* optimization)"""
    for node in near_nodes:
        if node == new_node.parent:
            continue
        
        new_cost = new_node.cost + distance(new_node, node)
        if new_cost < node.cost and is_collision_free(array, new_node, node):
            # Remove old parent connection
            if node.parent:
                node.parent.children.remove(node)
            
            # Set new parent
            node.parent = new_node
            node.cost = new_cost
            new_node.children.append(node)
            
            # Update costs for all descendants
            update_costs(node)

def update_costs(node):
    """Update costs for all descendants of a node"""
    for child in node.children:
        child.cost = node.cost + distance(node, child)
        update_costs(child)

def get_path_from_tree(node):
    """Extract path from tree node to root"""
    path = []
    current = node
    while current is not None:
        path.append((current.x, current.y))
        current = current.parent
    return path[::-1]

def birrt_star(array, start, goal, max_iter=3000, extend_length=5.0):
    """
    Improved Bi-RRT* algorithm implementation with better success rate
    
    Args:
        array: 2D numpy array representing the map (0=free, 1=obstacle)
        start: tuple (x, y) start position
        goal: tuple (x, y) goal position
        max_iter: maximum number of iterations
        extend_length: maximum extension distance for steering
    
    Returns:
        list of (x, y) coordinates representing the path, or False if no path found
    """
    
    # Validate start and goal positions
    if (start[0] < 0 or start[0] >= array.shape[0] or 
        start[1] < 0 or start[1] >= array.shape[1] or
        goal[0] < 0 or goal[0] >= array.shape[0] or 
        goal[1] < 0 or goal[1] >= array.shape[1]):
        print("[ERROR] Start or goal position is out of bounds")
        return False
    
    if array[int(start[0]), int(start[1])] == 1 or array[int(goal[0]), int(goal[1])] == 1:
        print("[ERROR] Start or goal position is in an obstacle")
        return False
    
    # Initialize trees
    start_node = BiRRTNode(start[0], start[1])
    goal_node = BiRRTNode(goal[0], goal[1])
    
    tree_start = [start_node]
    tree_goal = [goal_node]
    
    # RRT* parameters - adjusted for better performance
    gamma = 30.0  # Increased for larger search radius
    goal_sample_rate = 0.15  # Increased goal sampling rate
    
    for i in range(max_iter):
        # Calculate dynamic radius for RRT*
        radius = min(gamma * math.sqrt(math.log(len(tree_start) + len(tree_goal) + 1) / 
                                     (len(tree_start) + len(tree_goal) + 1)), extend_length * 2)
        
        # Sample random point with higher goal bias
        if random.random() < goal_sample_rate:
            if i % 2 == 0:
                random_node = BiRRTNode(goal[0], goal[1])
            else:
                random_node = BiRRTNode(start[0], start[1])
        else:
            # Sample from free space only
            max_attempts = 50
            for _ in range(max_attempts):
                random_x = random.randint(0, array.shape[0] - 1)
                random_y = random.randint(0, array.shape[1] - 1)
                if array[random_x, random_y] == 0:  # Free space
                    break
            random_node = BiRRTNode(random_x, random_y)
        
        # Decide which tree to extend (alternating)
        if i % 2 == 0:
            extending_tree = tree_start
            other_tree = tree_goal
        else:
            extending_tree = tree_goal
            other_tree = tree_start
        
        # Find nearest node in extending tree
        nearest_node = get_nearest_node(extending_tree, random_node)
        
        # Steer towards random node
        new_node = steer(nearest_node, random_node, extend_length)
        
        # Check if new node is collision-free
        if not is_collision_free(array, nearest_node, new_node):
            continue
        
        # RRT* optimization: choose best parent
        near_nodes = get_near_nodes(extending_tree, new_node, radius)
        best_parent = choose_parent(near_nodes, new_node, array)
        
        if best_parent is None:
            # Fallback to nearest node if no optimal parent found
            if is_collision_free(array, nearest_node, new_node):
                best_parent = nearest_node
            else:
                continue
        
        # Set parent and cost
        new_node.parent = best_parent
        new_node.cost = best_parent.cost + distance(best_parent, new_node)
        best_parent.children.append(new_node)
        
        # Add to tree
        extending_tree.append(new_node)
        
        # RRT* optimization: rewire tree
        if len(near_nodes) > 1:  # Only rewire if there are multiple near nodes
            rewire(extending_tree, new_node, near_nodes, array)
        
        # Try to connect to other tree with increased connection distance
        connection_distance = extend_length * 1.5
        nearest_other = get_nearest_node(other_tree, new_node)
        
        if distance(new_node, nearest_other) <= connection_distance:
            if is_collision_free(array, new_node, nearest_other):
                # Trees connected! Extract path
                if extending_tree == tree_start:
                    path_start = get_path_from_tree(new_node)
                    path_goal = get_path_from_tree(nearest_other)
                    path_goal.reverse()
                    path = path_start + path_goal[1:]  # Avoid duplicate connection point
                else:
                    path_start = get_path_from_tree(nearest_other)
                    path_goal = get_path_from_tree(new_node)
                    path_goal.reverse()
                    path = path_start + path_goal[1:]  # Avoid duplicate connection point
                
                print(f"[SUCCESS] Bi-RRT* found path after {i+1} iterations")
                return path
        
        # Early termination check - if trees are getting close
        if i % 100 == 0 and i > 500:
            min_distance = float('inf')
            for start_node_iter in tree_start:
                for goal_node_iter in tree_goal:
                    dist = distance(start_node_iter, goal_node_iter)
                    if dist < min_distance:
                        min_distance = dist
            
            if min_distance < extend_length * 3:
                print(f"[INFO] Trees are close (distance: {min_distance:.2f}), continuing search...")
    
    # If no path found, try to return partial path
    print(f"[WARNING] Bi-RRT* failed to find complete path after {max_iter} iterations")
    
    # Find closest node to goal in start tree
    closest_node = None
    closest_dist = float('inf')
    
    for node in tree_start:
        dist = distance(node, goal_node)
        if dist < closest_dist:
            closest_dist = dist
            closest_node = node
    
    if closest_node is not None and closest_dist < extend_length * 5:
        print(f"[INFO] Returning partial path (distance to goal: {closest_dist:.2f})")
        return get_path_from_tree(closest_node)
    
    return False
def bspline_planning(array, sn):
    try:
        array = np.array(array)
        x = array[:, 0]
        y = array[:, 1]
        N = 2
        t = range(len(x))
        x_tup = si.splrep(t, x, k=N)
        y_tup = si.splrep(t, y, k=N)

        x_list = list(x_tup)
        xl = x.tolist()
        x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]

        y_list = list(y_tup)
        yl = y.tolist()
        y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

        ipl_t = np.linspace(0.0, len(x) - 1, sn)
        rx = si.splev(ipl_t, x_list)
        ry = si.splev(ipl_t, y_list)
        path = [(rx[i],ry[i]) for i in range(len(rx))]
    except:
        path = array
    return path

def pure_pursuit(current_x, current_y, current_heading, path, index):
    global lookahead_distance
    closest_point = None
    v = speed
    for i in range(index,len(path)):
        x = path[i][0]
        y = path[i][1]
        distance = math.hypot(current_x - x, current_y - y)
        if lookahead_distance < distance:
            closest_point = (x, y)
            index = i
            break
    if closest_point is not None:
        target_heading = math.atan2(closest_point[1] - current_y, closest_point[0] - current_x)
        desired_steering_angle = target_heading - current_heading
    else:
        target_heading = math.atan2(path[-1][1] - current_y, path[-1][0] - current_x)
        desired_steering_angle = target_heading - current_heading
        index = len(path)-1
    if desired_steering_angle > math.pi:
        desired_steering_angle -= 2 * math.pi
    elif desired_steering_angle < -math.pi:
        desired_steering_angle += 2 * math.pi
    if desired_steering_angle > math.pi/6 or desired_steering_angle < -math.pi/6:
        sign = 1 if desired_steering_angle > 0 else -1
        desired_steering_angle = sign * math.pi/4
        v = 0.0
    return v,desired_steering_angle,index

def frontierB(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0.0:
                if i > 0 and matrix[i-1][j] < 0:
                    matrix[i][j] = 2
                elif i < len(matrix)-1 and matrix[i+1][j] < 0:
                    matrix[i][j] = 2
                elif j > 0 and matrix[i][j-1] < 0:
                    matrix[i][j] = 2
                elif j < len(matrix[i])-1 and matrix[i][j+1] < 0:
                    matrix[i][j] = 2
    return matrix

def assign_groups(matrix):
    group = 1
    groups = {}
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 2:
                group = dfs(matrix, i, j, group, groups)
    return matrix, groups

def dfs(matrix, i, j, group, groups):
    if i < 0 or i >= len(matrix) or j < 0 or j >= len(matrix[0]):
        return group
    if matrix[i][j] != 2:
        return group
    if group in groups:
        groups[group].append((i, j))
    else:
        groups[group] = [(i, j)]
    matrix[i][j] = 0
    dfs(matrix, i + 1, j, group, groups)
    dfs(matrix, i - 1, j, group, groups)
    dfs(matrix, i, j + 1, group, groups)
    dfs(matrix, i, j - 1, group, groups)
    dfs(matrix, i + 1, j + 1, group, groups) # sağ alt çapraz
    dfs(matrix, i - 1, j - 1, group, groups) # sol üst çapraz
    dfs(matrix, i - 1, j + 1, group, groups) # sağ üst çapraz
    dfs(matrix, i + 1, j - 1, group, groups) # sol alt çapraz
    return group + 1

def fGroups(groups):
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    top_five_groups = [g for g in sorted_groups[:5] if len(g[1]) > 2]    
    return top_five_groups

def calculate_centroid(x_coords, y_coords):
    n = len(x_coords)
    sum_x = sum(x_coords)
    sum_y = sum(y_coords)
    mean_x = sum_x / n
    mean_y = sum_y / n
    centroid = (int(mean_x), int(mean_y))
    return centroid

def findClosestGroup(matrix, groups, current, resolution, originX, originY):
    targetP = None
    distances = []
    paths = []
    score = []
    max_score = -1
    
    # Try multiple attempts for each group if initial planning fails
    max_attempts = 3
    
    for i in range(len(groups)):
        middle = calculate_centroid([p[0] for p in groups[i][1]], [p[1] for p in groups[i][1]])
        
        # Try multiple times with different parameters if path planning fails
        path = None
        for attempt in range(max_attempts):
            if attempt == 0:
                # First attempt with normal parameters
                path = birrt_star(matrix, current, middle, max_iter=2000, extend_length=3.0)
            elif attempt == 1:
                # Second attempt with more iterations and smaller steps
                path = birrt_star(matrix, current, middle, max_iter=3000, extend_length=2.0)
            else:
                # Third attempt with maximum effort
                path = birrt_star(matrix, current, middle, max_iter=4000, extend_length=1.5)
            
            if path:
                break
        
        if path:
            # Convert path coordinates
            path = [(p[1]*resolution+originX, p[0]*resolution+originY) for p in path]
            total_distance = pathLength(path)
            distances.append(total_distance)
            paths.append(path)
            print(f"[SUCCESS] Path found to group {i+1} (distance: {total_distance:.2f})")
        else:
            distances.append(float('inf'))
            paths.append([])
            print(f"[FAILED] No path found to group {i+1}")
    
    # Calculate scores for each group
    for i in range(len(distances)):
        if distances[i] == 0 or distances[i] == float('inf'):
            score.append(0)
        else:
            # Score based on frontier size divided by distance
            frontier_size = len(groups[i][1])
            score.append(frontier_size / distances[i])
    
    # Find the best scoring group that meets distance criteria
    for i in range(len(distances)):
        if distances[i] > target_error * 3 and distances[i] != float('inf'):
            if max_score == -1 or score[i] > score[max_score]:
                max_score = i
    
    if max_score != -1:
        targetP = paths[max_score]
        print(f"[INFO] Selected group {max_score+1} as target (score: {score[max_score]:.4f})")
    else:
        # If no good group found, try a different approach
        print("[INFO] No suitable group found, trying alternative approach...")
        
        # Try to find any reachable group
        for i in range(len(distances)):
            if distances[i] != float('inf') and len(paths[i]) > 0:
                targetP = paths[i]
                print(f"[INFO] Using alternative target from group {i+1}")
                break
        
        # Last resort: random exploration
        if targetP is None and len(groups) > 0:
            print("[INFO] Attempting random target selection...")
            for attempt in range(3):  # Try 3 random targets
                index = random.randint(0, len(groups)-1)
                target = groups[index][1]
                target = target[random.randint(0, len(target)-1)]
                
                path = birrt_star(matrix, current, target, max_iter=2000, extend_length=2.0)
                if path:
                    targetP = [(p[1]*resolution+originX, p[0]*resolution+originY) for p in path]
                    print(f"[SUCCESS] Random target found (attempt {attempt+1})")
                    break
    
    return targetP

def pathLength(path):
    # Handle None or empty path
    if path is None:
        return 0
    
    # Convert to numpy array if it isn't already
    if not isinstance(path, np.ndarray):
        if len(path) == 0:
            return 0
        path = np.array(path)
    
    # Check if array is empty or has less than 2 points
    if path.size == 0 or len(path) < 2:
        return 0
    
    # Ensure path is in the correct format
    for i in range(len(path)):
        path[i] = (path[i][0], path[i][1])
    
    points = np.array(path)
    differences = np.diff(points, axis=0)
    distances = np.hypot(differences[:,0], differences[:,1])
    total_distance = np.sum(distances)
    return total_distance

def costmap(data,width,height,resolution):
    data = np.array(data).reshape(height,width)
    wall = np.where(data == 100)
    for i in range(-expansion_size,expansion_size+1):
        for j in range(-expansion_size,expansion_size+1):
            if i  == 0 and j == 0:
                continue
            x = wall[0]+i
            y = wall[1]+j
            x = np.clip(x,0,height-1)
            y = np.clip(y,0,width-1)
            data[x,y] = 100
    data = data*resolution
    return data

def exploration(data,width,height,resolution,column,row,originX,originY):
        global pathGlobal #Global degisken
        data = costmap(data,width,height,resolution) #Engelleri genislet
        data[row][column] = 0 #Robot Anlık Konum
        data[data > 5] = 1 # 0 olanlar gidilebilir yer, 100 olanlar kesin engel
        data = frontierB(data) #Sınır noktaları bul
        data,groups = assign_groups(data) #Sınır noktaları gruplandır
        groups = fGroups(groups) #Grupları küçükten büyüğe sırala. En buyuk 5 grubu al
        if len(groups) == 0: #Grup yoksa kesif tamamlandı
            path = -1
        else: #Grup varsa en yakın grubu bul
            data[data < 0] = 1 #-0.05 olanlar bilinmeyen yer. Gidilemez olarak isaretle. 0 = gidilebilir, 1 = gidilemez.
            path = findClosestGroup(data,groups,(row,column),resolution,originX,originY) #En yakın grubu bul
            if path != None: #Yol varsa BSpline ile düzelt
                path = bspline_planning(path,len(path)*5)
            else:
                path = -1
        pathGlobal = path
        return

def localControl(scan):
    v = None
    w = None
    for i in range(60):
        if scan[i] < robot_r:
            v = 0.2
            w = -math.pi/4 
            break
    if v == None:
        for i in range(300,360):
            if scan[i] < robot_r:
                v = 0.2
                w = math.pi/4
                break
    return v,w


class navigationControl(Node):
    def __init__(self):
        super().__init__('Exploration')
        self.subscription = self.create_subscription(OccupancyGrid,'map',self.map_callback,10)
        self.subscription = self.create_subscription(Odometry,'odom',self.odom_callback,10)
        self.subscription = self.create_subscription(LaserScan,'scan',self.scan_callback,10)
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        print("[BILGI] KESİF MODU AKTİF - Bi-RRT* Algoritması Kullanılıyor")
        self.kesif = True
        threading.Thread(target=self.exp).start() #Kesif fonksiyonunu thread olarak calistirir.
        
    def exp(self):
        twist = Twist()
        while True: #Sensor verileri gelene kadar bekle.
            if not hasattr(self,'map_data') or not hasattr(self,'odom_data') or not hasattr(self,'scan_data'):
                time.sleep(0.1)
                continue
            if self.kesif == True:
                if isinstance(pathGlobal, int) and pathGlobal == 0:
                    column = int((self.x - self.originX)/self.resolution)
                    row = int((self.y- self.originY)/self.resolution)
                    exploration(self.data,self.width,self.height,self.resolution,column,row,self.originX,self.originY)
                    self.path = pathGlobal
                else:
                    self.path = pathGlobal
                if isinstance(self.path, int) and self.path == -1:
                    print("[BILGI] KESİF TAMAMLANDI")
                    sys.exit()
                self.c = int((self.path[-1][0] - self.originX)/self.resolution) 
                self.r = int((self.path[-1][1] - self.originY)/self.resolution) 
                self.kesif = False
                self.i = 0
                print("[BILGI] YENI HEDEF BELİRLENDİ - Bi-RRT* ile Hesaplandı")
                t = pathLength(self.path)/speed
                t = t - 0.2 #x = v * t formülüne göre hesaplanan sureden 0.2 saniye cikarilir. t sure sonra kesif fonksiyonu calistirilir.
                self.t = threading.Timer(t,self.target_callback) #Hedefe az bir sure kala kesif fonksiyonunu calistirir.
                self.t.start()
            
            #Rota Takip Blok Baslangic
            else:
                v , w = localControl(self.scan)
                if v == None:
                    v, w,self.i = pure_pursuit(self.x,self.y,self.yaw,self.path,self.i)
                if(abs(self.x - self.path[-1][0]) < target_error and abs(self.y - self.path[-1][1]) < target_error):
                    v = 0.0
                    w = 0.0
                    self.kesif = True
                    print("[BILGI] HEDEFE ULASILDI")
                    self.t.join() #Thread bitene kadar bekle.
                twist.linear.x = v
                twist.angular.z = w
                self.publisher.publish(twist)
                time.sleep(0.1)
            #Rota Takip Blok Bitis

    def target_callback(self):
        exploration(self.data,self.width,self.height,self.resolution,self.c,self.r,self.originX,self.originY)
        
    def scan_callback(self,msg):
        self.scan_data = msg
        self.scan = msg.ranges

    def map_callback(self,msg):
        self.map_data = msg
        self.resolution = self.map_data.info.resolution
        self.originX = self.map_data.info.origin.position.x
        self.originY = self.map_data.info.origin.position.y
        self.width = self.map_data.info.width
        self.height = self.map_data.info.height
        self.data = self.map_data.data

    def odom_callback(self,msg):
        self.odom_data = msg
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = euler_from_quaternion(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)


def main(args=None):
    rclpy.init(args=args)
    navigation_control = navigationControl()
    rclpy.spin(navigation_control)
    navigation_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
