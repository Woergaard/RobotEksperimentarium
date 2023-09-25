
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, pos):
            self.pos = pos      #configuration position, usually 2D/3D for planar robots  
            self.path = []      #the path with a integration horizon. this could be just a straight line for holonomic system
            self.parent = None
        
        def calc_distance_to(self, to_node):
            # node distance can be nontrivial as some form of cost-to-go function for e.g. underactuated system
            # use euclidean norm for basic holonomic point mass or as heuristics
            d = np.linalg.norm(np.array(to_node.pos) - np.array(self.pos))
            print(d)
            return d
        
    def __init__(self,
                 start,
                 goal,
                 robot_model,   #model of the robot
                 map,           #map should allow the algorithm to query if the path in a node is in collision. note this will ignore the robot geom
                 expand_dis=0.2,
                 path_resolution=0.05,
                 goal_sample_rate=5,
                 max_iter=500,
                 ):

        self.start = self.Node(start)
        self.end = self.Node(goal)
        self.robot = robot_model
        self.map = map
        
        self.min_rand = map.map_area[0]
        self.max_rand = map.map_area[1]

        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter

        self.node_list = []











    def draw_graph(self, rnd=None):
        # plt.clf()
        # # for stopping simulation with the esc key.
        # plt.gcf().canvas.mpl_connect(
        #     'key_release_event',
        #     lambda event: [exit(0) if event.key == 'escape' else None])
        plt.clf()
        if rnd is not None:
            plt.plot(rnd.pos[0], rnd.pos[1], "^k")

        # draw the map
        self.map.draw_map()

        for node in self.node_list:
            if node.parent:
                path = np.array(node.path)
                plt.plot(path[:, 0], path[:, 1], "-g")

        plt.plot(self.start.pos[0], self.start.pos[1], "xr")
        plt.plot(self.end.pos[0], self.end.pos[1], "xr")
        plt.axis(self.map.extent)
        plt.grid(True)
        plt.pause(0.01)


    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [ node.calc_distance_to(rnd_node)
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind


    def check_collision_free(self, node):
        if node is None:
            return False
        for p in node.path:
            if self.map.in_collision(np.array(p)):
                return False
        return True


import Ex4_grid, robot_models
