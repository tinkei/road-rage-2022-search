from commonroad_cc.collision_detection.pycrcc_collision_dispatch import create_collision_checker, create_collision_object
from commonroad.scenario.obstacle import StaticObstacle, ObstacleType, Obstacle
from commonroad.scenario.trajectory import State as StateTupleFactory
from commonroad.scenario.trajectory import State
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.geometry.shape import Polygon, ShapeGroup, Circle
from commonroad.common.util import Interval
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.trajectory import Trajectory
from automata.MotionAutomata import MotionPrimitive
import numpy as np
import math
import construction
import heapq
import copy
from typing import *




# ====================================================================================================
# Logging
# ====================================================================================================
# Level    Numeric value
# CRITICAL 50
# ERROR    40
# WARNING  30
# INFO     20
# DEBUG    10
# NOTSET   0

import os
import sys
import logging

def setupLogging(filename):

    logPath = './log/'
    logFile = filename#  + datetime.now(tz=hkt).strftime('%Y-%m-%d_%H-%M-%S.%f')
    os.makedirs(logPath, exist_ok=True)

    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    log_handler_console_out = logging.StreamHandler(sys.stdout)
    log_handler_console_out.setLevel(logging.INFO)
    log_handler_console_out.addFilter(lambda record: record.levelno >= logging.INFO)

    log_handler_console_err = logging.StreamHandler(sys.stderr) # Goes to stderr by default.
    log_handler_console_err.setLevel(logging.WARNING)

    log_handler_file_debug = logging.FileHandler("{0}{1}_debug.log".format(logPath, logFile), encoding='utf-8', mode='w')
    log_handler_file_debug.setLevel(logging.NOTSET)

    log_handler_file_info  = logging.FileHandler("{0}{1}_info.log".format( logPath, logFile), encoding='utf-8', mode='w')
    log_handler_file_info.setLevel(logging.INFO)

    log_handler_file_error = logging.FileHandler("{0}{1}_error.log".format(logPath, logFile), encoding='utf-8', mode='w')
    log_handler_file_error.setLevel(logging.WARNING)

    # Options: https://docs.python.org/3/library/logging.html#logrecord-attributes
    logging.basicConfig(
        level=logging.INFO,
        # format="%(asctime)s [%(processName)-12.12s %(process)-5.5d] [%(threadName)-12.12s] [%(levelname)-5.5s] [%(name)s %(filename)s %(funcName)s %(lineno)d]  %(message)s",
        # format="%(asctime)s [%(processName)-12.12s %(process)-5.5d] [%(threadName)-12.12s] [%(name)-6.6s %(filename)-8.8s] [%(levelname)-5.5s]  %(message)s",
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            log_handler_console_out,
            log_handler_console_err,
            log_handler_file_debug,
            log_handler_file_info,
            log_handler_file_error
        ]
    )

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.info('Logger created (GBFS).')




import time
from IPython.display import display, clear_output




# example Greedy Best First Search planner

class MotionPlanner:

    def __init__(self, scenario, planningProblem, automata):

        setupLogging('output_gbfs')

        # store input parameters
        self.scenario = scenario
        self.planningProblem = planningProblem
        self.automata = automata

        self.egoShape = automata.egoShape

        # create necessary attributes
        self.lanelet_network = self.scenario.lanelet_network
        self.priority_queue = PriorityQueue()
        self.obstacles = self.scenario.obstacles
        self.initial_state = self.planningProblem.initial_state

        # remove unneeded attributes of initial state
        if hasattr(self.initial_state, 'yaw_rate'):
            del self.initial_state.yaw_rate

        if hasattr(self.initial_state, 'slip_angle'):
            del self.initial_state.slip_angle

        # get lanelet id of the starting lanelet (of initial state)
        self.startLanelet_ids = self.scenario.lanelet_network.find_lanelet_by_position(
            [self.planningProblem.initial_state.position])[0]

        self.goalLanelet_ids = None
        # get lanelet id of the ending lanelet (of goal state),this depends on type of goal state
        if hasattr(self.planningProblem.goal.state_list[0], 'position'):
            if hasattr(self.planningProblem.goal.state_list[0].position, 'center'):
                self.goalLanelet_ids = self.scenario.lanelet_network.find_lanelet_by_position(
                    [self.planningProblem.goal.state_list[0].position.center])[0]

            elif hasattr(planningProblem.goal.state_list[0].position, 'shapes'):
                self.goalLanelet_ids = self.scenario.lanelet_network.find_lanelet_by_position(
                    [self.planningProblem.goal.state_list[0].position.shapes[0].center])[0]
                self.planningProblem.goal.state_list[0].position.center = \
                    self.planningProblem.goal.state_list[0].position.shapes[0].center

        # set specifications from given goal state
        if hasattr(self.planningProblem.goal.state_list[0], 'time_step'):
            self.desired_time = self.planningProblem.goal.state_list[0].time_step
        else:
            self.desired_time = Interval(0, np.inf)

        if hasattr(self.planningProblem.goal.state_list[0], 'velocity'):
            self.desired_velocity = self.planningProblem.goal.state_list[0].velocity
        else:
            self.desired_velocity = Interval(0, np.inf)

        if hasattr(self.planningProblem.goal.state_list[0], 'orientation'):
            self.desired_orientation = self.planningProblem.goal.state_list[0].orientation
        else:
            self.desired_orientation = Interval(-math.pi, math.pi)

        # create necessary attributes
        if hasattr(planningProblem.goal.state_list[0], 'position'):
            self.initial_distance = distance(planningProblem.initial_state.position,
                                             planningProblem.goal.state_list[0].position.center)
        else:
            self.initial_distance = 0

        # set lanelet costs to -1, except goal lanelet
        self.lanelet_cost = {}
        for lanelet in scenario.lanelet_network.lanelets:
            self.lanelet_cost[lanelet.lanelet_id] = -1

        if self.goalLanelet_ids is not None:
            for goal_lanelet_id in self.goalLanelet_ids:
                self.lanelet_cost[goal_lanelet_id] = 0

            # calculate costs for lanelets, this is a recursive method
            for goal_lanelet_id in self.goalLanelet_ids:
                visited_lanelets = []
                self.calc_lanelet_cost(self.scenario.lanelet_network.find_lanelet_by_id(goal_lanelet_id), 1, visited_lanelets)

        # construct commonroad boundaries and collision checker object
        build = ['section_triangles', 'triangulation']
        boundary = construction.construct(self.scenario, build, [], [])
        road_boundary_shape_list = list()
        initial_state = None
        for r in boundary['triangulation'].unpack():
            initial_state = StateTupleFactory(position=np.array([0, 0]), orientation=0.0, time_step=0)
            p = Polygon(np.array(r.vertices()))
            road_boundary_shape_list.append(p)
        road_bound = StaticObstacle(obstacle_id=scenario.generate_object_id(), obstacle_type=ObstacleType.ROAD_BOUNDARY,
                                    obstacle_shape=ShapeGroup(road_boundary_shape_list), initial_state=initial_state)
        self.collisionChecker = create_collision_checker(self.scenario)
        self.collisionChecker.add_collision_object(create_collision_object(road_bound))

    def map_obstacles_to_lanelets(self, time_step: int):
        """
        Find all obstacles that are located in every lanelet at time step t and returns a dictionary where obstacles are stored according to lanelet id.

        :param time_step: The time step in which the obstacle is in the current lanelet network. 
        :Return type: dict[lanelet_id] 
        """
        mapping = {}
        for lanelet in self.lanelet_network.lanelets:
            # map obstacles to current lanelet
            mapped_objs = self.get_obstacles(lanelet, time_step)
            # check if mapping is not empty
            if len(mapped_objs) > 0:
                mapping[lanelet.lanelet_id] = mapped_objs
        return mapping

    def get_obstacles(self, laneletObj: Lanelet, time_step: int) -> List[Obstacle]:
        """
        Returns the subset of obstacles, which are located in the given lanelet.

        :param laneletObj: specify the lanelet object to get its obstacles.
        :param time_step: The time step for the occupancy to check.
        
        """
        # output list
        res = list()

        # look at each obstacle
        for o in self.obstacles:
            if o.occupancy_at_time(time_step) is not None:
                o_shape = o.occupancy_at_time(time_step).shape

                # vertices to check
                vertices = list()

                # distinguish between shape and shapegroup and extract vertices
                if isinstance(o_shape, ShapeGroup):
                    for sh in o_shape.shapes:
                        # distinguish between type of shape (circle has no vertices)
                        if isinstance(sh, Circle):
                            vertices.append(sh.center)
                        else:
                            vertices.append(sh.vertices)
                else:
                    # distinguish between type of shape (circle has no vertices)
                    if isinstance(o_shape, Circle):
                        vertices = o_shape.center
                    else:
                        vertices = o_shape.vertices

                # check if obstacle is in lane
                if any(laneletObj.contains_points(np.array(vertices))):
                    res.append(o)
        return res

    def calc_lanelet_cost(self, curLanelet: Lanelet, dist: int, visited_lanelets: List[int]):
        """
        Calculates distances of all lanelets which can be reached through recursive adjacency/predecessor relationship by the current lanelet. 
        This is a recursive implementation.

        :param curLanelet: the current lanelet object (Often set to the goal lanelet).
        :param dist: the initial distance between 2 adjacent lanelets (Often set to 1). This value will increase recursively during the execution of this function.
        :param visited_lanelets: list of visited lanelet id. In the iterations, visited lanelets will not be considered. This value changes during the recursive implementation.

        The calculated costs will be stored in dictionary self.lanelet_cost[Lanelet].
        """
        if curLanelet.lanelet_id in visited_lanelets:
            return
        else:
            visited_lanelets.append(curLanelet.lanelet_id)

        if curLanelet.predecessor is not None:
            for pred in curLanelet.predecessor:
                if self.lanelet_cost[pred] == -1 or self.lanelet_cost[pred] > dist:
                    self.lanelet_cost[pred] = dist

            for pred in curLanelet.predecessor:
                self.calc_lanelet_cost(self.lanelet_network.find_lanelet_by_id(pred), dist + 1, visited_lanelets)

        if curLanelet.adj_left is not None and curLanelet.adj_left_same_direction:
            if self.lanelet_cost[curLanelet.adj_left] == -1 or self.lanelet_cost[curLanelet.adj_left] > dist:
                self.lanelet_cost[curLanelet.adj_left] = dist
                self.calc_lanelet_cost(self.lanelet_network.find_lanelet_by_id(curLanelet.adj_left), dist + 1, visited_lanelets)

        if curLanelet.adj_right is not None and curLanelet.adj_right_same_direction:
            if self.lanelet_cost[curLanelet.adj_right] == -1 or self.lanelet_cost[curLanelet.adj_right] > dist:
                self.lanelet_cost[curLanelet.adj_right] = dist
                self.calc_lanelet_cost(self.lanelet_network.find_lanelet_by_id(curLanelet.adj_right), dist + 1, visited_lanelets)

    def calc_lanelet_orientation(self, lanelet_id: int, pos: np.ndarray) -> float:
        """
        Returns lanelet orientation (angle in radian, counter-clockwise defined) at the given position and lanelet id.

        :param lanelet_id: id of the lanelet, based on which the orientation is calculated.
        :param pos: position, where orientation is calculated. (Often the position of the obstacle)

        """

        laneletObj = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
        return calcAngleOfPosition(laneletObj.center_vertices, pos)

    def calc_angle_to_goal(self, state: State) -> float:
        """
        Returns the orientation of the goal (angle in radian, counter-clockwise defined) with respect to position of the state.

        :param state: the angle between this state and the goal will be calculated.

        """

        curPos = state.position
        if hasattr(self.planningProblem.goal.state_list[0], 'position'):
            goalPos = self.planningProblem.goal.state_list[0].position.center
            return math.atan2(goalPos[1] - curPos[1], goalPos[0] - curPos[0])
        else:
            return 0

    def lanelets_of_position(self, lanelets: List[int], state: State, diff=math.pi / 5) -> List[int]:
        """
        Returns all lanelets, the angle between which and the orientation of the input state is smaller than pi/5.

        :param lanelets: lanelets, whose orientation is considered.
        :param state: the state, whose orientation is considered.
        :param diff: acceptable angle difference between the state and the lanelet.

        """

        correctLanelets = []
        for laneletId in lanelets:
            laneletObj = self.scenario.lanelet_network.find_lanelet_by_id(laneletId)
            laneletOrientationAtPosition = calcAngleOfPosition(laneletObj.center_vertices, state.position)
            if math.pi - abs(abs(laneletOrientationAtPosition - state.orientation) - math.pi) < diff:
                correctLanelets.append(laneletId)

        while len(correctLanelets) > 0:
            if self.lanelet_cost[correctLanelets[0]] == -1:
                correctLanelets.pop(0)
            else:
                break

        return correctLanelets

    def dist_to_closest_obstacle(self, lanelet_id: int, pos: np.ndarray, time_step: int) -> float:
        """
        Returns distance between the input position and the center of the closest obstacle in the respective lanelet (specified by input lanelet id).

        :param lanelet_id: the id of the lanelet where distance is calculated.
        :param pos: current position, from which the distance is calculated.
        :param time_step: current time step.

        """

        obstacles_in_lanelet = self.get_obstacles(self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id), time_step)
        shortestDist = math.inf
        for obstacleObj in obstacles_in_lanelet:
            if distance(pos, obstacleObj.occupancy_at_time(time_step).shape.center) < shortestDist:
                shortestDist = distance(pos, obstacleObj.occupancy_at_time(time_step).shape.center)
        return shortestDist

    def num_obstacles_in_lanelet_at_time_step(self, time_step: int, lanelet_id: int) -> int:
        """
        Returns the number of obstacles in the given lanelet at time step t.

        :param time_step: time step.
        :param lanelet_id: id of the lanelet whose obstacles are considered.

        """
        obstacles_in_lanelet = self.get_obstacles(self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id),
                                                  time_step)
        return len(obstacles_in_lanelet)

    def is_adjacent(self, start_lanelet_id: int, final_lanelet_id: int) -> bool:
        """
        Returns true if the the final lanelet is adjacent to the start lanelet.

        :param start_lanelet_id: id of the first lanelet (start lanelet).
        :param final_lanelet_id: id of the second lanelet (final lanelet).
        
        """

        laneletObj = self.scenario.lanelet_network.find_lanelet_by_id(start_lanelet_id)
        if laneletObj.adj_left is not None and laneletObj.adj_left_same_direction:
            if laneletObj.adj_left == final_lanelet_id:
                return True

        if laneletObj.adj_right is not None and laneletObj.adj_right_same_direction:
            if laneletObj.adj_right == final_lanelet_id:
                return True
        return False

    def is_successor(self, start_lanelet_id: int, final_lanelet_id: int) -> bool:
        """
        Returns true if the the final lanelet is a succrssor of the start lanelet.

        :param start_lanelet_id: id of the first lanelet (start lanelet).
        :param final_lanelet_id: id of the second lanelet (final lanelet).
        
        Return type: bool
        """

        laneletObj = self.scenario.lanelet_network.find_lanelet_by_id(start_lanelet_id)
        if laneletObj.successor is not None:
            for suc in laneletObj.successor:
                if suc == final_lanelet_id:
                    return True
        return False

    def is_goal_in_lane(self, lanelet_id: int, traversed_lanelets=None) -> bool:
        """
        Returns true if the goal is in the given lanelet or any successor (including all successors of successors) of the given lanelet.

        :param lanelet_id: the id of the given lanelet.
        :param traversed_lanelets: helper variable which stores potential path (a list of lanelet id) to goal lanelet. Initialized to None. 

        """
        if traversed_lanelets is None:
            traversed_lanelets = []
        if lanelet_id not in traversed_lanelets:
            traversed_lanelets.append(lanelet_id)
        else:
            return False
        reachable = False
        if lanelet_id in self.goalLanelet_ids:
            return True
        laneletObj = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
        if laneletObj.successor is not None:
            for suc in laneletObj.successor:
                if self.lanelet_cost[suc] >= 0:
                    reachable = self.is_goal_in_lane(suc, traversed_lanelets)
                    if reachable:
                        return True
        return reachable

    @staticmethod
    def calc_time_cost(path: List[State]) -> int:
        """
        Returns time cost (number of time steps) to perform the given path.

        :param path: the path on which the time cost is calculated.
        """
        return path[-1].time_step - path[0].time_step

    def calc_path_efficiency(self, path: List[State]) -> float:
        """
        Returns the path efficiency = travelled_distance / time_cost

        :param path: the path whose efficiency should be calculated.
        """
        return calc_travelled_distance(path) / self.calc_time_cost(path)

    def calc_heuristic_distance(self, state: State, distance_type=0) -> float:
        """
        Returns the heuristic distance between the current state and the goal state.

        :param state: the state, whose heuristic distance to the goal is calculated.
        :param distance_type: 0: euclideanDistance, 1: manhattanDistance, 2: chebyshevDistance, 3: sumOfSquaredDifference,
                              4: meanAbsoluteError, 5: meanSquaredError, 6: canberraDistance, 7: cosineDistance
        """

        curPos = state.position
        goalPos = self.planningProblem.goal.state_list[0].position.center

        return distance(curPos, goalPos, distance_type)

    def calc_heuristic_lanelet(self, path: List[State]):
        """
        Calculated the distance between every individual state of the path and the centers of their corresponding lanelets and sum them up.

        :param path: the path, whose heuristics is calculated.

        Returns the heuristic distance of the path (int), id of the end lanelet of the given path (list) and the start lanelet id (list). 
        End lanelet means the lanelet where the last state pf the path is in, start lanelet means the lanelet corresponding to the first state of the path.
        
        """
        end_lanelet_id = None
        dist = 0
        start_lanelet_id = self.scenario.lanelet_network.find_lanelet_by_position([path[0].position])[
            0]  # returns id of the start lanelet
        if not start_lanelet_id:
            return None, None, None
        for i in range(len(path)):
            lanelets_of_pathSegment = self.lanelets_of_position(
                self.scenario.lanelet_network.find_lanelet_by_position([path[i].position])[0], path[i])
            if not lanelets_of_pathSegment:
                return None, None, None  # return none if path element is not in a lanelet with correct orientation
            laneletObj = self.scenario.lanelet_network.find_lanelet_by_id(lanelets_of_pathSegment[0])
            dist = dist + findDistanceToNearestPoint(laneletObj.center_vertices,
                                                     path[i].position)  # distance to center line
            end_lanelet_id = lanelets_of_pathSegment
        return dist, end_lanelet_id, start_lanelet_id

    def reached_goal(self, path: List[State]) -> bool:
        """
        Goal-test every state of the path and returns true if one of the state satisfies all conditions for the goal region: position, orientation, velocity, time.

        :param path: the path to be goal-tested.

        """
        for i in range(len(path)):
            if self.planningProblem.goal.is_reached(path[i]):
                return True
        return False

    def remove_states_behind_goal(self, path: List[State]) -> List[State]:
        """
        Remove all states that are behind the state which statisfies the goal conditions and return the pruned path.

        :param path: the path to be pruned.
        """
        for i in range(len(path)):
            if self.planningProblem.goal.is_reached(path[i]):
                for j in range(i + 1, len(path)):
                    path.pop()
                return path
        return path

    def check_collision_free(self, path: List[State]) -> bool:
        """
        Checks if path collides with an obstacle. Returns true for no collision and false otherwise.

        :param path: The path you want to check
        """
        trajectory = Trajectory(path[0].time_step, path)

        # create a TrajectoryPrediction object consisting of the trajectory and the shape of the ego vehicle
        traj_pred = TrajectoryPrediction(trajectory=trajectory, shape=self.egoShape)

        # create a collision object using the trajectory prediction of the ego vehicle
        co = create_collision_object(traj_pred)

        # check collision for motion primitive
        if self.collisionChecker.collide(co):
            return False
        return True

    @staticmethod
    def translate_primitive_to_current_state(primitive: MotionPrimitive, path_current: List[State]) -> List[State]:
        """
        Uses the value of the given primitive, translate them towards the last state of current path and returns the list of new path.
        In the newly appended part of the path, the position, orientation and time step are changed, but the velocity is not changed.
        Attention: The input primitive itself will not be changed after this operation.

        :param primitive: the primitive to be translated.
        :param path_current: the path whose last state is the reference state for the translation.
        """
        return primitive.appendTrajectoryToState(path_current[-1])

    @staticmethod
    def append_path(path_current: List[State], pathNew: List[State]) -> List[State]:
        """
        Appends a new path to the current path and returns the whole path.

        :param path_current: current path which is to be extended.
        :param pathNew: new path which is going to be added to the current path.
        """
        path = path_current[:]
        path.extend(pathNew)
        return path

    @staticmethod
    def get_successor_primitives(cur_primitive: MotionPrimitive) -> List[MotionPrimitive]:
        """
        Returns all possible successor primitives of the current primitive

        :param cur_primitive: current primitive.
        """
        return cur_primitive.Successors

    def calc_heuristic_cost_sample(self, path: List[State], curPos: State):
        """
        Returns heuristic cost for the path.
        You should define your own cost function here.

        :param path: The path for which you want to calculate the cost
        :param curPos: Last state of your current path. Every state has the following variables: position, velocity, orientation, time_step
        """

        # *** below is just a sample code from a random previous student,
        # *** you are free to come up with your own ideas on how to compute the heuristic cost 
        # sample code 1
        
        # dist_pos = self.calc_heuristic_distance(path[-1])
        # if hasattr(self.planningProblem.goal.state_list[0], 'velocity'):
        #     v_mean_goal = (self.planningProblem.goal.state_list[0].velocity.start + 
        #                    self.planningProblem.goal.state_list[0].velocity.end) / 2
        #     dist_vel = abs(path[-1].velocity - v_mean_goal)
        # else:
        #     dist_vel = 0

        # if hasattr(self.planningProblem.goal.state_list[0], 'orientation'):
        #     o_mean_goal = (self.planningProblem.goal.state_list[0].orientation.start + 
        #                    self.planningProblem.goal.state_list[0].orientation.end) / 2
        #     dist_o = abs(path[-1].orientation - o_mean_goal)
        # else:
        #     dist_o = 0

        # weighs = np.zeros(3)
        # weighs[0] = 1.0
        # weighs[1] = 0.3
        # weighs[2] = 0.1

        # if dist_pos < 5:
        #     weighs[1] = 1.0
        #     weighs[2] = 1.0

        # return (dist_pos * weighs[0] + \
        #         dist_vel * weighs[1] + \
        #         dist_o   * weighs[2]   )

        # *** below is just a sample code from a random previous student,
        # *** you are free to come up with your own ideas on how to compute the heuristic cost 
        # sample code 2

        distStartState = self.calc_heuristic_distance(path[0])
        distLastState = self.calc_heuristic_distance(path[-1])
        if distLastState is None:
            return None
            
        if distStartState < distLastState:
            return None
            
        cost_lanelet, final_lanelet_id, start_lanelet_id = self.calc_heuristic_lanelet(path)
        if cost_lanelet is None or final_lanelet_id[0] is None:
            
            return None
        self.calc_path_efficiency(path)
        self.num_obstacles_in_lanelet_at_time_step(path[-1].time_step, final_lanelet_id[0])
        self.is_goal_in_lane(final_lanelet_id[0])
        if self.lanelet_cost[final_lanelet_id[0]] == -1:
            
            return None
        factor = 1
        if self.lanelet_cost[final_lanelet_id[0]] > self.lanelet_cost[start_lanelet_id[0]]:
            
            return None
        if self.lanelet_cost[final_lanelet_id[0]] < self.lanelet_cost[start_lanelet_id[0]]:
            factor = factor * 0.1
        laneletOrientationAtPosition = self.calc_lanelet_orientation(final_lanelet_id[0], path[-1].position)
        angleToGoal = self.calc_angle_to_goal(path[-1])

        orientationToGoalDiff = orientation_diff(angleToGoal, path[-1].orientation)
        if final_lanelet_id[0] in self.goalLanelet_ids:
            factor = factor * 0.07
        pathLength = calc_travelled_distance(path)
        cost_time = self.calc_time_cost(path)
        weigths = np.zeros(6)
        if distLastState < 0.5:
            factor = factor * 0.00001
        # elif math.pi - abs(abs(laneletOrientationAtPosition - path[-1].orientation)

        if hasattr(self.planningProblem.goal.state_list[0], 'velocity'):
            v_mean_goal = (self.planningProblem.goal.state_list[0].velocity.start + 
                           self.planningProblem.goal.state_list[0].velocity.end) / 2
            dist_vel = abs(path[-1].velocity -v_mean_goal)
        else:
            dist_vel = 0

        weigths[0] = 8.7
        weigths[1] = 0.01
        weigths[2] = 0.5
        weigths[3] = 0.1
        weigths[4] = 0.05
        weigths[5] = 1
        cost = weigths[0] * (cost_lanelet / len(path)) + \
               weigths[1] * abs(orientationToGoalDiff) + \
               weigths[3] * cost_time + \
               weigths[2] * distLastState + \
               weigths[4] * (100 - pathLength) + \
               weigths[5] * dist_vel

        if cost < 0: cost = 0
        return cost * factor

    def calc_heuristic_cost(self, path: List[State], curPos: State):
        """
        Returns heuristic cost for the path.
        You should define your own cost function here.

        :param path: The path for which you want to calculate the cost
        :param curPos: Last state of your current path. Every state has the following variables: position, velocity, orientation, time_step
        """

        # cost_lanelet, final_lanelet_id, start_lanelet_id = self.calc_heuristic_lanelet(path)
        # if cost_lanelet is None or final_lanelet_id[0] is None:
        #     return None
        # self.calc_path_efficiency(path)
        # self.num_obstacles_in_lanelet_at_time_step(path[-1].time_step, final_lanelet_id[0])
        # self.is_goal_in_lane(final_lanelet_id[0])
        # if self.lanelet_cost[final_lanelet_id[0]] == -1:
        #     return None
        # factor = 1
        # if self.lanelet_cost[final_lanelet_id[0]] > self.lanelet_cost[start_lanelet_id[0]]:
        #     return None
        # if self.lanelet_cost[final_lanelet_id[0]] < self.lanelet_cost[start_lanelet_id[0]]:
        #     factor = factor * 0.1
        # laneletOrientationAtPosition = self.calc_lanelet_orientation(final_lanelet_id[0], path[-1].position)

        # if final_lanelet_id[0] in self.goalLanelet_ids:
        #     factor = factor * 0.07
        # pathLength = calc_travelled_distance(path)
        # cost_time = self.calc_time_cost(path)
        # weigths = np.zeros(6)
        # if distLastState < 0.5:
        #     factor = factor * 0.00001
        # # elif math.pi - abs(abs(laneletOrientationAtPosition - path[-1].orientation)


        # weigths[0] = 8.7
        # weigths[1] = 0.01
        # weigths[2] = 0.5
        # weigths[3] = 0.1
        # weigths[4] = 0.05
        # weigths[5] = 1
        # cost = weigths[0] * (cost_lanelet / len(path)) + \
        #        weigths[1] * abs(orientationToGoalDiff) + \
        #        weigths[3] * cost_time + \
        #        weigths[2] * distLastState + \
        #        weigths[4] * (100 - pathLength) + \
        #        weigths[5] * dist_vel



        # ===================================================================================================
        # Custom implementation:
        # ===================================================================================================

        epsilon = 0.000001
        factor = 1

        try:

            # Scale weighting using normalized ratio of current distance to goal vs initial distance to goal.
            # `scale_factor` goes to 1 when near the goal.

            if hasattr(self.planningProblem.goal.state_list[0], 'position'):
                distance_from_curr  = self.calc_heuristic_distance(path[-1])
                # distance_from_prev  = self.calc_heuristic_distance(path[-2])
                distance_normalized = distance_from_curr / (self.initial_distance + epsilon)
                # distance_normalized_prev = distance_from_prev / (self.initial_distance + epsilon)
                scale_factor = 1 - distance_normalized + 0.05
            else:
                distance_from_curr  = np.inf
                distance_normalized = 1 
                scale_factor = -1
            if scale_factor > 1:
                scale_factor = 1
            elif scale_factor < 0:
                scale_factor = epsilon
            else:
                scale_factor = scale_factor ** 1.5



            # Calculate cost due to distance.
            weight_distance = 2
            distance_cost = weight_distance * distance_normalized



            # Calculate cost due to orientation.
            weight_orientation = 2
            if hasattr(self.planningProblem.goal.state_list[0], 'orientation'):

                goal_orientation_start_x = math.cos(self.planningProblem.goal.state_list[0].orientation.start)
                goal_orientation_start_y = math.sin(self.planningProblem.goal.state_list[0].orientation.start)
                goal_orientation_end_x   = math.cos(self.planningProblem.goal.state_list[0].orientation.end)
                goal_orientation_end_y   = math.sin(self.planningProblem.goal.state_list[0].orientation.end)
                goal_orientation_x = (goal_orientation_start_x + goal_orientation_end_x) / 2
                goal_orientation_y = (goal_orientation_start_y + goal_orientation_end_y) / 2
                goal_orientation = math.atan2(goal_orientation_y, goal_orientation_x)
                
                car_orientation_x = math.cos(path[-1].orientation)
                car_orientation_y = math.sin(path[-1].orientation)
                car_orientation = math.atan2(car_orientation_y, car_orientation_x)
                
                diff_orientation = car_orientation - goal_orientation
                if diff_orientation > np.pi:
                    diff_orientation -= 2 * np.pi
                elif diff_orientation <= -np.pi:
                    diff_orientation += 2 * np.pi

            else:

                goal_orientation = 0
                car_orientation  = 0
                diff_orientation = 0

            orientation_cost = weight_orientation * abs(diff_orientation / np.pi) * scale_factor



            # Calculate cost due to velocity.
            weight_velocity = 2 # 1
            v_mean_goal = None
            if hasattr(self.planningProblem.goal.state_list[0], 'velocity'):
                v_mean_goal = (self.planningProblem.goal.state_list[0].velocity.start + 
                            self.planningProblem.goal.state_list[0].velocity.end) / 2
                diff_velocity = abs(path[-1].velocity - v_mean_goal)
            else:
                diff_velocity = 0
            # When far from the goal, we want velocity to be as fast as possible.
            velocity_cost = weight_velocity * (1 - path[-1].velocity / 20) * (1 - scale_factor) 
            if velocity_cost < 0:
                velocity_cost = 0
            # When close to the goal, we want velocity to be as close as possible.
            velocity_diff_cost = 1.2 * weight_velocity * abs(diff_velocity) * scale_factor

            velocity_cost = 0 # i.e. disable



            # Calculate cost due to acceleration.
            weight_acceleration = 5 # 0.2
            accel = []
            accel_abs = []
            for k in range(1, len(path)):
                accel.append(path[k].velocity - path[k-1].velocity)
                accel_abs.append(abs(path[k].velocity - path[k-1].velocity))
            accel_cost = weight_acceleration * sum(accel_abs) / (len(path) - 1) * (1 - scale_factor)



            # Calculate cost due to acceleration jerk.
            weight_jerk = 0.2
            jerk = []
            jerk_abs = []
            for k in range(1, len(accel)):
                jerk.append(accel[k] - accel[k-1])
                jerk_abs.append(abs(accel[k] - accel[k-1]))
            jerk_cost = weight_jerk * sum(jerk_abs) / (len(accel) - 1) * (1 - scale_factor)



            # Calculate cost from path inefficiency.
            weight_efficiency = 0.25
            efficiency = self.calc_path_efficiency(path) / 0.1 # `/ 0.1` because each timestep is 0.1s.
            efficiency_cost = weight_efficiency * (1 - efficiency / 40) * (1 - scale_factor)
            if efficiency_cost < 0:
                efficiency_cost = 0



            # Calculate cost from deviating from lanelet center.
            weight_lanelet = 0.2 # 0.15
            lanelet_distance, final_lanelet_id, start_lanelet_id = self.calc_heuristic_lanelet(path)
            if lanelet_distance is None or final_lanelet_id[0] is None:
                return np.inf
            num_obstacles = self.num_obstacles_in_lanelet_at_time_step(path[-1].time_step, final_lanelet_id[0])
            is_goal_lane = self.is_goal_in_lane(final_lanelet_id[0])
            lanetlet_cost = weight_lanelet * lanelet_distance * (scale_factor + 1) / 2 / (num_obstacles + 1) ** 2

            # Favor high speed.
            # if num_obstacles == 0 and is_goal_lane and scale_factor < 0.7:
            #     velocity_cost /= 8 * (1 - scale_factor)
            #     efficiency_cost /= 4 * (1 - scale_factor)
            #     lanetlet_cost /= 16 * (1 - scale_factor)
            #     # velocity_diff_cost /= 4 * (1 - scale_factor)

            # if self.lanelet_cost[final_lanelet_id[0]] == -1:
            #     return None
            factor = 1
            # # if self.lanelet_cost[final_lanelet_id[0]] > self.lanelet_cost[start_lanelet_id[0]]:
            # #     return None
            # # if self.lanelet_cost[final_lanelet_id[0]] < self.lanelet_cost[start_lanelet_id[0]]:
            # #     factor = factor * 0.1
            # laneletOrientationAtPosition = self.calc_lanelet_orientation(final_lanelet_id[0], path[-1].position)

            # if final_lanelet_id[0] in self.goalLanelet_ids:
            #     factor = factor * 0.07
            # pathLength = calc_travelled_distance(path)
            # cost_time = self.calc_time_cost(path)
            # weigths = np.zeros(6)
            # if distLastState < 0.5:
            #     factor = factor * 0.00001
            # elif math.pi - abs(abs(laneletOrientationAtPosition - path[-1].orientation)



            # Calculate cost from obstacle.
            weight_obstacle = 0.1
            weight_obstacle_periphery = 0.02
            obstacle_count = 0
            obstacle_cost = 0
            obstacle_cost_periphery = 0

            # lanelet_distance, final_lanelet_id, start_lanelet_id = self.calc_heuristic_lanelet(path)
            if final_lanelet_id is not None and len(final_lanelet_id) > 0:

                dist_to_obstacle = self.dist_to_closest_obstacle(final_lanelet_id[0], path[-1].position, path[-1].time_step)
                if not np.isinf(dist_to_obstacle):
                    obstacle_count += 1
                obstacle_cost = weight_obstacle * (1 / dist_to_obstacle) * (1 - scale_factor)

                if start_lanelet_id is not None and len(start_lanelet_id) > 0:
                    if final_lanelet_id[0] != start_lanelet_id[0]:
                        dist_to_obstacle_s = self.dist_to_closest_obstacle(start_lanelet_id[0], path[-1].position, path[-1].time_step)
                        obstacle_cost -= weight_obstacle * (1 / dist_to_obstacle_s) * (1 - scale_factor)

                if obstacle_cost < 0:
                    obstacle_cost = 0

                if len(final_lanelet_id) > 1:
                    for k in range(1, len(final_lanelet_id)):
                        dist_to_obstacle_periphery = self.dist_to_closest_obstacle(final_lanelet_id[k], path[-1].position, path[-1].time_step)
                        obstacle_cost_periphery += weight_obstacle_periphery * (1 / dist_to_obstacle_periphery) * (1 - scale_factor)

                if len(start_lanelet_id) > 1:
                    for k in range(1, len(start_lanelet_id)):
                        dist_to_obstacle_periphery_s = self.dist_to_closest_obstacle(start_lanelet_id[k], path[-1].position, path[-1].time_step)
                        obstacle_cost_periphery -= weight_obstacle_periphery * (1 / dist_to_obstacle_periphery_s) * (1 - scale_factor)

                if obstacle_cost_periphery < 0:
                    obstacle_cost_periphery = 0

            else:
                
                obstacle_cost = 0
                obstacle_cost_periphery = 0

            # Reduce penalties of other things when near obstacle
            # if not np.isinf(dist_to_obstacle) and dist_to_obstacle < 20 and scale_factor < 0.7:
            #     orientation_cost   /= 20 / dist_to_obstacle
            #     velocity_diff_cost /= 20 / dist_to_obstacle
            #     lanetlet_cost      /= 20 / dist_to_obstacle



            # Calculate cost from passenger comfort.
            weight_curvature = 0.2
            curvature = 0
            try:
                curvature = curvature_of_polyline(np.array([path_i.position for path_i in path]))
                if np.isnan(curvature):
                    curvature = 10
            except:
                pass
            curvature_cost = weight_curvature * curvature * (1 - scale_factor) / (num_obstacles + 1) ** 2



            # Calculate cost from time step.
            weight_time = 0.2
            time_step_normalized = path[-1].time_step / self.planningProblem.goal.state_list[0].time_step.end
            if path[-1].time_step > self.planningProblem.goal.state_list[0].time_step.end:
                return np.inf
            if time_step_normalized >= 0.7:
                if path[-1].time_step < self.planningProblem.goal.state_list[0].time_step.start:
                    weight_time *= 1.1
                    if time_step_normalized >= 0.8:
                        weight_time *= 1.1
                    if time_step_normalized >= 0.9:
                        weight_time *= 1.1
            time_step_cost = weight_time * (1 - time_step_normalized) * scale_factor
            


            cost = 0
            cost += distance_cost
            cost += orientation_cost
            cost += velocity_cost
            cost += velocity_diff_cost
            cost += accel_cost
            cost += jerk_cost
            cost += efficiency_cost
            cost += lanetlet_cost
            cost += curvature_cost
            cost += obstacle_cost
            cost += obstacle_cost_periphery
            cost += time_step_cost



            logging.debug(' ')
            logging.debug('======================================')
            logging.debug('Time step: ' + str(path[-1].time_step))
            logging.debug('Time limit: ' + str(self.planningProblem.goal.state_list[0].time_step.end))
            logging.debug('Scale factor: ' + str(scale_factor))
            logging.debug('Goal in lane?: ' + str(is_goal_lane))
            logging.debug(' ')
            # logging.debug('Path length: ' + str(len(path))) # Is constant length of 5.
            # logging.debug('Dist from start: ' + str(self.initial_distance))
            # logging.debug('Dist from curr : ' + str(distance_from_curr))
            # logging.debug('Dist normalized: ' + str(distance_normalized))
            logging.debug('Dist cost         : {:.2f}'.format(distance_cost))
            # logging.debug(' ')
            # logging.debug('Car angle : ' + str(path[-1].orientation / np.pi * 180) + ' degrees')
            # logging.debug('Goal angle: ' + str(goal_orientation / np.pi * 180) + ' degrees')
            # logging.debug('Angle diff: ' + str(diff_orientation / np.pi * 180) + ' degrees')
            logging.debug('Orientation cost  : {:.2f}'.format(orientation_cost))
            # logging.debug(' ')
            logging.debug('Car velocity      : {:.2f}'.format(path[-1].velocity))
            # logging.debug('Velocity goal     : {:.2f}'.format(v_mean_goal if v_mean_goal else -1))
            # logging.debug('Velocity diff     : {:.2f}'.format(diff_velocity))
            # logging.debug('Velocity diff cost: {:.2f}'.format(velocity_diff_cost))
            logging.debug('Slowness cost     : {:.2f}'.format(velocity_cost))
            logging.debug('Accel cost        : {:.2f}'.format(accel_cost))
            logging.debug('Jerk cost         : {:.2f}'.format(jerk_cost))
            # logging.debug(' ')
            # logging.debug('Efficiency: ' + str(efficiency))
            logging.debug('Efficiency cost   : {:.2f}'.format(efficiency_cost))
            # logging.debug(' ')
            # logging.debug('Lanelet ID: ' + str(final_lanelet_id))
            # logging.debug('Lanelet distance: ' + str(lanelet_distance))
            logging.debug('Lanelet cost      : {:.2f}'.format(lanetlet_cost))
            # logging.debug(' ')
            # logging.debug('Curvature: ' + str(curvature))
            logging.debug('Curvature cost    : {:.2f}'.format(curvature_cost))
            # logging.debug(' ')
            # logging.debug('Distance to obstacle: ' + str(dist_to_obstacle))
            # logging.debug('Obstacle count: ' + str(obstacle_count))
            # logging.debug('Obstacle num: ' + str(num_obstacles))
            logging.debug('Obstacle cost     : {:.2f}'.format(obstacle_cost))
            logging.debug('Obstacle cost peri: {:.2f}'.format(obstacle_cost_periphery))
            # logging.debug(' ')
            logging.debug('Timestep cost     : {:.2f}'.format(time_step_cost))
            logging.debug(' ')
            logging.debug('Overall cost      : {:.2f}'.format(cost))
            logging.debug('======================================')
            logging.debug(' ')

        except Exception as e:

            cost = np.inf
            logging.error('Error:')
            logging.error(repr(e))



        if cost < 0: cost = 0
        return cost * factor

    def search_alg(self, list_successors_start_state: List[MotionPrimitive], max_tree_depth: int, dict_status = None):  # -> Tuple(List[State], List[MotionPrimitive]):        return None
        """
        Returns the path and the used primitives

        :param list_successors_start_state: All connectable primitives (successors) of the start state
        :param max_tree_depth: Maximum number of concatenated primitives
        :param dict_status: Used for the visualization in the jupyter notebook
        """

        # Here the greedy best first search algorithm is demonstrated. #

        # Every state has the following attributes: .position, .velocity, .orientation, .time_step
        # For more information please refer to the commonroad-io tutorial

        # initial values to be fed into the priority queue
        # the queue is a heap that always pops up the item with the lowest cost

        total_cost = 0
        path = [self.initial_state]
        tree_depth = 0
        path_appended = [self.initial_state]
        list_primitives = []
        priority = 0

        # if the goal state is reached by the current path
        if self.reached_goal(path):
            # return a pruned path and its corresponding primitives
            return self.remove_states_behind_goal(path), list_primitives

        # inputs for PriorityQueue: (total_cost,                        : total cost of the path so far
        #                            path,                              : current path
        #                            list_successors_primitive_last,    : list of successors of the last primitive in the path
        #                            tree_depth,                        : tree depth
        #                            path_appended,                     : the path that is appened in this iteration
        #                            list_primitives),                  : list of primitives of the current path
        #                            priority                           : cost of the path / value to be sorted in the heap
        list_primitives_copied = copy.copy(list_primitives)
        # add initial state into priority queue
        self.priority_queue.put((total_cost,                          # : initial cost is 0 
                                path,                                 # : initial path only contains the initial state
                                list_successors_start_state,          # : the successors of the initial state
                                tree_depth,                           # : initial depth is 0
                                path_appended,                        # : only initial state is added to the path
                                list_primitives_copied),              # : no primitives added to the list yet
                                priority)                             # : initial priority is 0


        # while the queue is not empty
        while not self.priority_queue.empty():
            # get the item with lowest cost from the queue
            (total_cost, path, list_successors_primitive_last, tree_depth, _, list_primitives) = self.priority_queue.get()
            list_primitives_copied = copy.copy(list_primitives)

            status = {'cost_current': total_cost, 
                      'tree_depth'  : tree_depth, 
                      'path_current': path}

            if dict_status is not None: dict_status.value = status

            # check whether the tree depth exceeds the maximum allowed number
            if tree_depth > max_tree_depth:
                print("reached max tree depth")
                return None
            else:
                tree_depth += 1

            # skip if the time step of the last state of path exceeds that of the goal state
            if path[-1].time_step > self.desired_time.end:
                continue
  
            # otherwise, expand all successors of the primitive
            for i in range(len(list_successors_primitive_last)):

                # get a primitive from the list of successors
                primitive_next = list_successors_primitive_last[i]

                # add new primitves to the current path
                # we attach the primitive to current path by translating the pritimive, and 
                # obtain a path containing the translated states
                # Note: all motion primitives start at position [0, 0]^T
                path_primitive_translated = self.translate_primitive_to_current_state(primitive_next, path)

                # skip if the path is None
                if path_primitive_translated is None:
                    continue

                # skip if the time step of the last state of path exceeds that of goal state
                if path_primitive_translated[0].time_step > self.desired_time.end:
                    continue

                # skip this translated path if it is not collision free
                if not self.check_collision_free(path_primitive_translated):
                    continue

                # otherwise, calculated the cost associated with this path
                # lower cost implies higher priority when expanding
                # in Greedy BFS, the final cost (f) = heuristic cost (h)
                cost_final = self.calc_heuristic_cost(path_primitive_translated, path[-1])

                # skip if the cost is not valid
                if cost_final is None:  
                    continue

                # if the new primitive passes all conditions listed above, add it into current path

                # update total cost
                total_cost = total_cost + cost_final

                # add primitive into the list of primitives
                list_primitives_copied.append(primitive_next)

                # if the goal state is reached by the new path_primitive_translated
                if self.reached_goal(path_primitive_translated):
                    # concacenate its predecessor and return result
                    path_new = path + path_primitive_translated

                    # return a pruned path and its corresponding primitives
                    return self.remove_states_behind_goal(path_new), list_primitives_copied

                # get new path by concatenating current path and the path translated from the primitve
                path_new = path + path_primitive_translated

                # push the new path with corresponding values into the priority queue
                
                self.priority_queue.put(
                    (total_cost, 
                     path_new, 
                     self.get_successor_primitives(primitive_next), 
                     tree_depth, 
                     path_primitive_translated, 
                     list_primitives_copied),
                     cost_final)
        # return None is no feasible path is found
        return None


class PriorityQueue:
    def __init__(self):
        self.elements = []
        self.count = 0

    def empty(self):
        """
        Test whether the queue is empty. Returns true if the queue is empty.
        """
        return len(self.elements) == 0

    def put(self, item, priority):
        """
        Put an item into the queue and count the number of elements in the queue. The number is saved in self.count.

        :param item: item to be put into the queue
        :param priority: the priority used to sort the queue. It's often the value of some cost function.
        """
        heapq.heappush(self.elements, (np.round(priority * 10000000), self.count, item))
        self.count = self.count + 1

    def get(self):
        """
        Pop the smallest item off the heap (Priority queue) if the queue is not empty.
        """
        if self.empty():
            return None
        return heapq.heappop(self.elements)[2]


def findClosestVertex(centerVertices: np.ndarray, pos: np.ndarray) -> int:
    """
    Return the index of the closest center vertice to the given position.
    
    :param centerVertices: The vertices of the center line of the Lanelet described as a polyline [[x0,x1,...,xn],[y0,y1,...,yn]]
    :param pos: The position, to which the closest vertex is calculated.
    
    """

    distances = []
    for vertex in centerVertices:
        distances.append(distance(vertex, pos, 0))
    return distances.index(min(distances))


def calcAngleOfPosition(centerVertices: np.ndarray, pos: np.ndarray) -> float:
    """
    Returns the angle (in world coordinate, radian) of the line defined by 2 nearest lanelet center vertices to the given position.

    :param centerVertices: Lanelet center vertices, whose distance to the given position is considered.
    :param pos: The position to be considered.
    """
    index_closestVert = findClosestVertex(centerVertices, pos)
    if index_closestVert + 1 >= centerVertices.size / 2.0:
        index_closestVert = index_closestVert - 1
    return math.atan2(centerVertices[index_closestVert + 1][1] - centerVertices[index_closestVert][1],
                      centerVertices[index_closestVert + 1][0] - centerVertices[index_closestVert][0])


def distToClosestPointOnLine(vertexA: np.ndarray, vertexB: np.ndarray, pos: np.ndarray):
    """
    Returns the distance of the given position to a line segment (e.g. the nearest lanelet center line segment to the given position).

    :param pos:
    :param vertexA: The first point of the line segment.
    :param vertexB: The second point of the line segment.

    """
    magAB2 = (vertexB[0] - vertexA[0]) ** 2 + (vertexB[1] - vertexA[1]) ** 2
    ABdotAP = (pos[0] - vertexA[0]) * (vertexB[0] - vertexA[0]) + (pos[1] - vertexA[1]) * (vertexB[1] - vertexA[1])
    s = ABdotAP / magAB2
    if s < 0:
        return distance(vertexA, pos, 0)
    elif s > 1:
        return distance(vertexB, pos, 0)
    else:
        newVertex = np.empty(2)
        newVertex[0] = vertexA[0] + (vertexB[0] - vertexA[0]) * s
        newVertex[1] = vertexA[1] + (vertexB[1] - vertexA[1]) * s
        return distance(newVertex, pos, 0)


def findDistanceToNearestPoint(centerVertices: np.ndarray, pos: np.ndarray):
    """
    Returns the closest euclidean distance to a polyline (e.g. defined by lanelet center vertices) according to the given position.

    :param centerVertices: The poyline, between which and the given position the distance is calculated.
    :param pos: The position to be considered.
    """
    distances = []
    for vertex in centerVertices:
        distances.append(distance(vertex, pos, 0))
    index_closestVert = distances.index(min(distances))

    if (index_closestVert + 1) < len(centerVertices):
        dist1 = distToClosestPointOnLine(centerVertices[index_closestVert], centerVertices[index_closestVert + 1], pos)
    else:
        dist1 = distance(centerVertices[index_closestVert], pos, 0)

    if index_closestVert > 0:
        dist2 = distToClosestPointOnLine(centerVertices[index_closestVert - 1], centerVertices[index_closestVert], pos)
    else:
        dist2 = distance(centerVertices[index_closestVert], pos, 0)

    return min(dist1, dist2)


def calc_travelled_distance(path: List[State]):
    """
    Returns the travelled distance of the given path.

    :param: The path, whose travelled euclidean distance is calculated.

    """
    dist = 0
    for i in range(len(path) - 1):
        dist = dist + distance(path[i].position, path[i + 1].position)
    return dist


def euclideanDistance(pos1: np.ndarray, pos2: np.ndarray):
    """
    Returns the euclidean distance between 2 points.

    :param pos1: the first point 
    :param pos2: the second point
    """
    return np.sqrt((pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1]))


def manhattanDistance(pos1: np.ndarray, pos2: np.ndarray):
    """
    Returns the manhattan distance between 2 points.

    :param pos1: the first point 
    :param pos2: the second point
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def chebyshevDistance(pos1: np.ndarray, pos2: np.ndarray):
    """
    Returns the chebyshev distance between 2 points.

    :param pos1: the first point 
    :param pos2: the second point
    """
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))


def sumOfSquaredDifference(pos1: np.ndarray, pos2: np.ndarray):
    """
    Returns the squared euclidean distance between 2 points.

    :param pos1: the first point 
    :param pos2: the second point
    """
    return (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2


def meanAbsoluteError(pos1: np.ndarray, pos2: np.ndarray):
    """
    Returns a half of the manhattan distance between 2 points.

    :param pos1: the first point 
    :param pos2: the second point
    """
    return 0.5 * (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))


def meanSquaredError(pos1: np.ndarray, pos2: np.ndarray):
    """
    Returns the mean of squared difference between 2 points.

    :param pos1: the first point 
    :param pos2: the second point
    """
    return 0.5 * ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def canberraDistance(pos1: np.ndarray, pos2: np.ndarray):
    """
    Returns the canberra distance between 2 points.

    :param pos1: the first point 
    :param pos2: the second point
    """
    return abs(pos1[0] - pos2[0]) / (abs(pos1[0]) + abs(pos2[0])) + abs(pos1[1] - pos2[1]) / (
                abs(pos1[1]) + abs(pos2[1]))


def cosineDistance(pos1: np.ndarray, pos2: np.ndarray):
    """
    Returns the cosine distance between 2 points.

    :param pos1: the first point 
    :param pos2: the second point
    """
    return 1 - (pos1[0] * pos2[0] + pos1[1] * pos2[1]) / (
                np.sqrt(pos1[0] ** 2 + pos2[0] ** 2) * np.sqrt(pos1[1] ** 2 + pos2[1] ** 2))


def distance(pos1: np.ndarray, pos2: np.ndarray, distance_type=0):
    """
    Returns the distance between 2 points, the type is specified by 'type'.

    :param pos1: the first point
    :param pos2: the second point
    :param distance_type: specifies which kind of distance is used:
        1: manhattanDistance,
        2: chebyshevDistance,
        3: sumOfSquaredDifference,
        4: meanAbsoluteError,
        5: meanSquaredError,
        6: canberraDistance,
        7: cosineDistance.
    """
    if distance_type == 0:
        return euclideanDistance(pos1, pos2)
    elif distance_type == 1:
        return manhattanDistance(pos1, pos2)
    elif distance_type == 2:
        return chebyshevDistance(pos1, pos2)
    elif distance_type == 3:
        return sumOfSquaredDifference(pos1, pos2)
    elif distance_type == 4:
        return meanAbsoluteError(pos1, pos2)
    elif distance_type == 5:
        return meanSquaredError(pos1, pos2)
    elif distance_type == 6:
        return canberraDistance(pos1, pos2)
    elif distance_type == 7:
        return cosineDistance(pos1, pos2)
    return


def curvature_of_polyline(polyline: np.ndarray):
    """
    Returns the curvature of the given polyline.

    :param polyline: The polyline to be calculated.
    """
    dx_dt = np.gradient(polyline[:, 0])
    dy_dt = np.gradient(polyline[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvatureArray = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
    curvature = 0
    for elem in curvatureArray:
        curvature = curvature + abs(elem)
    return curvature


def orientation_diff(orientation_1: float, orientation_2: float) -> float:
    """
    Returns the orientation difference between 2 orientations in radians.

    :param orientation_1: the first orientation.
    :param orientation_2: the second orientation.
    """
    return math.pi - abs(abs(orientation_1 - orientation_2) - math.pi)


def length_of_polyline(polyline: np.ndarray):
    """
    Returns the length of the polyline.

    :param polyline: The polyline, whose length is calculated.
    """

    dist = 0
    for i in range(0, len(polyline) - 1):
        dist = dist + distance(polyline[i], polyline[i + 1])
    return dist
