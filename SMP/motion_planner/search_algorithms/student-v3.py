# from matplotlib.patches import Rectangle
from commonroad.geometry.shape import Rectangle, Polygon, ShapeGroup, Circle
from SMP.motion_planner.node import Node, PriorityNode

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import GreedyBestFirstSearch, AStarSearch
from commonroad_dc.costs.evaluation import CostFunctionEvaluator
from commonroad.common.solution import Solution, CostFunction, VehicleType
from commonroad.scenario.trajectory import State
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from commonroad_route_planner.route_planner import RoutePlanner
from typing import List

import numpy as np
from IPython import display

from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.parameters_vehicle3 import parameters_vehicle3

class StudentMotionPlanner(AStarSearch):
    """
    Motion planner implementation by students.
    Note that you may inherit from any given motion planner as you wish, or come up with your own planner.
    Here as an example, the planner is inherited from the GreedyBestFirstSearch planner.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)

        # print('StudentMotionPlanner V0 - Revived 2019 Techniques in AI assignment.')
        # print('StudentMotionPlanner V1 - Improved weights and obstacle calculations.')
        # print('StudentMotionPlanner V2 - Squared spacing of motion primitives.')
        # print('StudentMotionPlanner V3 - Observe lane direction. Update heuristic cost weights.')
        # print('StudentMotionPlanner V3.1 - Use old primitive.')
        print('StudentMotionPlanner V3.2 - Add cost to vehicle orientation to lane.')

        self.iteration = 0
        self.print_every = 100
        self.goal_is_reached = False

        self.tsi = TrafficSigInterpreter(scenario.scenario_id.country_id, scenario.lanelet_network)

        print(f'Number of goal states: {len(self.planningProblem.goal.state_list)}')
        if hasattr(self.planningProblem.goal.state_list[0], 'position'): # and hasattr(self.planningProblem.goal.state_list[0].position, 'vertices'):
            print(f'Standard mode. Goal(s):')
            self.survival_mode = False
            for i, state in enumerate(self.planningProblem.goal.state_list):
                print(f'Goal state {i}:')
                print(state)
            self.initial_distance = self.calc_euclidean_distance(Node([[planningProblem.initial_state]], [], 0))
            print(f'Initial Euclidean distance: {self.initial_distance}')
        else:
            print(f'Survival mode. Only time goal:')
            self.survival_mode = True
            for i, state in enumerate(self.planningProblem.goal.state_list):
                print(f'Goal state {i}:')
                print(state)

        if self.automaton.type_vehicle == VehicleType.FORD_ESCORT:
            self.vehicle = parameters_vehicle1()
        elif self.automaton.type_vehicle == VehicleType.BMW_320i:
            self.vehicle = parameters_vehicle2()
        elif self.automaton.type_vehicle == VehicleType.VW_VANAGON:
            self.vehicle = parameters_vehicle3()

        # self.ce = CostFunctionEvaluator(CostFunction.TR1, VehicleType.FORD_ESCORT)
        # self.ce = CostFunctionEvaluator(CostFunction.SM1, VehicleType.BMW_320i)
        # Infer cost function from vehicle type, should be overwritten.
        if self.automaton.type_vehicle == VehicleType.FORD_ESCORT:
            self.init_cost_function(CostFunction.TR1)
        elif self.automaton.type_vehicle == VehicleType.BMW_320i:
            self.init_cost_function(CostFunction.SM1)

        # Instantiate a route planner with the scenario and the planning problem.
        route_planner = RoutePlanner(scenario, planningProblem, backend=RoutePlanner.Backend.PRIORITY_QUEUE)

        # Plan routes, and save the routes in a route candidate holder.
        candidate_holder = route_planner.plan_routes()

        # Option 1: retrieve all routes.
        # list_routes, num_route_candidates = candidate_holder.retrieve_all_routes()
        # print(f"Number of route candidates: {num_route_candidates}")
        # self.route = candidate_holder.retrieve_first_route() # Equivalent to: route = list_routes[0]

        # Option 2: retrieve the best route by orientation metric.
        self.route = candidate_holder.retrieve_best_route_by_orientation()

        # Identify lanelets in the correct direction of travel along the planned route.
        # Source: https://commonroad.in.tum.de/forum/t/identify-landlet-direction-of-travel-from-lanelet-id-or-position/885/2
        set_ids_lanelets = set(self.route.list_ids_lanelets)
        # obtain lanelets in the same direction as the route
        terminate = False
        while not terminate:
            num_ids_lanelets = len(set_ids_lanelets)
            for id_lanelet in list(set_ids_lanelets):
                lanelet = scenario.lanelet_network.find_lanelet_by_id(id_lanelet)

                # if left lanelet is in the same direction
                if lanelet.adj_left and lanelet.adj_left_same_direction:
                    set_ids_lanelets.add(lanelet.adj_left)

                # if right lanelet is in the same direction
                if lanelet.adj_right and lanelet.adj_right_same_direction:
                    set_ids_lanelets.add(lanelet.adj_right)

            terminate = (num_ids_lanelets == len(set_ids_lanelets))

        self.set_ids_lanelets_same_direction = set_ids_lanelets.copy()


    def init_cost_function(self, cost_func: CostFunction) -> None:
        """Must call this after __init__()."""
        self.ce = CostFunctionEvaluator(cost_func, self.automaton.type_vehicle)
        print(f'Created cost function {cost_func.name} for {self.automaton.type_vehicle.name}')



    def evaluation_function(self, node_current: PriorityNode) -> float:
        """
        Evaluation function of A* is f(n) = g(n) + h(n)
        """
        ########################################################################
        # TODO: Implement your own evaluation function here.                   #
        ########################################################################

        # print(' ')
        # print()
        # print('evaluation_function()')

        self.iteration += 1
        if self.iteration % (self.print_every * 10) == 0:
            display.clear_output(wait=True)
        if self.iteration % self.print_every == 0:
            print(f'A* | {self.scenario.scenario_id} | Iteration {self.iteration} | Path length: {len(node_current.list_paths[-1])}')

        # print(f'Iteration {self.iteration}')
        # for state in node_current.list_paths[-1]:
        #     print(state.attributes)
        # print()

        if not self.is_collision_free(node_current.list_paths[-1]):
            # print('Collision! Path rejected!')
            node_current.priority = np.inf
            return np.inf

        self.goal_is_reached = self.reached_goal(node_current.list_paths[-1])
        # if self.goal_is_reached:
        #     print(f'{self.scenario.scenario_id} | Goal is reached!!!')
            # node_current.list_paths = self.remove_states_behind_goal(node_current.list_paths)

        trajectory = Trajectory(node_current.list_paths[-1][0].time_step, node_current.list_paths[-1])
        if len(node_current.list_paths[-1]) > 1:
            try:
                cost = self.ce.evaluate_pp_solution(self.scenario, self.planningProblem, trajectory)
                g_cost = cost.total_costs
            except Exception as e:
                print(f'Error evaluating cost function for {self.scenario.scenario_id}.')
                print(repr(e))
                g_cost = 100
            # print('Partial cost:')
            # print(cost)
            # g_cost = cost.total_costs
            h_cost = self.heuristic_function(node_current=node_current)
            f_cost = g_cost + h_cost
            node_current.priority = g_cost
            if self.iteration % self.print_every == 0:
                print(f'Evaluation cost: {g_cost}')
                print(f'Heuristic cost : {h_cost}')
                print(f'Total cost     : {f_cost}')
                print()
            return f_cost

        else:
            # calculate g(n)
            node_current.priority += (len(node_current.list_paths[-1]) - 1) * self.scenario.dt
            # f(n) = g(n) + h(n)
            g_cost = node_current.priority
            h_cost = self.heuristic_function(node_current=node_current)
            f_cost = g_cost + h_cost
            if self.iteration % self.print_every == 0:
                print(f'Evaluation cost: {g_cost}')
                print(f'Heuristic cost : {h_cost}')
                print(f'Total cost     : {f_cost}')
                print()
            return f_cost



    def heuristic_function(self, node_current: PriorityNode) -> float:
        ########################################################################
        # TODO: Implement your own heuristic cost calculation here.            #
        # Hint:                                                                #
        #   Use the State of the current node and the information from the     #
        #   planning problem, as well as from the scenario.                    #
        #   Some helper functions for your convenience can be found in         #
        #   ./search_algorithms/base_class.py                                  #
        ########################################################################


        # ===================================================================================================
        # Custom implementation:
        # ===================================================================================================

        epsilon = 0.000001
        factor = 1
        path = node_current.list_paths[-1]
        partial_costs = {} # Nested dict with weights.

        try:

            if not self.is_collision_free(path):
                # print('Collision!')
                return np.inf

            # Scale weighting using normalized ratio of current distance to goal vs initial distance to goal.
            # `scale_factor` goes to 1 when near the goal.
            if hasattr(self.planningProblem.goal.state_list[0], 'position'): # and hasattr(self.planningProblem.goal.state_list[0].position, 'vertices'):
                # distance_from_curr  = self.calc_heuristic_distance(path[-1])
                distance_from_curr  = self.calc_euclidean_distance(node_current)
                # distance_from_prev  = self.calc_heuristic_distance(path[-2])
                distance_normalized = distance_from_curr / (self.initial_distance + epsilon)
                # print(f'Normalized distance from goal: {distance_normalized}')
                # distance_normalized_prev = distance_from_prev / (self.initial_distance + epsilon)
                scale_factor = 1 - distance_normalized + 0.05
            else:
                distance_from_curr  = np.inf
                distance_normalized = 1 
                scale_factor = -1
            if scale_factor > 1:
                scale_factor = 1
            elif scale_factor < 0:
                scale_factor = 1
            else:
                scale_factor = scale_factor ** 1.5

            # Calculate cost due to distance.
            # weight_distance = 2
            # distance_cost = weight_distance * distance_normalized
            partial_costs['distance'] = {
                'cost': distance_normalized,
                'weight': 10,
            }

            # calc_travelled_distance
            # print(mapping)
            # get_obstacles



            # Identify distance to obstacles in all lanes currently occupied by the ego vehicle, over time.
            margin = 1.01
            current_obstacle_distance_cost = 0
            future_obstacle_distance_cost = 0
            vertex_lanelet_ids_list = []
            for snapshot in path:

                ego = Rectangle(length=self.vehicle.l*margin, width=self.vehicle.w*margin, center=snapshot.position, orientation=snapshot.orientation)

                vertex_lanelet_ids_nested = self.lanelet_network.find_lanelet_by_position([v for v in ego.vertices])
                vertex_lanelet_ids = []
                # vertex_lane_correct = []
                for ids in vertex_lanelet_ids_nested: # Flatten it.
                    vertex_lanelet_ids += ids
                    # A point can be located on multiple lanelets. Some might be of a wrong travel direction, e.g. in an intersection. That is fine as long as one of the lanelets are correct.
                    correct_lane = [l_id in self.set_ids_lanelets_same_direction for l_id in ids]
                    # if len(ids)>1:
                    #     print('Correctness')
                    #     print(ids)
                    #     print(correct_lane)
                    # correct_lane = [any(correct_lane) for correctness in correct_lane]
                    # if len(ids)>1:
                    #     print(correct_lane)
                    # vertex_lane_correct += correct_lane
                    if not any(correct_lane):
                        return np.inf

                vertex_lanelet_ids = np.unique(np.array(vertex_lanelet_ids).flatten()).tolist()
                vertex_lanelet_ids_list.append(vertex_lanelet_ids)
                # print(f'Vertex lanelet_ids {vertex_lanelet_ids}')

                current_obstacle_distance = np.inf
                future_obstacle_distance = np.inf
                for lanelet_id in vertex_lanelet_ids:
                    dist_temp_now = self.calc_dist_to_closest_obstacle(lanelet_id, snapshot.position, snapshot.time_step, margin)
                    dist_temp_fut = self.calc_dist_to_closest_obstacle(lanelet_id, snapshot.position, snapshot.time_step+1, margin)
                    if dist_temp_now < current_obstacle_distance:
                        current_obstacle_distance = dist_temp_now
                    if dist_temp_fut < future_obstacle_distance:
                        future_obstacle_distance  = dist_temp_fut
                # print(f'Current closest distance to obstacle  : {current_obstacle_distance}')
                # print(f'Closest distance to obs next time step: {future_obstacle_distance}')

                current_obstacle_distance_cost += 1 / current_obstacle_distance
                future_obstacle_distance_cost  += 1 / future_obstacle_distance

            current_obstacle_distance_cost /= len(path)
            future_obstacle_distance_cost  /= len(path)

            # Calculate cost from obstacle.
            weight_obstacle = 10
            obstacle_cost = weight_obstacle * (current_obstacle_distance_cost + future_obstacle_distance_cost / 2)
            # print(f'obstacle_cost: {obstacle_cost}')
            partial_costs['obstacle'] = {
                'cost': current_obstacle_distance_cost + future_obstacle_distance_cost / 2,
                'weight': 10,
            }



            # Weight lanelet distance to goal
            lanelet_cost = 0
            center_lanelet_ids_nested = self.lanelet_network.find_lanelet_by_position([path[-1].position])
            center_lanelet_ids = []
            for ids in center_lanelet_ids_nested: # Flatten it.
                # center_lanelet_ids += ids
                correct_lane = [l_id in self.set_ids_lanelets_same_direction for l_id in ids]
                if not any(correct_lane):
                    return np.inf
                for correct, l_id in zip(correct_lane, ids):
                    if correct:
                        center_lanelet_ids.append(l_id)
            center_lanelet_ids = np.unique(np.array(center_lanelet_ids).flatten()).tolist()
            # print(f'Center lanelet_ids {center_lanelet_ids}')
            for lanelet_id in center_lanelet_ids:
                # print(f'{lanelet_id} lanelet cost to goal: {self.dict_lanelets_costs[lanelet_id]}')
                if self.dict_lanelets_costs[lanelet_id] >= 0:
                    lanelet_cost += self.dict_lanelets_costs[lanelet_id]
                else:
                    # lanelet_cost += 100
                    pass # Would already have returned np.inf if it's definitely in the wrong direction. Otherwise we end up here at an intersection.
            # if len(center_lanelet_ids) > 0:
            #     lanelet_cost /= len(center_lanelet_ids)
            # print(f'lanelet_cost {lanelet_cost}')
            # mapping = self.map_obstacles_to_lanelets(path[-1].time_step)
            partial_costs['lanelet'] = {
                'cost': lanelet_cost,
                'weight': 1,
            }
            if self.iteration % self.print_every == 0 or self.goal_is_reached:
                print(f'Current lanelet id: {center_lanelet_ids}. Right direction? {[l_id in self.set_ids_lanelets_same_direction for l_id in center_lanelet_ids]}')



            # Discourage the ego vehicle from travelling in the opposite lane.
            # Can't really return inf because it is possible we're at an intersection.
            opposite_cost_v_list = []
            for vertex_lanelet_ids in vertex_lanelet_ids_list:
                opposite_cost_v = 0
                for lanelet_id in vertex_lanelet_ids:
                    if not lanelet_id in self.set_ids_lanelets_same_direction:
                        if lanelet_id in self.scenario.lanelet_network.map_inc_lanelets_to_intersections.keys():
                            # print(f'Lanelet {lanelet_id} is an intersection')
                            opposite_cost_v += 2
                        else:
                            # print(f'Lanelet {lanelet_id} is NOT an intersection')
                            opposite_cost_v += 20
                            # return np.inf
                opposite_cost_v /= len(vertex_lanelet_ids)
                opposite_cost_v_list.append(opposite_cost_v)
            opposite_cost_v = sum(opposite_cost_v_list)
            opposite_cost_c = 0
            for lanelet_id in center_lanelet_ids:
                if not lanelet_id in self.set_ids_lanelets_same_direction:
                    if lanelet_id in self.scenario.lanelet_network.map_inc_lanelets_to_intersections.keys():
                        print(f'Lanelet {lanelet_id} is an intersection')
                        opposite_cost_c += 10
                    else:
                        # print(f'Lanelet {lanelet_id} is NOT an intersection')
                        opposite_cost_c += 100
                        # return np.inf
            opposite_cost_c /= len(center_lanelet_ids)
            # partial_costs['opposite'] = {
            #     'cost': opposite_cost_v + opposite_cost_c,
            #     'weight': 10,
            # }



            # Cost due to speed limit.
            speed_limit = self.tsi.speed_limit(frozenset(center_lanelet_ids))
            if speed_limit is None:
                overspeed_cost = 0
            else:
                # print(f'Speed limit: {speed_limit}')
                if path[-1].velocity > speed_limit:
                    overspeed_cost = (path[-1].velocity - speed_limit)
                else:
                    overspeed_cost = 0

            partial_costs['overspeed'] = {
                'cost': overspeed_cost,
                'weight': 5,
            }



            # Cost due to minimum speed.
            speed_req = self.tsi.required_speed(frozenset(center_lanelet_ids))
            if speed_req is None:
                underspeed_cost = 0
            else:
                # print(f'Speed requirement: {speed_req}')
                if path[-1].velocity < speed_req:
                    underspeed_cost = (speed_req - path[-1].velocity)
                else:
                    underspeed_cost = 0

            partial_costs['underspeed'] = {
                'cost': underspeed_cost,
                'weight': 1,
            }



            # Calculate cost from deviating from lanelet center.
            weight_lanelet = 0.2 # 0.15
            lanelet_distance, final_lanelet_id, start_lanelet_id = self.calc_heuristic_lanelet(path)
            if lanelet_distance is None or final_lanelet_id[0] is None:
                num_obstacles = 0
                lanelet_distance = 0
            else:
                num_obstacles = self.num_obstacles_in_lanelet_at_time_step(path[-1].time_step, final_lanelet_id[0])
                is_goal_lane = self.is_goal_in_lane(final_lanelet_id[0])
            ch_lanelet_cost = weight_lanelet * lanelet_distance * (scale_factor + 1) / 2 / (num_obstacles + 1) ** 2
            partial_costs['center'] = {
                'cost': lanelet_distance / len(path),
                'weight': 5,
            }

            # Favor high speed.
            # if num_obstacles == 0 and is_goal_lane and scale_factor < 0.7:
            #     velocity_cost /= 8 * (1 - scale_factor)
            #     efficiency_cost /= 4 * (1 - scale_factor)
            #     ch_lanelet_cost /= 16 * (1 - scale_factor)
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
            # elif np.pi - abs(abs(laneletOrientationAtPosition - path[-1].orientation)



            # if not np.isinf(dist_to_obstacle) and dist_to_obstacle < 20 and scale_factor < 0.7:
            #     orientation_cost   /= 20 / dist_to_obstacle
            #     velocity_diff_cost /= 20 / dist_to_obstacle
            #     ch_lanelet_cost    /= 20 / dist_to_obstacle



            # Calculate cost due to orientation w.r.t goal.
            # weight_orientation = 2
            if hasattr(self.planningProblem.goal.state_list[0], 'orientation'):

                goal_orientation_start_x = np.cos(self.planningProblem.goal.state_list[0].orientation.start)
                goal_orientation_start_y = np.sin(self.planningProblem.goal.state_list[0].orientation.start)
                goal_orientation_end_x   = np.cos(self.planningProblem.goal.state_list[0].orientation.end)
                goal_orientation_end_y   = np.sin(self.planningProblem.goal.state_list[0].orientation.end)
                goal_orientation_x = (goal_orientation_start_x + goal_orientation_end_x) / 2
                goal_orientation_y = (goal_orientation_start_y + goal_orientation_end_y) / 2
                goal_orientation = np.arctan2(goal_orientation_y, goal_orientation_x)

                diff_orientation = path[-1].orientation - goal_orientation
                if diff_orientation > np.pi:
                    diff_orientation -= 2 * np.pi
                elif diff_orientation <= -np.pi:
                    diff_orientation += 2 * np.pi
                # print(f'{diff_orientation=}')

            else:

                diff_orientation = 0

            # orientation_cost = weight_orientation * abs(diff_orientation / np.pi) * scale_factor
            partial_costs['orient2goal'] = {
                'cost': abs(diff_orientation / np.pi) * scale_factor,
                'weight': 0.5,
            }



            # Want the car's orientation to be aligned with the lanelet.
            orientation_cost = 0
            for laneletId in center_lanelet_ids:
                laneletObj = self.scenario.lanelet_network.find_lanelet_by_id(laneletId)
                laneletOrientationAtPosition = self.calc_angle_of_position(laneletObj.center_vertices, path[-1].position)
                angle_diff = np.pi - abs(abs(laneletOrientationAtPosition - path[-1].orientation) - np.pi)
                orientation_cost += abs(angle_diff)
            orientation_cost /= len(center_lanelet_ids) / np.pi

            partial_costs['orientation'] = {
                'cost': orientation_cost,
                'weight': 0.8,
            }



            # Calculate cost due to velocity.
            weight_velocity = 2 # 1
            velocity_cost = 0
            for goal_state in self.planningProblem.goal.state_list:
                if hasattr(goal_state, 'velocity'):
                    if not self.planningProblem.goal._check_value_in_interval(path[-1].velocity, goal_state.velocity):
                        if path[-1].velocity > goal_state.velocity.end:
                            velocity_cost = abs(path[-1].velocity - goal_state.velocity.end)
                        elif path[-1].velocity < goal_state.velocity.start:
                            velocity_cost = abs(path[-1].velocity - goal_state.velocity.start)
            # When far from the goal, we want velocity to be as fast as possible.
            velocity_cost *= (1 - scale_factor) 
            if velocity_cost < 0:
                velocity_cost = 0

            partial_costs['velocity'] = {
                'cost': velocity_cost / 20, # Normalzied to 20m/s.
                'weight': 1#0.2,
            }



            # Calculate cost due to acceleration.
            if len(path) > 1:
                accel = []
                accel_abs = []
                for k in range(1, len(path)):
                    accel.append(path[k].velocity - path[k-1].velocity)
                    accel_abs.append(abs(path[k].velocity - path[k-1].velocity))
                accel_cost = sum(accel_abs) / (len(path) - 1) * (1 - scale_factor)
            else:
                accel_cost = 0

            partial_costs['acceleration'] = {
                'cost': accel_cost, # Normalzied to 20m/s.
                'weight': 3#0.1,
            }



            # Calculate cost due to acceleration jerk.
            if len(path) > 2:
                jerk = []
                jerk_abs = []
                for k in range(1, len(accel)):
                    jerk.append(accel[k] - accel[k-1])
                    jerk_abs.append(abs(accel[k] - accel[k-1]))
                jerk_cost = sum(jerk_abs) / (len(accel) - 1) * (1 - scale_factor)
            else:
                jerk_cost = 0

            partial_costs['jerk'] = {
                'cost': jerk_cost,
                'weight': 2#0.5,
            }



            # Calculate cost from path inefficiency.
            weight_efficiency = 0.25
            efficiency = self.calc_path_efficiency(path) / 0.1 # `/ 0.1` because each timestep is 0.1s.
            efficiency_cost = weight_efficiency * (1 - efficiency / 40) * (1 - scale_factor)
            if efficiency_cost < 0:
                efficiency_cost = 0

            partial_costs['efficiency'] = {
                'cost': efficiency_cost,
                'weight': 5,
            }



            # Calculate cost from passenger comfort.
            weight_curvature = 0.2
            curvature = 0
            try:
                curvature = self.calc_curvature_of_polyline(np.array([path_i.position for path_i in path]))
                if np.isnan(curvature):
                    curvature = 10
            except:
                pass
            curvature_cost = weight_curvature * curvature * (1 - scale_factor) / (num_obstacles + 1) ** 2
            partial_costs['curvature'] = {
                'cost': curvature * (1 - scale_factor) / (num_obstacles + 1),
                'weight': 2,
            }



            # Calculate cost from time step.
            weight_time = 0.1
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
            partial_costs['time'] = {
                'cost': (1 - time_step_normalized) * scale_factor,
                'weight': 0.5,
            }



            # cost = 0
            # cost += distance_cost
            # cost += orientation_cost
            # cost += velocity_cost
            # cost += velocity_diff_cost
            # cost += accel_cost
            # cost += jerk_cost
            # cost += efficiency_cost
            # cost += lanelet_cost
            # cost += ch_lanelet_cost
            # cost += curvature_cost
            # cost += obstacle_cost
            # cost += time_step_cost

            cost = 0
            for key, value in partial_costs.items():
                if self.iteration % self.print_every == 0 or self.goal_is_reached:
                    print(f"Cost: {key:<16} = {value['cost']: 9.4f} | Weighted = {value['weight'] * value['cost']: 9.4f}")
                cost += value['weight'] * value['cost']
            # cost = cost * 2#len(partial_costs.items()) * 2


        except Exception as e:

            cost = np.inf
            # logging.error('Error:')
            # logging.error(repr(e))
            print(f'Error: {self.scenario.scenario_id}')
            print(repr(e))
            print('')
            raise e


        if cost < 0: cost = 0
        return cost
