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
        print('StudentMotionPlanner V2 - Squared spacing of motion primitives.')

        self.tsi = TrafficSigInterpreter(scenario.scenario_id.country_id, scenario.lanelet_network)

        if hasattr(self.planningProblem.goal.state_list[0], 'position'): # and hasattr(self.planningProblem.goal.state_list[0].position, 'vertices'):
            print(f'Standard mode. Goal:')
            for i, state in enumerate(self.planningProblem.goal.state_list):
                print(f'Goal state {i}:')
                print(state)
            self.initial_distance = self.calc_euclidean_distance(Node([[planningProblem.initial_state]], [], 0))
            print(f'Initial Euclidean distance: {self.initial_distance}')
            self.survival_mode = False
        else:
            print(f'Survival mode. Only time goal:')
            for i, state in enumerate(self.planningProblem.goal.state_list):
                print(f'Goal state {i}:')
                print(state)
            self.survival_mode = True

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

        self.iteration = 0

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
        if self.iteration % 1000 == 0:
            display.clear_output(wait=True)
        if self.iteration % 100 == 0:
            # display.clear_output(wait=True)
            print(f'A* | {self.scenario.scenario_id} | Iteration {self.iteration} | Path length: {len(node_current.list_paths[-1])}')

        # print(f'Iteration {self.iteration}')
        # for state in node_current.list_paths[-1]:
        #     print(state.attributes)
        # print()

        if not self.is_collision_free(node_current.list_paths[-1]):
            # print('Collision! Path rejected!')
            node_current.priority = np.inf
            return np.inf

        if self.reached_goal(node_current.list_paths[-1]):
            print('Goal is reached!!!')
            node_current.list_paths = self.remove_states_behind_goal(node_current.list_paths)

        # TODO: Switch to JB1 if survival mode!
        trajectory = Trajectory(node_current.list_paths[-1][0].time_step, node_current.list_paths[-1])
        if len(node_current.list_paths[-1]) > 1:
            try:
                cost = self.ce.evaluate_pp_solution(self.scenario, self.planningProblem, trajectory)
                g_cost = cost.total_costs
            except Exception as e:
                print('Error evaluating cost function.')
                print(repr(e))
                g_cost = 100
            # print('Partial cost:')
            # print(cost)
            # g_cost = cost.total_costs
            h_cost = self.heuristic_function(node_current=node_current)
            f_cost = g_cost + h_cost
            node_current.priority = g_cost
            # print(f'Evaluation cost: {g_cost}')
            # print(f'Heuristic cost : {h_cost}')
            # print(f'Total cost     : {f_cost}')
            return f_cost

        else:
            # calculate g(n)
            node_current.priority += (len(node_current.list_paths[-1]) - 1) * self.scenario.dt
            # f(n) = g(n) + h(n)
            g_cost = node_current.priority
            h_cost = self.heuristic_function(node_current=node_current)
            f_cost = g_cost + h_cost
            # print(f'Evaluation cost: {g_cost}')
            # print(f'Heuristic cost : {h_cost}')
            # print(f'Total cost     : {f_cost}')
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
                #print('Collision!')
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
                scale_factor = epsilon
            else:
                scale_factor = scale_factor ** 1.5



            # Calculate cost due to distance.
            # weight_distance = 2
            # distance_cost = weight_distance * distance_normalized
            partial_costs['distance'] = {
                'cost': distance_normalized,
                'weight': 10,
            }



            # Calculate cost due to orientation.
            # weight_orientation = 2
            if hasattr(self.planningProblem.goal.state_list[0], 'orientation'):

                goal_orientation_start_x = np.cos(self.planningProblem.goal.state_list[0].orientation.start)
                goal_orientation_start_y = np.sin(self.planningProblem.goal.state_list[0].orientation.start)
                goal_orientation_end_x   = np.cos(self.planningProblem.goal.state_list[0].orientation.end)
                goal_orientation_end_y   = np.sin(self.planningProblem.goal.state_list[0].orientation.end)
                goal_orientation_x = (goal_orientation_start_x + goal_orientation_end_x) / 2
                goal_orientation_y = (goal_orientation_start_y + goal_orientation_end_y) / 2
                goal_orientation = np.arctan2(goal_orientation_y, goal_orientation_x)

                car_orientation_x = np.cos(path[-1].orientation)
                car_orientation_y = np.sin(path[-1].orientation)
                car_orientation = np.arctan2(car_orientation_y, car_orientation_x)

                diff_orientation = car_orientation - goal_orientation
                if diff_orientation > np.pi:
                    diff_orientation -= 2 * np.pi
                elif diff_orientation <= -np.pi:
                    diff_orientation += 2 * np.pi

            else:

                goal_orientation = 0
                car_orientation  = 0
                diff_orientation = 0

            # orientation_cost = weight_orientation * abs(diff_orientation / np.pi) * scale_factor
            partial_costs['orientation'] = {
                'cost': abs(diff_orientation / np.pi) * scale_factor,
                'weight': 10#0.5,
            }



            # Calculate cost due to velocity.
            weight_velocity = 2 # 1
            v_mean_goal = None
            if hasattr(self.planningProblem.goal.state_list[0], 'velocity'):
                v_mean_goal = (self.planningProblem.goal.state_list[0].velocity.start + 
                            self.planningProblem.goal.state_list[0].velocity.end) / 2
                diff_velocity = path[-1].velocity - v_mean_goal
            else:
                diff_velocity = 0
            # When far from the goal, we want velocity to be as fast as possible.
            velocity_cost = weight_velocity * (1 - path[-1].velocity / 20) * (1 - scale_factor) 
            if velocity_cost < 0:
                velocity_cost = 0
            # When close to the goal, we want velocity to be as close as possible.
            velocity_diff_cost = weight_velocity * abs(diff_velocity) * scale_factor
            velocity_cost = 0 # i.e. disable

            partial_costs['velocity'] = {
                'cost': abs(diff_velocity) / 20 * scale_factor, # Normalzied to 20m/s.
                'weight': 10#0.2,
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
                'weight': 30#0.1,
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
                'weight': 20#0.5,
            }



            # Calculate cost from path inefficiency.
            weight_efficiency = 0.25
            efficiency = self.calc_path_efficiency(path) / 0.1 # `/ 0.1` because each timestep is 0.1s.
            efficiency_cost = weight_efficiency * (1 - efficiency / 40) * (1 - scale_factor)
            if efficiency_cost < 0:
                efficiency_cost = 0

            partial_costs['efficiency'] = {
                'cost': efficiency_cost,
                'weight': 10,
            }


            # calc_travelled_distance
            # print(mapping)
            # get_obstacles



            # Identify all obstaceles in all lanes currently occupied by the ego vehicle.
            margin = 1.1
            ego = Rectangle(length=self.vehicle.l*margin, width=self.vehicle.w*margin, center=path[-1].position, orientation=path[-1].orientation)

            vertex_lanelet_ids = self.lanelet_network.find_lanelet_by_position([v for v in ego.vertices])
            current_lanelet_ids = []
            for ids in vertex_lanelet_ids: # Flatten it.
                current_lanelet_ids += ids
            current_lanelet_ids = np.unique(np.array(current_lanelet_ids).flatten()).tolist()
            #print(f'Vertex lanelet_ids {current_lanelet_ids}')
            num_obstacles = 0
            for lanelet_id in current_lanelet_ids:
                num_obstacles += self.num_obstacles_in_lanelet_at_time_step(path[-1].time_step, lanelet_id)
            #print(f'num_obstacles {num_obstacles}')
            dist_to_obstacle = np.inf
            future_distance = np.inf
            for lanelet_id in current_lanelet_ids:
                dist_temp_now = self.calc_dist_to_closest_obstacle(lanelet_id, path[-1].position, path[-1].time_step, margin)
                dist_temp_fut = self.calc_dist_to_closest_obstacle(lanelet_id, path[-1].position, path[-1].time_step+1, margin)
                if dist_temp_now < dist_to_obstacle:
                    dist_to_obstacle = dist_temp_now
                if dist_temp_fut < future_distance:
                    future_distance = dist_temp_fut
            # print(f'Current distance to obstacle  : {dist_to_obstacle}')
            # print(f'Distance to obs next time step: {future_distance}')

            # Calculate cost from obstacle.
            weight_obstacle = 2
            obstacle_cost = weight_obstacle * (1 / dist_to_obstacle + 1 / future_distance)
            # print(f'obstacle_cost: {obstacle_cost}')
            partial_costs['obstacle'] = {
                'cost': 1 / dist_to_obstacle + 1 / future_distance,
                'weight': 50#5,
            }



            # Weight lanelet distance to goal
            lanelet_cost = 0
            center_lanelet_ids = self.lanelet_network.find_lanelet_by_position([path[-1].position])
            current_lanelet_ids = []
            for ids in center_lanelet_ids: # Flatten it.
                current_lanelet_ids += ids
            current_lanelet_ids = np.unique(np.array(current_lanelet_ids).flatten()).tolist()
            # print(f'Center lanelet_ids {current_lanelet_ids}')
            for lanelet_id in current_lanelet_ids:
                # print(f'{lanelet_id} lanelet cost to goal: {self.dict_lanelets_costs[lanelet_id]}')
                if self.dict_lanelets_costs[lanelet_id] >= 0:
                    lanelet_cost += self.dict_lanelets_costs[lanelet_id]
                else:
                    lanelet_cost += 100
            if len(current_lanelet_ids) > 0:
                lanelet_cost /= len(current_lanelet_ids)
            # print(f'lanelet_cost {lanelet_cost}')
            # mapping = self.map_obstacles_to_lanelets(path[-1].time_step)
            partial_costs['lanelet'] = {
                'cost': lanelet_cost,
                'weight': 1,
            }



            # Cost due to speed limit.
            speed_limit = self.tsi.speed_limit(frozenset(current_lanelet_ids))
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
                'weight': 50,
            }



            # Cost due to minimum speed.
            speed_req = self.tsi.required_speed(frozenset(current_lanelet_ids))
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
                'weight': 5,
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
                'weight': 80,
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
                'weight': 10,
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
                'weight': 1,
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
                # print(f"Cost: {key:<16} | {value['weight'] * value['cost']: 9.4f}")
                cost += value['weight'] * value['cost']
            cost *= 2


        except Exception as e:

            cost = np.inf
            # logging.error('Error:')
            # logging.error(repr(e))
            # print(repr(e))
            raise e


        if cost < 0: cost = 0
        return cost * factor
