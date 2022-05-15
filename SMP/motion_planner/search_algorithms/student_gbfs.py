import gc
import psutil
import traceback

# from matplotlib.patches import Rectangle
from commonroad.geometry.shape import Rectangle, Polygon, ShapeGroup, Circle
from SMP.motion_planner.node import Node, PriorityNode

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import GreedyBestFirstSearch, AStarSearch
from SMP.motion_planner.search_algorithms.student import StudentMotionPlanner # This is custom Astar search.
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

class StudentMotionPlannerGBFS(StudentMotionPlanner):
    """
    Motion planner implementation by students.
    Note that you may inherit from any given motion planner as you wish, or come up with your own planner.
    Here as an example, the planner is inherited from the GreedyBestFirstSearch planner.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)



    def evaluation_function(self, node_current: PriorityNode) -> float:
        """
        Evaluation function of GBFS is f(n) = h(n)
        """
        ########################################################################
        # TODO: Implement your own evaluation function here.                   #
        ########################################################################

        # print(' ')
        # print()
        # print('evaluation_function()')

        # print(f'Iteration {self.iteration}')
        # for state in node_current.list_paths[-1]:
        #     print(state.attributes)
        # print()

        h_cost = self.heuristic_function(node_current=node_current)

        self.iteration += 1
        if self.iteration % (self.print_every * 10) == 0:
            # display.clear_output(wait=True)
            pass
        if self.iteration % self.print_every == 0:
            # display.clear_output(wait=True)
            print(f'GBFS | {self.scenario.scenario_id} | Iter {self.iteration} | Path {len(node_current.list_paths[-1])} | Processed: {self.frontier.count} | Frontier count: {len(self.frontier.list_elements)} | Cost: {h_cost:.4f} | Memory use: {psutil.virtual_memory().percent:.1f}%')
            gc.collect()

        if len(self.frontier.list_elements) > self.frontier_limit:
            raise MemoryError(f'Manually aborting scenario {self.scenario.scenario_id} as frontier size grows too large.')
        if psutil.virtual_memory().percent > self.sys_memory_cap:
            raise MemoryError(f'Manually aborting scenario {self.scenario.scenario_id} as memory use exceeded limit.')

        if not self.is_collision_free(node_current.list_paths[-1]):
            # print('Collision! Path rejected!')
            node_current.priority = np.inf
            return np.inf

        node_current.priority = h_cost
        return node_current.priority
