from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import numpy as np
class tsp():
    def __init__(self,distance_file, cities_file, num_vehicles=1):
        self.city_names = None
        self.location = None
        self.num_vehicles = num_vehicles
        self.distance_file = distance_file
        self.cities_file = cities_file

    def create_data_model(self):
        distance_df = pd.read_excel(self.distance_file, index_col=0)
        self.city_names = distance_df.index
        self.location = data=pd.read_excel(self.cities_file,index_col='name').to_dict()['location']
        data = {}
        data['distance_matrix'] = distance_df.values/1000
        data['num_vehicles'] = self.num_vehicles
        data['depot'] = 0
        return data
    
    def get_solution(self, routing, solution, manager):
        #print('总行驶里程: {} 公里'.format(solution.ObjectiveValue()))
        route_list=[]
        route_location_list=[]
        distance_list = []
        for vehicle_id in range(self.num_vehicles):
            print('vehicle_id:{}'.format(vehicle_id))
            index = routing.Start(vehicle_id)
            #plan_output = '车辆的路径:\n'
            plan_output = []
            plan_location_output=[]
            route_distance = 0
            while not routing.IsEnd(index):
                #plan_output += ' {} ->'.format(city_names[manager.IndexToNode(index)])
                city_name = self.city_names[manager.IndexToNode(index)]
                plan_output.append(city_name)
                plan_location_output.append(self.location[city_name].split(','))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
                #plan_output += city_names[manager.IndexToNode(index)]
            distance_list.append(route_distance)
            route_list.append(plan_output)
            route_location_list.append(plan_location_output)
        print(route_list)
        print(distance_list)
        print(route_location_list)
        return route_list, route_location_list, distance_list

    def add_dimension(self, routing, transit_callback_index):
        # 添加距离约束
        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            10000,  # 车辆最大行驶距离
            True,  # start cumul to zero
            dimension_name)
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        # 尽量减少车辆之间的最大距离
        distance_dimension.SetGlobalSpanCostCoefficient(100)

    def work(self): 
        data = self.create_data_model()
        # 定义路由，比如10个节点，4辆车
        manager = pywrapcp.RoutingIndexManager (len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
        # 创建 Routing Model.
        routing = pywrapcp.RoutingModel(manager)
        # 计算两点之间的距离
        def distance_callback(from_index, to_index):
            # 将路由变量Index转化为 距离矩阵ditance_matrix的节点index
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        # 定义每条边arc的代价
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        #添加约束条件
        self.add_dimension(routing, transit_callback_index)
        
        # 设置启发式搜索
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        # 求解路径规划
        solution = routing.SolveWithParameters(search_parameters)
        return routing, solution, manager

