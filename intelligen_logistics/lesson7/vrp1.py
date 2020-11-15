#!/usr/bin/env python
# coding: utf-8

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import numpy as np 

# 数据加载
distance_df = pd.read_excel('distance.xlsx', index_col=0)

# 设置城市名
city_names = distance_df.index

# 设置数据
def create_data_model():
    data = {}
    data['distance_matrix'] = distance_df.values/1000
    data['num_vehicles'] = 4
    data['depot'] = 0
    return data
 
 
# 输出结果
def print_solution(manager, num_vehicles, routing, solution):
    
    #print('总行驶里程: {} 公里'.format(solution.ObjectiveValue()))
    route_list=[]
    distance_list = []
    for vehicle_id in range(num_vehicles):
        print('vehicle_id:{}'.format(vehicle_id))
        index = routing.Start(vehicle_id)
        #plan_output = '车辆的路径:\n'
        plan_output = []
        route_distance = 0
        while not routing.IsEnd(index):
            #plan_output += ' {} ->'.format(city_names[manager.IndexToNode(index)])
            plan_output.append(city_names[manager.IndexToNode(index)])
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
            #plan_output += city_names[manager.IndexToNode(index)]
        distance_list.append(route_distance)
        route_list.append(plan_output)
    print(route_list)
    print(distance_list)
 
def main():
    
    # 初始化数据
    data = create_data_model()
 
    # 创建路线管理，tsp_size（城市数量）, num_vehicles（车的数量）, depot（原点）
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
 
    # 创建 Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # 计算两点之间的距离
    def distance_callback(from_index, to_index):
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        #print(data['distance_matrix'][from_node][to_node])
        return data['distance_matrix'][from_node][to_node]

 
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

 
    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

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
 
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
 
    # 求解路径规划
    solution = routing.SolveWithParameters(search_parameters)
    # 输出结果
    if solution:
        print_solution(manager, data['num_vehicles'], routing, solution)
    else:
        print(solution)
 
if __name__ == '__main__':
    main()

