from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np

class MySolver():
    def __init__(self, N_FAMILY, N_DAYS, MIN_OCCUPANCY, MAX_OCCUPANCY, DESIRED, FAMILY_SIZE):
        self.N_FAMILY = N_FAMILY
        self.N_DAYS = N_DAYS
        self.MIN_OCCUPANCY = MIN_OCCUPANCY
        self.MAX_OCCUPANCY = MAX_OCCUPANCY
        self.FAMILY_SIZE = FAMILY_SIZE
        self.DESIRED = DESIRED
        self.resdict = {0:'OPTIMAL', 1:'FEASIBLE', 2:'INFEASIBLE', 3:'UNBOUNDED', 
               4:'ABNORMAL', 5:'MODEL_INVALID', 6:'NOT_SOLVED'}

    def solveLP(self, pcost_mat, acost_mat, thrs=25):
        solver = pywraplp.Solver('santa', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
        x = {}  #famliyID 在第几天参观
        #每一天都有哪些家庭来参观
        candidates = [[] for i in range(self.N_DAYS)]
        for i in range(self.N_FAMILY):
            for j in self.DESIRED[i,:]:
                candidates[j].append(i)
                #定义变量，x[i,j]第i个家庭，第j天是否参加
                x[i,j] = solver.BoolVar('x[%i,%i]'%(i,j))
        print(len(x))
        #每天人数125-300，统计每天的人数
        daliy_occupancy = [solver.Sum([x[i,j] * self.FAMILY_SIZE[i] for i in candidates[j]]) for j in range(self.N_DAYS)]
        #家庭在10选择中出现的次数
        famliy_presence = [solver.Sum([x[i,j] for j in self.DESIRED[i,:]]) for i in range(self.N_FAMILY)]
        
        #目标函数
        preference_cost = solver.Sum([pcost_mat[i,j] *x[i,j] for i in range(self.N_FAMILY) for j in self.DESIRED[i,:]])
        solver.Minimize(preference_cost)
        #设置约束条件
        #当天出现人数不超过前一天25
        for j in range(self.N_DAYS-1):
            solver.Add(daliy_occupancy[j] - daliy_occupancy[j+1] <=thrs)
            solver.Add(daliy_occupancy[j+1] - daliy_occupancy[j] <=thrs)
            
        for i in range(self.N_FAMILY):
            solver.Add(famliy_presence[i] == 1)
        for j in range(self.N_DAYS):
            solver.Add(daliy_occupancy[j] >= self.MIN_OCCUPANCY)
            solver.Add(daliy_occupancy[j] <= self.MAX_OCCUPANCY)
        result = solver.Solve()
        print('LP solver result:', self.resdict[result])
        temp = [(i,j,x[i,j].solution_value()) for i in range(self.N_FAMILY) for j in self.DESIRED[i,:] if x[i,j].solution_value() >0]
        df = pd.DataFrame(temp, columns=['family_id', 'day','result'])
        return df

    def solveMIP(self, families, min_occupancy, max_occupancy, pcost_mat, acost_mat):
        solver = pywraplp.Solver('santa', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        n_famliy = len(families) 
        x = {}  #famliyID 在第几天参观
        #每一天都有哪些家庭来参观
        candidates = [[] for i in range(self.N_DAYS)]
        for i in families:
            for j in self.DESIRED[i,:]:
                candidates[j].append(i)
                #定义变量，x[i,j]第i个家庭，第j天是否参加
                x[i,j] = solver.BoolVar('x[%i,%i]'%(i,j))
        print(n_famliy, len(x))
        #每天人数125-300，统计每天的人数
        daliy_occupancy = [solver.Sum([x[i,j] * self.FAMILY_SIZE[i] for i in candidates[j]]) for j in range(self.N_DAYS)]
        #家庭在10选择中出现的次数
        famliy_presence = [solver.Sum([x[i,j] for j in self.DESIRED[i,:]]) for i in families]
        
        #目标函数
        preference_cost = solver.Sum([pcost_mat[i,j] *x[i,j] for i in families for j in self.DESIRED[i,:]])
        solver.Minimize(preference_cost)
        #设置约束条件
        #当天出现人数不超过前一天25
        for j in range(self.N_DAYS-1):
            solver.Add(daliy_occupancy[j] - daliy_occupancy[j+1] <=25)
            solver.Add(daliy_occupancy[j+1] - daliy_occupancy[j] <=25)
            
        for i in range(n_famliy):
            solver.Add(famliy_presence[i] == 1)
        for j in range(self.N_DAYS):
            solver.Add(daliy_occupancy[j] >= min_occupancy[j])
            solver.Add(daliy_occupancy[j] <= max_occupancy[j])
        result = solver.Solve()
        print('MIP solver result:', self.resdict[result])
        temp = [(i,j) for i in families for j in self.DESIRED[i,:] if x[i,j].solution_value() >0]
        df = pd.DataFrame(temp, columns=['family_id', 'day'])
        return df

    def solveSanta(self, pcost_mat, acost_mat, thrs=25):
        df = self.solveLP(pcost_mat, acost_mat, thrs)
        threshold = 0.99
        assigned_df = df[df['result'] >= threshold].copy()
        unassigned_family = df[(df['result'] < threshold) &(df['result'] > 1-threshold)].family_id.unique()
        print('{} unassigned families, {} assigned families'.format(len(unassigned_family), len(assigned_df)))
        assigned_df['family_size'] = self.FAMILY_SIZE[assigned_df['family_id']]
        occupancy = assigned_df.groupby(['day'])['family_size'].sum().values
        min_accupancy = np.array([max(0, self.MIN_OCCUPANCY - o) for o in occupancy])
        max_accupancy = np.array([max(0, self.MAX_OCCUPANCY -o) for o in occupancy])

        rdf = self.solveMIP(unassigned_family, min_accupancy, max_accupancy, pcost_mat, acost_mat)
        df = pd.concat([assigned_df[['family_id', 'day']], rdf]).sort_values('family_id')
        return df.day.values



