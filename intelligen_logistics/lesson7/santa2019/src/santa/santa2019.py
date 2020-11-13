#kaggle路径如下：https://www.kaggle.com/c/santa-2019-revenge-of-the-accountants/overview/evaluation

import sys
sys.path.append('./')
'''
sys.path.append('../model')
sys.path.append('../util')
sys.path.append('../source')
sys.path.append('../')

'''
from model.my_solver import MySolver
from util.common import get_penalty, SantaTools
from util.configure import *  
import numpy as np
import pandas as pd

pcost_mat = np.full(shape=(N_FAMILY, N_DAYS), fill_value= np.inf)
acost_mat = np.zeros((MAX_OCCUPANCY, MAX_OCCUPANCY), dtype='float64')

#数据加载
data = pd.read_csv('E:\\private\\学习\\2020-bi-programing-practice\\随便写写\\intelligen_logistics\\L22\\santa\\family_data.csv',index_col='family_id')
DESIRED = data.iloc[:,:-1].values -1
for i in range(N_FAMILY):
    n_people = data.loc[i, 'n_people']
    pcost_mat[i,:] = get_penalty(n_people, i, DESIRED[i], N_DAYS)
for i in range(acost_mat.shape[0]):
    for j in range(acost_mat.shape[1]):
        acost_mat[i,j] = max(0, ((i+1-125)/400)*((i+1)** (1/2+abs(i-j+1)/50)))
FAMILY_SIZE = data['n_people'].values

if __name__ == '__main__':
    solver = MySolver(N_FAMILY, N_DAYS, MIN_OCCUPANCY, MAX_OCCUPANCY, DESIRED, FAMILY_SIZE)
    prediction = solver.solveSanta(pcost_mat, acost_mat)
    tools = SantaTools(FAMILY_SIZE, DESIRED, pcost_mat, acost_mat)
    pcost_value, occupancy = tools.pcost(prediction, FAMILY_SIZE, pcost_mat)
    acost_value, _ = tools.acost(occupancy, acost_mat)
    print('pcost_value:{}, acost_value:{:.2f}, (occ.min:{}, occ.max:{})'.format(pcost_value, acost_value, occupancy.min(), occupancy.max()))
    score = tools.cost_function(prediction)
    print('score:{}'.format(score))
    new_prediction = prediction.copy()
    tools.find_better_day_for_family(new_prediction)
    
    final = tools.stochastic_product_search(top_k=2,
        fam_size=8, 
        original=new_prediction, 
        n_iter=10000,
        verbose=1000,
        verbose2=50000,
        random_state=2019
        )
    final = tools.seed_finding(2019, final)

