import pandas as pd 
import numpy as np
from numba import njit
from itertools import product
from util.configure import *

#计算单个preference penalty
def get_penalty(n_people, i, n_choice, N_DAYS):
    temp = np.zeros((N_DAYS))
    for i in range(N_DAYS):
        if i == n_choice[0]:
            temp[i] = 0
        elif i== n_choice[1]:
            temp[i] = 50
        elif i == n_choice[2]:
            temp[i] = 50 + n_people*9
        elif i == n_choice[3]:
            temp[i] = 100 + n_people*9
        elif i==n_choice[4]:
            temp[i] = 200 + n_people*9
        elif i==n_choice[5]:
            temp[i] = 200 + n_people*18
        elif i==n_choice[6]:
            temp[i] = 300 + n_people*18
        elif i == n_choice[7]:
            temp[i] = 300 + n_people*36
        elif i== n_choice[8]:
            temp[i] = 400 + n_people*36
        elif i == n_choice[9]:
            temp[i] = 500 + n_people*(36+199)
        else:
            temp[i] = 500 + n_people*(36+398)
    return temp

class SantaTools():
    def __init__(self, FAMILY_SIZE, DESIRED, pcost_mat, acost_mat):
        self.FAMILY_SIZE = FAMILY_SIZE
        self.DESIRED = DESIRED
        self.pcost_mat = pcost_mat
        self.acost_mat = acost_mat

    # preference cost
    @staticmethod
    @njit(nopython=True)
    def pcost(prediction, FAMILY_SIZE, pcost_mat):
        daily_occupancy = np.zeros(N_DAYS, dtype=np.int64)
        penalty = 0
        for (i, p) in enumerate(prediction):
            n = FAMILY_SIZE[i]
            penalty += pcost_mat[i, p]
            daily_occupancy[p] += n
        return penalty, daily_occupancy

    # accounting cost
    @staticmethod
    @njit(fastmath=True)
    def acost(daily_occupancy, acost_mat ):
        do = np.zeros(N_DAYS+1, dtype=np.int64)
        do[:N_DAYS] = daily_occupancy 
        do[N_DAYS:] = do[N_DAYS-1]
        accounting_cost = 0
        n_out_of_range = 0
        for day in range(N_DAYS):
            n_pj = do[day + 1]
            n    = do[day]
            n_out_of_range += (n > MAX_OCCUPANCY) or (n < MIN_OCCUPANCY)
            if n <= MAX_OCCUPANCY and n_pj <= MAX_OCCUPANCY:
                accounting_cost += acost_mat[n-1, n_pj-1]
        return accounting_cost, n_out_of_range

    def cost_function(self, prediction):
        penalty, daily_occupancy = self.pcost(prediction, self.FAMILY_SIZE, self.pcost_mat)
        accounting_cost, n_out_of_range = self.acost(daily_occupancy, self.acost_mat)
        return penalty + accounting_cost + n_out_of_range*100000000

    def find_better_day_for_family(self, prediction):
        fobs = np.argsort(self.FAMILY_SIZE)
        score = self.cost_function(prediction)
        original_score = np.inf
        
        while original_score>score:
            original_score = score
            for family_id in fobs:
                for pick in range(10):
                    day = self.DESIRED[family_id, pick]
                    oldvalue = prediction[family_id]
                    prediction[family_id] = day
                    new_score = self.cost_function(prediction)
                    if new_score<score:
                        score = new_score
                    else:
                        prediction[family_id] = oldvalue

            print(score, end='\r')
        print(score)

    def stochastic_product_search(self, top_k, fam_size, original, 
                                verbose=1000, verbose2=50000,
                                n_iter=500, random_state=2019):
        """
        original (np.array): The original day assignments.
        
        At every iterations, randomly sample fam_size families. Then, given their top_k
        choices, compute the Cartesian product of the families' choices, and compute the
        score for each of those top_k^fam_size products.
        """
        
        best = original.copy()
        best_score = self.cost_function(best)
        
        np.random.seed(random_state)

        for i in range(n_iter):
            fam_indices = np.random.choice(range(self.DESIRED.shape[0]), size=fam_size)
            changes = np.array(list(product(*self.DESIRED[fam_indices, :top_k].tolist())))

            for change in changes:
                new = best.copy()
                new[fam_indices] = change

                new_score =self.cost_function(new)

                if new_score < best_score:
                    best_score = new_score
                    best = new
                    
            if verbose and i % verbose == 0:
                print(f"Iteration #{i}: Best score is {best_score:.2f}  ", end='\r')
                
            if verbose2 and i % verbose2 == 0:
                print(f"Iteration #{i}: Best score is {best_score:.2f}  ")
        
        print(f"Final best score is {best_score:.2f}")
        return best

    def seed_finding(self, seed, prediction_input):
        prediction = prediction_input.copy()
        np.random.seed(seed)
        best_score = self.cost_function(prediction)
        original_score = best_score
        best_pred = prediction.copy()
        print("SEED: {}   ORIGINAL SCORE: {}".format(seed, original_score))
        for t in range(N_DAYS):
            for i in range(N_FAMILY):
                for j in range(10):
                    di = prediction[i]
                    prediction[i] = self.DESIRED[i, j]
                    cur_score = self.cost_function(prediction)

                    KT = 1
                    if t < 5:
                        KT = 1.5
                    elif t < 10:
                        KT = 4.5
                    else:
                        if cur_score > best_score + 100:
                            KT = 3
                        elif cur_score > best_score + 50 :
                            KT = 2.75
                        elif cur_score > best_score + 20:
                            KT = 2.5
                        elif cur_score > best_score + 10:
                            KT = 2
                        elif cur_score > best_score:
                            KT = 1.5
                        else:
                            KT = 1

                    prob = np.exp(-(cur_score - best_score) / KT)
                    if np.random.rand() < prob:
                        best_score = cur_score
                    else:
                        prediction[i] = di
            if best_score < original_score:
                print("NEW BEST SCORE on seed {}: {}".format(seed, best_score))
                original_score = best_score
                best_pred = prediction.copy()

        return best_pred