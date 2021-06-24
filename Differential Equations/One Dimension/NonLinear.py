import math
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import pandas as pd

## Int vector such that [h,x,f(x),f'(x) .... f^n(x)] in a column matrix
condition_vect = np.array([[0.1,1,1,1,1]])

def adjustment_funtion(vector):
    return vector

xmin = 0
xmax = 5


search_min = -0.4
search_max = 0.4
search_step = 0.1


def check_well_posedness(condition_vect):
    known_conditions = []
    ## Seperate each peice of info into a row
    for vect in condition_vect:
        ## See additional information defined by definition of ODE
        vect_transformed = adjustment_funtion(vect.T).T
        k = 0
        for value in vect_transformed[2:len(vect)]:
            if not np.isnan(value):
                ## [x, order of derivative, value], each row is a piece of information
                known_conditions.append([vect[1], k, value])
            k += 1

    ## Remove Duplicates
    known_conditions = np.unique(known_conditions,axis=0)

    ##Find if value is defined twice differently
    range_of_def = []
    for i in known_conditions:
        range_of_def.append(i[0:2])
    if len(range_of_def) != len(np.unique(range_of_def,axis=0)):
        return False, known_conditions, 'No Solution, contradictory conditions'
    elif len(known_conditions) == len(condition_vect[0]) - 2:
        return True, known_conditions,''
    else:
        return False, known_conditions, 'Non unique solution, wrong number of few conditions'

def nonlin_IVP(xmin, xmax, vect_initial, print=True):

    #Check for valid input information
    ## Column int Vect?
    if vect_initial.shape[1] != 1 or len(vect_initial.shape) != 2:
        print('Invalid input vector, not a column')
        exit(0)
    ## positive h?
    if vect_initial[0][0] <= 0:
        print('Requires a positive non 0 value for h')
        exit(0)
    ## Valid xmin and xmax?
    if xmax-xmin <= 0:
        print('xmax must be greater than xmin')
        exit(0)

    h = vect_initial[0][0]

    # Build the Stepping matrices
    matrix_step_neg = build_matrix_step_nonlin(-h, vect_initial)
    matrix_step_pos = build_matrix_step_nonlin(h, vect_initial)

    ## Create df_output Space

    df_output = pd.DataFrame(vect_initial.T)

    ## Do the positive steps first
    vect = vect_initial.copy()
    while vect[1][0] <= xmax + h:
        vect = np.matmul(matrix_step_pos, vect)
        df_output.loc[len(df_output)] = vect.T[0]

    ## Do the negative steps next
    vect = vect_initial.copy()
    while vect[1][0] > xmin:
        vect = np.matmul(matrix_step_neg, vect)
        df_output.loc[len(df_output)] = vect.T[0]

    # Reprder
    df_output.sort_values(1, inplace=True)

    if print == True:
        ## Plot df_output
        plt.plot(df_output[1], df_output[2])
        plt.title(str(vect_initial.T))
        plt.show()

    return df_output
def build_matrix_step_nonlin(h, vect_initial):
    base_matrix = np.identity(len(vect_initial))
    base_matrix[1][0] = np.sign(h)
    for i in range(2, len(base_matrix)):
        for j in range(i, len(base_matrix)):
            base_matrix[i][j] = h ** (j - i) / math.factorial(j - i)
    matix_step = base_matrix
    return matix_step
def find_best_initial_nonlin(search_min, search_max, search_step, vect_initial, condition_vect):
    # Find All Possible initial vectors
    search_permutations = [*it.combinations_with_replacement(np.arange(search_min, search_max + search_step, search_step),
                                            np.count_nonzero(np.isnan(vect_initial)))]
    nan_location = np.argwhere(np.isnan(vect_initial))
    nan_location = nan_location[nan_location != 0]
    vect_initial_poss = np.zeros((len(search_permutations), len(vect_initial)))
    for i in range(0, len(search_permutations)):
        vect_initial_poss[i] = vect_initial.T
        vect_initial_poss[i][nan_location] = search_permutations[i]
        ## This ensures only valid input vectors are met
        vect_initial_poss[i] = adjustment_funtion(vect_initial_poss[i].T)


    ## List of scores
    ls_distance = []

    # Run over range of possible vectors
    for vect_initial in vect_initial_poss:
        output = nonlin_IVP(xmin,xmax, np.reshape(vect_initial, (len(vect_initial), 1)), False)
        ls_distance_single = []
        h = vect_initial[0]
        for vect in condition_vect:
            index = int(math.floor((vect[1] - xmin) / h))
            dist = np.linalg.norm(np.nan_to_num(output.T[index] - vect))
            ls_distance_single.append(dist)
            ls_distance_single.append(dist)
        ls_distance.append(sum(ls_distance_single))

    df_results = pd.DataFrame(vect_initial_poss)
    df_results['dist'] = ls_distance

    best_index = df_results[['dist']].idxmin()

    best_int_vect = vect_initial_poss[best_index].T
    dist = df_results['dist'][best_index].values
    return (best_int_vect, dist, df_results)


if check_well_posedness(condition_vect)[0]:
    conditions = check_well_posedness(condition_vect)[1]
    (vector_initial_best, dist, df_results) = find_best_initial_nonlin(search_min, search_max, search_step, condition_vect[0].T, condition_vect)

    df_output = nonlin_IVP(xmin, xmax, vector_initial_best)

else:
    (bool, conditions, message)  = check_well_posedness(condition_vect)
    print(message)