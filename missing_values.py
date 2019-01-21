#!/usr/bin/env python
# coding: utf-8
import numpy as np
from math import isnan as isnan_np
import matplotlib.pyplot as plt

def drop(df, del_list, axis=0):
    '''axis: 0 = row, 1 = column'''
    if axis == 0:
        rows = list(range(len(df)))
        for i in del_list:
            del rows[rows.index(i)]
        return df.iloc[rows, :]
    elif axis == 1:
        cols = list(df.columns)
        for i in del_list:
            del cols[cols.index(i)]
        return df.loc[:, cols]

def isnan(number):
    if str(number) == 'nan' and isnan_np(number) == True:
        return True
    else:
        return False

def isna(series):
    a = []
    for i in series:
        a.append(isnan(i))
    return a

def mean(series):
    summ = 0
    n = 0
    for value in series:
        if isnan(value):
            continue
        summ += value
        n += 1
    return summ / n
    
def median(series): 
    l = []
    for i in series:
        if isnan(i):
            continue
        l.append(i)
    l.sort()
    return l[len(l)//2]

def mod(series):
    l = {}
    for i in series:
        if isnan(i):
            continue
        if i in l:
            l[i]+=1
        else:
            l[i] = 0
    m_k = list(l.keys())[0]
    m_v = 0
    for key in l:
        if l[key] > m_v:
            m_k = key
            m_v = l[key]
    return m_k

def unknown(series):
    return 'unknown'

def replace(df, columns, mode='mean', inplace=False):
    '''mode: mean, median, mod, unknown'''
    if mode == 'mean':
        f = mean
    elif mode == 'median':
        f = median
    elif mode == 'mod':
        f = mod
    elif mode == 'unknown':
        f = unknown
    else:
        raise AttributeError(f'Wrong mode, got {mode}')
    
    if not inplace:
        df = df.copy(deep=True)
        
    for col in columns:
        replace_value = f(df[col])
        change_list = [row for row in df.index if isnan(df.loc[row, col])]
        df.loc[change_list, col] = replace_value
            
    if not inplace:
        return df       
    
def replace_LR(df, target, v_columns=None):        
    if v_columns is None:
        v_columns = list(df.columns)
        del v_columns[v_columns.index(target)]
        
    nan_list = [row for row in df.index if isnan(df.loc[row, column])]
    valid_list = [row for row in df.index if not isnan(df.loc[row, column])]
    
    for col in v_columns:
        valid_list = [row for row in valid_list if not isnan(df.loc[row, col])]
        
    if len(nan_list) * 2 > len(valid_list):
        raise ValueError('Too holey dataset')
        
    model = LinReg()
    model.fit(df.iloc[valid_list, v_columns], df.iloc[valid_list, column])
    df.iloc[nan_list, v_columns] =  model.predict(df.iloc[nan_list, v_columns])
    

def plot_nan(df):
    fig, ax = plt.subplots(4,1, figsize=(20,8))
    a = np.array([isna(df[i]) for i in df.columns])
    n = len(a[0]) // 4 + 1
    for i in range(4):
        ax[i].matshow(a[:,i * n:(i+1) * n])
        
def dist_obj(df, number):
    w = []
    for i in df:
        if i == number:
            w.append(0)
        else:
            w.append(1)
    return np.array(w)

def dist_numb(df, number):      
    if isnan(number):
        return np.array([1] * len(df))
    return np.array(mashtab([(i - number)**2 for i in df]))

def mashtab(series):
    minimum = min(series)
    maximum = max(series)
    d = maximum - minimum
    return [1 if isnan(row) else (row - minimum) / d for row in series]

def standart(series):
    mat_waiting = mean(series)
    d = np.std(series, ddof=1)
    return [(i - mat_waiting) / d for i in series]
        
def knn(df, column, v_columns=None, weight=None, mode=None, n_neighbours=None, inplace=False):
    if inplace:
        df = df.copy(deep=True)
    
    # a couple of parametrs initializing
    if v_columns is None:
        v_columns = list(df.columns)
        del v_columns[v_columns.index(column)]
        
    if weight:
        weight = {v_columns[col] : weight[col] for col in range(len(v_columns))}
    else:
        weight = {col : 1./len(v_columns) for col in v_columns}
        
    if mode is None:
        mode = mod
    elif mode == 'mean':
        mode = mean
    elif mode == 'median':
        mode = median
    elif mode == 'mod':
        mode = mod
    else:
        raise AttributeError(f'Wrong mode, got {mode}')
    
    if n_neighbours is None:
        n_neighbours = len(df)//10
    elif n_neighbours > len(df)//2:
        raise AttributeError(f'n_neighbours can\' be more than len(df)//2, got {n_neighbours}')
    
    # end of parametr initializing
        
    nan_list = [row for row in df.index if isnan(df.loc[row, column])]
    valid_list = [row for row in df.index if not isnan(df.loc[row, column])]
    for row in nan_list:
        dist_list = []
        for col in v_columns:
            
            # initialize distance function
            
            if str(df[col].dtype) in ['object', 'boolean']:
                f = dist_obj
            elif str(df[col].dtype) in ['float', 'float32', 'float64', 'int', 'int32', 'int64', 'uint8']:
                f = dist_numb
            else:
                raise ValueError('Unnknown data type')
            a = f(df.loc[valid_list, col], df.loc[row, col])
            dist_list.append(a * weight[col])
        else:
            
            # pick neighbours
            
            dist_list = np.array(dist_list).transpose()
            dist_list = [(sum(dist_list[row])**0.5, valid_list[row]) for row in range(len(valid_list))]
            dist_list.sort(key=lambda x: x[0])
            dist_list = np.array(dist_list[:n_neighbours])
            # uncomment to see neighbours distance and number
#             print(row)
#             for i in dist_list:
#                 print(f'{i[0]:4.2f}\t{i[1]:.0f}')
#             print('\n')
            
            # nearest neighbour has largest weight
    
            w_df = []
            for i in dist_list:
                if i[0] == 0:
                    df.loc[row, column] = df.loc[i[1], column]
                    break
                w_df += [df.loc[i[1], column]] * int(round(1 / i[0] * 100, 0))
            else:
                df.loc[row, column] = mode(w_df)
                
        if inplace:
            return df