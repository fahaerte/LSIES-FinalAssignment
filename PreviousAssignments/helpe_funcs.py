import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy import distance
from datetime import time

def make_groups(IDs, stations_df):
    # IDs - list of station IDs
    # coordinates - array of station's coordinates, 
    #             - indexable by station ID
    # IDs and coordinates are indexed identically
    # function returns indexes of grouped stations
    from geopy import distance
    
    max_distance = 500  # maximum distance for grouped stations in meters
    min_distance = 0
    print("create groups of sensors based on distance between them")
    tdata = np.empty(len(IDs), 'float')
    
    ds = pd.DataFrame(data={'d': tdata}, index=IDs)
    tdata = [[] for _ in range(len(IDs))]
    groups_df = pd.DataFrame(data={'grp': tdata}, index=IDs)
    
    for sid in IDs:
        single = stations_df['coords'][sid]
        

        for sidd in IDs:
            tcoord = stations_df.loc[sidd]['coords']
            ds.loc[sidd]['d'] = float(distance.distance(single, tcoord).meters)

        group = ds.nsmallest(4, 'd')
        group = group[group['d'] > min_distance]
        group = group[group['d'] < max_distance]
        group = list(group.index)
        
        groups_df.loc[sid]['grp'] = group

        
    return groups_df.copy()
        
    
def calculate_correlations(df_data1, df_data2):
    # Function finds intervals of overlapping data, that was actually
    # measured by stations in group. The function then calculates 
    # correlations of measured data for lags from interval [-10; 10].
    # These correlations are then averaged over for each lag value 
    # { ur(lag) = mean(r(lag,i))) } and variance of these values is calculated
    # { var(ur) = E[ur(lag) - r(lag,i)] }. Used lag and correlation coeficcient
    # is chosen by optimizing = -ru(lag) + var(lag)
    
    # Function returns chosel lag and correlation coefficient for each partner
    # station in the group
    df_indata = pd.DataFrame(columns=['d1', 'd2'], index=df_data1.index)
    # 1) find datapoints present in both datasets with matching timestamps
    #mindexer = df_data1.notna and df_data2.notna
    df_indata['d1'] = df_data1
    
    corrs=[]
    for lag in range(-10, 10):
        df_indata['d2'] = df_data2.shift(periods=lag)
        temp = df_indata.corr(min_periods = 5)
        #print(temp['d1']['d2'])
        corrs.append(temp['d1']['d2'])
        
    return corrs

def calculate_linear_regression(df_data1, df_data2, lag):
    pass
    
    

def init_ax_resamp(ax, resample_pers, rmse_values):
  ax.set_xlabel('Resample period [mins]')
  ax.set_ylabel('RMSE', color='r')
  ax.plot(resample_pers, rmse_values[0], color='r')
  ax.plot(resample_pers, rmse_values[1], color='b')
  ax.plot(resample_pers, rmse_values[2], color='g')
  
  ax.grid(True)
  return ax


def indexer_day(dafr):
    day_str = time.fromisoformat('07:00:00')
    day_stp = time.fromisoformat('18:59:59') 
    return dafr.index.indexer_between_time(day_str, day_stp)

def indexer_evening(dafr):
    evening_str = time.fromisoformat('19:00:00')
    evening_stp = time.fromisoformat('22:59:59')
    return dafr.index.indexer_between_time(evening_str, evening_stp)

def indexer_night(dafr):
    night_str = time.fromisoformat('23:00:00')
    night_stp = time.fromisoformat('06:59:59')
    return dafr.index.indexer_between_time(night_str, night_stp)

def evaluate_resample(df_orig, df_compare):
    resample_pers = [1, 2, 5, 10, 15, 20, 30, 60]
    
    sensor_names = list(df_orig.columns)
    result_list = []
    
    resample_vars_d = pd.DataFrame(data=np.zeros([len(sensor_names), len(resample_pers)]), columns=resample_pers, index=sensor_names)
    resample_vars_e = resample_vars_d.copy()
    resample_vars_n = resample_vars_d.copy()

    df_orig_d = df_orig.iloc[indexer_day(df_orig)]
    df_orig_e = df_orig.iloc[indexer_evening(df_orig)]
    df_orig_n = df_orig.iloc[indexer_night(df_orig)]

    df_compare_d = df_compare.iloc[indexer_day(df_compare)]
    df_compare_e = df_compare.iloc[indexer_evening(df_compare)]
    df_compare_n = df_compare.iloc[indexer_night(df_compare)]

    df_resamp_d = pd.DataFrame()
    df_resamp_e = pd.DataFrame()
    df_resamp_n = pd.DataFrame()

    sqrt_lam = lambda x: np.sqrt(x)

    for per in resample_pers:
        # downsample data - get mean
        df_resamp_d = df_compare_d.resample(str(per)+'T').mean()
        df_resamp_e = df_compare_e.resample(str(per)+'T').mean()
        df_resamp_n = df_compare_n.resample(str(per)+'T').mean()
        
        # unpsample data - simulate using the averages for all data points
        df_resamp_d = df_resamp_d.resample('1T').ffill()
        df_resamp_e = df_resamp_e.resample('1T').ffill()
        df_resamp_n = df_resamp_n.resample('1T').ffill()
        
        # calculate errors   
        df_resamp_d = ((df_orig_d - df_resamp_d) ** 2)
        df_resamp_e = ((df_orig_e - df_resamp_e) ** 2)
        df_resamp_n = ((df_orig_n - df_resamp_n) ** 2)
        
        # calculate means
        df_resamp_d = df_resamp_d.mean()
        df_resamp_e = df_resamp_e.mean()
        df_resamp_n = df_resamp_n.mean()
        
        # calculate square roots
        resample_vars_d[per] = df_resamp_d.apply(np.sqrt)
        resample_vars_e[per] = df_resamp_e.apply(np.sqrt)
        resample_vars_n[per] = df_resamp_n.apply(np.sqrt)
        
        # downsample and get means
        
        
        
    if (resample_vars_d[0].isna().sum() == df_orig_d.shape[0]): 
        resample_vars_d[0]=np.zeros(len(sensor_names)) 
        resample_vars_n[0]=np.zeros(len(sensor_names)) 
        resample_vars_e[0]=np.zeros(len(sensor_names)) 
        
    result_list = [ resample_vars_d, resample_vars_e, resample_vars_n ]
    
    return result_list
