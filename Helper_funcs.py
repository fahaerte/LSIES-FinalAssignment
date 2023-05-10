import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy import distance
from datetime import time, datetime
from csv import reader

columns_sensors_positions = ['sensor_name', 'latitude', 'longitude', 'region']
folders = ['region_1_mustamäe_kristiine', 'region_2_data_kesklinn', 'region_3_kadriorg_lasnamäe',
           'region_4_ülemiste']
start_time = datetime.strptime('2022.08.01 00:00:00', '%Y.%m.%d %H:%M:%S')
end_time = datetime.strptime('2022.08.13 23:59:00', '%Y.%m.%d %H:%M:%S')


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
            tcoord = stations_df['coords'][sidd]
            ds.loc[sidd]['d'] = float(distance.distance(single, tcoord).meters)

        group = ds.nsmallest(4, 'd')
        group = group[group['d'] > min_distance]
        group = group[group['d'] < max_distance]
        group = list(group.index)

        groups_df.loc[sid]['grp'] = group

    return groups_df.copy()

################ interpolation task functions ################
#------------------------------------------------------------#
def calculate_correlations(df_data1, df_data2):
    # Function finds intervals of overlapping data, that was actually
    # measured by stations in group. The function then calculates 
    # correlations of measured data for lags from interval [-10; 10].
    # These correlations are then averaged over for each lag value 

    # Function returns chosel lag and correlation coefficient for each partner
    # station in the group
    df_indata = pd.DataFrame(columns=['d1', 'd2'], index=df_data1.index)
    # 1) find datapoints present in both datasets with matching timestamps
    # mindexer = df_data1.notna and df_data2.notna
    df_indata['d1'] = df_data1

    corrs = []
    for lag in range(-10, 10):
        df_indata['d2'] = df_data2.shift(periods=lag)
        temp = df_indata.corr(min_periods=5)
        # print(temp['d1']['d2'])
        corrs.append(temp['d1']['d2'])

    return corrs


def calculate_linear_regression(df_data1, df_data2, lag):
    # calculates beta and alpha for interpolation
    # y = beta*x + alpha
    df_indata = pd.DataFrame(columns=['d1', 'd2'], index=df_data1.index)
    df_indata['d1'] = df_data1
    df_indata['d2'] = df_data2.shift(periods=lag)
    tcov = df_indata.cov(min_periods = 5)
    tcov = df_indata.cov(min_periods = 5)
    tvar = df_indata['d2'].var()
    beta = tcov['d2']['d1']/tvar
    alpha = df_indata['d1'].mean() - beta*df_indata['d2'].mean()
    return beta, alpha


def init_ax_resamp(ax, resample_pers, rmse_values):
    ax.set_xlabel('Resample period [mins]')
    ax.set_ylabel('RMSE', color='r')
    ax.plot(resample_pers, rmse_values[0], color='r')
    ax.plot(resample_pers, rmse_values[1], color='b')
    ax.plot(resample_pers, rmse_values[2], color='g')

    ax.grid(True)
    return ax

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
        df_resamp_d = df_compare_d.resample(str(per) + 'T').mean()
        df_resamp_e = df_compare_e.resample(str(per) + 'T').mean()
        df_resamp_n = df_compare_n.resample(str(per) + 'T').mean()

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
    
    print(df_orig_d.shape[0])

    if (resample_vars_d[1].isna().sum() == df_orig_d.shape[0]):
        resample_vars_d[1] = np.zeros(len(sensor_names))
        resample_vars_n[1] = np.zeros(len(sensor_names))
        resample_vars_e[1] = np.zeros(len(sensor_names))

    result_list = [resample_vars_d, resample_vars_e, resample_vars_n]

    return result_list

def fill_missing_values(df_data1, df_data2, lags, betas, alphas):
    # fills missing values in df_data1 by averaging linear interpolations
    # defined by:
    # y = beta*x + alpha, where
    #   - x     ... are columns of df_data2
    #   - betas ... is list of beta coefficients, their order
    # corresponding to order of df_data2 columns
    #   - alphas ... is list of alpha coefficients, their order
    # corresponding to order of df_data2 columns
    # Function returns DataFrame with filled with available data
    # and RMSE between interpolated data and available original data
    idx = 0
    for label in df_data2.columns:
        df_data2[label] = df_data2[label].shift(periods=lags[idx]) * betas[idx] + alphas[idx]
        idx = idx + 1
        
    df_fill = df_data2.apply(np.mean, axis=1)
    indexer = (df_data1.isna()) & (df_fill.notna())
    rmsev = []
    df_err = (df_fill - df_data1)**2
    rmsev.append(np.sqrt(df_err.mean()))
    
    for label in df_data2.columns:
        df_err = (df_data2[label] - df_data1)**2
        rmsev.append(np.sqrt(df_err.mean()))
    
    df_data1[indexer] = df_fill[indexer]
    
    return df_data1.copy(), rmsev

def fill_missing_values_assim(df_data1, df_data2, lags, betas, alphas):
    # fills missing values in df_data1 by kalman filter assimilation
    # of linear interpolations
    # defined by:
    # y = beta*x + alpha, where
    #   - x     ... are columns of df_data2
    #   - betas ... is list of beta coefficients, their order
    # corresponding to order of df_data2 columns
    #   - alphas ... is list of alpha coefficients, their order
    # corresponding to order of df_data2 columns
    # Function returns DataFrame with filled with available data
    # and RMSE between interpolated data and available original data
    idx = 0
    rmsev = np.zeros(len(alphas))
    #) calculate linear regression + rmse for each member separately
    for label in df_data2.columns:
        df_data2[label] = df_data2[label].shift(periods=lags[idx]) * betas[idx] + alphas[idx]
        
        df_err = (df_data2[label] - df_data1)**2
        rmsev[idx] = np.sqrt(df_err.mean())
        
        idx = idx + 1
    #) use rmse as approximation of variation for kalman coefficients  
    df_fill = df_data2.apply(lambda row: apply_assimilation(row, rmsev), axis=1)
    
    indexer = (df_data1.isna()) & (df_fill.notna())
    
    #) get rmse for reference
    df_err = (df_fill - df_data1)**2    

    rmsevv          = np.zeros(len(rmsev) + 1)
    rmsevv[-1]      = (np.sqrt(df_err.mean()))

    idx = 0
    for idx in range(0, len(rmsev)):
        rmsevv[idx]    = rmsev[idx]
    
    df_data1[indexer] = df_fill[indexer]
    
    return df_data1.copy(), rmsevv

def apply_assimilation(row, vars):
    #) function is meant to be used with 'pd.DataFrame.apply' method
    #) call: df.apply(lambda row: apply_assimilation(row, vars), axis=1)
    #) row ... row of a dataframe filled with input data for assimilation
    #) vars ... np.array() of approximated variances 
    # for each data source. The order of columns in 
    # DataFrame must corespond to order of variances
    
    # 1) calculate the kalman coefficient, but only 
    #) for the present values -> different coefficient
    #) if only 2 values available, or all 3
    if row.isna().sum() == len(row):
        return np.nan
    rmt= vars[row.notna()]
    rmt1 = np.zeros([len(rmt)])
    idxs = np.arange(len(rmt))
    for i in idxs:
        rmt1[i] = np.prod(rmt[idxs != i])

    k_kal = (rmt1) / sum(rmt1)
        
    row = row.dropna()
    row = row * k_kal
    
    #2 ) return assimilated data
    return row.sum()

##################### indexing functions #####################
#------------------------------------------------------------#

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





def import_sensor_positions(dir_path, file):
    sensor_positions_df = pd.DataFrame(columns=columns_sensors_positions, index=[0])
    with open(dir_path + file, 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            data = {columns_sensors_positions[0]: row[1],
                    columns_sensors_positions[1]: float(row[0].split(' ')[0].replace('(', '')),
                    columns_sensors_positions[2]: float(row[0].split(' ')[1].replace(')', '')),
                    columns_sensors_positions[3]: ''}
            data_df = pd.DataFrame(data, columns=columns_sensors_positions, index=[0])
            sensor_positions_df = pd.concat([sensor_positions_df, data_df])

    sensor_positions_df = sensor_positions_df.dropna()

    for folder in folders:
        for file in os.listdir(dir_path + folder):
            sensor_name = file.split('-')[0]
            region = int(folder.split('_')[1])
            sensor_positions_df.loc[sensor_positions_df.sensor_name == sensor_name, 'region'] = region

    sensor_positions_df = sensor_positions_df.drop_duplicates(subset='sensor_name', keep="last")
    return sensor_positions_df


def import_sensor_data(dir_path):
    list_df = []
    sensor_names = []
    region_list = []

    for folder in folders:
        for file in os.listdir(dir_path + folder):
            sensor_name = file.split('-')[0]
            region_list.append(int(folder.split('_')[1]))
            sensor_names.append(file.split('-')[0])
            df = pd.read_csv(dir_path + folder + "/" + file, index_col=None, header=0)
            df['Time'] = df.apply(lambda row: datetime.strptime(row['Time'], '%Y-%m-%d %H:%M:%S'), axis=1)
            df.rename(columns={'dt_sound_level_dB': sensor_name}, inplace=True)
            list_df.append(df)

    tindex = pd.date_range(start_time, end_time, freq='1min')

    df_data_incomplete = pd.DataFrame(index=tindex, columns=sensor_names)

    idx = 0
    for df in list_df:
        # get rid of redundant datapoints
        df = df[df.Time >= start_time]
        df = df[df.Time <= end_time]
        df.drop_duplicates(subset='Time', keep='first', inplace=True)

        # index data by Time
        df.index = pd.to_datetime(df['Time'])
        df.drop(columns=['Time'], inplace=True)
        df = df.reindex(tindex)
        df_data_incomplete[sensor_names[idx]] = df[sensor_names[idx]]
        idx = idx + 1

    return df_data_incomplete


def compute_period_indices(df, start, end):
    if pd.to_datetime(end).hour < pd.to_datetime(pd.to_datetime(start)).hour:
        return np.where(np.logical_or(
            (df.index.time > pd.to_datetime(start).time()),
            (df.index.time <= pd.to_datetime(end).time())))[0]
    else:
        return np.where(np.logical_and(df.index.time > pd.to_datetime(start).time(),
                     df.index.time <= pd.to_datetime(end).time()))[0]


def get_sensors_based_on_region(df):
    d = {'sensor names': df['sensor_name'], 'region': df['region']}
    sensors = pd.DataFrame(d)
    sensors_region1 = sensors[sensors['region'] == 1]['sensor names']
    sensors_region2 = sensors[sensors['region'] == 2]['sensor names']
    sensors_region3 = sensors[sensors['region'] == 3]['sensor names']
    sensors_region4 = sensors[sensors['region'] == 4]['sensor names']
    return sensors_region1, sensors_region2, sensors_region3, sensors_region4
