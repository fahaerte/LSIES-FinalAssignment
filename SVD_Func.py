import numpy as np
import matplotlib.pyplot as plt

color1 = 'r'
color2 = 'b'


def print_values(rmse_values, sing_values, steps):
    print()
    for rank in range(len(rmse_values)):
        if rank % steps == 0:
            print(f'Rank: {rank + 1} RMSE: {rmse_values[rank]} Amount Singular Values: {sing_values[rank]}')


def calc_RMSE_sinVal_based_on_time_period(U, s, Vt, df, period_index, print_values=False):
    rank_range = range(1, len(df.columns) + 1)
    rmse_list = []
    amount_sing_values = []

    for rank in rank_range:
        Sigma = np.zeros((df.shape[0], df.shape[1]))
        Sigma[:rank, :rank] = np.diag(s[:rank])
        df_recon = U.dot(Sigma.dot(Vt))

        rmse = np.sqrt(np.mean((df.iloc[period_index] - df_recon[period_index]) ** 2))
        rmse_list.append(rmse.mean())
        amount_sing_values.append(Sigma[rank - 1][rank - 1])

    if print_values:
        print_values(rmse_list, amount_sing_values, 2)

    return rmse_list, amount_sing_values


def calc_RMSE_sinVal(data):
    U, s, Vt = np.linalg.svd(data)
    return calc_RMSE_sinVal(U, s, Vt, data)


def init_ax(ax, rank_values, rmse_values, sing_values, color1, color2):
    ax.set_xlabel('Rank')
    ax.set_ylabel('Singular Values', color=color1)
    ax.plot(rank_values, sing_values, color=color1)
    ax2 = ax.twinx()
    ax2.set_ylabel('RMSE', color=color2)
    ax2.plot(rank_values, rmse_values, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax.grid(True)
    ax2.grid(False)
    return ax


def plot_rmse_sinval_global_given_SVD(U, s, Vt, df):
    rmse_list, amount_sing_values = calc_RMSE_sinVal(U, s, Vt, df)
    plot_rmse_sinval(rmse_list, amount_sing_values)


def plot_rmse_sinval(rmse_list, amount_sing_values, title):
    rank_range_all_regions = range(1, len(rmse_list) + 1)
    amount_sing_values[0] = amount_sing_values[1] * 2

    fig, ax = plt.subplots()
    ax = init_ax(ax, rank_range_all_regions, rmse_list, amount_sing_values, color1, color2)
    ax.set_title(title)
    plt.show()


def plot_rmse_sinval_regions(data, sensors_region1, sensors_region2, sensors_region3, sensors_region4):
    rmse_list_reg1, amount_sing_val_reg1 = calc_RMSE_sinVal(data[sensors_region1])
    rmse_list_reg2, amount_sing_val_reg2 = calc_RMSE_sinVal(data[sensors_region2])
    rmse_list_reg3, amount_sing_val_reg3 = calc_RMSE_sinVal(data[sensors_region3])
    rmse_list_reg4, amount_sing_val_reg4 = calc_RMSE_sinVal(data[sensors_region4])

    plot_rmse_sinval_regions(rmse_list_reg1, amount_sing_val_reg1, rmse_list_reg2, amount_sing_val_reg2, rmse_list_reg3,
                             amount_sing_val_reg3, rmse_list_reg4, amount_sing_val_reg4)


def plot_rmse_sinval_regions(rmse_list_reg1, amount_sing_val_reg1, rmse_list_reg2, amount_sing_val_reg2, rmse_list_reg3,
                             amount_sing_val_reg3, rmse_list_reg4, amount_sing_val_reg4):
    rank_range_reg1 = range(1, len(rmse_list_reg1) + 1)
    rank_range_reg2 = range(1, len(rmse_list_reg2) + 1)
    rank_range_reg3 = range(1, len(rmse_list_reg3) + 1)
    rank_range_reg4 = range(1, len(rmse_list_reg4) + 1)

    amount_sing_val_reg1[0] = amount_sing_val_reg1[1] * 2
    amount_sing_val_reg2[0] = amount_sing_val_reg2[1] * 2
    amount_sing_val_reg3[0] = amount_sing_val_reg3[1] * 2
    amount_sing_val_reg4[0] = amount_sing_val_reg4[1] * 2

    fig, ax = plt.subplots(2, 2)
    ax[0, 0] = init_ax(ax[0, 0], rank_range_reg1, rmse_list_reg1, amount_sing_val_reg1, color1, color2)
    ax[0, 1] = init_ax(ax[0, 1], rank_range_reg2, rmse_list_reg2, amount_sing_val_reg2, color1, color2)
    ax[1, 0] = init_ax(ax[1, 0], rank_range_reg3, rmse_list_reg3, amount_sing_val_reg3, color1, color2)
    ax[1, 1] = init_ax(ax[1, 1], rank_range_reg4, rmse_list_reg4, amount_sing_val_reg4, color1, color2)

    ax[0, 0].set_title('Region 1')
    ax[0, 1].set_title('Region 2')
    ax[1, 0].set_title('Region 3')
    ax[1, 1].set_title('Region 4')

    fig.tight_layout()
    plt.show()


def keep_low_rank_model_of_svd(rank, df, s):
    s_new = np.zeros((df.shape[0], df.shape[1]))
    s_new[:rank, :rank] = np.diag(s[:rank])
    return s_new


def apply_svd(df, sensor, u, sigma, vt):
    sensor_index = df.columns.get_loc(sensor)
    df_recon = u.dot(sigma.dot(vt))
    reconstructed_sensor = df_recon[:, sensor_index]
    return reconstructed_sensor


def compare_results(stat, dic_element, data, data_recon):
    rmse = np.sqrt(np.mean((data - data_recon) ** 2)).mean()
    mad = np.median(abs(data - np.median(data_recon)))
    stat[dic_element + '_rmse'] = rmse
    stat[dic_element + '_mad'] = mad
    return stat


def random_walk_uni(length, a, b):
    data = np.zeros(length)
    noise = a + (b - a) * np.random.random(length)
    for i in range(1, length):
        if noise[i] > 0:
            data[i] = data[i - 1] + noise[i]
        else:
            i = i - 1
    data = data / max(data)
    return data
