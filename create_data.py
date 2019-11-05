# -------------------------------------
# IMPORTS
# -------------------------------------

from os.path import expanduser
from types import SimpleNamespace
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.utils_initial_dataset import InitialDataset
from lib.utils_calc_psi import plot_psi
from lib.utils_generate_data import generate_init_data_diverse
from lib.utils_generate_n import n_r, n_i
from lib.utils_generate_W import get_grid, gauss_dist
from lib.utils_general import diff_same_length

# -------------------------------------
# PARAMETERS
# -------------------------------------

args = SimpleNamespace(seed=42,
                       toy_data=False
                       )

size_dataset_toy = 150  # if run on GPU machine, set to 5000 (toydata 8)
size_dataset = 150000
size_dataset_test = 3000
size_dataset_gauss = 10

args.grid, args.stepsize = get_grid(half_width=5,
                                    nmbr_points=10000)

# -------------------------------------
# PATHS
# -------------------------------------

home = expanduser("~")
path_data = home + '/ml_data/data_initial_diverse/'
path_data_test = home + '/ml_data/data_test/'
path_data_toy = home + '/MEGA/MEGAsync/Projektarbeit/Code/' + 'toy_data/initial_data/'
path_data_gauss = 'toy_data/data_gauss/'
path_plots = 'plots/'

# -------------------------------------
# GAUSS DATA
# -------------------------------------

Potentials = {}
Potentials['idx'] = []
Potentials['k'] = []

W = 10 * gauss_dist(0, 1, args.grid) + 1
dW = diff_same_length(W, args.stepsize)

# nr,ni,k
for i, k in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    nr = n_r(W, dW, args.grid, k)
    ni = n_i(dW, nr, args.grid, k)
    np.save(path_data_gauss + 'n_real_' + str(i), nr)
    np.save(path_data_gauss + 'n_imag_' + str(i), ni)
    Potentials['idx'].append(i)
    Potentials['k'].append(k)

    df = pd.DataFrame(Potentials, columns=['idx', 'k'])
    export_csv = df.to_csv(r'' + path_data_gauss + 'k_values.csv', index=None, header=True)

csv_file = path_data_gauss + 'k_values.csv'
k_frame = pd.read_csv(csv_file)

# psis
for idx in range(size_dataset_gauss):
    print('dataset toy sample ', idx)

    nr = np.load(path_data_gauss + 'n_real_' + str(idx) + '.npy')
    ni = np.load(path_data_gauss + 'n_imag_' + str(idx) + '.npy')
    k = k_frame['k'][idx]

    _, _, psir, psii = plot_psi(args.grid,
                                args.stepsize,
                                n=nr + 1j * ni,
                                Amp=1,
                                k=k,
                                plot=False)

    np.save(path_data_gauss + 'psir_' + str(idx), psir)
    np.save(path_data_gauss + 'psii_' + str(idx), psii)

# hermitean psis
csv_file = path_data_gauss + 'k_values.csv'
k_frame = pd.read_csv(csv_file)

for idx in range(size_dataset_gauss):
    if idx % 100 == 0:
        print('dataset toy hermitean sample ', idx)

    nr = np.load(path_data_gauss + 'n_real_' + str(idx) + '.npy')
    ni = np.load(path_data_gauss + 'n_imag_' + str(idx) + '.npy')
    k = k_frame['k'][idx]

    _, _, psir, psii = plot_psi(args.grid,
                                args.stepsize,
                                n=nr,
                                Amp=1,
                                k=k,
                                plot=False)

    np.save(path_data_gauss + 'psir_hermitean_' + str(idx), psir)
    np.save(path_data_gauss + 'psii_hermitean_' + str(idx), psii)

# -------------------------------------
# GENERATE k, n
# -------------------------------------

# # dataset
# generate_init_data_diverse(size_dataset,
#                            path=path_data,
#                            x=args.grid,
#                            dx=args.stepsize)
#
# # test data
# generate_init_data_diverse(size_dataset_test,
#                            path=path_data_test,
#                            x=args.grid,
#                            dx=args.stepsize)
#
# toy data
# generate_init_data_diverse(size_dataset_toy,
#                            path=path_data_toy,
#                            x=args.grid,
#                            dx=args.stepsize)

# -------------------------------------
# GENERATE psi FOR n, k
# -------------------------------------
#
# # dataset
# csv_file = path_data + 'k_values.csv'
# k_frame = pd.read_csv(csv_file)
#
# for idx in range(size_dataset):
#     if idx%100 == 0:
#          print('dataset sample ', idx)
#
#     nr = np.load(path_data + 'n_real_' + str(idx) + '.npy')
#     ni = np.load(path_data + 'n_imag_' + str(idx) + '.npy')
#     k = k_frame['k'][idx]
#
#     _, _, psir, psii = plot_psi(args.grid,
#                                 args.stepsize,
#                                 n=nr + 1j * ni,
#                                 Amp=1,
#                                 k=k,
#                                 plot=False)
#
#     np.save(path_data + 'psir_' + str(idx), psir)
#     np.save(path_data + 'psii_' + str(idx), psii)
#
#
# # test data
# csv_file = path_data_test + 'k_values.csv'
# k_frame = pd.read_csv(csv_file)
#
# for idx in range(size_dataset_test):
#     if idx%100 == 0:
#          print('dataset test sample ', idx)
#
#     nr = np.load(path_data_test + 'n_real_' + str(idx) + '.npy')
#     ni = np.load(path_data_test + 'n_imag_' + str(idx) + '.npy')
#     k = k_frame['k'][idx]
#
#     _, _, psir, psii = plot_psi(args.grid,
#                                 args.stepsize,
#                                 n=nr + 1j * ni,
#                                 Amp=1,
#                                 k=k,
#                                 plot=False)
#
#     np.save(path_data_test + 'psir_' + str(idx), psir)
#     np.save(path_data_test + 'psii_' + str(idx), psii)


# # toy data
# csv_file = path_data_toy + 'k_values.csv'
# #k_frame = pd.read_csv(csv_file)
# k_frame = pd.read_csv(csv_file)
#
# for idx in range(size_dataset_toy):
#     if idx%100 == 0:
#         print('dataset toy sample ', idx)
#
#     nr = np.load(path_data_toy + 'n_real_' + str(idx) + '.npy')
#     ni = np.load(path_data_toy + 'n_imag_' + str(idx) + '.npy')
#     k = k_frame['k'][idx]
#
#     _, _, psir, psii = plot_psi(args.grid,
#                                 args.stepsize,
#                                 n=nr + 1j * ni,
#                                 Amp=1,
#                                 k=k,
#                                 plot=False)
#
#     np.save(path_data_toy + 'psir_' + str(idx), psir)
#     np.save(path_data_toy + 'psii_' + str(idx), psii)

# -------------------------------------
# GENERATE psi FOR HERMITEAN n, k
# -------------------------------------

# # dataset
# csv_file = path_data + 'k_values.csv'
# k_frame = pd.read_csv(csv_file)
#
# for idx in range(size_dataset):
#     if idx%100 == 0:
#          print('dataset hermitean sample ', idx)
#
#     nr = np.load(path_data + 'n_real_' + str(idx) + '.npy')
#     ni = np.load(path_data + 'n_imag_' + str(idx) + '.npy')
#     k = k_frame['k'][idx]
#
#     _, _, psir, psii = plot_psi(args.grid,
#                                 args.stepsize,
#                                 n=nr,
#                                 Amp=1,
#                                 k=k,
#                                 plot=False)
#
#     np.save(path_data + 'psir_hermitean_' + str(idx), psir)
#     np.save(path_data + 'psii_hermitean_' + str(idx), psii)
#
#
# # test data
# csv_file = path_data_test + 'k_values.csv'
# k_frame = pd.read_csv(csv_file)
#
# for idx in range(size_dataset_test):
#     if idx%100 == 0:
#          print('dataset test hermitean sample ', idx)
#
#     nr = np.load(path_data_test + 'n_real_' + str(idx) + '.npy')
#     ni = np.load(path_data_test + 'n_imag_' + str(idx) + '.npy')
#     k = k_frame['k'][idx]
#
#     _, _, psir, psii = plot_psi(args.grid,
#                                 args.stepsize,
#                                 n=nr,
#                                 Amp=1,
#                                 k=k,
#                                 plot=False)
#
#     np.save(path_data_test + 'psir_hermitean_' + str(idx), psir)
#     np.save(path_data_test + 'psii_hermitean_' + str(idx), psii)


# # toy data
# csv_file = path_data_toy + 'k_values.csv'
# k_frame = pd.read_csv(csv_file)
#
# for idx in range(size_dataset_toy):
#     if idx%100 == 0:
#         print('dataset toy hermitean sample ', idx)
#
#     nr = np.load(path_data_toy + 'n_real_' + str(idx) + '.npy')
#     ni = np.load(path_data_toy + 'n_imag_' + str(idx) + '.npy')
#     k = k_frame['k'][idx]
#
#     _, _, psir, psii = plot_psi(args.grid,
#                                 args.stepsize,
#                                 n=nr,
#                                 Amp=1,
#                                 k=k,
#                                 plot=False)
#
#     np.save(path_data_toy + 'psir_hermitean_' + str(idx), psir)
#     np.save(path_data_toy + 'psii_hermitean_' + str(idx), psii)

# -------------------------------------
# TEST DATA
# -------------------------------------
#
# csv_file = path_data_toy + 'k_values.csv'
# k_frame = pd.read_csv(home + '/MEGA/MEGAsync/Projektarbeit/Code/' + csv_file)
#
# print('PLOTTING SAMPLES.')
# plt.close()
# plt.figure(figsize=(12, 6))
# for i in range(6):
#     idx = np.random.randint(100)
#     nr = np.load(home + '/MEGA/MEGAsync/Projektarbeit/Code/' + path_data_toy + 'n_real_' + str(idx) + '.npy')
#     ni = np.load(home + '/MEGA/MEGAsync/Projektarbeit/Code/' + path_data_toy + 'n_imag_' + str(idx) + '.npy')
#     psir = np.load(home + '/MEGA/MEGAsync/Projektarbeit/Code/' + path_data_toy + 'psir_' + str(idx) + '.npy')
#     psii = np.load(home + '/MEGA/MEGAsync/Projektarbeit/Code/' + path_data_toy + 'psii_' + str(idx) + '.npy')
#     k = k_frame['k'][idx]
#
#     print('Plot for k = ', k)
#     plt.subplot(3, 2, i+1)
#     plt.plot(args.grid, nr, label='nr')
#     plt.plot(args.grid, ni, label='ni')
#     plt.plot(args.grid, psir, label='psir')
#     plt.plot(args.grid, psii, label='psii')
#
# plt.legend(loc='upper right')
# plt.savefig(home + '/MEGA/MEGAsync/Projektarbeit/Code/' + path_plots + 'CreateDataTest.pdf')
#
# print('PLOTTING HERMITEAN SAMPLES.')
# plt.close()
# plt.figure(figsize=(12, 6))
# for i in range(6):
#     idx = np.random.randint(100)
#     nr = np.load(home + '/MEGA/MEGAsync/Projektarbeit/Code/' + path_data_toy + 'n_real_' + str(idx) + '.npy')
#     psir = np.load(home + '/MEGA/MEGAsync/Projektarbeit/Code/' + path_data_toy + 'psir_hermitean_' + str(idx) + '.npy')
#     psii = np.load(home + '/MEGA/MEGAsync/Projektarbeit/Code/' + path_data_toy + 'psii_hermitean_' + str(idx) + '.npy')
#     k = k_frame['k'][idx]
#
#     print('Plot for k = ', k)
#     plt.subplot(3, 2, i+1)
#     plt.plot(args.grid, nr, label='nr')
#     plt.plot(args.grid, ni, label='ni')
#     plt.plot(args.grid, psir, label='psir')
#     plt.plot(args.grid, psii, label='psii')
#
# plt.legend(loc='upper right')
# plt.savefig(home + '/MEGA/MEGAsync/Projektarbeit/Code/' + path_plots + 'CreateDataTestHermitean.pdf')
