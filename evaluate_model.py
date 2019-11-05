# ---------------------------------------------
# IMPORTS
# ---------------------------------------------

# libraries
from os.path import expanduser
from types import SimpleNamespace
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
import matplotlib as mpl

from lib.utils_ODEOpt import CIDataset

# own files
from lib.utils_generate_W import get_grid
from lib.utils_initial_dataset import ToTensor, calc_mean_std
from lib.utils_train import get_dataloaders
from lib.utils_test_model import Norm, DN, plot_bars, R
from lib.utils_calc_psi import init_val_in, get_psi
from lib.LSTMFFout import LSTMFFout
from lib.UNet_featurek import UNet_featurek
from lib.utils_ODEMultistep import WrapperODEMultistep

# ---------------------------------------------
# PARAMETERS
# ---------------------------------------------

args = SimpleNamespace(batch_size=6,
                       normalize=True,
                       plot_hermitian_calculated=False,
                       plot_mean_median_bars=False,
                       plot_reflections=False,
                       plot_matrices=False,
                       plot_predictions=True,
                       testset_size=3000,
                       device='cpu',  # torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       which_model='LSTMFFout',
                       toy_data=False,
                       half_width=5,
                       nmbr_points=10000,
                       amp=1.0
                       )
args.grid, args.stepsize = get_grid(args.half_width, args.nmbr_points)
print('Device: ', args.device)

# ---------------------------------------------
# PATHS
# ---------------------------------------------

home = expanduser("~")
path_model = 'models/' + args.which_model + '.pt'
if not args.toy_data:
    path_data_test = home + '/ml_data/data_test/'
    path_here = home + '/MEGAsync/Projektarbeit/Code/'
else:
    path_here = home + '/MEGA/MEGAsync/Projektarbeit/Code/'
    path_data_test = path_here + 'toy_data/initial_data/'
    args.testset_size = 150
path_plots = 'plots/'

# ---------------------------------------------
# PLOT CONFIGURATIONS
# ---------------------------------------------

# Change to the directory which contains the current script
# dirFile = os.path.dirname(os.path.join(path_here,
#                                        'evaluate_model.py'))
# Load style file
plt.style.use('file://' + path_here + 'PaperDoubleFig.mplstyle')
# Make some style choices for plotting
colourWheel = ['#329932',
               '#ff6961',
               'b',
               '#6a3d9a',
               '#fb9a99',
               '#e31a1c',
               '#fdbf6f',
               '#ff7f00',
               '#cab2d6',
               '#6a3d9a',
               '#ffff99',
               '#b15928',
               '#67001f',
               '#b2182b',
               '#d6604d',
               '#f4a582',
               '#fddbc7',
               '#f7f7f7',
               '#d1e5f0',
               '#92c5de',
               '#4393c3',
               '#2166ac',
               '#053061']
dashesStyles = [[3, 1],
                [1000, 1],
                [2, 1, 10, 1],
                [4, 1, 1, 1, 1, 1]]

# ---------------------------------------------
# CREATE TEST DATASET
# ---------------------------------------------

if args.normalize:
    args.nr_mean, args.nr_std, \
    args.ni_mean, args.ni_std = calc_mean_std(args.testset_size,
                                              path_data_test)
    transform = transforms.Compose([Norm(args),
                                    ToTensor()
                                    ])
else:
    transform = transforms.Compose([ToTensor()
                                    ])

testset = CIDataset(csv_file=path_data_test + 'k_values.csv',
                    root_dir=path_data_test,
                    transform=transform
                    )

# ---------------------------------------------
# CREATE TEST DATALOADER
# ---------------------------------------------

test_loader, _ = get_dataloaders(testset,
                                 batch_size=args.batch_size,
                                 validation_split=0,
                                 shuffle_dataset=False,
                                 random_seed=42)

# ---------------------------------------------
# LOAD MODEL
# ---------------------------------------------

print('Loading model...')
if args.which_model == 'ODEMultistep':
    model = WrapperODEMultistep(path_model=path_model,
                                args=args)
else:
    model = torch.load(path_model, map_location=args.device)
    model.device = args.device
model.to(args.device)
print('Model on CUDA: ', next(model.parameters()).is_cuda)
model.eval()

# ---------------------------------------------
# INITIAL VALUES
# ---------------------------------------------

# create all initial values
print('\nCREATING INITIAL VALUES.')
iv = {}
for i in range(1, 11):
    iv[i] = init_val_in(x=args.grid,
                        dx=args.stepsize,
                        k=i,
                        A=1)

print('Initial Values for k=1: ', iv[1])

# ---------------------------------------------
# PREDICTION PLOTS HERMITIAN
# ---------------------------------------------

if args.plot_hermitian_calculated:

    print('\nPLOTTING HERMITIAN POTENTIAL.')

    plt.close('all')

    fig, ax = plt.subplots()

    alphaVal = 0.6
    linethick = 1

    idx = np.random.randint(args.testset_size)
    item = testset[idx]
    k_nr, ni = item['k_n_r'], item['n_i']
    k_nr, ni = DN(k_nr, 'r', args), DN(ni, 'i', args)
    psir_hermitian = item['psir_hermitian']
    psii_hermitian = item['psii_hermitian']

    plt.plot(args.grid,
             k_nr[1:].detach().cpu().numpy(),
             color=colourWheel[0 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[0 % len(dashesStyles)],
             lw=linethick,
             label=r'$n_r$',
             alpha=alphaVal)
    plt.plot(args.grid,
             psir_hermitian.detach().cpu().numpy(),
             color=colourWheel[1 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[1 % len(dashesStyles)],
             lw=linethick,
             label=r'$Re(\psi)$',
             alpha=alphaVal)
    plt.plot(args.grid,
             psii_hermitian.detach().cpu().numpy(),
             color=colourWheel[2 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[2 % len(dashesStyles)],
             lw=linethick,
             label=r'$Im(\psi)$',
             alpha=alphaVal)
    plt.plot(args.grid,
             (psir_hermitian ** 2 + psii_hermitian ** 2).detach().cpu().numpy(),
             color=colourWheel[3 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[3 % len(dashesStyles)],
             lw=linethick,
             label=r'$I$',
             alpha=alphaVal)

    ax.set_xlabel('')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.major.formatter._useMathText = True
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_label_coords(0.53, 1.01)
    ax.yaxis.tick_right()
    nameOfPlot = 'Hermitian Potential: Refractive Index, Wave Amplitude, k = {:.0f}'.format(k_nr[0])
    plt.ylabel(nameOfPlot, rotation=0)

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    legend = ax.legend(frameon=False,
                       loc='upper center',
                       bbox_to_anchor=(0.5, -0.05),
                       ncol=4)

    # ax.legend(frameon=False, loc='upper left', ncol=2, handlelength=4)

    fig.savefig(path_plots + 'onlynr.pdf', bbox_inches="tight", dpi=300)

    # ---------------------------------------------
    # FQ/MAX BARS HERMITIAN
    # ---------------------------------------------

    print('\nPLOTTING HERMITIAN FQ/MAX BARS.')

    hermit_mean_max_dev = np.zeros(10)
    hermit_max_devs = [[] for i in range(10)]
    how_many_k = np.zeros(10)

    for j, batch in enumerate(test_loader):

        knr = batch['k_n_r'].to(args.device)
        psir_hermitian = batch['psir_hermitian'].to(args.device)
        psii_hermitian = batch['psii_hermitian'].to(args.device)

        for one_knr, one_psir_hermitian, one_psii_hermitian in \
                zip(knr, psir_hermitian, psii_hermitian):
            k = int(
                np.around(
                    DN(one_knr[0].detach().cpu().numpy(), 'r', args)))

            how_many_k[k - 1] += 1
            Int = one_psir_hermitian ** 2 + one_psii_hermitian ** 2

            rel_err = torch.max(torch.abs(Int - args.amp) / args.amp)  # all of it
            hermit_max_devs[k - 1].append(rel_err)
            hermit_mean_max_dev[k - 1] += rel_err

        if j % 10 == 0:
            print('{}/{}'.format(j, len(test_loader)))

    hermit_mean_max_dev = hermit_mean_max_dev / how_many_k
    hermit_median_max_dev = [np.median(i) for i in hermit_max_devs]

    plot_bars(bar_heights=hermit_mean_max_dev,
              bar_labels=list(range(1, 11)),
              path=path_plots + 'bars_hermitian_dev_max.pdf',
              path_here=path_here,
              nameOfPlot='Hermitian Case: Mean of Maximal Deviatian from CI-Value, w.r.t. $k$')

    plot_bars(bar_heights=hermit_median_max_dev,
              bar_labels=list(range(1, 11)),
              path=path_plots + 'bars_hermitian_dev_max_median.pdf',
              path_here=path_here,
              nameOfPlot='Hermitian Case: Median of Maximal Deviatian from CI-Value, w.r.t. $k$')

    # ---------------------------------------------
    # FQ/REFL BARS HERMITiAN
    # ---------------------------------------------

    print('\nPLOTTING HERMITiAN FQ/REFL BARS.')

    hermit_mean_refl = np.zeros(10)
    hermit_refls = [[] for i in range(10)]
    how_many_k = np.zeros(10)

    for j, batch in enumerate(test_loader):

        knr = batch['k_n_r'].to(args.device)
        psir_hermitian = batch['psir_hermitian']
        psii_hermitian = batch['psii_hermitian']

        psi_hermitian = psir_hermitian.detach().cpu().numpy() + \
                        1j * psii_hermitian.detach().cpu().numpy()

        for one_knr, one_psi_hermitian in zip(knr, psi_hermitian):
            k = int(
                np.around(
                    DN(one_knr[0].detach().cpu().numpy(), 'r', args)))

            how_many_k[k - 1] += 1

            R_coeff = R(psi_xmin1=one_psi_hermitian[-1],
                        psi_x0=one_psi_hermitian[-2],
                        deltax=args.stepsize,
                        k=k)

            hermit_refls[k - 1].append(R_coeff)
            hermit_mean_refl[k - 1] += R_coeff

        if j % 10 == 0:
            print('{}/{}'.format(j, len(test_loader)))

    hermit_mean_refl = hermit_mean_refl / how_many_k
    hermit_median_refl = [np.median(i) for i in hermit_refls]

    plot_bars(bar_heights=hermit_mean_refl,
              bar_labels=list(range(1, 11)),
              path=path_plots + 'bars_hermitian_refl.pdf',
              path_here=path_here,
              nameOfPlot='Hermitian Case: Mean of Reflection Coefficient, w.r.t. $k$')

    plot_bars(bar_heights=hermit_median_refl,
              bar_labels=list(range(1, 11)),
              path=path_plots + 'bars_hermitian_refl_median.pdf',
              path_here=path_here,
              nameOfPlot='Hermitian Case: Median of Reflection Coefficient, w.r.t. $k$')

    # ---------------------------------------------
    # PREDICTION PLOTS CALCULATED NUMPY
    # ---------------------------------------------

    print('\nPLOTTING NUMPY CALCULATED.')

    plt.close('all')

    fig, ax = plt.subplots()

    alphaVal = 0.6
    linethick = 1
    alphaVal_ni = 1.
    linethick_ni = 2

    idx = np.random.randint(args.testset_size)

    item = testset[idx]
    k_nr, ni = item['k_n_r'], item['n_i']
    k_nr, ni = DN(k_nr, 'r', args), DN(ni, 'i', args)
    psir, psii = item['psir'], item['psii']

    plt.plot(args.grid,
             k_nr[1:].detach().cpu().numpy(),
             color=colourWheel[0 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[0 % len(dashesStyles)],
             lw=linethick,
             label=r'$n_r$',
             alpha=alphaVal)
    plt.plot(args.grid,
             ni.detach().cpu().numpy(),
             color=colourWheel[1 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[1 % len(dashesStyles)],
             lw=linethick,
             label=r'$n_i$',
             alpha=alphaVal_ni)
    plt.plot(args.grid,
             psir.detach().cpu().numpy(),
             color=colourWheel[2 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[2 % len(dashesStyles)],
             lw=linethick,
             label=r'$Re(\psi)$',
             alpha=alphaVal)
    plt.plot(args.grid,
             psii.detach().cpu().numpy(),
             color=colourWheel[3 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[3 % len(dashesStyles)],
             lw=linethick,
             label=r'$Im(\psi)$',
             alpha=alphaVal)
    plt.plot(args.grid,
             (psir ** 2 + psii ** 2).detach().cpu().numpy(),
             color=colourWheel[4 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[4 % len(dashesStyles)],
             lw=linethick_ni,
             label=r'$I$',
             alpha=alphaVal_ni)

    ax.set_xlabel('')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.major.formatter._useMathText = True
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_label_coords(0.58, 1.01)
    ax.yaxis.tick_right()
    nameOfPlot = 'CI Potential: Refractive Index, Wave Amplitude, k = {:.0f}'.format(k_nr[0])
    plt.ylabel(nameOfPlot, rotation=0)
    ax.legend(frameon=False,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.05),
              ncol=5)
    plt.savefig(path_plots + 'nilabel_numpy.pdf', bbox_inches="tight", dpi=300)

    # ---------------------------------------------
    # FQ/MAX BARS CALCULATED NUMPY
    # ---------------------------------------------

    print('\nPLOTTING NUMPY CALCULATED FQ/MAX BARS.')

    nilabel_mean_max_dev = np.zeros(10)  # tresh, k_real
    nilabel_max_devs = [[] for i in range(10)]
    how_many_k = np.zeros(10)

    for j, batch in enumerate(test_loader):

        knr = batch['k_n_r'].to(args.device)
        psir = batch['psir'].to(args.device)
        psii = batch['psii'].to(args.device)

        for one_knr, one_psir, one_psii in zip(knr, psir, psii):
            k = int(np.around(DN(one_knr[0].detach().cpu().numpy(), 'r', args)))

            how_many_k[k - 1] += 1
            Int = one_psir ** 2 + one_psii ** 2
            rel_err = torch.max(torch.abs(Int - args.amp) / args.amp)  # all of it
            nilabel_max_devs[k - 1].append(rel_err)
            nilabel_mean_max_dev[k - 1] += rel_err

        if j % 10 == 0:
            print('{}/{}'.format(j, len(test_loader)))

    nilabel_mean_max_dev = nilabel_mean_max_dev / how_many_k
    nilabel_median_max_dev = [np.median(i) for i in nilabel_max_devs]

    plot_bars(bar_heights=nilabel_mean_max_dev,
              bar_labels=list(range(1, 11)),
              path=path_plots + 'bars_nilabel_dev_max.pdf',
              path_here=path_here,
              nameOfPlot='Calculated Potential: Mean of Maximal Deviation from CI-Value, w.r.t. $k$')

    plot_bars(bar_heights=nilabel_median_max_dev,
              bar_labels=list(range(1, 11)),
              path=path_plots + 'bars_nilabel_dev_max_median.pdf',
              path_here=path_here,
              nameOfPlot='Calculated Potential: Median of Maximal Deviation from CI-Value, w.r.t. $k$')

    # ---------------------------------------------
    # FQ/REFL BARS CALCULATED NUMPY
    # ---------------------------------------------

    print('\nPLOTTING NUMPY CALCULATED FQ/REFL BARS.')

    nilabel_mean_refl = np.zeros(10)  # tresh, k_real
    nilabel_refls = [[] for i in range(10)]
    how_many_k = np.zeros(10)

    for j, batch in enumerate(test_loader):

        knr = batch['k_n_r'].to(args.device)
        psir = batch['psir'].to(args.device)
        psii = batch['psii'].to(args.device)

        psi = psir.detach().cpu().numpy() + 1j * psii.detach().cpu().numpy()

        for one_knr, one_psi in zip(knr, psi):
            k = int(np.around(DN(one_knr[0].detach().cpu().numpy(), 'r', args)))

            how_many_k[k - 1] += 1

            R_coeff = R(psi_xmin1=one_psi[-1],
                        psi_x0=one_psi[-2],
                        deltax=args.stepsize,
                        k=k)

            nilabel_refls[k - 1].append(R_coeff)
            nilabel_mean_refl[k - 1] += R_coeff

        if j % 10 == 0:
            print('{}/{}'.format(j, len(test_loader)))

    # print('how_many_k: ', how_many_k)

    nilabel_mean_refl = nilabel_mean_refl / how_many_k
    nilabel_median_refl = [np.median(i) for i in nilabel_refls]

    plot_bars(bar_heights=nilabel_mean_refl,
              bar_labels=list(range(1, 11)),
              path=path_plots + 'bars_nilabel_refl.pdf',
              path_here=path_here,
              nameOfPlot='Calculated Potential: Mean of Reflection Coefficient, w.r.t. $k$')

    plot_bars(bar_heights=nilabel_median_refl,
              bar_labels=list(range(1, 11)),
              path=path_plots + 'bars_nilabel_refl_median.pdf',
              path_here=path_here,
              nameOfPlot='Calculated Potential: Median of Reflection Coefficient, w.r.t. $k$')

# ---------------------------------------------
# PREDICTION PLOTS ON LABELS
# ---------------------------------------------

if args.plot_predictions:
    print('\nPLOTTING PREDICTION FOR LABEL K.')

    plt.close('all')

    fig, ax = plt.subplots()

    alphaVal = 0.6
    linethick = 1
    alphaVal_ni = 1.
    linethick_ni = 2

    idx = np.random.randint(args.testset_size)

    data = testset[idx]['k_n_r'].view(1, -1).to(args.device)
    out = model(data)
    label = testset[idx]['n_i'].to(args.device)

    k = DN(data[0][0].detach().cpu().numpy(), 'r', args)
    nr = DN(data[0][1:].detach().cpu().numpy(), 'r', args)
    ni_pred = DN(out[0].detach().cpu().numpy(), 'i', args)

    Int, psir, psii = get_psi(args=args,
                              n=nr + 1j * ni_pred,
                              Amp=1,
                              k=k)

    plt.plot(args.grid,
             nr,
             color=colourWheel[0 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[0 % len(dashesStyles)],
             lw=linethick,
             label=r'$n_r$',
             alpha=alphaVal)
    plt.plot(args.grid,
             ni_pred,
             color=colourWheel[1 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[1 % len(dashesStyles)],
             lw=linethick_ni,
             label=r'$n_{i,predicted}$',
             alpha=alphaVal_ni)
    plt.plot(args.grid,
             DN(label.detach().cpu().numpy(), 'i', args),
             color=colourWheel[2 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[2 % len(dashesStyles)],
             lw=linethick,
             label=r'$n_{i,label}$',
             alpha=alphaVal)
    plt.plot(args.grid,
             psir,
             color=colourWheel[3 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[3 % len(dashesStyles)],
             lw=linethick,
             label=r'$Re(\psi)$',
             alpha=alphaVal)
    plt.plot(args.grid,
             psii,
             color=colourWheel[4 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[4 % len(dashesStyles)],
             lw=linethick,
             label=r'$Im(\psi)$',
             alpha=alphaVal)
    plt.plot(args.grid,
             Int,
             color=colourWheel[5 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[5 % len(dashesStyles)],
             lw=linethick,
             label=r'$I$',
             alpha=alphaVal)

    ax.set_xlabel('')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.major.formatter._useMathText = True
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_label_coords(0.58, 1.01)
    ax.yaxis.tick_right()
    nameOfPlot = 'Predicted Potential: Refractive Index, Wave Amplitude, k = {:.0f}' \
        .format(k)
    plt.ylabel(nameOfPlot, rotation=0)
    ax.legend(frameon=False,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.05),
              ncol=6)
    plt.savefig(path_plots + args.which_model + '_prediction.pdf', bbox_inches="tight", dpi=300)

    # ---------------------------------------------
    # PREDICTION PLOTS ON DIFFERENT K VALUES
    # ---------------------------------------------

    print('\nPLOTTING PREDICTION FOR DIFFERENT K.')

    plt.close('all')

    fig, ax = plt.subplots()

    alphaVal = 0.6
    linethick = 1
    alphaVal_ni = 1.
    linethick_ni = 2

    idx = np.random.randint(args.testset_size)

    data = testset[idx]['k_n_r'].view(1, -1).to(args.device)
    k = DN(data[0][0].detach().cpu().numpy(), 'r', args)
    nr = DN(data[0][1:].detach().cpu().numpy(), 'r', args)

    k_new = np.random.randint(1, 11)

    data[0][0] = (k_new - args.nr_mean) / args.nr_std

    out = model(data)
    label = testset[idx]['n_i'].to(args.device)

    ni_pred = DN(out[0].detach().cpu().numpy(), 'i', args)

    Int, psir, psii = get_psi(args=args,
                              n=nr + 1j * ni_pred,
                              Amp=1,
                              k=k_new)

    plt.plot(args.grid,
             nr,
             color=colourWheel[0 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[0 % len(dashesStyles)],
             lw=linethick,
             label=r'$n_r$',
             alpha=alphaVal)
    plt.plot(args.grid,
             ni_pred,
             color=colourWheel[1 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[1 % len(dashesStyles)],
             lw=linethick_ni,
             label=r'$n_{i,predicted}$',
             alpha=alphaVal_ni)
    plt.plot(args.grid,
             psir,
             color=colourWheel[2 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[2 % len(dashesStyles)],
             lw=linethick,
             label=r'$Re(\psi)$',
             alpha=alphaVal)
    plt.plot(args.grid,
             psii,
             color=colourWheel[3 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[3 % len(dashesStyles)],
             lw=linethick,
             label=r'$Im(\psi)$',
             alpha=alphaVal)
    plt.plot(args.grid,
             Int,
             color=colourWheel[4 % len(colourWheel)],
             linestyle='-',
             dashes=dashesStyles[4 % len(dashesStyles)],
             lw=linethick,
             label=r'$I$',
             alpha=alphaVal)

    ax.set_xlabel('')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.major.formatter._useMathText = True
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_label_coords(0.58, 1.01)
    ax.yaxis.tick_right()
    nameOfPlot = 'Prediction for different k value: original k = {:.0f}, new k: {}' \
        .format(k, k_new)
    plt.ylabel(nameOfPlot, rotation=0)
    ax.legend(frameon=False,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.05),
              ncol=5)
    plt.savefig(path_plots + args.which_model + '_differentk.pdf', bbox_inches="tight", dpi=300)

# ---------------------------------------------
# FQ/MAX BARS FOR LABEL PREDICTIONS
# ---------------------------------------------

if args.plot_mean_median_bars:
    print('\nPLOTTING ON LABEL K PREDICTED FQ/MAX BARS.')

    Amp = 1

    nipred_mean_max_dev = np.zeros(10)
    nipred_max_devs = [[] for i in range(10)]
    how_many_k = np.zeros(10)

    for j, batch in enumerate(test_loader):

        knr = batch['k_n_r'].to(args.device)
        ni = batch['n_i'].to(args.device)

        for one_knr in knr:
            k = int(np.around(DN(one_knr[0].detach().cpu().numpy(), 'r', args)))

            how_many_k[k - 1] += 1
            ni_pred = model(one_knr.view(1, -1))[0]

            nr = DN(one_knr[1:].detach().cpu().numpy(), 'r', args)
            ni_pred = DN(ni_pred.detach().cpu().numpy(), 'i', args)

            Int_pred, _, _ = get_psi(args,
                                     n=nr + 1j * ni_pred,
                                     Amp=Amp,
                                     k=k)

            rel_err = np.max((np.abs(Int_pred - Amp) / Amp))  # all of it
            nipred_max_devs[k - 1].append(rel_err)
            nipred_mean_max_dev[k - 1] += rel_err

        if j % 10 == 0:
            print('{}/{}'.format(j, len(test_loader)))

    nipred_mean_max_dev = nipred_mean_max_dev / how_many_k
    nipred_median_max_dev = [np.median(i) for i in nipred_max_devs]

    plot_bars(bar_heights=nipred_mean_max_dev,
              bar_labels=list(range(1, 11)),
              path=path_plots + args.which_model + '_bars_nipred_dev_max.pdf',
              path_here=path_here,
              nameOfPlot='Predicted Potential: Mean of Maximal Deviation from CI-Value, w.r.t. $k$')

    plot_bars(bar_heights=nipred_median_max_dev,
              bar_labels=list(range(1, 11)),
              path=path_plots + args.which_model + '_bars_nipred_dev_max_median.pdf',
              path_here=path_here,
              nameOfPlot='Predicted Potential: Median of Maximal Deviation from CI-Value, w.r.t. $k$')
# ---------------------------------------------
# FQ/REFL BARS FOR LABEL PREDICTIONS
# ---------------------------------------------

if args.plot_reflections:
    print('\nPLOTTING ON LABEL K PREDICTED FQ/REFL BARS.')

    Amp = 1

    nipred_mean_refl = np.zeros(10)
    nipred_refls = [[] for i in range(10)]
    how_many_k = np.zeros(10)

    for j, batch in enumerate(test_loader):

        knr = batch['k_n_r'].to(args.device)
        ni = batch['n_i'].to(args.device)

        for one_knr in knr:
            k = int(np.around(DN(one_knr[0].detach().cpu().numpy(), 'r', args)))

            how_many_k[k - 1] += 1
            ni_pred = model(one_knr.view(1, -1))[0]

            nr = DN(one_knr[1:].detach().cpu().numpy(), 'r', args)
            ni_pred = DN(ni_pred.detach().cpu().numpy(), 'i', args)

            _, psir_pred, psii_pred = get_psi(args,
                                              n=nr + 1j * ni_pred,
                                              Amp=Amp,
                                              k=k)

            psi_pred = psir_pred + 1j*psii_pred

            R_coeff = R(psi_xmin1=psi_pred[-1],
                        psi_x0=psi_pred[-2],
                        deltax=args.stepsize,
                        k=k)

            nipred_refls[k - 1].append(R_coeff * 10000)
            nipred_mean_refl[k - 1] += R_coeff * 10000

        if j % 10 == 0:
            print('{}/{}'.format(j, len(test_loader)))

    nipred_mean_refl = nipred_mean_refl / how_many_k
    nipred_median_refl = [np.median(i) for i in nipred_refls]

    plot_bars(bar_heights=nipred_mean_refl,
              bar_labels=list(range(1, 11)),
              path=path_plots + args.which_model + '_bars_nipred_refl.pdf',
              path_here=path_here,
              nameOfPlot='Predicted Potential: Mean of Reflection Coefficient, w.r.t. $k [x \cdot 10^{(-4)}]$')

    plot_bars(bar_heights=nipred_median_refl,
              bar_labels=list(range(1, 11)),
              path=path_plots + args.which_model + '_bars_nipred_refl_median.pdf',
              path_here=path_here,
              nameOfPlot='Predicted Potential: Median of Reflection Coefficient, w.r.t. $k [x \cdot 10^{(-4)]}$')

# ---------------------------------------------
# FQ/MAX MATRIX FOR DIFFERENT K VALUES
# ---------------------------------------------

if args.plot_matrices:

    mpl.rcParams.update({'font.size': 8})

    print('\nPLOTTING ON DIFFERENT K PREDICTED FQ/MAX MATRIX.')

    # Calculation

    Amp = 1
    diffk_mean_max_dev = np.zeros([10, 10])
    diffk_mean_refl = np.zeros([10, 10])
    diffk_how_many_k = np.zeros([10, 10])

    for j, batch in enumerate(test_loader):

        knr = batch['k_n_r'].to(args.device)
        ni = batch['n_i'].to(args.device)

        for one_knr, one_ni in zip(knr, ni):
            k_real = int(
                np.around(
                    DN(one_knr[0].detach().cpu().numpy(), 'r', args)))

            k_new = np.random.randint(1, 11)

            diffk_how_many_k[k_real - 1, k_new - 1] += 1

            one_knr[0] = (k_new - args.nr_mean) / args.nr_std

            ni_pred = model(one_knr.view(1, -1))[0]

            nr = DN(one_knr[1:].detach().cpu().numpy(), 'r', args)
            ni_pred = DN(ni_pred.detach().cpu().numpy(), 'i', args)

            Int_pred, psir_pred, psii_pred = get_psi(args,
                                                     n=nr + 1j * ni_pred,
                                                     Amp=Amp,
                                                     k=k_new)

            rel_err = np.max((np.abs(Int_pred - Amp) / Amp))

            diffk_mean_max_dev[k_real - 1, k_new - 1] += rel_err

            psi_pred = psir_pred + 1j * psii_pred

            R_coeff = R(psi_xmin1=psi_pred[-1],
                        psi_x0=psi_pred[-2],
                        deltax=args.stepsize,
                        k=k_new)

            diffk_mean_refl[k_real - 1, k_new - 1] += R_coeff

        if j % 10 == 0:
            print('{}/{}'.format(j, len(test_loader)))

    diffk_mean_max_dev = diffk_mean_max_dev / diffk_how_many_k
    diffk_mean_refl = diffk_mean_refl / diffk_how_many_k

    # map for the frequencies

    k_real = list(range(1, 11))
    k_new = list(range(1, 11))

    textcolors = ["black", "white"]
    textcolor_tresh = np.max(diffk_mean_max_dev) / 2

    plt.close('all')

    fig, ax = plt.subplots()
    alphaVal = 0.6

    im = ax.imshow(diffk_mean_max_dev,
                   cmap='magma',
                   alpha=alphaVal)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(k_new)))
    ax.set_yticks(np.arange(len(k_real)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(k_new)
    ax.set_yticklabels(k_real)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")

    # this should fix the alignment error
    ax.set_xticks(np.arange(len(k_new) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(k_real) + 1) - .5, minor=True)

    # Loop over data dimensions and create text annotations.
    for i in range(len(k_real)):
        for j in range(len(k_new)):
            text = ax.text(j,
                           i,
                           np.around(diffk_mean_max_dev[i, j], decimals=2),
                           ha="center",
                           va="center",
                           color=textcolors[int(diffk_mean_max_dev[i, j] < textcolor_tresh)])

    plt.ylabel('original $k$-values')
    plt.xlabel('new $k$-values')

    ax.set_title(
        "Prediction for different $k$: Mean of Max. Dev. from CI")
    # fig.tight_layout()
    plt.savefig(path_plots + args.which_model + '_matrix_MeanMaxDev.pdf', dpi=300)  # bbox_inches="tight"

    # ---------------------------------------------
    # FQ/REFL MATRIX FOR DIFFERENT K VALUES
    # ---------------------------------------------

    print('\nPLOTTING ON DIFFERENT K PREDICTED FQ/REFL MATRIX.')

    # map for the frequencies

    k_real = list(range(1, 11))
    k_new = list(range(1, 11))

    textcolors = ["black", "white"]
    textcolor_tresh = np.max(diffk_mean_refl) / 2

    plt.close('all')

    fig, ax = plt.subplots()
    alphaVal = 0.6

    im = ax.imshow(diffk_mean_refl,
                   cmap='magma',
                   alpha=alphaVal)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(k_new)))
    ax.set_yticks(np.arange(len(k_real)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(k_new)
    ax.set_yticklabels(k_real)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")

    # this should fix the alignment error
    ax.set_xticks(np.arange(len(k_new) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(k_real) + 1) - .5, minor=True)

    # Loop over data dimensions and create text annotations.
    for i in range(len(k_real)):
        for j in range(len(k_new)):
            text = ax.text(j,
                           i,
                           np.around(diffk_mean_refl[i, j]*10000, decimals=2), # large number for better visability
                           ha="center",
                           va="center",
                           color=textcolors[int(diffk_mean_refl[i, j] < textcolor_tresh)])

    plt.ylabel('original $k$-values')
    plt.xlabel('new $k$-values')

    ax.set_title(
        "Prediction for different $k$: Mean of Reflection Coefficient $[x \cdot 10^{(-4)}]$")
    # fig.tight_layout()
    plt.savefig(path_plots + args.which_model + '_matrix_refl.pdf', dpi=300)  # , bbox_inches="tight"
