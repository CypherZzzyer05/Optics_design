"""Display all held-out forward predictions and training curves, without training."""
import argparse
import json
from pathlib import Path

import numpy as np
from forward_metrics import valleys

ROOT = Path(__file__).resolve().parent


def plot_results(run, destination=None, show=False, save=True):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = Path(run)
    destination = Path(destination) if destination else run
    if save:
        destination.mkdir(parents=True, exist_ok=True)
    history = json.loads((run/'history.json').read_text(encoding='utf-8'))
    metrics = json.loads((run/'metrics.json').read_text(encoding='utf-8'))
    with np.load(run/'predictions.npz', allow_pickle=False) as archive:
        data = {key:archive[key].copy() for key in archive.files}
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(8.8, 4.5), layout='constrained')
    epochs = [row['epoch'] for row in history]
    ax.plot(epochs, [r['train_mse'] for r in history], label='Training (dropout/AMP active)')
    ax.plot(epochs, [r['val_mse'] for r in history], label='Validation (FP32)')
    ax.axhline(metrics['train_mean_baseline']['val']['mse'], color='gray', ls='--', label='Training-mean spectrum: validation MSE')
    ax.axvline(metrics['best_epoch'], color='black', ls=':', label=f'Best epoch {metrics["best_epoch"]}')
    ax.set(xlabel='Epoch', ylabel='Full-spectrum reflectance MSE', yscale='log',
           title='Forward Metaformer pilot: 80 training / 10 validation / 10 test')
    ax.grid(alpha=.2); ax.legend(fontsize=9)
    if save:
        fig.savefig(destination/'learning_curve.png', dpi=180)
    wave = data['wavelength_nm']
    order = np.argsort(data['test_sample_ids'])
    fig, axes = plt.subplots((len(order)+1)//2, 2, figsize=(13, 2.45*((len(order)+1)//2)),
                             layout='constrained', squeeze=False)
    for ax, row in zip(axes.flat, order):
        truth, pred = data['test_target'][row], data['test_prediction'][row]
        ax.plot(wave, truth, color='#185c89', label='FDTD reference', lw=1.1)
        ax.plot(wave, pred, color='#d97423', label='Forward Metaformer', lw=1.1)
        ax.plot(wave, data['train_mean_spectrum'], color='gray', ls='--', alpha=.6, lw=.8, label='Training mean')
        error = float(np.mean((truth-pred)**2))
        ax.set(xlim=(wave[0],wave[-1]), ylim=(-.02,1.03), xlabel='Wavelength (nm)', ylabel='R',
               title=f'Test sample {int(data["test_sample_ids"][row])} | MSE {error:.4f}')
        ax.grid(alpha=.16)
    for ax in axes.flat[len(order):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8, loc='lower right')
    fig.suptitle(f'All {len(order)} held-out spectra, sorted by sample ID; no selection of best-looking cases\n'
                 'Historical v1 numerical data; Au material accuracy is not certified', fontsize=12)
    if save:
        fig.savefig(destination/'test_spectra.png', dpi=180)
    # A fixed subset (first three test IDs), chosen independently of prediction quality.
    chosen = order[:3]
    fig, axes = plt.subplots(len(chosen), 2, figsize=(11, 2.7*len(chosen)),
                             layout='constrained', squeeze=False)
    for axes_row, row in zip(axes, chosen):
        reference = valleys(data['test_target'][row], wave, prominence=.05, limit=2)
        for ax, valley in zip(axes_row, reference):
            center = valley['center_nm']
            ax.plot(wave, data['test_target'][row], color='#185c89', label='FDTD reference')
            ax.plot(wave, data['test_prediction'][row], color='#d97423', label='Forward Metaformer')
            ax.plot(wave, data['train_mean_spectrum'], color='gray', ls='--', alpha=.7, label='Training mean')
            ax.set(xlim=(center-20,center+20), ylim=(-.02,1.03), xlabel='Wavelength (nm)', ylabel='R',
                   title=f'Sample {int(data["test_sample_ids"][row])}: reference valley {center:.1f} nm')
            ax.grid(alpha=.2)
        for ax in axes_row[len(reference):]:
            ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Valley details: first three held-out sample IDs (fixed selection)', fontsize=12)
    if save:
        fig.savefig(destination/'valley_details.png', dpi=180)
    if show:
        plt.show()
    else:
        plt.close('all')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'results/forward_au100_seed42')
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--no-save', action='store_true', help='View committed results without rewriting images')
    args = parser.parse_args()
    plot_results(args.run, args.destination, args.show, save=not args.no_save)


if __name__ == '__main__':
    main()
