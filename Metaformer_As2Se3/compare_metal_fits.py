"""Probe copied Au/Ag material fits in a fresh session; never run a solver.

This does not inspect or change the running comparison projects. Their saved
fits are audited separately after the matrix completes. Candidate fit errors
are diagnostics, not a certification of device spectra or time-domain stability.
"""
import argparse
from pathlib import Path
import numpy as np

from audit_gold_mesh_materials import fit_metrics
from fdtd_pipeline.config import PROJECT_ROOT, load_config
from fdtd_pipeline.lumerical import FDTDSimulator
from fdtd_pipeline.pipeline import atomic_json, checksum

METALS = ("Au (Gold) - Palik", "Ag (Silver) - Palik (0-2um)")
CANDIDATES = ((8, 1.), (8, 10.), (12, 1.), (12, 10.))


def permittivity_errors(measured, fitted):
    """Report dispersion and loss separately; a small k error can hide lost loss."""
    metrics = fit_metrics(measured, fitted)
    data, fit = np.asarray(measured)**2, np.asarray(fitted)**2
    metrics['max_relative_complex_permittivity_error'] = float(np.max(
        abs(fit-data)/np.maximum(abs(data), 1e-12)))
    metrics['max_relative_imaginary_permittivity_error'] = float(np.max(
        abs(fit.imag-data.imag)/np.maximum(abs(data.imag), 1e-12)))
    metrics['minimum_data_imaginary_permittivity'] = float(data.imag.min())
    metrics['minimum_fit_imaginary_permittivity'] = float(fit.imag.min())
    return metrics


def probe(config, destination, resample=False, coefficients=None, weights=None, improve_stability=True, metals=None):
    candidates = CANDIDATES
    if coefficients is not None or weights is not None:
        candidates = tuple((c, w) for c in (coefficients or (8, 12)) for w in (weights or (1., 10.)))
    if any(c < 1 or not np.isfinite(w) or w <= 0 for c, w in candidates):
        raise ValueError('Positive coefficients and finite positive weights required')
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError('Use a new directory so existing material diagnostics remain intact')
    wavelengths = config.simulation.wavelength_grid()
    frequency = 299792458/(wavelengths*1e-9)
    band = (float(frequency.min()), float(frequency.max()))
    report = {'scope': 'Fresh in-memory copies or sampled-material imports only; no FSP loaded/saved, no solver run, no defaults edited.',
              'band_nm': [float(wavelengths.min()), float(wavelengths.max())],
              'wavelength_points': len(wavelengths), 'device_validation_pending': True,
              'resample_library_data': resample,
              'improve_numerical_stability': improve_stability,
              'materials': {}, 'references': [
                  'https://optics.ansys.com/hc/en-us/articles/360034915053-Tips-for-improving-the-quality-of-optical-material-fits',
                  'https://optics.ansys.com/hc/en-us/articles/360034930033-copymaterial-Script-command']}
    with FDTDSimulator(config.simulation, config.lumapi_path) as simulator:
        report['lumerical_version'] = simulator.version
        f = simulator.fdtd
        selected_metals = tuple(metals or METALS)
        for position, metal in enumerate(selected_metals):
            label = metal[:2] if sum(name[:2] == metal[:2] for name in selected_metals) == 1 else f'{metal[:2]}_{position}'
            if not f.materialexists(metal):
                raise ValueError('Library material missing: '+metal)
            original_samples = np.asarray(f.getmaterial(metal, 'sampled 3d data'))
            data_wavelengths = 299792458/original_samples[:, 0].real*1e9
            source_range = [float(data_wavelengths.min()), float(data_wavelengths.max())]
            if source_range[0] > wavelengths.min() or source_range[1] < wavelengths.max():
                raise ValueError('Selected metal data does not cover the complete source band')
            measured = np.asarray(f.getindex(metal, frequency)).reshape(-1)
            sampled = None
            if resample:
                raw = np.asarray(f.getmaterial(metal, 'sampled 3d data'))
                knots = raw[:, 0].real
                sampled_frequency = np.unique(np.r_[frequency, knots[(knots >= band[0]) & (knots <= band[1])]])
                sampled_index = np.asarray(f.getindex(metal, sampled_frequency)).reshape(-1)
                sampled = np.c_[sampled_frequency, sampled_index**2]
            records = []
            for index, candidate in enumerate((None, *candidates)):
                name = metal
                settings = {'library_default': True}
                if candidate is not None:
                    coefficients, weight = candidate
                    if resample:
                        name = f.addmaterial('Sampled data')
                        f.setmaterial(name, 'sampled data', sampled)
                    else:
                        name = f.copymaterial(metal)
                    settings = {'library_default': False, 'max coefficients': coefficients,
                                'tolerance': 0., 'imaginary weight': weight,
                                'make fit passive': 1, 'improve numerical stability': int(improve_stability)}
                    for key, value in settings.items():
                        if key != 'library_default':
                            f.setmaterial(name, key, value)
                heldout_frequency = 299792458/(np.linspace(wavelengths.min(), wavelengths.max(), 4001)*1e-9)
                original_curve = np.asarray(f.getindex(metal, heldout_frequency)).reshape(-1)
                prepared_curve = np.asarray(f.getindex(name, heldout_frequency)).reshape(-1)
                preparation_error = float(np.max(abs(original_curve-prepared_curve)))
                if preparation_error > 1e-7:
                    raise ValueError('Preparing the material changed the held-out index curve')
                fitted = np.asarray(f.getfdtdindex(name, frequency, *band)).reshape(-1)
                if fitted.shape != measured.shape or not np.isfinite(fitted).all():
                    raise ValueError('Native fit has a missing or nonfinite frequency axis')
                passive = bool(np.all(fitted.imag >= -1e-8))
                metrics = permittivity_errors(measured, fitted) if passive else None
                path = destination/f'{label}_{index}.csv'
                np.savetxt(path, np.c_[wavelengths, measured.real, measured.imag, fitted.real, fitted.imag],
                           delimiter=',', header='wavelength_nm,data_n,data_k,fit_n,fit_k', comments='')
                records.append({'settings': settings, 'metrics': metrics, 'passivity_check_passed': passive,
                                'minimum_fit_k': float(fitted.imag.min()),
                                'heldout_data_preparation_max_index_error': preparation_error,
                                'prepared_data_points': len(sampled) if candidate is not None and resample else None,
                                'source_data_range_nm': source_range,
                                'csv': path.name, 'csv_sha256': checksum(path)})
                if passive:
                    print(f'{metal[:2]} candidate {index}: max relative epsilon error '
                          f'{metrics["max_relative_complex_permittivity_error"]:.5g}; '
                          f'loss error {metrics["max_relative_imaginary_permittivity_error"]:.5g}', flush=True)
                else:
                    print(f'{metal[:2]} candidate {index}: rejected; minimum fitted k '
                          f'{fitted.imag.min():.5g}', flush=True)
                if candidate is not None:
                    f.deletematerial(name)
            report['materials'][metal] = records
    atomic_json(destination/'comparison.json', report)
    atomic_json(destination/'checksums.json', {p.name: checksum(p) for p in destination.iterdir()
                                             if p.is_file() and p.name != 'checksums.json'})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/as2se3_au_fine8.json')
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--resample', action='store_true', help='Include original knots and dense same-library samples; verify held-out data equality')
    parser.add_argument('--coefficients', type=int, nargs='+')
    parser.add_argument('--weights', type=float, nargs='+')
    parser.add_argument('--relax-stability', action='store_true', help='Diagnostic only; disables numerical-stability constraint, so any resulting fit needs independent time-domain validation')
    parser.add_argument('--metals', nargs='+', help='Exact installed material names; each source is reported separately')
    args = parser.parse_args()
    probe(load_config(args.config), args.destination.resolve(), args.resample,
          args.coefficients, args.weights, not args.relax_stability, args.metals)
