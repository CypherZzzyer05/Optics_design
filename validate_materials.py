"""Export the actual FDTD material fits and their error against imported data."""
import argparse
from pathlib import Path
import numpy as np
from fdtd_pipeline.config import PROJECT_ROOT, load_config
from fdtd_pipeline.lumerical import FDTDSimulator
from fdtd_pipeline.pipeline import atomic_json


def validate(config, output):
    output.mkdir(parents=True,exist_ok=True)
    settings=config.simulation
    names=[settings.resonator_material or settings.silicon_material,settings.substrate_material]
    if settings.metal_material:
        names.append(settings.metal_material)
    records={}
    with FDTDSimulator(settings,config.lumapi_path) as simulator:
        grid=settings.wavelength_grid();frequency=299792458/(grid*1e-9)
        for i,name in enumerate(names):
            measured=np.asarray(simulator.fdtd.getindex(name,frequency)).reshape(-1)
            fitted=np.asarray(simulator.fdtd.getfdtdindex(name,frequency,frequency.min(),frequency.max())).reshape(-1)
            if not np.isfinite(measured).all() or not np.isfinite(fitted).all() or np.any(fitted.imag < -1e-8):
                raise ValueError('Material fit contains nonfinite data or gain')
            csv=output/f'material_{i}.csv'
            np.savetxt(csv,np.column_stack([grid,measured.real,measured.imag,fitted.real,fitted.imag]),delimiter=',',
                       header='wavelength_nm,data_n,data_k,fit_n,fit_k',comments='')
            records[name]={'max_n_error':float(np.max(np.abs(measured.real-fitted.real))),
                           'max_k_error':float(np.max(np.abs(measured.imag-fitted.imag))),
                           'rms_n_error':float(np.sqrt(np.mean((measured.real-fitted.real)**2))),
                           'rms_k_error':float(np.sqrt(np.mean((measured.imag-fitted.imag)**2))),
                           'minimum_fit_k':float(fitted.imag.min()),'csv':str(csv.resolve())}
    report={'config':config.metadata(),'fits':records,
            'note':'Select tolerances for the intended loss-sensitive observable; this report alone does not certify a fit.'}
    atomic_json(output/'material_fit.json',report)
    return report


if __name__=='__main__':
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/as2se3_au_pilot.json')
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'data/material_validation')
    args=parser.parse_args()
    print(json.dumps(validate(load_config(args.config),args.output.resolve()),indent=2))
