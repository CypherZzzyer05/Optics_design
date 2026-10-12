"""Read material data and FDTD fits from completed native projects, without solving."""
import json
from pathlib import Path
import numpy as np
from fdtd_pipeline.config import PROJECT_ROOT
from fdtd_pipeline.lumerical import FDTDSimulator
from fdtd_pipeline.pipeline import atomic_json,checksum
from audit_gold_mesh_case import settings_from_recipe
from gold_mesh_research import validate_saved_case


def fit_metrics(measured,fitted):
    measured,fitted=np.asarray(measured),np.asarray(fitted)
    if measured.shape!=fitted.shape or not measured.size or not np.isfinite(measured).all() or not np.isfinite(fitted).all():
        raise ValueError('Material data and fit must be finite with matching dimensions')
    if np.any(fitted.imag < -1e-8):raise ValueError('The material fit contains gain')
    difference=measured-fitted
    return {'max_n_error':float(np.max(abs(difference.real))),
            'max_k_error':float(np.max(abs(difference.imag))),
            'rms_n_error':float(np.sqrt(np.mean(difference.real**2))),
            'rms_k_error':float(np.sqrt(np.mean(difference.imag**2))),
            'minimum_fit_n':float(fitted.real.min()),'minimum_fit_k':float(fitted.imag.min()),
            'maximum_imaginary_permittivity_error':float(np.max(abs((measured**2-fitted**2).imag)))}


def audit_materials(output,destination,targets):
    """Called only after the serial matrix completes, when solver RAM is released."""
    destination.mkdir(parents=True,exist_ok=True)
    config=json.loads((PROJECT_ROOT/'configs/as2se3_au_fine8.json').read_text())
    result={};fits={}
    for case_id in ('Au8','bare8','Au4','bare4'):
        directory=output/case_id;state=validate_saved_case(directory)
        settings=settings_from_recipe(state['recipe']);project=directory/'simulation.fsp'
        before=checksum(project)
        if before!=state['project_sha256']:raise ValueError('Native material project checksum differs')
        wavelengths=np.unique(np.r_[np.linspace(settings.wavelength_start_nm,settings.wavelength_stop_nm,1000),targets])
        frequencies=299792458/(wavelengths*1e-9)
        names=(settings.resonator_material,settings.substrate_material)
        if state['recipe']['gold_present']:names=(*names,settings.metal_material)
        materials={};fits[case_id]={}
        with FDTDSimulator(settings,config['lumapi_path']) as simulator:
            simulator.fdtd.load(str(project))
            for key,value in (('wavelength start',settings.wavelength_start_nm),('wavelength stop',settings.wavelength_stop_nm)):
                if not np.isclose(simulator.fdtd.getnamed('source',key)*1e9,value,rtol=0,atol=1e-6):
                    raise ValueError('Saved source band differs from fit query band')
            for index,name in enumerate(names):
                measured=np.asarray(simulator.fdtd.getindex(name,frequencies)).reshape(-1)
                fitted=np.asarray(simulator.fdtd.getfdtdindex(name,frequencies,frequencies.min(),frequencies.max())).reshape(-1)
                if len(fitted)!=len(wavelengths):raise ValueError('Material diagnostic frequency axis differs')
                metrics=fit_metrics(measured,fitted);fits[case_id][name]=fitted
                csv=destination/f'{case_id}_material_{index}.csv'
                np.savetxt(csv,np.c_[wavelengths,measured.real,measured.imag,fitted.real,fitted.imag],delimiter=',',
                           header='wavelength_nm,data_n,data_k,fit_n,fit_k',comments='')
                metrics.update(csv=csv.name,csv_sha256=checksum(csv),at_target_wavelengths=[])
                for target in targets:
                    i=int(np.argmin(abs(wavelengths-target)))
                    metrics['at_target_wavelengths'].append({'wavelength_nm':float(wavelengths[i]),
                        'data_n':float(measured[i].real),'data_k':float(measured[i].imag),
                        'fit_n':float(fitted[i].real),'fit_k':float(fitted[i].imag)})
                materials[name]=metrics
        if checksum(project)!=before:raise ValueError('Material diagnostic modified the native file')
        result[case_id]={'project_sha256':before,'sample_sha256':state['sample_sha256'],
                         'fit_frequency_limits_hz':[float(frequencies.min()),float(frequencies.max())],
                         'materials':materials}
    agreement={f'{left}_vs_{right}':{name:float(np.max(abs(fits[left][name]-fits[right][name])))
               for name in sorted(fits[left].keys() & fits[right].keys())} for left,right in
               (('Au8','Au4'),('bare8','bare4'),('Au8','bare8'),('Au4','bare4'))}
    report={'cases':result,'shared_material_fit_differences':agreement,
            'shared_fits_match':all(value<=1e-7 for pair in agreement.values() for value in pair.values()),
            'scope':'getindex data and getfdtdindex fit over the saved 700-1700 nm source band; native FSP bytes unchanged; no solver or fit-setting mutation.',
            'accuracy_not_certified':True}
    atomic_json(destination/'material_audit.json',report)
    atomic_json(destination/'checksums.json',{path.name:checksum(path) for path in destination.iterdir()
                                            if path.is_file() and path.name!='checksums.json'})
    return report
