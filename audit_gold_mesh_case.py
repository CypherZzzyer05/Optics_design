"""Reload a completed native research project without modifying its bytes."""
import argparse
import json
from pathlib import Path
import numpy as np

from fdtd_pipeline.config import PROJECT_ROOT,SimulationSettings
from fdtd_pipeline.lumerical import mesh_grid_summary,validate_spectrum,validate_power_integration
from fdtd_pipeline.pipeline import atomic_json,checksum
from gold_mesh_research import GoldMeshSimulator,geometry_source,validate_saved_case


def settings_from_recipe(recipe):
    values=dict(recipe['simulation'])
    values['resonator_nk_csv']=str(PROJECT_ROOT/values['resonator_nk_csv'])
    for key in ('research_wavelengths_nm','power_validation_wavelengths_nm'):values[key]=tuple(values[key])
    return SimulationSettings(**values)


def audit_case(directory):
    state=validate_saved_case(directory);recipe=state['recipe'];settings=settings_from_recipe(recipe)
    config=json.loads((PROJECT_ROOT/'configs/as2se3_au_fine8.json').read_text())
    g=geometry_source();project=directory/'simulation.fsp';before=checksum(project)
    if before!=state['project_sha256']:raise ValueError('Native project checksum changed')
    with np.load(directory/'sample.npz') as saved, GoldMeshSimulator(settings,config['lumapi_path'],
            recipe['gold_present'],recipe['vacuum_reference'],recipe['electric_wavelengths_nm'],
            recipe['magnetic_wavelengths_nm']) as sim:
        # Recreate the declared recipe to get an independent Au reference grid,
        # then load the completed native result. Nothing is saved in this audit.
        sim.build(g)
        expected_boundaries={key:sim.fdtd.getnamed('FDTD',key) for key in
                             ('x min bc','x max bc','y min bc','y max bc','z min bc','z max bc','index')}
        reference={a:v.copy() for a,v in sim.reference_Au_grid.items()}
        sim.fdtd.load(str(project))
        sim.fdtd.cwnorm()
        sim.grid_matches_Au_layout=not recipe['vacuum_reference']
        structure=sim.native_structure_check(g)
        for key,expected in expected_boundaries.items():
            if sim.fdtd.getnamed('FDTD',key)!=expected:raise ValueError('Native boundary/background changed')
        grids={a:np.asarray(sim.fdtd.getresult('FDTD',a)).reshape(-1) for a in ('x','y','z')}
        mesh=mesh_grid_summary(grids,settings.maximum_mesh_nm)
        for a,v in grids.items():
            if not np.allclose(v*1e9,saved['mesh_'+a+'_nm'],rtol=0,atol=1e-6):
                raise ValueError('Native and exported spatial grids differ')
            if not recipe['vacuum_reference'] and (v.shape!=reference[a].shape or not np.allclose(v,reference[a],rtol=0,atol=1e-15)):
                raise ValueError('Native bare/Au comparison does not retain the same reference grid')
        status=int(np.asarray(sim.fdtd.getresult('FDTD','status')).item())
        r,t=sim.fdtd.getresult('R','T'),sim.fdtd.getresult('T','T')
        if not np.allclose(r['lambda'],t['lambda'],rtol=0,atol=1e-15):raise ValueError('Native R/T grids differ')
        result=validate_spectrum(np.asarray(r['lambda'])*1e9,r['T'],-np.asarray(t['T']),settings,status)
        errors={key:float(np.max(abs(result[key]-saved[key]))) for key in
                ('wavelength_nm','reflectance','transmittance','absorptance')}
        if any(value>1e-10 for value in errors.values()):raise ValueError('Native spectrum differs from export')
        power=None
        if settings.power_validation_wavelengths_nm:
            r,t=sim.fdtd.getresult('R_powercheck','T'),sim.fdtd.getresult('T_powercheck','T')
            power,*_=validate_power_integration(result['wavelength_nm'],result['reflectance'],result['transmittance'],
                    np.asarray(r['lambda'])*1e9,r['T'],-np.asarray(t['T']))
        fields=sim.read_fields();field_errors={}
        for key,value in fields.items():
            native=np.asarray(value);export=np.asarray(saved[key])
            if native.shape!=export.shape or not np.allclose(native,export,rtol=5e-7,atol=1e-7):
                raise ValueError('Native near field or normalization differs: '+key)
            field_errors[key]=float(np.max(abs(native-export)))
    if checksum(project)!=before:raise ValueError('Audit modified native project bytes')
    report={'case_id':recipe['case_id'],'sample_sha256':state['sample_sha256'],
            'project_sha256':before,'native_geometry_verified':True,'native_spectra_match_export':True,
            'native_fields_match_export':True if sim.field_names else None,
            'field_monitor_count':len(sim.field_names),'actual_mesh':mesh,'status':status,
            'structure':structure,'native_vs_export_spectral_errors':errors,
            'native_vs_export_field_errors':field_errors,'power_integration_validation':power,
            'field_export_dtype':'float32 intensity enhancement; coordinates and source normalization float64',
            'note':'Native reload verifies this saved experiment; it does not prove material accuracy or mesh convergence.'}
    atomic_json(directory/'native_audit.json',report)
    print(f"Native audit passed: {recipe['case_id']}",flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True)
    p.add_argument('--output',type=Path,default=PROJECT_ROOT/'data/gold_mesh_comparison')
    args=p.parse_args();audit_case(args.output/args.case)
