"""Controlled Au/air replacement and 8/4 nm research, with measured E/H profiles.

The bare control retains the finite substrate, lower air domain, T monitor and
metal-region mesh. Setting metal_material=None in the legacy builder would
instead make the substrate semi-infinite and would confound this experiment.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from fdtd_pipeline.config import Geometry, PROJECT_ROOT, load_config
from fdtd_pipeline.lumerical import FDTDSimulator, mesh_grid_summary, validate_spectrum
from fdtd_pipeline.pipeline import atomic_json, atomic_npz, checksum, directory_lock

C = 299792458.
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
MATERIAL_SHA = '9e6ee6d3b584f33226d7201e2b2e12490726cb3b8256726754e5aca9cae64ce9'
PLANES = ('mid', 'top')


def geometry_source():
    source = PROJECT_ROOT/'research_results/double_dip_mesh_5_4/comparison.json'
    values = json.loads(source.read_text(encoding='utf-8'))['geometry_values']
    return Geometry(*values[:5], theta_odd_deg=values[5], theta_even_deg=values[6])


def incident_squared(intensity, vector='E', index=1.):
    intensity = np.asarray(intensity, dtype=float).reshape(-1)
    if not np.isfinite(intensity).all() or np.any(intensity <= 0):
        raise ValueError('Incident intensity must be positive and finite')
    if vector == 'E':
        return 2*intensity/(index*EPS0*C)
    if vector == 'H':
        return 2*intensity*index/(MU0*C)
    raise ValueError('Only electric and magnetic field normalization is supported')


def field_profile(dataset, intensity, expected_wavelengths, vector='E'):
    """Preserve native spatial axes; sort the native frequency axis explicitly."""
    coordinates = {a: np.asarray(dataset[a], dtype=float).reshape(-1)*1e9 for a in ('x','y','z')}
    wavelength = np.asarray(dataset['lambda'], dtype=float).reshape(-1)*1e9
    order = np.argsort(wavelength)
    expected = np.asarray(expected_wavelengths, dtype=float)
    if (wavelength.shape != expected.shape or not np.isfinite(wavelength).all()
            or not np.allclose(wavelength[order], expected, rtol=0, atol=1e-5)):
        raise ValueError('Native field frequencies do not match the requested samples')
    shape = tuple(len(coordinates[a]) for a in ('x','y','z')) + (len(wavelength),)
    raw = np.asarray(dataset[vector])
    if raw.shape != shape+(3,) or not np.isfinite(raw).all():
        raise ValueError('Unexpected or nonfinite native vector-field dimensions')
    squared = np.sum(np.abs(raw)**2, axis=-1)
    denominator = incident_squared(intensity, vector)
    if len(denominator) != len(wavelength):
        raise ValueError('Source intensity and field frequency dimensions differ')
    enhancement = (squared/denominator.reshape((1,1,1,-1)))[...,order]
    if not np.isfinite(enhancement).all():
        raise ValueError('Nonfinite field enhancement')
    return coordinates, wavelength[order], enhancement.astype(np.float32), denominator[order]


class GoldMeshSimulator(FDTDSimulator):
    def __init__(self, settings, lumapi_path=None, gold=True, vacuum=False,
                 electric_wavelengths=(), magnetic_wavelengths=()):
        super().__init__(settings, lumapi_path)
        if settings.metal_material != 'Au (Gold) - Palik':
            raise ValueError('Use the Au-based finite-stack builder for both controls')
        self.gold, self.vacuum = bool(gold), bool(vacuum)
        self.electric_wavelengths = tuple(electric_wavelengths)
        self.magnetic_wavelengths = tuple(magnetic_wavelengths)
        self.field_names = []
        self.project_path = None

    def build(self, geometry, include_silicon=True, substrate_index=None):
        if substrate_index is not None:
            raise ValueError('The controlled study keeps the same finite SiO2 substrate')
        super().build(geometry, include_silicon=not self.vacuum, substrate_index=None)
        f = self.fdtd
        f.setnamed('source', 'amplitude', 1.)
        f.cwnorm()
        self.field_names = []
        for vector, wavelengths in (('E',self.electric_wavelengths), ('H',self.magnetic_wavelengths)):
            if not wavelengths:
                continue
            for plane, height in (('mid', geometry.h/2), ('top', geometry.h+10)):
                name = f'nf_{plane}_{vector}'
                f.addpower(name=name, monitor_type='2D Z-normal', x=0., y=0.,
                           x_span=geometry.p*1e-9, y_span=geometry.p*1e-9, z=height*1e-9)
                f.set('override global monitor settings', True)
                f.set('sample spacing', 'custom')
                f.set('custom frequency samples', (C/(np.asarray(wavelengths)*1e-9))[::-1])
                f.set('spatial interpolation', 'specified position')
                f.set('down sample x', 1)
                f.set('down sample y', 1)
                for component in ('Ex','Ey','Ez','Hx','Hy','Hz','Px','Py','Pz'):
                    f.set('output '+component, component.startswith(vector))
                f.set('output power', False)
                self.field_names.append((name, vector, wavelengths, height))
        # Capture the Au-built grid before replacing the layer with background air.
        reference = {a: np.asarray(f.getresult('FDTD',a)).reshape(-1) for a in ('x','y','z')}
        self.reference_Au_grid = reference
        if not self.gold or self.vacuum:
            f.select('back_metal')
            f.delete()
        if self.vacuum:
            f.select('substrate')
            f.delete()
        actual = {a: np.asarray(f.getresult('FDTD',a)).reshape(-1) for a in ('x','y','z')}
        mesh_grid_summary(actual, self.settings.maximum_mesh_nm)
        self.grid_matches_Au_layout = all(reference[a].shape == actual[a].shape
            and np.allclose(reference[a], actual[a], rtol=0, atol=1e-15) for a in reference)
        if not self.vacuum and not self.grid_matches_Au_layout:
            raise ValueError('Removing Au changed the generated mesh; resolve this before the controlled comparison')
        self.native_structure_check(geometry)
        if self.project_path is not None:
            # Installed 2024 R1 supports the original no-argument CPU command.
            report = f.runsystemcheck()
            atomic_json(self.project_path.with_suffix('.resources.json'), jsonable(report))
            recommended = report.get('Memory_Recommended_Bytes',report.get('Memory_Recommended'))
            if recommended is None:
                raise ValueError('Native system check did not return recommended memory')
            print(f"Preflight: {float(recommended)/1024**3:.2f} GiB recommended", flush=True)

    def native_structure_check(self, geometry):
        f, s = self.fdtd, self.settings
        def near(name, key, expected, scale=1e9):
            actual = float(np.asarray(f.getnamed(name,key)).item())*scale
            if not np.isclose(actual, expected, rtol=0, atol=1e-6):
                raise ValueError(f'Native {name}/{key} differs from the controlled setup')
        if int(f.getnamednumber('back_metal')) != int(self.gold and not self.vacuum):
            raise ValueError('Native Au presence differs from the study recipe')
        if int(f.getnamednumber('substrate')) != int(not self.vacuum):
            raise ValueError('Native substrate presence differs from the recipe')
        near('FDTD','x span',geometry.p)
        near('FDTD','y span',geometry.p)
        near('FDTD','z min',-s.substrate_depth_nm-s.metal_thickness_nm-s.bottom_padding_nm)
        near('FDTD','z max',geometry.h+s.air_padding_nm)
        near('T','z',-s.substrate_depth_nm-s.metal_thickness_nm-.5*s.bottom_padding_nm)
        near('source','amplitude',1.,1.)
        near('source','polarization angle',s.polarization_deg,1.)
        near('source','wavelength start',s.wavelength_start_nm)
        near('source','wavelength stop',s.wavelength_stop_nm)
        if not self.vacuum:
            near('substrate','z min',-s.substrate_depth_nm)
            near('substrate','z max',0.)
            if f.getnamed('substrate','material') != s.substrate_material:
                raise ValueError('Substrate material changed')
            for i, (center,angle) in enumerate(zip(geometry.centers(),geometry.angles())):
                for key,value in (('x',center[0]),('y',center[1]),('x span',geometry.l),('y span',geometry.l),('z min',0.),('z max',geometry.h)):
                    near(f'si_{i}',key,value)
                near(f'si_{i}','rotation 1',-angle,1.)
                if f.getnamed(f'si_{i}','material') != s.resonator_material:
                    raise ValueError('Resonator material changed')
        if self.gold and not self.vacuum:
            near('back_metal','z min',-s.substrate_depth_nm-s.metal_thickness_nm)
            near('back_metal','z max',-s.substrate_depth_nm)
            if f.getnamed('back_metal','material') != s.metal_material:
                raise ValueError('Au material changed')
        for name,vector,wavelengths,height in self.field_names:
            near(name,'z',height)
            for a in ('x','y'):
                near(name,'down sample '+a,1.,1.)
            if f.getnamed(name,'spatial interpolation') != 'specified position':
                raise ValueError('Field probes must retain the same physical plane positions')
        return {'gold_present':self.gold and not self.vacuum,
                'finite_substrate':not self.vacuum, 'geometry_verified':True,
                'grid_matches_Au_layout':self.grid_matches_Au_layout}

    def read_fields(self):
        result = {}
        for name,vector,wavelengths,height in self.field_names:
            data = self.fdtd.getresult(name,vector)
            intensity = np.asarray(self.fdtd.sourceintensity(np.asarray(data['f']).reshape(-1))).reshape(-1)
            axes,w,enhancement,denominator = field_profile(data,intensity,wavelengths,vector)
            if len(axes['z']) != 1 or not np.isclose(axes['z'][0],height,rtol=0,atol=1e-5):
                raise ValueError('Field plane moved from its specified physical position')
            for a in ('x','y'):
                delta = np.diff(axes[a])
                if len(delta)<1 or np.any(delta<=0) or np.max(delta)>=10:
                    raise ValueError('Field recording coordinates must be increasing and below 10 nm spacing')
            result.update({name+'_enhancement':enhancement, name+'_wavelength_nm':w,
                           name+'_incident_squared':denominator})
            result.update({name+'_'+a+'_nm':value for a,value in axes.items()})
        return result

    def run(self, geometry, project_path, *args, **kwargs):
        self.project_path = Path(project_path).resolve()
        self.project_path.parent.mkdir(parents=True,exist_ok=True)
        result = super().run(geometry,self.project_path,*args,**kwargs)
        result.update(self.read_fields(), gold_present=np.bool_(self.gold and not self.vacuum),
                      vacuum_reference=np.bool_(self.vacuum), geometry_values=geometry.array(include_quadrant=True),
                      grid_matches_Au_layout=np.bool_(self.grid_matches_Au_layout))
        result.update({f'reference_Au_grid_{axis}_nm':grid*1e9 for axis,grid in self.reference_Au_grid.items()})
        return result


def jsonable(value):
    if isinstance(value,dict): return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    return value


def make_recipe(case_id, output):
    config = load_config(PROJECT_ROOT/'configs/as2se3_au_fine8.json')
    if checksum(Path(config.simulation.resonator_nk_csv)) != MATERIAL_SHA:
        raise ValueError('Measured material CSV bytes changed')
    g = geometry_source()
    if case_id == 'bare8_preliminary':
        settings = replace(config.simulation,research_grid=True,spectrum_points=3000)
        gold,vacuum,electric,magnetic = False,False,(),()
    else:
        plan = json.loads((output/'sampling_plan.json').read_text(encoding='utf-8'))
        mesh = 4 if case_id.endswith('4') else 8
        gold,vacuum = case_id.startswith('Au'),case_id.startswith('vacuum')
        downsample = 2 if mesh==4 else 1
        checks = tuple(plan['power_check_wavelengths_nm']) if downsample>1 else ()
        settings = replace(config.simulation,research_grid=True,spectrum_points=len(plan['spectral_wavelengths_nm']),
                           mesh_nm=mesh,maximum_mesh_nm=mesh,monitor_downsample_xy=downsample,
                           research_wavelengths_nm=tuple(plan['spectral_wavelengths_nm']),
                           power_validation_wavelengths_nm=checks)
        electric,magnetic = tuple(plan['electric_wavelengths_nm']),tuple(plan['magnetic_wavelengths_nm'])
    recipe = {'version':1 if case_id=='bare8_preliminary' else 2,'case_id':case_id,'geometry_values':g.array(include_quadrant=True).tolist(),
              'simulation':asdict(settings),'gold_present':gold,'vacuum_reference':vacuum,
              'electric_wavelengths_nm':electric,'magnetic_wavelengths_nm':magnetic,
              'material_csv_sha256':MATERIAL_SHA,'layer_control':'Au volume replaced by air; finite substrate and all planes fixed'}
    recipe['simulation']['resonator_nk_csv'] = 'materials/as2se3_joseph_700nm.csv'
    recipe = jsonable(recipe)
    fingerprint = hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
    return config,g,settings,recipe,fingerprint


def run_case(case_id,output,preflight_only=False):
    output.mkdir(parents=True,exist_ok=True)
    with directory_lock(output):
        config,g,s,recipe,fingerprint = make_recipe(case_id,output)
        directory = output/case_id
        directory.mkdir(exist_ok=True)
        state_path = directory/'state.json'
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding='utf-8'))
            if state['config_fingerprint'] != fingerprint:
                raise ValueError('Research recipe changed; use a new output directory')
            if state['state']=='success':
                validate_saved_case(directory)
                print(f'Verified and reused {case_id}',flush=True)
                return
        state = {'state':'preflight' if preflight_only else 'running','recipe':recipe,'config_fingerprint':fingerprint}
        atomic_json(state_path,state)
        try:
            with GoldMeshSimulator(s,config.lumapi_path,recipe['gold_present'],recipe['vacuum_reference'],
                                   recipe['electric_wavelengths_nm'],recipe['magnetic_wavelengths_nm']) as sim:
                project = directory/'simulation.fsp'
                if preflight_only:
                    sim.project_path = project
                    sim.build(g)
                    sim.fdtd.save(str(project))
                    state.update(state='preflight_complete')
                else:
                    print(f'Running {case_id}: {s.maximum_mesh_nm:g} nm; {s.spectrum_points} spectral points',flush=True)
                    result = sim.run(g,project)
                    atomic_npz(directory/'sample.npz',**result,config_fingerprint=np.array(fingerprint))
                    state.update(state='success',sample_sha256=checksum(directory/'sample.npz'),
                                 project_sha256=checksum(project),elapsed_seconds=float(result['elapsed_seconds']),
                                 status=int(result['status']))
            atomic_json(state_path,state)
            if not preflight_only: validate_saved_case(directory)
        except BaseException as error:
            state.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
            atomic_json(state_path,state)
            raise


def validate_saved_case(directory):
    state = json.loads((directory/'state.json').read_text(encoding='utf-8'))
    if state['state']!='success' or checksum(directory/'sample.npz')!=state['sample_sha256']:
        raise ValueError('Research result is incomplete or its checksum changed')
    recipe=state['recipe']
    values=dict(recipe['simulation']); values['resonator_nk_csv']=str(PROJECT_ROOT/values['resonator_nk_csv'])
    from fdtd_pipeline.config import SimulationSettings
    for key in ('research_wavelengths_nm','power_validation_wavelengths_nm'): values[key]=tuple(values[key])
    s=SimulationSettings(**values)
    with np.load(directory/'sample.npz') as z:
        if str(z['config_fingerprint'])!=state['config_fingerprint'] or not np.array_equal(z['geometry_values'],recipe['geometry_values']):
            raise ValueError('Research result provenance or geometry differs')
        validate_spectrum(z['wavelength_nm'],z['reflectance'],z['transmittance'],s,int(z['status']))
        mesh_grid_summary({a:z['mesh_'+a+'_nm']/1e9 for a in ('x','y','z')},s.maximum_mesh_nm)
        if bool(z['gold_present'])!=bool(recipe['gold_present'] and not recipe['vacuum_reference']):
            raise ValueError('Saved Au presence differs')
        for vector,waves in (('E',recipe['electric_wavelengths_nm']),('H',recipe['magnetic_wavelengths_nm'])):
            if not waves: continue
            for plane in PLANES:
                name=f'nf_{plane}_{vector}'
                if not np.allclose(z[name+'_wavelength_nm'],waves,rtol=0,atol=1e-5):
                    raise ValueError('Saved field wavelength grid differs')
                if not np.isfinite(z[name+'_enhancement']).all() or np.any(z[name+'_enhancement']<0):
                    raise ValueError('Invalid saved field enhancement')
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','verify'))
    parser.add_argument('--case',required=True,choices=('bare8_preliminary','Au8','bare8','Au4','bare4','vacuum8','vacuum4'))
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'data/gold_mesh_comparison')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    if args.command=='run':run_case(args.case,args.output.resolve(),args.preflight_only)
    else: print(json.dumps(validate_saved_case(args.output/args.case),indent=2))
