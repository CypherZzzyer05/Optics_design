"""Independently reload a completed FSP and verify geometry, mesh, and exported spectra."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_double_mesh import read_case
from fdtd_pipeline.config import PROJECT_ROOT, Geometry
from fdtd_pipeline.lumerical import FDTDSimulator, mesh_grid_summary, validate_spectrum
from fdtd_pipeline.pipeline import atomic_json, checksum


def audit(directory, case_id, lumapi_path, output):
    case = read_case(directory, case_id)
    values = case['entry']['geometry_values']
    geometry = Geometry(*values[:5], theta_odd_deg=values[5], theta_even_deg=values[6])
    settings = case['settings']
    project = directory / 'projects' / (case_id + '.fsp')
    if not project.is_file():
        raise FileNotFoundError('Native audit requires the original completed FSP')
    project_sha256 = checksum(project)
    objects = []
    with FDTDSimulator(settings, lumapi_path) as simulator:
        f = simulator.fdtd
        f.load(str(project))
        for i, (center, angle) in enumerate(zip(geometry.centers(), geometry.angles())):
            name = f'si_{i}'
            dimensions = {key: float(f.getnamed(name, key))*1e9
                          for key in ('x', 'y', 'x span', 'y span', 'z min', 'z max')}
            expected = dict(zip(dimensions, (*center, geometry.l, geometry.l, 0, geometry.h)))
            if not all(np.isclose(dimensions[key], value, rtol=0, atol=1e-6) for key, value in expected.items()):
                raise ValueError(f'Native dimensions or center differ: {name}')
            rotation = float(f.getnamed(name, 'rotation 1'))
            material = str(f.getnamed(name, 'material'))
            axis = str(f.getnamed(name, 'first axis'))
            if not np.isclose(rotation, -angle) or axis != 'z' or material != settings.resonator_material:
                raise ValueError(f'Native material or angle differs: {name}')
            objects.append({'name': name, **dimensions, 'project_angle_deg': angle,
                            'native_rotation_deg': rotation, 'material': material, 'axis': axis})
        substrate = {key: float(f.getnamed('substrate', key))*1e9 for key in ('z min', 'z max')}
        substrate['material'] = str(f.getnamed('substrate', 'material'))
        metal = {key: float(f.getnamed('back_metal', key))*1e9 for key in ('z min', 'z max')}
        metal['material'] = str(f.getnamed('back_metal', 'material'))
        if (not np.isclose(substrate['z min'], -settings.substrate_depth_nm)
                or not np.isclose(substrate['z max'], 0)
                or substrate['material'] != settings.substrate_material
                or not np.isclose(metal['z max'], substrate['z min'])
                or not np.isclose(metal['z min'], substrate['z min']-settings.metal_thickness_nm)
                or metal['material'] != settings.metal_material):
            raise ValueError('Native substrate/metal stack differs from the intended finite stack')
        for axis in ('x', 'y'):
            if not np.isclose(float(f.getnamed('FDTD', axis+' span'))*1e9, geometry.p):
                raise ValueError('Native period differs')
        if not np.isclose(float(f.getnamed('source', 'polarization angle')), settings.polarization_deg):
            raise ValueError('Native polarization differs')
        if not np.isclose(float(f.getnamed('FDTD', 'auto shutoff min')), settings.auto_shutoff_min):
            raise ValueError('Native automatic shutoff threshold differs')
        for key, wanted in (('wavelength start', settings.wavelength_start_nm),
                            ('wavelength stop', settings.wavelength_stop_nm)):
            if not np.isclose(float(f.getnamed('source', key))*1e9, wanted):
                raise ValueError('Native source wavelength band differs')
        global_mesh = {axis: float(f.getnamed('global_mesh_limit', 'd'+axis))*1e9 for axis in ('x', 'y', 'z')}
        if not all(np.isclose(step, settings.maximum_mesh_nm) for step in global_mesh.values()):
            raise ValueError('Native full-domain mesh limit differs')
        coordinates = {axis: np.asarray(f.getresult('FDTD', axis)).reshape(-1) for axis in ('x', 'y', 'z')}
        actual_mesh = mesh_grid_summary(coordinates, settings.maximum_mesh_nm)
        for axis, grid in coordinates.items():
            if not np.allclose(grid*1e9, case['data'][f'mesh_{axis}_nm'], rtol=0, atol=1e-6):
                raise ValueError('Native grid differs from exported grid')
        status = int(np.asarray(f.getresult('FDTD', 'status')).item())
        r, t = f.getresult('R', 'T'), f.getresult('T', 'T')
        if not np.allclose(r['lambda'], t['lambda'], rtol=1e-10, atol=1e-15):
            raise ValueError('Native R/T wavelength grids differ')
        spectrum = validate_spectrum(np.asarray(r['lambda'])*1e9, r['T'], -np.asarray(t['T']), settings, status)
        errors = {}
        for key in ('wavelength_nm', 'reflectance', 'transmittance', 'absorptance'):
            errors[key] = float(np.max(np.abs(spectrum[key]-case['data'][key])))
            if not np.allclose(spectrum[key], case['data'][key], rtol=0, atol=1e-8):
                raise ValueError(f'Exported {key} differs from the saved native simulation')
    if checksum(project) != project_sha256:
        raise ValueError('Native audit modified the source project')
    report = {'case_id': case_id, 'project_sha256': project_sha256, 'sample_sha256': checksum(case['path']),
              'native_geometry_verified': True, 'native_spectra_match_export': True,
              'all_actual_mesh_cells_below_10_nm': True, 'status': status, 'objects': objects,
              'substrate': substrate, 'metal': metal, 'actual_mesh': actual_mesh,
              'native_vs_export_max_abs_errors': errors,
              'note': 'Independent native reload verifies the saved result, not material accuracy or mesh convergence.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=PROJECT_ROOT / 'data/double_dip_controls8')
    parser.add_argument('--case-id', default='Au_full_difference')
    parser.add_argument('--lumapi-path', default='D:/Program Files/Lumerical/v241/api/python/lumapi.py')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.directory / (args.case_id + '_native_audit.json')
    print(json.dumps(audit(args.directory, args.case_id, args.lumapi_path, output), indent=2))
