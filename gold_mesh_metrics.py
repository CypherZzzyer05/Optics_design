"""Area-weighted, fixed-region field metrics and direct-frequency sampling plans."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

from fdtd_pipeline.config import PROJECT_ROOT
from fdtd_pipeline.pipeline import atomic_json,checksum
from gold_mesh_research import geometry_source,validate_saved_case


def axis_weights(coordinates,lower,upper):
    coordinates=np.asarray(coordinates,dtype=float)
    if len(coordinates)<2 or not np.isfinite(coordinates).all() or np.any(np.diff(coordinates)<=0):
        raise ValueError('Spatial coordinates must be finite and strictly increasing')
    bounds=np.r_[lower,(coordinates[1:]+coordinates[:-1])/2,upper]
    bounds=np.clip(bounds,lower,upper)
    weights=np.diff(bounds)
    if np.any(weights<0) or not np.isclose(weights.sum(),upper-lower):
        raise ValueError('Invalid spatial integration weights')
    return weights


def weighted_statistics(values,weights,mask=None):
    values,weights=np.asarray(values,dtype=float),np.asarray(weights,dtype=float)
    if values.shape!=weights.shape or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('Invalid intensity enhancement or integration shape')
    mask=np.ones(values.shape,dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    if mask.shape!=values.shape or np.any(weights<0) or not np.isfinite(weights).all():
        raise ValueError('Invalid physical region or area weights')
    keep=mask & (weights>0)
    if not np.any(keep):raise ValueError('No sampled area in the selected physical region')
    v,w=values[keep],weights[keep]
    order=np.argsort(v); cumulative=np.cumsum(w[order]); total=float(cumulative[-1])
    percentile=float(v[order[min(np.searchsorted(cumulative,.95*total),len(v)-1)]])
    return {'mean_intensity_enhancement':float(np.sum(v*w)/total),
            'p95_intensity_enhancement':percentile,'maximum_intensity_enhancement':float(v.max()),
            'sampled_area_nm2':total,'sampled_nodes':int(len(v))}


def physical_masks(x,y,geometry,edge_margin_nm=10):
    """Evaluate the same physical squares, excluding points near their sharp edges."""
    xx,yy=np.meshgrid(x,y,indexing='ij')
    points=np.stack((xx,yy),axis=-1)
    inside=np.zeros(xx.shape,dtype=bool); bulk=np.zeros(xx.shape,dtype=bool)
    distance=np.full(xx.shape,np.inf)
    for center,angle in zip(geometry.centers(),geometry.angles()):
        local=(points-np.asarray(center)) @ geometry.rotation_matrix(angle)
        absolute=np.abs(local); extent=geometry.l/2
        inside|=np.max(absolute,axis=-1)<=extent
        bulk|=np.max(absolute,axis=-1)<=extent-edge_margin_nm
        delta=np.maximum(absolute-extent,0)
        distance=np.minimum(distance,np.linalg.norm(delta,axis=-1))
    return {'all':np.ones(xx.shape,dtype=bool),'pillar_bulk':bulk,
            'air_away_from_edges':(~inside)&(distance>=edge_margin_nm)}


def profile_statistics(sample,name,geometry):
    x,y=sample[name+'_x_nm'],sample[name+'_y_nm']
    intensity=sample[name+'_enhancement']
    if intensity.shape!=(len(x),len(y),1,len(sample[name+'_wavelength_nm'])):
        raise ValueError('Expected one XY plane with a preserved frequency axis')
    weights=np.outer(axis_weights(x,-geometry.p/2,geometry.p/2),axis_weights(y,-geometry.p/2,geometry.p/2))
    masks=physical_masks(x,y,geometry) if '_mid_' in name else {'all':np.ones(weights.shape,dtype=bool)}
    return [{'wavelength_nm':float(w),'regions':{region:weighted_statistics(intensity[:,:,0,i],weights,mask)
             for region,mask in masks.items()}} for i,w in enumerate(sample[name+'_wavelength_nm'])]


def sampling_plan(wavelength,gold,bare):
    """Direct subset of real 3000-point spectra; no fabricated finer spectrum."""
    wavelength=np.asarray(wavelength,dtype=float)
    if len(wavelength)!=3000 or np.any(np.diff(wavelength)<=0):
        raise ValueError('Sampling selection requires the original 3000-point grid')
    source=json.loads((PROJECT_ROOT/'research_results/double_dip_mesh_5_4/comparison.json').read_text())
    targets=[d['wavelength_nm'] for d in source['fine_pair']['dips']]
    selected=set(range(0,len(wavelength),3)); selected.add(len(wavelength)-1)
    checks=set(np.round(np.linspace(0,len(wavelength)-1,31)).astype(int))
    electric=set(np.round(np.linspace(0,len(wavelength)-1,26)).astype(int))
    magnetic=set()
    def window(center,radius):return np.flatnonzero(abs(wavelength-center)<=radius)
    for center in targets:
        selected.update(window(center,24)); checks.update(window(center,12))
        electric.update(window(center,3)); magnetic.add(int(np.argmin(abs(wavelength-center))))
    bare_peaks,_=find_peaks(np.asarray(bare['reflectance']),prominence=.02)
    bare_peaks=sorted(bare_peaks,key=lambda i:bare['reflectance'][i],reverse=True)[:2]
    for idx in bare_peaks:
        center=wavelength[idx]
        selected.update(window(center,12)); electric.update(window(center,3)); magnetic.add(int(idx))
        checks.update(np.flatnonzero(abs(wavelength-center)<=1))
    for data in (gold,bare):
        for signal in ('reflectance','transmittance','absorptance'):
            for sign in (1,-1):
                indices,_=find_peaks(sign*np.asarray(data[signal]),prominence=.02)
                for idx in indices:selected.update(window(wavelength[idx],6))
    selected.update(electric); selected.update(magnetic); selected.update(checks)
    # The entire source band has approximately 1 nm coverage, with dense peak windows.
    return {'version':1,'spectral_wavelengths_nm':wavelength[sorted(selected)].tolist(),
            'power_check_wavelengths_nm':wavelength[sorted(checks)].tolist(),
            'electric_wavelengths_nm':wavelength[sorted(electric)].tolist(),
            'magnetic_wavelengths_nm':wavelength[sorted(magnetic)].tolist(),
            'gold_target_wavelengths_nm':targets,'bare_peak_wavelengths_nm':wavelength[bare_peaks].tolist(),
            'scope':'Direct shared samples across 700-1700 nm plus dense reference-feature windows; field maxima concern sampled frequencies and two XY planes.'}


def make_plan(output):
    state=validate_saved_case(output/'bare8_preliminary')
    gold_path=PROJECT_ROOT/'research_results/double_dip_controls8/samples/Au_full_difference.npz'
    bare_path=output/'bare8_preliminary/sample.npz'
    with np.load(gold_path) as gold,np.load(bare_path) as bare:
        if not np.allclose(gold['wavelength_nm'],bare['wavelength_nm'],rtol=0,atol=1e-5):
            raise ValueError('Preliminary spectral grids differ')
        plan=sampling_plan(gold['wavelength_nm'],gold,bare)
    plan.update(gold_reference_sha256=checksum(gold_path),bare_reference_sha256=checksum(bare_path),
                geometry_values=state['recipe']['geometry_values'],
                unchanged_criteria={'center_nm':2.,'absolute_R_change':.03,'relative_width_change':.1,
                                    'relative_field_mean_change':.1,'relative_field_p95_change':.1})
    path=output/'sampling_plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        raise ValueError('Sampling plan changed; use another research output directory')
    atomic_json(path,plan)
    print(json.dumps({'spectral_points':len(plan['spectral_wavelengths_nm']),
                      'electric_points':len(plan['electric_wavelengths_nm']),
                      'magnetic_points':len(plan['magnetic_wavelengths_nm']),
                      'power_check_points':len(plan['power_check_wavelengths_nm'])}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=PROJECT_ROOT/'data/gold_mesh_comparison')
    args=p.parse_args();make_plan(args.output)
