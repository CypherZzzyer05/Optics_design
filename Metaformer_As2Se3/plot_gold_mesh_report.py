"""Create scientific plots only from the completed, audited four-case matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from fdtd_pipeline.config import PROJECT_ROOT
from analyze_gold_mesh_results import CASES,load_audited
from gold_mesh_research import geometry_source


COLORS={'Au8':'#df9b28','bare8':'#158aa0','Au4':'#a65d0e','bare4':'#174d78'}


def plots(output,destination):
    report=json.loads((destination/'analysis.json').read_text())
    if not report['complete_matrix']:raise ValueError('The comparison matrix is incomplete')
    cases={name:load_audited(output,name) for name in CASES}
    for name in CASES:
        if cases[name]['state']['sample_sha256']!=report['cases'][name]['sample_sha256']:
            raise ValueError('Analysis and plot source checksums differ')
    figures=destination/'figures';figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,2,figsize=(10,9),sharex=True,sharey='row')
    for column,mesh in enumerate((8,4)):
        for row,(key,label) in enumerate((('reflectance','R'),('transmittance','T'),('absorptance','A (total)'))):
            ax=axes[row,column]
            for stack in ('Au','bare'):
                name=stack+str(mesh);z=cases[name]['data']
                ax.plot(z['wavelength_nm'],z[key],color=COLORS[name],label=name,lw=1.15)
            ax.set_ylabel(label);ax.set_ylim(-.02,1.02);ax.grid(alpha=.2)
            if row==0:ax.set_title(f'{mesh} nm controlled Au / air comparison');ax.legend()
            if row==2:ax.set_xlabel('Wavelength (nm)')
    fig.tight_layout();fig.savefig(figures/'spectra.png',dpi=190);plt.close(fig)
    targets=[row['wavelength_nm'] for row in report['comparisons']['Au_8nm_vs_4nm']['field_at_common_wavelengths'][:2]]
    targets=sorted(set(targets))
    fig,axes=plt.subplots(2,2,figsize=(10,6))
    bare_peaks=report['comparisons']['bare_8nm_vs_4nm']['own_peak_mesh_comparison']['reference_peaks']
    for row,stack in enumerate(('Au','bare')):
        centers=targets if stack=='Au' else [peak['center_nm'] for peak in bare_peaks]
        for col,center in enumerate(centers):
            ax=axes[row,col]
            for mesh in (8,4):
                name=stack+str(mesh);z=cases[name]['data']
                ax.plot(z['wavelength_nm'],z['reflectance'],color=COLORS[name],label=name)
            ax.set_xlim(center-15,center+15);ax.set_ylabel('R');ax.legend();ax.grid(alpha=.2)
            ax.set_title(f'{stack}: near {center:.2f} nm')
            if row==1:ax.set_xlabel('Wavelength (nm)')
    fig.tight_layout();fig.savefig(figures/'mesh_zoom.png',dpi=190);plt.close(fig)
    geometry=geometry_source()
    for plane in ('mid','top'):
        for target in targets:
            profile=f'nf_{plane}_E';arrays={}
            for name in CASES:
                z=cases[name]['data'];w=z[profile+'_wavelength_nm'];i=int(np.argmin(abs(w-target)))
                if abs(w[i]-target)>1e-5:raise ValueError('A map frequency is missing')
                arrays[name]=z[profile+'_enhancement'][:,:,0,i]
            # One identical color scale for all four cases; no percentile clipping.
            lower=min(float(np.min(v[v>0])) for v in arrays.values());upper=max(float(v.max()) for v in arrays.values())
            fig,axes=plt.subplots(2,2,figsize=(8,7),layout='constrained')
            for ax,name in zip(axes.flat,CASES):
                z=cases[name]['data']
                image=ax.pcolormesh(z[profile+'_x_nm'],z[profile+'_y_nm'],arrays[name].T,
                    shading='auto',cmap='magma',norm=LogNorm(vmin=lower,vmax=upper),rasterized=True)
                for center,angle in zip(geometry.centers(),geometry.angles()):
                    corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])*geometry.l/2
                    corners=corners @ geometry.rotation_matrix(angle).T+center
                    ax.plot(corners[:,0],corners[:,1],color='cyan',lw=.65,alpha=.7)
                ax.set_title(name);ax.set_aspect('equal');ax.set_xlabel('x (nm)');ax.set_ylabel('y (nm)')
            fig.colorbar(image,ax=list(axes.flat),label='|E|^2 / |E_inc|^2 (log color scale)',shrink=.8)
            fig.suptitle(f'{plane} XY plane, {target:.3f} nm; shared color scale')
            fig.savefig(figures/f'E_{plane}_{target:.3f}.png',dpi=180);plt.close(fig)
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={'analysis_sha256':digest(destination/'analysis.json'),
              'case_sha256':{name:report['cases'][name]['sample_sha256'] for name in CASES},
              'figures':{path.name:digest(path) for path in figures.glob('*.png')}}
    (figures/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return figures


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'data/gold_mesh_comparison')
    parser.add_argument('--destination',type=Path,default=PROJECT_ROOT/'research_results/gold_mesh_comparison')
    args=parser.parse_args();print(plots(args.output,args.destination))
