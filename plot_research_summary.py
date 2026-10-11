"""Plot angle-dependent candidate spectra from the preserved real FDTD results."""
import argparse
from pathlib import Path
import numpy as np
from fdtd_pipeline.config import PROJECT_ROOT


def plot(study, output, verification=None, refit=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,2,figsize=(11,7))
    colors=('#7f8b9b','#d48623','#6c48ab')
    for column,metal in enumerate(('Au','Ag')):
        for case,label,color in zip(('aligned','opposite_10','opposite_20'),('0 / 0 deg','+10 / -10 deg','+20 / -20 deg'),colors):
            with np.load(study/'samples'/f'{metal}_{case}.npz',allow_pickle=False) as data:
                w,r=data['wavelength_nm'].copy(),data['reflectance'].copy()
            axes[0,column].plot(w,r,color=color,label=label,lw=1.6)
            axes[1,column].plot(w,r,color=color,label=label,lw=1.6)
        axes[0,column].set(xlim=(700,1700),ylim=(.15,1.05),title=f'{metal} | full near-IR band',ylabel='Reflectance')
        axes[0,column].axvspan(1688,1700,color='orange',alpha=.25)
        axes[1,column].set(xlim=(840,885),ylim=(.15,.8),title=f'{metal} | dominant dip (20 nm mesh)',ylabel='Reflectance')
        for ax in axes[:,column]:
            ax.grid(alpha=.2);ax.set_xlabel('Wavelength (nm)');ax.legend(fontsize=8)
    if verification and verification.exists():
        with np.load(verification,allow_pickle=False) as data:
            axes[1,0].plot(data['wavelength_nm'],data['reflectance'],'--',color='#008c7d',lw=1.3,label='+20 / -20 deg, 10 nm mesh')
    if refit and refit.exists():
        with np.load(refit,allow_pickle=False) as data:
            axes[1,0].plot(data['wavelength_nm'],data['reflectance'],':',color='#c23c51',lw=1.5,label='+20 / -20 deg, refitted As2Se3')
    axes[1,0].legend(fontsize=7)
    fig.suptitle('As2Se3 tetramer candidates | exploratory FDTD comparison',fontsize=15)
    fig.tight_layout();output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=170);plt.close(fig)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=PROJECT_ROOT/'research_results/near_ir_study')
    parser.add_argument('--verification',type=Path,default=PROJECT_ROOT/'research_results/candidate_verification/Au_opposite_20_mesh_10.npz')
    parser.add_argument('--refit',type=Path,default=PROJECT_ROOT/'research_results/near_ir_refit/samples/Au_opposite_20.npz')
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'docs/figures/candidate_comparison.png')
    args=parser.parse_args();print(plot(args.study,args.output,args.verification,args.refit))
