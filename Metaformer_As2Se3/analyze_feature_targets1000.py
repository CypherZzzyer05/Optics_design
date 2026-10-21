"""Characterize saved spectra and select held-out double-dip inverse targets."""
import csv
import json
from pathlib import Path

import numpy as np

from explore_double_dips import double_pair
from fdtd_pipeline.config import Geometry
from fdtd_pipeline.pipeline import checksum

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'results/feature_targets_1000'


def analyze():
    path=ROOT/'examples/au100_metal8_1000/dataset.npz'
    audit=json.loads((ROOT/'results/inverse_au100_1000/data_audit.json').read_text())
    assert checksum(path)==audit['dataset_sha256']
    with np.load(path,allow_pickle=False) as archive:
        data={name:archive[name].copy() for name in archive.files}
    wavelength=data['wavelength_nm']
    rows=[]
    for index,spectrum in enumerate(data['spectra']):
        result=double_pair(wavelength,spectrum)
        if not result['dominant_pair']:
            continue
        pair=result['dominant_pair']
        # The legacy lambda/width descriptor is deliberately not called Q.
        dips=[{k:v for k,v in dip.items() if k!='lambda_over_width'} for dip in pair['dips']]
        row={'sample_id':int(data['sample_ids'][index]),'split':str(data['split_names'][index]),
             'score':pair['score'],'pair':{**pair,'dips':dips},
             'resolved_dip_count_above_prominence_0p1':sum(d['resolved'] for d in result['dips']),
             'geometry_values':data['geometry_values'][index].tolist()}
        rows.append(row)
    rows.sort(key=lambda row:-row['score'])
    passing={row['sample_id'] for row in rows}
    training_ids=data['sample_ids'][data['train_indices']]
    associations={}
    names=data['parameter_names'].tolist()
    for column,name in enumerate(names):
        values=data['geometry_values'][data['train_indices'],column]
        edges=np.quantile(values,[0,.25,.5,.75,1])
        bins=[]
        for index in range(4):
            selected=training_ids[(values>=edges[index])&
                ((values<=edges[index+1]) if index==3 else values<edges[index+1])]
            count=sum(int(sample) in passing for sample in selected)
            bins.append({'lower':float(edges[index]),'upper':float(edges[index+1]),
                         'samples':len(selected),'double_dip_count':count,'fraction':count/len(selected)})
        associations[name]=bins
    p_cut=associations['p'][3]['lower']
    dy_cut=associations['alpha_y'][0]['upper']
    region=(data['geometry_values'][:,0]>=p_cut)&(data['geometry_values'][:,4]<=dy_cut)
    region_counts={}
    for split in ('train','val','test'):
        ids=data[split+'_indices']
        inside=ids[region[ids]]
        outside=ids[~region[ids]]
        region_counts[split]={'inside_samples':len(inside),
            'inside_double_dip':sum(int(data['sample_ids'][i]) in passing for i in inside),
            'outside_samples':len(outside),
            'outside_double_dip':sum(int(data['sample_ids'][i]) in passing for i in outside)}
    table=np.atleast_1d(np.genfromtxt(ROOT/'results/inverse_au100_1000/test_predictions.csv',
                                    delimiter=',',names=True))
    by_id={int(row['sample_id']):row for row in table}
    fresh_test=[row for row in rows if row['split']=='test' and row['sample_id']>=100]
    selected=[row.copy() for row in fresh_test if row['resolved_dip_count_above_prominence_0p1']==2][:3]
    assert len(selected)==3
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for row in selected:
        record=by_id[row['sample_id']]
        row['inverse_predicted_geometry_values']=[float(record['pred_'+unit+'_'+name])
            for name,unit in zip(names,data['parameter_units'])]
        Geometry(*row['inverse_predicted_geometry_values'][:5],
                 theta_odd_deg=row['inverse_predicted_geometry_values'][5],
                 theta_even_deg=row['inverse_predicted_geometry_values'][6])
        index=int(np.flatnonzero(data['sample_ids']==row['sample_id'])[0])
        np.savetxt(OUTPUT/f"target_{row['sample_id']:06d}.csv",
            np.column_stack([wavelength,data['spectra'][index],data['transmittance'][index],data['absorptance'][index]]),
            delimiter=',',header='wavelength_nm,reflectance,transmittance,absorptance',comments='')
    result={'dataset_sha256':audit['dataset_sha256'],'samples':len(data['spectra']),
        'double_dip_count':len(rows),'double_dip_training_count':sum(int(i) in passing for i in training_ids),
        'double_dip_test_count':sum(row['split']=='test' for row in rows),
        'double_dip_previously_unseen_test_count':len(fresh_test),
        'criteria':{'max_dip_reflectance':.2,'minimum_prominence':.4,'minimum_width_grid_intervals':5,
                    'minimum_pair_separation_nm':30,'minimum_intervening_rise':.2,'edge_exclusion_nm':30},
        'selection':'Highest legacy double-pair score among previously unseen test structures with only two resolved dips above prominence 0.10',
        'training_only_quartile_associations':associations,'selected_targets':selected,'ranking':rows,
        'exploratory_region_rule':{'p_min_nm':p_cut,'dy_max_nm':dy_cut,'counts':region_counts,
                                  'threshold_source':'Highest P and lowest dY quartiles of the 800 training structures'},
        'inverse_checkpoint_sha256':checksum(ROOT/'models/inverse_au100_1000.pt'),
        'trained_weights_or_test_metrics_changed':False,'predicted_geometry_fdtd_validated':False,
        'quality_level':'pipeline_validation','stack':'Generator 1.0: Au100 below finite 350nm SiO2, As2Se3 tetramer above',
        'limitations':['Associations are marginal observations, not isolated parameter sweeps',
                       'Only two resolved dips above prominence 0.10 does not exclude weaker or under-resolved features',
                       'Half-prominence widths are not modal linewidths or Q factors',
                       'New model geometries require fresh FDTD spectra; do not equate the reference spectrum with model output']}
    (OUTPUT/'analysis.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    with (OUTPUT/'double_dip_ranking.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['sample_id','split','score','resolved_dips_prominence_0p1',*names,
                         'dip1_nm','dip1_R','dip1_half_prominence_width_nm','dip2_nm','dip2_R','dip2_half_prominence_width_nm'])
        for row in rows:
            values=[v for dip in row['pair']['dips'] for v in
                    (dip['wavelength_nm'],dip['reflectance_min'],dip['half_prominence_width_nm'])]
            writer.writerow([row['sample_id'],row['split'],row['score'],row['resolved_dip_count_above_prominence_0p1'],
                             *row['geometry_values'],*values])
    plot(result,data)
    print(json.dumps({'double_dip_count':len(rows),'fresh_test':len(fresh_test),
        'selected':[{'sample_id':r['sample_id'],'geometry':r['geometry_values'],
                    'predicted':r['inverse_predicted_geometry_values'],'dips':r['pair']['dips']} for r in selected]},indent=2))


def draw_cell(ax,values,title):
    from matplotlib.patches import Polygon,Rectangle
    g=Geometry(*values[:5],theta_odd_deg=values[5],theta_even_deg=values[6])
    ax.add_patch(Rectangle((-g.p/2,-g.p/2),g.p,g.p,facecolor='#f2f5f8',edgecolor='#758499',linestyle='--'))
    for vertices,label,color in zip(g.vertices(),('III','II','IV','I'),('#3265b5','#e19d47','#e19d47','#3265b5')):
        ax.add_patch(Polygon(vertices,facecolor=color,edgecolor='white',alpha=.9))
        center=vertices.mean(0)
        ax.text(*center,label,ha='center',va='center',color='white',fontweight='bold')
    ax.set(xlim=(-g.p*.56,g.p*.56),ylim=(-g.p*.56,g.p*.56),xlabel='x（nm）',ylabel='y（nm）',title=title)
    ax.set_aspect('equal')
    ax.grid(alpha=.15)


def plot(result,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':10})
    row=result['selected_targets'][0]
    index=int(np.flatnonzero(data['sample_ids']==row['sample_id'])[0])
    truth,pred=row['geometry_values'],row['inverse_predicted_geometry_values']
    fig,axes=plt.subplots(2,3,figsize=(16,10),layout='constrained')
    fig.suptitle(f"未参与训练的双谷目标：样本 {row['sample_id']} 与 Metaformer 逆向结构",fontsize=17,fontweight='bold')
    draw_cell(axes[0,0],truth,'产生目标光谱的原始结构')
    draw_cell(axes[0,1],pred,'逆向网络直接预测的结构（待回仿真）')
    ax=axes[0,2]
    for bottom,height,color,label in ((-450,100,'#d3a72f','Au 100 nm'),(-350,350,'#dce4ee','SiO2 350 nm'),(0,truth[2],'#3265b5','As2Se3 四聚体')):
        width=.52 if bottom==0 else .9
        ax.add_patch(Rectangle((.5-width/2,bottom),width,height,facecolor=color,edgecolor='white'))
        ax.text(.5,bottom+height/2,label,ha='center',va='center',color='white' if bottom==0 else '#263441')
    ax.set(xlim=(0,1),ylim=(-520,truth[2]+80),xticks=[],ylabel='z（nm）',title='原始层序示意（非截面）')
    wave=data['wavelength_nm']
    ax=axes[1,0]
    for name,field,color in (('R','spectra','#2457bd'),('T','transmittance','#269b8e'),('A','absorptance','#bd7539')):
        ax.plot(wave,data[field][index],label=name,color=color)
    ax.set(title='原始 FDTD 目标光谱',xlabel='波长（nm）',ylabel='源功率归一化 R / T / A',ylim=(-.02,1.02))
    ax.legend()
    ax.grid(alpha=.2)
    ax=axes[1,1]
    ax.plot(wave,data['spectra'][index],color='#2457bd')
    centers=[d['wavelength_nm'] for d in row['pair']['dips']]
    for d in row['pair']['dips']:
        ax.plot(d['wavelength_nm'],d['reflectance_min'],'o',color='#b25a35')
        ax.annotate(f"{d['wavelength_nm']:.1f} nm\nR={d['reflectance_min']:.3f}",
                    (d['wavelength_nm'],d['reflectance_min']),xytext=(8,18),textcoords='offset points')
    ax.set(xlim=(min(centers)-35,max(centers)+35),ylim=(-.02,1.02),title='目标双谷局部放大',xlabel='波长（nm）',ylabel='R')
    ax.grid(alpha=.2)
    ax=axes[1,2]
    ax.axis('off')
    labels=['P','L','H','dX','dY','θ奇','θ偶']
    table=ax.table(cellText=[[label,f'{a:.2f}',f'{b:.2f}'] for label,a,b in zip(labels,truth,pred)],
                   colLabels=['参数','原始结构','网络预测'],loc='upper center',cellLoc='center',bbox=[0,.4,1,.58])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    ax.text(0,.3,'长度单位 nm，角度单位 °。\nI/III 同角，II/IV 同角。\n\n实线光谱来自原始 FDTD 样本；\n预测结构尚无回仿真光谱。\n数据级别：pipeline_validation。',va='top',linespacing=1.5)
    fig.savefig(OUTPUT/'target_and_inverse_structure.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,(name,label) in zip(axes.flat,(('p','P'),('l','L'),('h','H'),('alpha_x','dX'),('alpha_y','dY'))):
        bins=result['training_only_quartile_associations'][name]
        ax.bar(np.arange(4),[100*b['fraction'] for b in bins],color='#3265b5')
        ax.set(title=label,xticks=np.arange(4),xticklabels=[f"{b['lower']:.0f}–{b['upper']:.0f}" for b in bins],
               xlabel='训练集参数四分位区间（nm）',ylabel='满足双主谷判据的比例（%）',ylim=(0,36))
        for i,b in enumerate(bins):
            ax.text(i,100*b['fraction']+.7,f"{b['double_dip_count']}/{b['samples']}",ha='center',fontsize=9)
        ax.grid(axis='y',alpha=.2)
    axes[1,2].axis('off')
    axes[1,2].text(.03,.95,'仅使用 800 条训练样本。\n\n这是多参数随机样本的边际关联，\n未固定其余六个参数。\n\n不能据此宣称单参数因果规律；\n还需逐参数受控扫描验证。',va='top',linespacing=1.7,fontsize=12)
    fig.suptitle('双主谷出现比例：训练集中的初步结构关联',fontsize=17,fontweight='bold')
    fig.savefig(OUTPUT/'training_associations.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    analyze()
