"""Validate the trained checkpoint and summarize the fixed 1000-sample run."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

from fdtd_pipeline.config import Geometry
from fdtd_pipeline.pipeline import checksum
from predict_inverse import load_inverse
from spectrum_dataset import SpectrumGeometryDataset
from train_metaformer import evaluate, normalized_errors, regression_metrics

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'results/inverse_au100_1000'


def circular_constant(values):
    """Minimize squared geodesic error using training labels only."""
    values=np.asarray(values,dtype=float)%1
    cuts=np.unique(np.r_[0.,(values+.5)%1,1.])
    candidates=list(cuts[:-1])
    for left,right in zip(cuts[:-1],cuts[1:]):
        midpoint=(left+right)/2
        unwrapped=midpoint-((midpoint-values+.5)%1-.5)
        candidates.append(float(np.clip(unwrapped.mean(),left,right))%1)
    return min(candidates,key=lambda c:np.mean(((c-values+.5)%1-.5)**2))


def summarize():
    torch.set_num_threads(4)
    pointer=json.loads((ROOT/'runs/inverse_au100_metal8_1000/latest_run.json').read_text())
    assert pointer['state']=='completed'
    run=Path(pointer['run_dir'])
    report=json.loads((run/'metrics.json').read_text(encoding='utf-8'))
    audit=json.loads((OUTPUT/'data_audit.json').read_text(encoding='utf-8'))
    dataset=ROOT/'examples/au100_metal8_1000/dataset.npz'
    assert checksum(dataset)==report['dataset_sha256']==audit['dataset_sha256']
    assert checksum(run/'best.pt')==report['best_checkpoint_sha256']
    assert report['model_parameters']==3998087
    assert report['config']['seed']==44 and report['config']['model']['input_normalization']=='train_standard'
    assert report['config']['model']['spectral_frontend']=='linear'
    with np.load(dataset,allow_pickle=False) as archive:
        data={key:archive[key].copy() for key in archive.files}
    names=data['parameter_names'].tolist()
    test=SpectrumGeometryDataset(dataset,'test',allow_validation_data=True)
    table=np.atleast_1d(np.genfromtxt(run/'test_predictions.csv',delimiter=',',names=True))
    assert np.array_equal(table['sample_id'].astype(int),test.sample_ids)
    target=np.column_stack([table['true_01_'+name] for name in names])
    prediction=np.column_stack([table['pred_01_'+name] for name in names])
    np.testing.assert_allclose(target,test.labels.numpy(),atol=6e-10,rtol=0)
    score=lambda pred,truth:regression_metrics(pred,truth,test.lower_values,test.upper_values,names,'square_periodic')
    recomputed=score(prediction,target)
    assert abs(recomputed['normalized_mse']-report['metrics']['test']['normalized_mse'])<1e-9
    model,metadata=load_inverse(run/'best.pt')
    for name,expected in metadata['source_sha256'].items():
        assert checksum(ROOT/name)==expected
    assert metadata['input_normalization_fit_sample_ids']==data['sample_ids'][data['train_indices']].tolist()
    train_spectra=torch.from_numpy(data['spectra'][data['train_indices']].astype(np.float32))
    torch.testing.assert_close(model.spectrum_mean,train_spectra.mean(0),rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(model.spectrum_scale,train_spectra.std(0,unbiased=False).clamp_min(.02),rtol=1e-6,atol=1e-7)
    _,reloaded,_=evaluate(model,DataLoader(test,batch_size=8),torch.device('cpu'))
    reload_difference=float(np.max(np.abs(reloaded-prediction)))
    assert reload_difference<1e-5
    constant=data['geometry_01'][data['train_indices']].mean(0).astype(float)
    for index in (5,6):
        constant[index]=circular_constant(data['geometry_01'][data['train_indices'],index])
    baseline_prediction=np.broadcast_to(constant,target.shape)
    baseline=score(baseline_prediction,target)
    fresh=~np.isin(test.sample_ids,audit['previously_seen_test_sample_ids'])
    assert int(fresh.sum())==audit['previously_unseen_test_count']
    fresh_metrics=score(prediction[fresh],target[fresh])
    old_path=ROOT/'models/inverse_au100_selected.pt'
    old_model,old_metadata=load_inverse(old_path)
    assert old_metadata['parameter_names']==names
    assert old_metadata['config']['model']==metadata['config']['model']
    _,old_prediction,_=evaluate(old_model,DataLoader(test,batch_size=8),torch.device('cpu'))
    old_fresh_metrics=score(old_prediction[fresh],target[fresh])
    physical=test.lower_values+prediction*(test.upper_values-test.lower_values)
    invalid=[]
    for sample_id,values in zip(test.sample_ids,physical):
        try:
            Geometry(*values[:5],theta_odd_deg=values[5],theta_even_deg=values[6])
        except ValueError as error:
            invalid.append({'sample_id':int(sample_id),'reason':str(error)})
    errors=normalized_errors(prediction,target,'square_periodic')
    base_errors=normalized_errors(baseline_prediction,target,'square_periodic')
    result={'dataset_sha256':audit['dataset_sha256'],'data_audit':'data_audit.json',
        'seed':44,'model_parameters':report['model_parameters'],'epochs_completed':report['epochs_completed'],
        'best_epoch':report['best_epoch'],'training_seconds':report['training_seconds'],
        'optimizer_steps':report['optimizer_steps'],'best_validation_mse':report['best_val_mse'],
        'train_metrics':report['metrics']['train'],'validation_metrics':report['metrics']['val'],
        'test_metrics':recomputed,'training_only_constant_baseline':baseline,
        'improvement_vs_constant_percent':100*(1-recomputed['normalized_mse']/baseline['normalized_mse']),
        'length_mse_improvement_vs_constant_percent':100*(1-np.mean(errors[:,:5]**2)/np.mean(base_errors[:,:5]**2)),
        'angle_mse_improvement_vs_constant_percent':100*(1-np.mean(errors[:,5:]**2)/np.mean(base_errors[:,5:]**2)),
        'previously_unseen_test':{'sample_ids':test.sample_ids[fresh].tolist(),'new_model':fresh_metrics,
            'previous_100_sample_model':old_fresh_metrics,'old_checkpoint_sha256':checksum(old_path),
            'mse_reduction_vs_previous_model_percent':100*(1-fresh_metrics['normalized_mse']/old_fresh_metrics['normalized_mse'])},
        'predicted_geometry_valid_count':len(test)-len(invalid),'invalid_predicted_geometries':invalid,
        'max_reloaded_cpu_vs_exported_cuda_prediction_difference':reload_difference,
        'normalization_fit_only_on_training_rows_verified':True,'test_metrics_recomputed_from_csv':True,
        'best_checkpoint_selected_by_validation_only':True,'source_data_modified':False,
        'fdtd_run_performed_by_this_script':False,
        'subsequent_fdtd_report':'../inverse_roundtrip_207/report.md',
        'quality_level':'pipeline_validation',
        'material_fit_quality':'known_severe_Au_loss_deficiency',
        'limitations':['One fixed training seed, no new hyperparameter search',
            '12 original test structures appeared in the earlier 100-sample dataset; fresh-88 results reported separately',
            'Geometry prediction errors do not establish reconstructed optical spectra or BIC Q factors',
            'Archived Au FDTD fit severely underestimates Ohmic loss; real-device validation requires corrected material data']}
    for name in ('metrics.json','history.json','test_predictions.csv','val_predictions.csv','training.log'):
        shutil.copy2(run/name,OUTPUT/name)
    checkpoint=ROOT/'models/inverse_au100_1000.pt'
    shutil.copy2(run/'best.pt',checkpoint)
    result['checkpoint']={'path':str(checkpoint.relative_to(ROOT)),'sha256':checksum(checkpoint)}
    (OUTPUT/'summary.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    plot_results(result,prediction,target,test.lower_values,test.upper_values)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def plot_results(result,prediction,target,lower,upper):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':10})
    history=json.loads((OUTPUT/'history.json').read_text())
    score=result['test_metrics']
    base=result['training_only_constant_baseline']
    blue,gray,green='#2457bd','#a2adbb','#008a78'
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    fig.suptitle('Metaformer：1000 个 FDTD 数值样本的逆向训练与验证',fontsize=17,fontweight='bold')
    fig.supxlabel('Au 底层结构（生成器 1.0） · 700–1700 nm · 800 / 100 / 100 · seed 44')
    ax=axes[0,0]
    ax.plot([h['epoch'] for h in history],[h['train_mse'] for h in history],label='训练（含 dropout）',color=blue)
    ax.plot([h['epoch'] for h in history],[h['val_mse'] for h in history],label='验证',color=green)
    ax.axvline(result['best_epoch'],linestyle='--',color='#bd7d20',label='验证选中轮次')
    ax.set(title='训练过程',xlabel='轮次',ylabel='周期归一化结构 MSE')
    ax.grid(alpha=.2)
    ax.legend()
    keys=list(score['mae_nm'])
    ax=axes[0,1]
    x=np.arange(5)
    ax.bar(x-.18,[score['mae_nm'][name] for name in keys],.36,color=blue,label='模型')
    ax.bar(x+.18,[base['mae_nm'][name] for name in keys],.36,color=gray,label='训练集常数基线')
    ax.set(title='100 个测试结构：尺寸误差',ylabel='MAE（nm）',xticks=x,xticklabels=['P','L','H','dX','dY'])
    ax.legend()
    ax.grid(axis='y',alpha=.2)
    ax=axes[1,0]
    keys=list(score['mae_deg'])
    x=np.arange(2)
    ax.bar(x-.18,[score['mae_deg'][name] for name in keys],.36,color=blue,label='模型')
    ax.bar(x+.18,[base['mae_deg'][name] for name in keys],.36,color=gray,label='训练集周期最优常数')
    ax.set(title='旋转角误差（按正方形 90° 周期）',ylabel='周期 MAE（°）',xticks=x,xticklabels=['奇数象限角','偶数象限角'],ylim=(0,45))
    ax.legend(fontsize=9)
    ax.grid(axis='y',alpha=.2)
    fresh=result['previously_unseen_test']
    ax=axes[1,1]
    ax.axis('off')
    ax.text(.02,.97,f"测试 MSE：{score['normalized_mse']:.6f}\n常数基线：{base['normalized_mse']:.6f}\n"
        f"相对降低：{result['improvement_vs_constant_percent']:.1f}%\n\n"
        f"88 个此前未出现的测试结构\n新模型 MSE：{fresh['new_model']['normalized_mse']:.6f}\n"
        f"旧 100 样本模型 MSE：{fresh['previous_100_sample_model']['normalized_mse']:.6f}\n"
        f"同一测试结构上降低：{fresh['mse_reduction_vs_previous_model_percent']:.1f}%\n\n"
        '仅以验证集选择权重；单种子训练。\n已知 Au 损耗拟合严重缺失。\n后续回仿真另见样本 207 报告。',
        va='top',fontsize=12,linespacing=1.45)
    fig.savefig(OUTPUT/'validation_overview.png',dpi=160)
    plt.close(fig)
    truth=lower+target*(upper-lower)
    pred=lower+prediction*(upper-lower)
    fig,axes=plt.subplots(2,4,figsize=(16,7),layout='constrained')
    labels=['P','L','H','dX','dY','奇数象限角','偶数象限角']
    for i,ax in enumerate(axes.flat):
        if i==7:
            ax.axis('off')
            ax.text(.05,.9,'100 个测试结构\n\n长度：真实值与预测值\n角度：真实值与周期误差\n\n角度周期为 90°，\n不使用普通散点距离\n判断跨边界的预测误差。\n\n已知 Au 损耗拟合严重缺失，\n本图用于数值流程验证。',va='top',linespacing=1.7)
        elif i<5:
            ax.scatter(truth[:,i],pred[:,i],s=17,alpha=.6,color=blue)
            ax.plot([lower[i],upper[i]],[lower[i],upper[i]],'--',color=gray)
            ax.set(title=labels[i],xlabel='真实值（nm）',ylabel='预测值（nm）')
        else:
            error=(pred[:,i]-truth[:,i]+45)%90-45
            ax.scatter(truth[:,i],error,s=17,alpha=.6,color=green)
            ax.axhline(0,linestyle='--',color=gray)
            ax.set(title=labels[i],xlabel='真实角度（°）',ylabel='周期预测误差（°）',ylim=(-45,45))
        ax.grid(alpha=.2)
    fig.suptitle('留出测试集：七参数预测分布',fontsize=17,fontweight='bold')
    fig.savefig(OUTPUT/'parameter_predictions.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    summarize()
