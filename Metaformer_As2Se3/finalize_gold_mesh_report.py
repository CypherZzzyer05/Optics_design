"""Wait for this local study, then prepare artifacts for human/agent visual QA.

This helper starts no solver, sends no messages and performs no Git publication.
The final PDF must be visually reviewed before delivery.
"""
import json
import os
from pathlib import Path
import subprocess
import time
from fdtd_pipeline.config import PROJECT_ROOT
from fdtd_pipeline.pipeline import atomic_json,directory_lock

SCIENCE=Path('D:/Anaconda3/envs/transformer_env/python.exe')
DEPENDENCIES=Path('C:/Users/lz/.cache/codex-runtimes/codex-primary-runtime/dependencies')
DOCUMENT=DEPENDENCIES/'python/python.exe'
RENDER=DEPENDENCIES/'native/poppler/Library/bin/pdftoppm.exe'


def finish():
    output=PROJECT_ROOT/'data/gold_mesh_comparison'
    destination=PROJECT_ROOT/'research_results/gold_mesh_comparison'
    status_path=output/'report_preparation.json'
    state={'state':'waiting_for_completed_matrix','started_at_unix':time.time()}
    with directory_lock(output/'report_coordinator'):
        atomic_json(status_path,state)
        try:
            while True:
                workflow=json.loads((output/'workflow.json').read_text())
                if workflow['state']=='cases_completed':break
                if workflow['state']=='needs_attention':raise RuntimeError('Simulation workflow needs attention: '+workflow.get('error','unknown'))
                if time.time()-state['started_at_unix']>24*3600:raise TimeoutError('Still waiting after 24 hours; no solver was stopped')
                time.sleep(30)
            env=os.environ.copy();env['MPLBACKEND']='Agg';env['OMP_NUM_THREADS']='1'
            def run(name,command):
                state.update(state='preparing',current_step=name,updated_at_unix=time.time());atomic_json(status_path,state)
                print('Report preparation: '+name,flush=True)
                subprocess.run([str(value) for value in command],cwd=PROJECT_ROOT,env=env,check=True)
            run('analyze',[SCIENCE,PROJECT_ROOT/'analyze_gold_mesh_results.py'])
            run('plot',[SCIENCE,PROJECT_ROOT/'plot_gold_mesh_report.py'])
            run('archive',[SCIENCE,PROJECT_ROOT/'archive_gold_mesh_results.py'])
            run('author_pdf',[DOCUMENT,PROJECT_ROOT/'author_gold_mesh_report.py'])
            render_directory=PROJECT_ROOT/'tmp/pdfs/gold_mesh';render_directory.mkdir(parents=True,exist_ok=True)
            pdf=PROJECT_ROOT/'output/pdf/Au_8nm_4nm_comparison_report.pdf'
            run('render_for_visual_review',[RENDER,'-r','110','-png',pdf,render_directory/'page'])
            report=json.loads((destination/'analysis.json').read_text());decision=report['decision']
            lines=['# As2Se3 四聚体：Au 镀层及 8/4 nm 网格对照报告',
                   '四组同结构对照及两组真空归一化均已完成，原生 FSP 光谱和 E/H 场复读通过。',
                   '当前结构建议：'+('移除 Au。' if decision['remove_gold'] else '保留 Au。')+
                   f"正式精细结果采用 {decision['selected_final_mesh_nm']} nm 网格。",
                   '标准：反射谷中心 2 nm、谷底反射率 3 个百分点、半突出度宽度 10%，区域平均及面积加权 P95 场强度增强 10%。']
            for key,c in report['comparisons'].items():
                rows=c['field_at_common_wavelengths']
                lines.append(f"- {key}: 最大光谱绝对差 R={c['maximum_absolute_R_difference']:.4f}、T={c['maximum_absolute_T_difference']:.4f}、A={c['maximum_absolute_A_difference']:.4f}；"
                    f"目标频率/核心区域最大场平均值相对变化 {100*max(row['relative_mean_change'] for row in rows):.1f}%，P95 {100*max(row['relative_p95_change'] for row in rows):.1f}%；"
                    f"全部替代判据通过={c['small_effect_under_current_criteria']}。")
            auxiliary=report['comparisons']['bare_8nm_vs_4nm']['own_peak_mesh_comparison']
            lines.append('无金结构自身反射峰及其共同峰波长处的E/H场网格检查：'+
                         f"同时通过={auxiliary['own_peaks_and_sampled_fields_stable']}。未保留Au目标双谷与网格误差分开判断。")
            lines.append('局限：固定结构、法向 x 偏振；总吸收不等于 As2Se3 单独吸收；场限于两个 XY 平面和选定频率；数值检查不证明材料拟合或实验精度。报告图表和原始证据见已核验的 Git 版本。')
            summary=destination/'report_summary.md'
            summary.write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
            state.update(state='ready_for_visual_review',pdf=str(pdf),render_directory=str(render_directory),
                         summary=str(summary),completed_at_unix=time.time(),not_yet_delivered=True,
                         delivery='local desktop PDF and Optics_design repository')
            atomic_json(status_path,state)
        except BaseException as error:
            state.update(state='needs_attention',error=str(error),updated_at_unix=time.time());atomic_json(status_path,state)
            raise


if __name__=='__main__':finish()
