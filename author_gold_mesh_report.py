"""Author a Chinese PDF from verified analysis and its scientific plots.

Run with the bundled document Python (reportlab), after analysis and plotting.
No numerical simulation or result fabrication is performed here.
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak

ROOT=Path(__file__).resolve().parent
NAMES={'Au8':'镀金 8 nm','bare8':'无金 8 nm','Au4':'镀金 4 nm','bare4':'无金 4 nm'}
METRICS=('mean_intensity_enhancement','p95_intensity_enhancement')


def author(analysis_path,output):
    report=json.loads(analysis_path.read_text(encoding='utf-8'))
    if not report.get('complete_matrix') or set(report['cases'])!=set(NAMES):
        raise ValueError('Do not author a final report from a partial matrix')
    if any(case['status']!=2 for case in report['cases'].values()):raise ValueError('Unconverged case')
    figures=analysis_path.parent/'figures'
    manifest=json.loads((figures/'manifest.json').read_text(encoding='utf-8'))
    if manifest['analysis_sha256']!=hashlib.sha256(analysis_path.read_bytes()).hexdigest():
        raise ValueError('Plot generation does not match the current analysis')
    for name,case in report['cases'].items():
        if manifest['case_sha256'][name]!=case['sample_sha256']:raise ValueError('Plot case provenance differs')
    for name,digest in manifest['figures'].items():
        if hashlib.sha256((figures/name).read_bytes()).hexdigest()!=digest:raise ValueError('A plotted figure changed')
    pdfmetrics.registerFont(TTFont('Chinese','C:/Windows/Fonts/msyh.ttc',subfontIndex=0))
    body=ParagraphStyle('body',fontName='Chinese',fontSize=9.2,leading=15,spaceAfter=7,wordWrap='CJK')
    title=ParagraphStyle('title',parent=body,fontSize=20,leading=29,spaceAfter=16,textColor=colors.HexColor('#174d78'))
    heading=ParagraphStyle('heading',parent=body,fontSize=14,leading=21,spaceAfter=12,textColor=colors.HexColor('#174d78'))
    small=ParagraphStyle('small',parent=body,fontSize=7.6,leading=11)
    story=[]
    def p(text,style=body):story.append(Paragraph(escape(text),style))
    def page(name):story.append(PageBreak());p(name,heading)
    def table(rows,widths=None):
        entries=[[Paragraph(escape(str(x)),small) for x in row] for row in rows]
        t=Table(entries,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eff5')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#174d78')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f9')]),
            ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
            ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.append(t);story.append(Spacer(1,9))
    def image(path,max_height=465):
        if not path.exists():raise FileNotFoundError(path)
        pic=Image(str(path));scale=min(480/pic.imageWidth,max_height/pic.imageHeight)
        pic.drawWidth=pic.imageWidth*scale;pic.drawHeight=pic.imageHeight*scale;story.append(pic)
    def number(value):return f'{value:.3g}'
    decision=report['decision'];criteria=report['criteria']
    p('Au 镀层与 8/4 nm 网格对照',title)
    p('As2Se3 四聚体 | 700-1700 nm | 固定七参数结构')
    p('生成时间：'+datetime.now().strftime('%Y-%m-%d %H:%M')+'（本机时间）')
    p(('当前结构建议去掉金层。' if decision['remove_gold'] else '当前结构建议保留金层。')+
      ('8 nm 满足已确认的替代容差，可以采用。' if decision['adopt_8nm'] else '8 nm 未满足全部替代容差；精细结果采用 4 nm。'))
    p('建议仅针对本报告的固定结构和指定光谱、场采样。未通过容差意味着不能在这个精度等级下直接替代，并不证明某材料或网格在所有设计中更优。')
    table([['指标','已确认的“影响不大”标准'],['反射谷中心','位移 <= 2 nm'],['谷底反射率','绝对变化 <= 0.03（3 个百分点）'],
           ['半突出度线宽','相对变化 <= 10%'],['场强度增强','区域平均值及面积加权 P95 相对变化 <= 10%']], [160,320])
    rows=[['案例','求解耗时 / min','最大实际步长 x / y / z (nm)']]
    for name,case in report['cases'].items():
        rows.append([NAMES[name],f"{case['elapsed_seconds']/60:.1f}",' / '.join(f"{case['actual_mesh'][axis]['maximum_step_nm']:.4f}" for axis in 'xyz')])
    table(rows,[105,110,265])
    p('全部案例求解终止状态为 2，结果经独立重载原生 FSP 校验。耗时为本机本次运行，不包含审计，也不是通用性能基准。无金组保留原金层区域的数值网格，因此不能把它的耗时解释为去金后的最终速度收益。',small)
    page('控制变量与方法')
    g=report['geometry_values']
    table([['参数','值'],*[[name,f'{value:.9f} '+('deg' if i>4 else 'nm')] for i,(name,value) in enumerate(zip(('P','L','H','dX','dY','奇数象限角','偶数象限角'),g))]],[160,320])
    p('堆叠：As2Se3 方柱 z=0..H；350 nm SiO2 基板 z=-350..0；150 nm Au z=-500..-350。无金对照只将 Au 体积替换为空气，保留有限基板、计算域、R/T 监测位置、光源、周期边界、PML 和数值网格。')
    p('法向入射、x 偏振。四组采用共同的 1692 个原生频率点：约 1 nm 全波段覆盖，重要谱线窗口保留约 0.333 nm 采样。无插值产生的“伪实算”细谱。4 nm 的主功率监测器降采样 2，并增加 181 个全空间功率频率核验；近场监测均不降采样。')
    p('电场强度增强 FE=|E|^2/|Einc|^2；磁场 FH=|H|^2/|Hinc|^2。从原生 CW 源强度 I 计算 |Einc|^2=2I/(epsilon0*c)，|Hinc|^2=2I/(mu0*c)。8/4 nm 真空参考分别校验该归一化。以上为强度倍数，振幅倍数为其平方根。')
    p('两个 XY 平面为 z=H/2 与 H+10 nm。E 记录 94 个频率，H 记录 4 个频率。平均值和 P95 按原生网格面积加权。中截面柱内及空气区域排除距离柱边缘 10 nm 内的节点；原生节点掩膜是区域积分近似，报告对应采样面积。')
    table([['案例','柱内采样面积 (nm^2)','空气采样面积 (nm^2)'],
           *[[NAMES[name],f"{case['region_quadrature']['pillar_bulk']['sampled_area_nm2']:.1f}",
               f"{case['region_quadrature']['air_away_from_edges']['sampled_area_nm2']:.1f}"]
             for name,case in report['cases'].items()]],[120,180,180])
    p(f"精确去边缘方柱多边形总面积为 {report['exact_eroded_pillar_area_nm2']:.1f} nm^2。采样区域面积随网格存在离散差异，上表明确列出，不能将节点数当成面积。",small)
    p('As2Se3 采用 Joseph 2020 薄膜 n/k 数据；末端 1688-1700 nm 的 12 nm 保持端点，属于探索性外推。Au 使用 Palik 原生材料拟合；数值网格比较不证明材料拟合或实验精度。单个尖角最大值不决定结构选择。')
    page('镀金与去金：全波段光谱')
    image(figures/'spectra.png',500)
    rows=[['比较','max |delta R|','max |delta T|','max |delta A|','双谷保留 / 全部判据']]
    for key in ('Au_vs_bare_8nm','Au_vs_bare_4nm'):
        c=report['comparisons'][key];rows.append([key.replace('Au_vs_bare_',''),*[f'{c[x]:.4f}' for x in
        ('maximum_absolute_R_difference','maximum_absolute_T_difference','maximum_absolute_A_difference')],
        f"{c['reference_pair_preserved']} / {c['small_effect_under_current_criteria']}"])
    table(rows,[90,80,80,80,150])
    p('A=1-R-T 为整个结构的总吸收，不是 As2Se3 的单独吸收。金属背板使透射受抑，改变能量分配；反射谷变深不等价于全部材料内场都增强。',small)
    page('8 nm 与 4 nm：光谱差异')
    image(figures/'mesh_zoom.png',350)
    rows=[['比较 / 谷','位移 nm','谷底 |delta R|','宽度变化 %','通过']]
    for key in ('Au_8nm_vs_4nm',):
        c=report['comparisons'][key]
        if not c['tracked_spectral_features']:rows.append([key,'目标双谷未能同时跟踪','','','否'])
        for i,row in enumerate(c['tracked_spectral_features']):rows.append([key+f' / {i+1}',f"{row['center_shift_nm']:.3f}",
           f"{row['absolute_R_change']:.5f}",f"{100*row['relative_width_change']:.2f}",str(row['passes'])])
    table(rows,[165,70,90,90,65])
    auxiliary=report['comparisons']['bare_8nm_vs_4nm']['own_peak_mesh_comparison']
    rows=[['无金自己的反射峰','位移 nm','峰值 |delta R|','宽度变化 %','通过']]
    for i,row in enumerate(auxiliary['tracked_peaks']):
        if row['first'] is None:rows.append([f'峰 {i+1}','未能匹配','','','否']);continue
        rows.append([f'峰 {i+1}',f"{row['center_shift_nm']:.3f}",f"{row['absolute_R_change']:.5f}",
                     f"{100*row['relative_width_change']:.2f}",str(row['passes'])])
    table(rows,[165,70,90,90,65])
    p('无金自身峰采用同数值门槛作辅助网格比较；未保留Au目标双谷时，双谷替代门槛不适用于判断其自身网格误差。峰比较不替代已确认的双谷选择判据。',small)
    peak_fields=auxiliary['field_at_shared_preliminary_peak_wavelengths']
    p(f"无金组两个预扫峰的共同采样波长处，8/4 nm 的 E/H 核心区域平均值最大相对变化为 {100*max(row['relative_mean_change'] for row in peak_fields):.1f}%，P95 为 {100*max(row['relative_p95_change'] for row in peak_fields):.1f}%；该辅助峰及场检查同时通过={auxiliary['own_peaks_and_sampled_fields_stable']}。",small)
    p('线宽指反射谷的半突出度宽度，不是已验证模态的 FWHM 或 Q。比较均使用直接采样的谱线；中心分辨率受约 0.333 nm 频率采样限制。')
    page('相同目标波长：电场和磁场增强')
    p('每个数值为“面积加权平均 / P95”的强度增强因子。mid-bulk 为柱内去边缘区域；mid-air 为柱外去边缘空气；top 为柱顶上方 10 nm 全单胞。')
    targets=sorted(set(row['wavelength_nm'] for row in report['comparisons']['Au_8nm_vs_4nm']['field_at_common_wavelengths']))
    for vector in ('E','H'):
        rows=[[vector+' 波长 nm / 区域',*NAMES.values()]]
        for wavelength in targets:
            for plane,region,label in (('top','all','top'),('mid','pillar_bulk','mid-bulk'),('mid','air_away_from_edges','mid-air')):
                values=[]
                for case in report['cases'].values():
                    row=min(case['field_profiles'][f'nf_{plane}_{vector}'],key=lambda x:abs(x['wavelength_nm']-wavelength))
                    if abs(row['wavelength_nm']-wavelength)>1e-5:raise ValueError('Missing common field sample')
                    s=row['regions'][region];values.append(' / '.join(number(s[metric]) for metric in METRICS))
                rows.append([f'{wavelength:.3f} / {label}',*values])
        table(rows,[140,85,85,85,85])
    rows=[['比较','最大平均值相对变化','最大 P95 相对变化','全部场指标通过']]
    for key,c in report['comparisons'].items():
        field=c['field_at_common_wavelengths'];rows.append([key,f"{100*max(x['relative_mean_change'] for x in field):.1f}%",
            f"{100*max(x['relative_p95_change'] for x in field):.1f}%",str(all(x['passes'] for x in field))])
    table(rows,[190,110,110,70])
    p('相对变化分母为比较中第二组：Au/无Au 以无Au为参考，8/4 nm 以4 nm为参考。判据检查所有列出的共同波长和核心区域；任一超差即不宣称可直接替代。',small)
    for plane in ('mid','top'):
        for wavelength in targets:
            page(f'电场空间分布：{plane}，{wavelength:.3f} nm')
            image(figures/f'E_{plane}_{wavelength:.3f}.png',535)
            p('四幅图使用完全相同的对数色标，显示 FE 强度倍数。青色轮廓是柱体在 XY 的投影。mid 平面穿过柱体，top 平面在柱顶上方。热点的单节点极值可随尖角离散变化，应结合面积平均与 P95 解释。',small)
    page('各自谱形与采样场极值')
    p('下表采用镀金组的两个主要反射谷，以及无金组的两个主要反射峰，按各自谱形列出。E 为该谱线附近最近的实测场频率；频率差明确保留，不能把稀疏 H 采样称为谱线中心场。')
    rows=[['案例 / 特征','中心 nm / R','E 频率差 nm','top E 平均 / P95','H 频率差 nm']]
    for name,case in report['cases'].items():
        kind='minimum' if name.startswith('Au') else 'maximum'
        chosen=[row for row in case['fields_near_own_spectral_features'] if row['spectral_feature']['kind']==kind][:2]
        for row in chosen:
            feature=row['spectral_feature'];e=row['nearest_measured_fields']['nf_top_E'];h=row['nearest_measured_fields']['nf_top_H'];s=e['regions']['all']
            rows.append([NAMES[name]+' / '+('峰' if feature['kind']=='maximum' else '谷'),f"{feature['center_nm']:.3f} / {feature['value']:.4f}",
                         f"{e['offset_nm']:.3f}",' / '.join(number(s[m]) for m in METRICS),f"{h['offset_nm']:.3f}"])
    table(rows,[115,115,80,105,65])
    rows=[['案例','mid-bulk E 最大采样 P95','对应波长 nm','top E 最大采样平均','对应波长 nm']]
    for name,case in report['cases'].items():
        m=case['sampled_field_extrema']['nf_mid_E']['pillar_bulk']['p95_intensity_enhancement']
        t=case['sampled_field_extrema']['nf_top_E']['all']['mean_intensity_enhancement']
        rows.append([NAMES[name],number(m['value']),f"{m['wavelength_nm']:.3f}",number(t['value']),f"{t['wavelength_nm']:.3f}"])
    table(rows,[100,115,85,100,80])
    p('这些是 94 个 E 采样频率和两个平面的统计极值，不能证明整个三维空间或连续全波段的全局最大增强。它们也包含背板导致的背景干涉，不单独证明某个共振模态。')
    page('可复核性、限制与最终采用方案')
    p('采用方案：'+('保留 Au' if not decision['remove_gold'] else '移除 Au')+f"；最终精细网格 {decision['selected_final_mesh_nm']} nm。未满足容差的具体指标见光谱和场对照表；8 nm 可作为探索阶段结果，但是否用于正式训练需按误差等级另行标识。")
    p('每组均有配置指纹、源和几何核验、实际网格坐标、自动终止状态、R/T/A 原生复读、四个 E/H 监测器复读和原始文件 SHA256。float32 场导出的复读容差远小于 10% 决策阈值。以下为原始 NPZ 校验值：')
    for name,case in report['cases'].items():p(NAMES[name]+': '+case['sample_sha256'],small)
    p('分析 JSON SHA256: '+hashlib.sha256(analysis_path.read_bytes()).hexdigest(),small)
    p('限制：固定单一结构及法向 x 偏振；没有偏振、角度、制造公差或实验验证。背板距离采用有限350 nm SiO2；不能把结论直接用于毫米厚基片背面的镀金。4 nm 与8 nm 对照不是无限细网格收敛证明；此前 5/4 nm 双谷光谱检查通过，仍不能替代本次场精度检查。没有宣称 Fano、BIC、模态身份或模态 Q。')
    p('方法来源（Ansys 官方）：')
    for name,url in [('源强度与 CW 归一化','https://optics.ansys.com/hc/en-us/articles/360034925373-sourceintensity-Script-command'),
                     ('频域场监测器','https://optics.ansys.com/hc/en-us/articles/360034902393-Frequency-domain-monitor-Simulation-object'),
                     ('Python 原生向量数据维度','https://optics.ansys.com/hc/en-us/articles/39744236202771-Accessing-Simulation-Results-Python-API'),
                     ('网格细化方法','https://optics.ansys.com/hc/en-us/articles/360034382594-Understanding-Mesh-Refinement-and-Conformal-Mesh-in-FDTD')]:
        story.append(Paragraph(escape(name)+': <link href="'+url+'" color="#174d78">'+escape(url)+'</link>',small))
    output.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.setFont('Chinese',8);canvas.setFillColor(colors.HexColor('#657580'))
        canvas.drawString(42,25,'As2Se3 固定结构 / Au 与网格对照');canvas.drawRightString(A4[0]-42,25,str(doc.page))
    SimpleDocTemplate(str(output),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=40,bottomMargin=42,
                      title='Au镀层与8/4 nm网格对照',author='Optics design').build(story,onFirstPage=footer,onLaterPages=footer)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,default=ROOT/'research_results/gold_mesh_comparison/analysis.json')
    parser.add_argument('--output',type=Path,default=ROOT/'output/pdf/Au_8nm_4nm_comparison_report.pdf')
    args=parser.parse_args();print(author(args.analysis,args.output))
