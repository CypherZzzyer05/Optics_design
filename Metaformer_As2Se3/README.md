# As₂Se₃ 四聚体：七参数近红外逆向设计

由原 Metaformer 项目建立的独立项目，原 `D:\projects\LumOpt_test` 未修改。开发仓库为 `D:\projects\Metaformer_As2Se3`，代码编辑在 Git worktree `D:\projects\Metaformer_As2Se3_dev`、分支 `codex/as2se3-metal-quadrants` 中进行。

目标波段 **700–1700 nm**。四个 As₂Se₃ 方柱位于有限厚 SiO₂ 基板上，Au 或 Ag 膜位于基板底部。七个输出按 `[P, L, H, dX, dY, theta_odd_deg, theta_even_deg]` 排列；前五项为 nm，后两项为 deg。第一/第三象限共享奇数角，第二/第四象限共享偶数角，各柱绕自身中心旋转。两个角独立；`+δ/−δ` 是可选扫描路径。

代码内部使用 `p,l,h,alpha_x,alpha_y` 表示五个长度。dX/dY 是未旋转方柱的名义边缘间距，中心位置由 `(L+dX)/2`、`(L+dY)/2` 确定；旋转后的碰撞和周期单元越界由多边形检查拒绝。方柱具有90°周期等价性，训练对两个角分别使用周期损失。

## 已完成的真实结果

14组 Au/Ag/裸基板对照和5组 As₂Se₃ 材料重拟合对照均已由本机 Lumerical FDTD 求解。固定 P=660、L=190、H=300、dX=dY=120 nm，角度从0/0到+10/−10、+20/−20时，约860 nm的反射谷加深并略微蓝移。20 nm网格下，Au主谷R约0.366→0.255→0.220，Ag约0.362→0.256→0.219。材料重拟合改变绝对谷深，但保留这组趋势。

这些是候选筛选结果，**尚未证明最终网格收敛、材料拟合精度或模态来源**。七输出网络与数据接口已实现并通过42项软件测试；尚未训练具有可靠泛化精度的七参数模型。

![候选结构光谱](docs/figures/candidate_comparison.png)

详细数值、计算条件和限制见[初步研究结果](docs/initial_research_results.md)。完整原始光谱、来源指纹和测试记录见 `research_results/`；大型 `.fsp` 和训练权重不提交到Git。

## 使用

在 PyCharm 中打开本项目工作树，选择 `D:\Anaconda3\envs\transformer_env\python.exe`，或安装 `requirements.txt`。Lumerical 路径在配置的 `lumapi_path` 中；迁移机器时需要修改。以下命令在本项目根目录运行。

```powershell
# 检查真实 FDTD 接口及材料
python generate_dataset.py doctor --config configs/as2se3_au_pilot.json
# 固定尺寸的 Au/Ag 角度对照；可断点续算
python explore_spectra.py plan
python explore_spectra.py run
# 读取已提交结果绘图，无需运行 FDTD
python plot_research_summary.py
# 检查七输出网络并保存结构图；不依赖尚未采集的训练数据
python draw_metaformer.py --sampling-config configs/as2se3_au_pilot.json --output-dir docs/figures/as2se3
# 随机七参数数据采集及两轮训练流程检查
python generate_dataset.py generate --config configs/as2se3_au_pilot.json
python train_metaformer.py --config configs/train_as2se3_pilot.json
python plot_training.py --output-root runs/as2se3_au_pilot
# 更细网格验证单个候选；可能需要较长求解时间
python verify_candidate.py --case Au_opposite_20 --mesh 10 5
```

`configs/as2se3_ag_pilot.json` 用于 Ag 数据采集；`configs/as2se3_au_refit.json` 用于 As₂Se₃ 损耗拟合对照。更改结构、材料、网格、波段或采样计划时使用新的输出目录，避免混合不兼容数据。

已加入专用于本项目的 PyCharm `.run/` 入口。数据生成前先完成环境检查；训练入口需要先生成七参数数据集。两轮训练配置用于接口验证。

## 材料和下一步

As₂Se₃ 使用 Joseph 等实测薄膜 n/k，来源与许可见 [materials/README.md](materials/README.md)。数据覆盖至1688 nm，1688–1700 nm采用显式端点延拓。当前350 nm基板为薄膜结构；若实物使用毫米厚基片，必须重新设置基板和背面金属传播模型。

研究流程、角度均值/差值扫描、尺寸扫描及目标谱导入方法见[近红外研究计划](docs/near_ir_research_plan.md)。用户尚未给出目标谱的具体形状、中心和线宽，因此当前筛选明显反射谷。支持用 `explore_spectra.py --target-csv` 比较指定目标。

原五/六参数模型的历史说明归档至 [legacy_metaformer_readme.md](docs/legacy_metaformer_readme.md)；其中原项目的数据、训练记录和4–8 μm结果属于历史参考。本项目没有复制其大型数据及权重。
