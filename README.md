# 独立 As₂Se₃ / Au、Ag / 七参数近红外项目

本项目由 `D:\projects\LumOpt_test` 的当前代码快照建立，原项目未修改。
独立仓库：`D:\projects\Metaformer_As2Se3`；开发 Git 工作树：`D:\projects\Metaformer_As2Se3_dev`；分支：`codex/as2se3-metal-quadrants`。

目标波段 **700–1700 nm**。参数为 P、L、H、dX、dY、ω奇数象限、ω偶数象限；第一/第三象限一组，第二/第四象限一组。提供真实 FDTD、Au/Ag 对照扫描、特征谱筛选、七参数数据导出与 Metaformer 训练。具体结构条件、材料范围和使用方法见 [近红外研究计划](docs/near_ir_research_plan.md)。以下内容保留为原模型的参考说明。

# MetaFormer：Python 控制 Lumerical FDTD 与数据集生成

项目参考 Gao 等人的 [Meta-Attention Deep Learning for Smart Development of Metasurface Sensors](https://pmc.ncbi.nlm.nih.gov/articles/PMC11558086/)（DOI：10.1002/advs.202405750）。现已接入本机 **Lumerical FDTD 2024 R1**：Python 建模、调整参数、运行三维仿真、导出光谱、构建数据集，并完成 100 样本的 MetaFormer 训练流程。

## 在 PyCharm 中展示和运行

打开 `D:\projects\LumOpt_test`，使用原项目解释器 `D:\Anaconda3\envs\transformer_env\python.exe`。已确认 Python 3.10、PyTorch 2.5.1、NumPy、Matplotlib 和 CUDA 可用。迁移环境时可执行 `python -m pip install -r requirements.txt`。

共享运行配置保存在 `.run/`，在 PyCharm 顶部运行下拉框依次选择：

1. **01 FDTD installation check**：启动会话，检查 API、材料和资源。
2. **02 Fresnel validation**：运行裸介质界面，与解析反射/透射率比较。
3. **03 Generate smoke dataset**：生成 100 个真实三维样本，已复用原来的 4 个；再次运行会跳过完整样本。
4. **04 Browse spectra and MetaFormer**：拖动滑块浏览光谱、几何和标签，检查实际数据的前向和反向传播。
5. **05 Generate dataset batch**：使用批量配置，每次处理最多 10 个缺失/失败样本；重复运行可续跑同一计划。
6. **06 Train MetaFormer (100 samples)**：执行实际优化训练，按验证集选择模型，支持早停、学习率调整和检查点保存。
7. **07 View training results**：训练完成后查看损失曲线、测试误差及预测对照，默认打开最近一次训练结果。
8. **08 Train ResNet + MetaFormer**：增加一维残差卷积前端，使用相同数据划分和训练设置训练。
9. **09 View ResNet training results**：查看残差网络版本的损失曲线与预测结果。
10. **10 Compare MetaFormer and ResNet**：核对数据与设置一致后，比较两个已保存模型的实际误差。

直接右键运行 `pycharm_demo.py` 也可浏览已生成数据。04 中的接口检查使用随机初始化模型；06 执行实际训练，07 展示已保存的训练结果。

## 脚本入口

以下命令在项目根目录中运行，使用上述 Python 解释器：

```powershell
python generate_dataset.py doctor --config configs/smoke.json
python generate_dataset.py plan --config configs/smoke.json
python generate_dataset.py reference --config configs/smoke.json
python generate_dataset.py generate --config configs/smoke.json
python pycharm_demo.py --check-model

# 无窗口保存预览，验证 MetaFormer 输入接口
python pycharm_demo.py --no-show --check-model

# 训练当前 100 个真实样本，保存最佳验证模型
python train_metaformer.py --config configs/train_smoke_100.json
python plot_training.py

# 试验一维残差网络前端，原模型保留在原输出目录
python train_metaformer.py --config configs/train_resnet_100.json
python plot_training.py --output-root runs/metaformer_resnet_100
python compare_training.py
# 中断后使用相同配置，从指定运行目录的 last.pt 续训
python train_metaformer.py --config configs/train_smoke_100.json --resume runs/metaformer_smoke_100/20260929_153313_618394

# 本次只运行最多 10 个缺失/失败样本
python generate_dataset.py generate --config configs/dataset.json --limit 10
# 省略 --limit 可执行完整采样计划
```

停止后使用**相同配置和输出目录**重新运行即可续跑。每个成功样本立即保存，SHA256 校验通过的样本不会重复求解。修改范围、波长、网格、采样数或其他配置时应指定新的 `output_dir`，程序拒绝混用不同设置的数据。

## 几何和仿真参数

修改 `configs/smoke.json` 或 `configs/dataset.json` 可调整参数、样本数和输出目录。输入尺寸、波长和物理标签的单位统一为 **nm**，传给 FDTD 时转为米。标签顺序固定为 `[p, l, h, alpha_x, alpha_y]`。

| 参数 | 含义 | 当前工程示例范围 (nm) |
| --- | --- | --- |
| p | x/y 周期 | 3000–3400 |
| l | 单个硅方块的边长 | 850–1050 |
| h | 硅方块高度 | 800–1100 |
| alpha_x | x 方向相邻方块的边缘间隙 | 150–350 |
| alpha_y | y 方向相邻方块的边缘间隙 | 150–350 |

**这些范围用于工程流程验证，尚未核对论文 Table S3，不能视为已复现论文的数据分布。** 原表位于[补充材料末页](https://advanced.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002/advs.202405750&file=advs9498-sup-0001-SuppMat.pdf)。取得原表后更新 `bounds_nm` 和 `parameter_range_source`，指定新输出目录采集。当前 **4–8 μm** 波长窗口也是可配置的工程窗口。

四个方块中心为 `x = ±(l + alpha_x)/2`、`y = ±(l + alpha_y)/2`。程序拒绝不能放入周期单元的几何（`2*l + gap < p`）。每个样本通过 Python 创建四个 Si 方块、延伸进下方 PML 的半无限基底、x/y 周期边界、z 向 PML、沿 −z 入射且电场沿 x 的平面波、R/T 功率监视器。R 监视器在光源后方；本入射方向对应 `R = T_R`、`T = -T_T`。监视器直接记录 1000 点均匀波长采样，避免插值改变窄共振。参见 [Ansys Python API](https://optics.ansys.com/hc/en-us/articles/360041579954-Script-Commands-as-Methods-Python-API)及[功率正负号定义](https://optics.ansys.com/hc/en-us/articles/360034405354-transmission-Script-command)。

材料使用安装库中的 `Si (Silicon) - Palik` 和 `SiO2 (Glass) - Palik`，其与论文引用光学常数的等价性尚未验证。`absorptance = 1 - R - T` 表示两监测面之间的功率损耗，包含基底传播吸收和数值误差。

`smoke.json` 提供 100 个样本计划，使用 80 nm 局部网格、4 级全局网格、10 ps 上限、`1e-5` 自动关断阈值，数据标记为 `pipeline_validation`，输出到 `data/smoke_100/`。`dataset.json` 同样设为 100 个样本，使用 40 nm 局部网格、20 ps 上限、`1e-6` 阈值。正式采集前须在代表性几何上比较更细网格、更远边界和更长仿真时间的光谱，尤其检查高 Q 共振。达到自动关断并不能单独证明网格已收敛。默认计算资源为 1 进程、4 线程，可修改 `processes` 和 `threads`。

## 数据输出

生成文件位于 `data/`（Git 忽略）：

```text
 data/smoke_100/
   manifest.json                完整配置、采样计划、状态、划分、文件校验值
   generation.log               每个样本的进度
   errors.jsonl                 失败原因和 traceback（发生失败时生成）
   samples/sample_000000.npz    实际几何、归一化标签、R/T、状态和耗时
   samples/sample_000000.csv    wavelength_nm,reflectance,transmittance,absorptance
   projects/sample_000000.fsp   可在 FDTD 中打开的工程（可配置关闭保留）
   dataset.npz                  汇总数据集
   summary.json                 成功/失败/待运行数量及划分数量
   dataset_preview.png          光谱和结构预览
   model_interface_check.json   实际数据前向/反向检查结果
 data/reference/
   fresnel.fsp
   fresnel.npz
   summary.json                 Fresnel 解析对照误差
```

默认仅接收 `status=2` 的自动关断样本。不收敛、发散、NaN、异常负功率或违反被动能量范围的数据会被排除并记录；程序保留数值容差，不对曲线取绝对值或截断来掩盖问题。`quality_tolerance=0.03` 是功率绝对容差，正式配置可收紧。操作系统锁会阻止同目录并发写入。

若未达到自动关断即触及时间上限，程序会将时间上限逐次加倍后重新求解，最多四次（验证配置为 10/20/40/80 ps）。几何、材料、网格、波长和自动关断阈值保持一致；只有达到收敛的结果才进入数据集。单样本保存 `attempts_json`，汇总数据保存 `simulation_time_ps` 和 `run_attempts`，可核查实际时间上限及重试次数。

| dataset.npz 字段 | 形状与用途 |
| --- | --- |
| spectra | `[N,1000]`，反射率 float32，直接作为输入 |
| geometry_01 | `[N,5]`，归一化结构标签 float32 |
| geometry_nm | `[N,5]`，实际尺寸 float64 |
| wavelength_nm | `[1000]`，共享的递增波长网格 |
| lower_nm / upper_nm | `[5]`，标签归一化和还原范围 |
| transmittance / absorptance | `[N,1000]`，辅助检查 |
| train_indices / val_indices / test_indices | 各划分在已成功数据行中的索引 |
| simulation_time_ps / run_attempts | `[N]`，实际使用的时间上限和本次求解次数 |
| sample_ids / status / metadata_json | 计划 ID、求解状态、来源和配置 |

固定随机种子生成几何，并预先指定约 80%/10%/10% 的训练/验证/测试划分。续跑时样本不会跨划分移动。100 个样本计划划分为 80/10/10；扩展时保留原 4 个样本的划分，补齐余下配额。

## 接入 MetaFormer

```python
from torch.utils.data import DataLoader
from metaformer import InverseMetaformer, geometry_mse
from spectrum_dataset import SpectrumGeometryDataset

dataset = SpectrumGeometryDataset("data/dataset/dataset.npz", split="train")
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = InverseMetaformer()
spectrum, target = next(iter(loader))
prediction = model(spectrum)
loss = geometry_mse(prediction, target)
geometry_nm = dataset.denormalize(prediction)
```

工程验证数据须显式传入 `allow_validation_data=True`；当前训练配置已显式允许这轮流程验证训练，数据质量标记保持为 `pipeline_validation`。标签按固定物理范围 `(geometry_nm - lower_nm)/(upper_nm - lower_nm)` 归一化；反射率保留光源功率归一化后的值，不再对每条曲线做 min/max 处理。

沿用 `metaformer.py`：1000 点输入、100 个片段、128 维嵌入、10 层、10 个注意力头，每头 64 维，输出 5 个 [0,1] 参数。论文没有完全公开实现，当前平均池化和前置归一化等选择不能视为作者代码的逐行复刻。

## 实际训练与模型保存

`configs/train_smoke_100.json` 指向已生成的 100 个样本。使用 AdamW、初始学习率 `3e-4`、权重衰减 `1e-3`、batch size 8、梯度裁剪 1.0、随机种子 42；CUDA 下启用混合精度。最多 200 轮，验证误差不再显著改善时早停，学习率也按验证误差降低。训练前检查划分无重复、无遗漏且全部求解状态为 2，即使允许工程验证数据也不接收未收敛样本。

仅 80 个训练样本参与反向传播；10 个验证样本用于选择最佳模型，另 10 个测试样本在训练完成后评估。每次新训练创建独立时间目录，避免覆盖原有模型；同一输出根目录的并发训练受到锁保护。中断后 `--resume` 恢复模型、优化器、学习率调度器、混合精度状态、随机状态和采样顺序，拒绝配置或数据 SHA256 不一致的续训。

本机 RTX 3050 Ti 已完成 **52 轮、520 次优化更新**，第 **17 轮**为最佳验证模型，随后因验证误差未改善而早停。测试集归一化 MSE 为 **0.063927**，仅根据训练集标签均值预测的基线为 **0.100630**，本次测试 MSE 降低约 **36.5%**。这些结果来自 10 个留出样本，不能据此认定已复现论文精度。

| 测试参数 | 最佳模型 MAE (nm) |
| --- | --- |
| p | 120.51 |
| l | 32.42 |
| h | 50.54 |
| alpha_x | 34.69 |
| alpha_y | 51.32 |

当前 `alpha_y` 的测试 MAE 尚未优于均值基线（49.93 nm），需要更充分的数据和后续模型验证。训练损失下降而验证误差在后期回升，保存并使用早期的最佳验证模型。

本次输出目录为 `runs/metaformer_smoke_100/20260929_153313_618394/`，包含：

```text
run.json                配置、数据 SHA256、波长网格、单位范围和划分 ID
training.log            每轮训练日志
history.json            损失、学习率和实际优化步骤
status.json             训练状态
best.pt                 验证误差最低的模型权重及推理元数据
last.pt                 最后一轮完整状态，可用于断点续训
metrics.json            训练/验证/测试指标、物理 MAE/RMSE 和训练均值基线
train_predictions.csv   最佳模型的训练集预测
val_predictions.csv     最佳模型的验证集预测
test_predictions.csv    留出测试集的真实与预测参数（归一化值及 nm）
training_preview.png    训练曲线和测试对照
validation_report.json  本次保存权重重新载入与预测一致性复查
```

`latest_run.json` 在输出根目录记录最近一次运行。训练模型权重和原始数据 SHA256 均已复查，保存权重重新载入后的 100 个预测与 CSV 一致。用于推理时加载 `best.pt` 中的 `model_state`，按 `metadata.config.model` 建模，并使用保存的 `lower_nm` / `upper_nm` 将输出还原为物理尺寸；输入波长网格须与保存的 `wavelength_nm` 一致。

## 一维 ResNet 前端试验与注意力检查

原 Transformer 编码层包含注意力与 MLP 的残差连接。新增 `spectral_frontend="resnet1d"` 会先通过 32 通道卷积和 3 个残差块提取局部峰形，卷积核为 7，残差块的扩张率为 1/2/4，波长顺序及 1000 点长度保持一致。随后按每 10 个点形成一个片段，输入原有的 10 层、128 维、10 头 MetaFormer。残差分支使用 GroupNorm，末端归一化初始为零，从恒等路径开始学习。默认 `linear` 保持旧模型结构和已有权重可加载；旧格式检查点的可选模型字段会补入默认值后检查续训兼容性。

使用 `configs/train_resnet_100.json` 进行了实际 GPU 训练，其他超参数、种子及 80/10/10 划分与原模型相同。新增模型为 4,080,901 个参数，完成 **88 轮、880 次优化更新**，选择第 **53 轮**的最佳验证模型。

| 归一化 MSE | 原 MetaFormer | ResNet + MetaFormer |
| --- | --- | --- |
| 训练集（最佳模型） | 0.051435 | 0.026243 |
| 验证集 | 0.047217 | 0.036733 |
| 原测试集 | 0.063927 | 0.062174 |

验证 MSE 降低约 **22.2%**，原测试集 MSE 只降低约 **2.7%**。逐参数测试 MAE 如下，残差网络尚未使全部参数同步改善：

| 参数 | 原模型 MAE (nm) | ResNet MAE (nm) |
| --- | --- | --- |
| p | 120.51 | 74.22 |
| l | 32.42 | 36.38 |
| h | 50.54 | 34.84 |
| alpha_x | 34.69 | 40.44 |
| alpha_y | 51.32 | 62.36 |

这是同一批已评估过的 10 个测试样本的对照结果，不构成对新增样本的泛化验证。后期训练损失继续下降而验证误差回升，仍可观察到过拟合。不能将剩余误差直接归因于注意力实现错误，也不能据此确认两个间隙在当前光谱中不可辨识；后者需用单参数物理扫参验证。

注意力实现已与 [PyTorch SDPA](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html) 的独立参考计算对照，前向值、输入梯度和投影参数梯度一致。对两个实际保存模型进一步检查，十层 Q/K/V 权重均已改变，梯度有限且非零，注意力每行归一化正常。保留已学习的 V 与输出投影、将路由改为均匀分布后，原模型验证 MSE 从 0.04722 升到 0.15097，ResNet 版本从 0.03673 升到 0.13750。该消融说明预测依赖学习到的注意力路由，并不能单独证明泛化良好。检查未执行优化更新，未改动保存权重。

新增运行目录为 `runs/metaformer_resnet_100/20260929_164633_176760/`。除常规训练输出外，还保存了 `comparison.json` / `comparison.png`、`attention_check.json` 和 `validation_report.json`。原模型的注意力检查记录保存在原运行目录；两个模型的检查点、原始数据及重新加载预测均已核对。

## 网络结构图

[完整网络结构图](docs/figures/metaformer_architecture.png) 按 `metaformer.py` 和 `configs/train_resnet_100.json` 的实际实现绘制，包含主流程、1D ResNet 残差块、Encoder 的两条残差连接、多头自注意力的 Q/K/V 计算，以及网络外的尺寸单位还原和训练损失。图中虚线给出原 MetaFormer 基线的路径；ResNet 是本项目增加的前端。

在 PyCharm 选择 **11 Draw MetaFormer Architecture**，或者运行：

```powershell
python draw_metaformer.py
```

脚本用实际前向传播核对各层及十组注意力矩阵的形状，再将 PNG、可编辑 SVG、PDF、内部模块放大图和形状报告保存到 `docs/figures/`。生成结构图只读取模型配置和数据元信息，不执行训练。输出的五个归一化值依次为 `p, l, h, alpha_x, alpha_y`，物理单位转换位于网络外，MSE 在归一化尺寸上计算。

## 共同旋转角：六参数扩展

旋转模式在五个长度参数之后追加 `rotation_deg`，范围为 **−45°～45°**。四个硅方块各自绕自身中心旋转同一个角度，中心位置、周期单元和 x 偏振光源保持原定义。项目角度以从 +z 观察的逆时针为正；[Lumerical 原生旋转属性](https://optics.ansys.com/hc/en-us/articles/360034901473-Rectangle-Simulation-Object) 使用顺时针度数，因此脚本写入 `rotation 1 = -rotation_deg`，旋转轴为 `z`。

`alpha_x/alpha_y` 仍表示未旋转排布的边缘间隙，故中心距离仍是 `l + alpha`；旋转后的实际最小间隙会改变。采样会剔除方块相交、接触或越过周期边界的组合。由于原长度范围在部分大角度下不允许独立方块，合法角度的边际分布并非均匀覆盖全部区间。

旋转数据采用 schema 2，标签顺序为 `p, l, h, alpha_x, alpha_y, rotation_deg`，单位依次为 `nm, nm, nm, nm, nm, deg`：

- `geometry_nm` / `lower_nm` / `upper_nm` 继续仅保存五个长度；`rotation_deg` 单独保存角度。
- `geometry_values` / `lower_values` / `upper_values` 保存六列物理值，单位由 `parameter_units` 指定。
- `geometry_01` 为六列归一化训练标签；角度转换为 `(theta + 45) / 90`。
- 网络仍输入 1000 点反射光谱，回归头改为 `128 → 256 → 6 → Sigmoid`。
- 长度误差使用 `mae_nm` / `rmse_nm`，角度误差使用 `mae_deg` / `rmse_deg`；预测 CSV 将角度列标为 `deg`。

当前对称结构、各向同性材料和 x 偏振反射率输入下，`+theta` 与 `-theta` 互为镜像等效，单条输入不能唯一判别符号。六参数配置使用 `angle_symmetry="square_mirror"`：保留原始有符号标签，但训练与模型选择在角度项比较 `|theta|/45`，防止相反标签把角度回归抵消到零。报告同时列出等效角误差和原始有符号误差；预测角度是一个等效候选，最终仍应检查预测几何及重新仿真。五参数配置与旧检查点读取方式保持兼容。

PyCharm 新增入口：**12 Rotation Pilot FDTD**、**13 Rotation Dataset Preview**、**14 Generate Rotation 100**、**15 Train Rotation 100**、**16 Rotation Pilot Training Results**、**17 Validate Rotation Physics**、**18 Rotation Training Check** 和 **19 Draw Rotation Architecture**。完成 100 样本采样和训练后，用 **20 Rotation 100 Dataset Preview** 查看完整数据，**21 Rotation 100 Training Results** 查看训练结果。

要操作滑块，请在顶部运行下拉框选择 **13 Rotation Dataset Preview** 并点击运行，脚本会打开独立的交互窗口。直接打开 `dataset_preview.png` 只会看到静态图片，里面的滑块无法操作。窗口中的“样本”切换真实仿真数据；“旋转角”让四块绕各自固定中心共同旋转，点击“还原角度”返回该样本的真实角度。改变预览角度不会自动运行 FDTD，光谱和训练标签仍对应原样本，窗口会标明“未仿真”；相交或越界的预览结构显示红色，不能用于采样。独立窗口使用 TkAgg 及阻塞事件循环，避免内嵌图片窗口不响应控件，参见 [Matplotlib 交互窗口说明](https://matplotlib.org/stable/users/explain/figure/interactive.html)。

**22 Compare Rotation Sample Spectra** 将六条真实反射谱放在一张图上，上方对照样本 00、01 的结构及参数，右下方放大两条光谱的峰区。图像与数值记录保存在 `data/rotation_pilot/sample_spectra_comparison.png` / `.svg` / `.json`。00 和 01 的最大反射率差异约为 **0.74546**，出现在 **4688.69 nm**；它们的五个尺寸和旋转角同时变化，不能把这组差异单独归因于角度。该对照与下面固定尺寸的 ±20° 镜像对照含义不同。

日常查看旋转影响使用 **23 Positive Rotation Comparison**：仅显示 **0°、+20°、+30°** 三条真实反射谱，五个尺寸固定为 `p=3200, l=900, h=1000, alpha_x=250, alpha_y=300 nm`。0° 和 +20° 复用已验证的数据，+30° 已补算并达到 `status=2`；后续运行会校验并复用已保存结果。图中不叠加负角度镜像曲线，主图为 `data/rotation_positive/positive_rotation_comparison.png`，三个独立结构图为 `data/rotation_positive/structures/angle_0.png`、`angle_20.png`、`angle_30.png`，均另存可编辑 SVG。来源、校验值、原生对象属性和数值差异记录在 `comparison_report.json`。20°、30° 相对 0° 的最大反射率差异分别为 **0.38807**、**0.56768**。

```powershell
python plot_sample_comparison.py
# 只保存静态图
python plot_sample_comparison.py --no-show
python plot_positive_rotation.py --angles 0 20 30
```

六个真实样本的流程检查使用单独的 `data/rotation_pilot/` 与 `runs/metaformer_rotation_pilot/`。其中训练仅执行两轮，用于验证六输出和真实优化更新，不用于评价拟合精度：

```powershell
python generate_dataset.py generate --config configs/rotation_pilot.json
python pycharm_demo.py --dataset data/rotation_pilot/dataset.npz --no-show --check-model
python train_metaformer.py --config configs/train_rotation_pilot.json
python validate_rotation.py
python draw_metaformer.py --config configs/train_rotation_pilot.json --output-dir docs/figures/rotation
```

本机已完成六个真实旋转样本，全部 `status=2`，光谱为 `(6,1000)`、标签为 `(6,6)`，划分为 **4/1/1**；其中一个样本在 20 ps 重试后收敛。六输出 ResNet + MetaFormer 有 **4,081,158** 个参数，已完成两轮 GPU 训练、四次优化更新，第六个回归输出的权重确实发生改变。重新载入最佳权重的全部预测与 CSV 一致，原五参数数据及两个旧模型的 SHA256 和预测也已核对。结果记录在 `data/rotation_pilot/validation_report.json` 和 `runs/metaformer_rotation_pilot/20260929_200333_359358/validation_report.json`。

另外用相同五个尺寸执行了 **0°、+20°、−20°** 三次真实 FDTD 对照，原生对象属性确认四个中心保持不变。20° 相对 0° 的最大反射率变化为 **0.3881**，正负 20° 的最大反射率差异为 **2.30×10⁻⁸**、透射率差异为 **2.94×10⁻⁸**，支持当前镜像等效角处理。该对照属于现有粗网格下的工程检查，不替代网格收敛或其他角度验证。查看[旋转与光谱对照](data/rotation_validation/rotation_validation.png)、[六参数网络结构图](docs/figures/rotation/metaformer_architecture.png)及 `data/rotation_validation/validation_report.json`。

100 样本旋转数据和正式训练入口使用独立配置与输出目录，先采样，再训练：

```powershell
python generate_dataset.py generate --config configs/rotation_100.json
python train_metaformer.py --config configs/train_rotation_100.json
python plot_training.py --output-root runs/metaformer_rotation_100
```

本次已运行的是上述六样本流程检查，新增的 100 样本旋转批次尚未运行。`train_rotation_100.json` 仍显式允许工程验证数据，完整批次训练也不能直接作为论文精度复现结论。

## 验证与后续

本机已完成 **100 个真实三维样本**，全部达到 `status=2`，失败和待运行数量均为 0。
汇总光谱为 `(100,1000)`、标签为 `(100,5)`，训练/验证/测试划分为 **80/10/10**。
原来的 4 个样本及其划分已保留；新增样本中有 5 个未在 10 ps 内自动关断，均在 20 ps 重试后收敛。
已逐个核对 100 份 NPZ、CSV、FSP 和 SHA256，检查标签归一化、汇总数据一致性及划分无重叠。
详细结果保存在 `data/smoke_100/validation_report.json`，单样本求解记录的耗时中位数约 30.6 秒。
Fresnel 对照最大反射绝对误差为 0.00174、能量误差为 0.000117。
真实数据通过了现有 MetaFormer 的 CUDA 前向/反向传播检查，并完成上述实际优化训练；当前数据仍属于粗网格工程流程验证。

```powershell
python -m unittest discover -s tests -v
```

17 项测试已通过，覆盖功率符号、波长顺序、能量检查、不收敛拒绝、时间上限延时重试、标签还原、划分稳定性、断点续跑、配置混用拒绝、损坏文件重生成和并发保护。测试中的假求解器仅用于自动化检查；实际采集入口只调用真实 Lumerical。

新增 5 项训练检查已通过，覆盖划分泄漏与遗漏拒绝、工程数据授权与收敛状态、物理单位指标、实际优化更新、CPU 中断续训一致性和配置混用保护。

新增 4 项残差与注意力检查已通过，覆盖与 PyTorch SDPA 的输出/梯度对照、残差恒等路径与输入梯度、ResNet 和 Q/K/V 的实际优化更新、旧模型默认字段兼容和不同前端续训拒绝。

新增 7 项旋转检查已通过，覆盖固定中心与镜像顶点、相交/越界拒绝、原生旋转属性、镜像角损失及梯度、nm/deg 指标分离、六列数据断点续跑与训练、单位边界一致性和五参数配置兼容，共 **33 项**测试通过。

新增 3 项预览交互检查使用鼠标事件验证角度滑块、四块共同旋转、非法角度提示、样本切换和还原、真实光谱/标签保持对应，以及单样本和旧五参数浏览界面，共 **36 项**测试通过；本机独立 Tk 窗口的鼠标回调也已实测。

下一阶段扩大经网格收敛检查的数据集、评估泛化能力，并将预测结构送回 FDTD 验证。双环境、BRIS/SRIS 和表面包覆尚未实现，当前训练使用裸结构的逆向设计数据。
