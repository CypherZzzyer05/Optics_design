# MetaformerPro

## 当前分支：按新数据和误差继续优化（2026-10-27）

独立工作树 `D:/projects/MetaformerPro_refine/Metaformer_As2Se3`，分支 `codex/metaformerpro-topmetal-refine`。沿用厚SiO₂ / Au100 / As₂Se₃新1000样本，比较标准化下限、保留100个波长分块位置的输出层、192维编码器；每个候选三个种子，旧圆周头作为固定基线。只优化逆向网络。

[本轮结果](Metaformer_As2Se3/results/topmetal_refine/report.md)。PyCharm解释器仍为 `D:/Anaconda3/envs/transformer_env/python.exe`，运行项 **18 MetaformerPro refine train** 执行对照，**19 MetaformerPro refined spectrum to structure** 使用本轮冻结权重。输入仍为 `inputs/target_topmetal.csv`，输出JSON和正确层序的结构图。旧权重保留，新输出单独放在 `models/metaformerpro_refined.pt`。

九次新训练已完成，三个方案均未通过替换门槛，继续采用0.2.0圆周头seed42。三种子平均镜像验证MSE：基线0.024448、调整标准化0.024715、保留分块位置0.026661、192维编码器0.025442。两种扩大容量方案分别为215万和127万参数，在本批数据上未改善验证结果。新输出文件与原部署权重逐字节相同。

替换要求在训练前固定：三种子平均镜像验证MSE至少降低1%，同时保护五尺寸、dX和dY误差；验证选型后只评估基线与被选中的方案。既有100条测试此前已被查看，本轮是重复比较，不是新增盲测。

## 历史：厚 SiO₂ / Au / 四聚体，新1000样本

分支 `codex/metaformerpro-topmetal-1000`，独立项目 `D:/projects/MetaformerPro_topmetal/Metaformer_As2Se3`。使用生成器2.2.0新数据：厚SiO₂延伸至下PML、100 nm修正JC Au、As₂Se₃四聚体；金属局部8 nm网格。1000个样本全部查收核对，固定800/100/100划分，两种现有输出头各用三个种子从头训练。

[本轮训练、测试和光谱覆盖报告](Metaformer_As2Se3/results/topmetal1000/report.md)。新权重单独保存为 `models/metaformerpro_topmetal.pt`；历史 `models/metaformerpro.pt` 仍对应旧层序。新数据中只有3条满足既定严格双主谷标准，没有两个严格主谷都在1000 nm以上的样本，不能将该目标视为已被训练集充分覆盖。

验证选中圆周角度头，部署seed42。新批次三种子平均测试：镜像等价MSE为0.025325，比同批Sigmoid头低3.4%；五尺寸MSE低6.3%。P/L/H/dX/dY平均误差约4.41/1.43/4.11/6.39/14.53 nm，奇/偶角约12.48/12.93°。dY与角度仍需改善；部署模型的100条测试结构均通过几何检查，尚未进行新候选的FDTD回验。

PyCharm打开上述内部项目，解释器 `D:/Anaconda3/envs/transformer_env/python.exe`。运行项 **16 MetaformerPro topmetal train** 重新训练；**17 MetaformerPro topmetal predict** 读取 `inputs/target_topmetal.csv`，输出七参数与正确层序的结构图。输入要求700–1700 nm等间距1000点、反射率0–1。推理图的光谱为输入目标，候选结构尚需电磁回验。训练和图表缓存位于D盘；原始数据包及旧权重保留。

## 历史：旧层序 MetaformerPro 0.2

MetaformerPro 是本项目的七参数光谱逆向模型名称。昨日的 `compact_raw_mirror` 记为 0.1，本轮在分支 `codex/metaformerpro` 比较圆周角度输出、参数查询池化与重叠分块，保留原模型和历史数据。[本轮结果](Metaformer_As2Se3/results/metaformerpro_v02/report.md)。

本轮完成5方案×3种子对照，验证集选中圆周角度头，记为 **MetaformerPro 0.2**。三个种子平均测试：五尺寸MSE降低3.4%、原始标签周期MSE降低8.3%，镜像等价MSE略升1.3%；五个尺寸的MAE均有改善，偶数角仍退步。参数量320,938，推理输出仍为7个物理参数，内部圆周编码不是新增自由度。当前0.2权重和0.1三个基线均保留，尚无全指标一致胜出的版本。

项目工作树：`D:/projects/MetaformerPro`。PyCharm 打开内部 `Metaformer_As2Se3` 目录，仍使用 `D:/Anaconda3/envs/transformer_env/python.exe`；新配置为 `13 MetaformerPro train v02` 和 `14 MetaformerPro predict`。新权重为 `models/metaformerpro.pt`，三个 0.1 基线快照独立保留。

所有当前工作仅优化逆向模型。沿用现有1000条数值数据，固定划分，验证集选型后再评估既有测试集。未新增正向网络、ResNet或LSTM，也未将新生成器数据混入历史训练集。

已在 `codex/metaformerpro-structure-view` 分支接通固定光谱输入和结构展示：PyCharm运行项 **15 MetaformerPro spectrum to structure** 读取 `inputs/target_spectrum.csv`，加载选中权重，保存七参数JSON、俯视图及三维图，并自动打开图片。示例为固定207号目标谱，使用方法见[输入说明](Metaformer_As2Se3/README.md#输入固定光谱并展示结构)。当前权重对应1.0旧层序；输入目标谱与候选回仿真光谱有明确区分。

## 历史模型记录

本分支 `codex/inverse-feature-optimization` 在独立工作树中使用现有 1000 条光谱，比较小型 Transformer、斜率/曲率多尺度编码、保留相位的 Fourier token 和镜像等价联合损失。原始数据与历史训练代码、权重保持原样。

[本轮实验与结果](Metaformer_As2Se3/results/inverse_features1000/report.md)。新入口为 `optimize_spectral_inverse.py` 和 `predict_spectral_inverse.py`；PyCharm 使用配置 11/12。所有方案只训练逆向映射。

七个方案、每个三个种子的验证选中 `compact_raw_mirror`：320,552 参数，约比原模型少 92%；同一镜像等价口径的平均测试 MSE 从 0.026834 降至 0.021085（降低 21.4%）。原始标签周期 MSE 从 0.032750 变为 0.033056（略升 0.9%），两种评价同时保留。最终模型使用原始光谱和镜像联合损失，未选用导数/Fourier 特征分支。

**材料更正：1000 样本服务器数据也沿用了已知几乎丢失 Au 欧姆损耗的拟合。以下结果用于数值流程和模型对照，真实金器件性能需要修正材料后重新验证。** [证据](Metaformer_As2Se3/results/inverse_roundtrip_207/report.md)

此前分支：`codex/train-au100-metal8-1000`。已完成生成器 1.0 的 1000 样本逆向训练，固定划分 800/100/100；[训练结果](Metaformer_As2Se3/results/inverse_au100_1000/report.md)包含数据核对、参数误差、图表和使用方式。

模型保持 3,998,087 个参数，使用训练集光谱标准化。100 条测试 MSE 为 0.033422；在 88 个此前未出现的相同测试结构上，比旧 100 样本模型低 34.1%。五个尺寸参数明显改善，两个角仍约 21°。

已完成[一个新测试目标的真实 FDTD 回仿真](Metaformer_As2Se3/results/inverse_roundtrip_207/report.md)：网络直接输出的结构保留双谷形状，谷位置约 771.1/877.2 nm，相对目标偏移约 1.0/4.0 nm；第二谷仍超出 2 nm 参考标准。

此前的[100 样本配置对照](Metaformer_As2Se3/docs/inverse_optimization.md)及旧权重保留。训练和推理入口位于 [Metaformer_As2Se3](Metaformer_As2Se3/README.md)。
Optic design
