# MetaformerPro

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
