# 正向 Metaformer 首轮验证

本分支 `codex/metaformer-forward-pilot` 首先补齐“七参数结构 → 反射光谱”的训练和独立推理入口。开发工作树为 `D:/projects/Optics_design_forward_pilot`。现有逆向网络、逆向训练器和 FDTD 生成器均保持原样；透射谱联合输出及逆向光谱一致性训练留待后续。

## 网络

`[B,7]` 归一化物理参数 → 5 个长度加两组 sin/cos 角度，共 9 个特征 → Linear(9,128) → 广播成 100 个波长分块 token → 原有正弦位置编码 → 3 层 Transformer Encoder（4 头，每头 64 维，FFN 256）→ LayerNorm → 每个 token 输出 10 点反射率 → `[B,1000]`。

两个方柱角按 `sin(4θ), cos(4θ)` 编码，θ 使用弧度，保证 90° 周期；物理自由度仍为 7。位置编码使用固定波长网格上的 token 序号。与原项目共用 EncoderBlock 的实现，但权重独立；未加入 ResNet、LSTM 或场图分支。这是基于现有编码块的本地正向实现，不宣称逐行复现原文源码。

## 训练与评价

输入数据见 [examples](../examples/au100_metal8_100/README.md)。固定 80/10/10 划分、seed=42、AdamW、lr=3e-4、batch=8；最多 150 轮，验证集 MSE 改善阈值 min_delta=1e-5，早停 25 轮。训练损失为原始反射率的整谱 MSE。本轮不搜索超参数、不用测试集选择模型。

额外计算两个最显著目标谷附近的误差。目标谷突出度至少 0.05，预测谷至少 0.02，在目标 ±15 nm 内匹配未使用的最近预测谷；缺失明确计数。报告线宽为半突出度宽度，不当作模态 FWHM 或 Q。整谱 MSE 较低不代表已经学会尖锐双谷，必须结合全部测试谱和缺失谷数判断。

常数基线是仅用训练集计算的平均光谱，用于判断网络是否已经学到结构相关变化；它不参与推理。训练结束后重载最佳检查点，使用 CPU、batch=1 核对 GPU 测试预测。运行文件记录源代码哈希、原始数据哈希、配置、划分与真实优化步数。

## 已完成的首轮

模型 598,026 参数，完成 73 轮、730 次真实更新，按上述验证规则选择第 48 轮。10 条测试谱 MSE 为 0.00315906，训练均值基线为 0.00556541；20 个主要目标谷匹配到 10 个，已匹配谷中心平均误差约 7.81 nm、谷底反射率平均误差约 0.382，仍明显不足。

[全部测试图、局部图和训练曲线](../results/forward_au100_seed42/README.md)。可直接使用随分支保存的权重 `models/forward_au100_seed42.pt` 推理，不必重新训练。CPU 单条网络前向中位约 2.91 ms（4 线程、FP32，已预热，不含加载/CSV）；CPU 检查点重载与 GPU 预测最大绝对差 2.68e-7。

这些结果只验证第一版正向训练与推理流程，以及宽带趋势的部分学习；尚不能作为可靠的窄双谷代理。后续应先定位整谱 MSE 对窄谷关注不足、表示能力及数据覆盖的影响，保持同一测试划分进行独立比较，不能把本次测试结果用于反复调参后再作为未见测试成绩。

## PyCharm

打开 `D:/projects/Optics_design_forward_pilot/Metaformer_As2Se3`，选择解释器 `D:/Anaconda3/envs/transformer_env/python.exe`（Python 3.10，PyTorch 2.5.1 CUDA 12.1）。不需要 Lumerical 或新建环境。

- `06 Train forward Metaformer`：新建训练运行，保留历史运行。
- `07 View forward pilot results`：查看本分支保存的曲线及全部 10 条测试谱。
- `08 Predict spectrum from geometry`：直接输入七个物理参数，输出 1000 点预测 CSV。

命令行在同一个项目目录执行：

```powershell
python -m unittest tests.test_forward_metaformer -v
python train_forward_metaformer.py --config configs/train_forward_au100_pilot.json
python plot_forward_results.py --run results/forward_au100_seed42 --show --no-save
python predict_forward.py --geometry 650 180 300 120 120 0 0
```

七个输入依次为 P、L、H、dX、dY、奇数象限角、偶数象限角，单位为前五项 nm、后两项 deg。尺寸边界和波长网格随检查点保存；推理遇到越界值会报错，不自动截断。几何在参数范围内仍需检查旋转后实际间隙和可加工性。

## 范围

这是 100 条历史数值数据上的首轮正向验证。它学习的是已保存求解器输出，材料拟合精度和实验结果尚未认证；不能混用新版中间金层数据，也不能据此宣称可靠的全空间预测或完整双向设计系统已完成。
