# As₂Se₃ 四聚体与背面金属膜：700–1700 nm

## 几何和固定实验条件

七个自由度按 `p,l,h,alpha_x,alpha_y,theta_odd_deg,theta_even_deg` 排列，分别对应 P、L、H、dX、dY、ω奇数象限、ω偶数象限。前五项单位 nm，后两项单位 deg。dX、dY 为**未旋转时边缘间隙**，旋转后中心位置保持不变，实际边缘最短距离随角度变化。第一和第三象限同角，第二和第四象限同角；不强制两组角互为相反数。`theta_even=-theta_odd` 是待研究子空间，不是模型约束。

从上往下：空气 / As₂Se₃ 四个正方柱 / 350 nm SiO₂ 层 / 150 nm Au 或 Ag / 空气。金属位于基板**底部**。这里明确采用有限厚度基板；如果实际器件使用毫米厚基板，350 nm 仿真层代表的腔结构不能直接代替实际器件，必须重新设置厚度。当前基板厚度和金属厚度作为实验条件固定，不加入网络标签。

正角为俯视逆时针，Lumerical 原生旋转属性使用相反符号。四个中心依次位于 III、II、IV、I，角度依次为奇、偶、偶、奇。检查各多边形不相交、不接触、不越过周期边界。整体保留 180° 旋转对称；通常不再有四重旋转对称。

## 研究步骤

1. 用实测复折射率建立有损 As₂Se₃，核对材料拟合。数据来源和最后 12 nm 的明确延拓见 `materials/README.md`。
2. 每种金属先算裸基板+金属、未旋转四聚体；区分基板腔特征与四聚体带来的变化。
3. 固定 P=660、L=190、H=300、dX=dY=120 nm，比较共同旋转 15°、相反旋转 ±10°/±20°、仅奇数象限旋转 20°及交换两组角度的对照。Au/Ag 共 14 个工程探索样本。已保存计划的 `mirror_opposite_20` 是历史案例 ID；其几何实际为 −20°/+20°。在 dX=dY 时，该案例是 +20°/−20° 结构的整体 90° 旋转，可用来比较 x/y 偏振响应，不能作为镜像等谱校验。
4. 将角度写成平均角 `m=(ω奇+ω偶)/2` 与差角 `δ=(ω奇-ω偶)/2`。沿 m=0 扫 δ，观察谱线位置、深度及分裂；沿 δ=0 扫 m，分离共同旋转与相对旋转的影响。随后扫 dX−dY 检查各向异性，再扫 L/H 定位共振区间。
5. 对候选做 20→10→5 nm 网格细化；金属 z 网格 5→2.5 nm，检查特征波长、深度和线宽变化。候选附近谱线至少跨 5 个点，进一步用窄波段高分辨采样确认。计算时间和自动关断也需验证。
6. 只有收敛后才讨论规律。可检验假设：差角增大可能改变模式耦合和谱线分裂；基板厚度可能控制腔相位；金属损耗可能改变反射谷深度。这些是研究假设，当前不预称为 Fano、准 BIC 或完美吸收。此类命名还需场分布、模态或拟合证据。
7. 明确目标中心、形状、线宽后，用前向 FDTD 确認候选，再扩展数据训练七输出 Metaformer。12 个 pilot 样本只能验证流程，不能支持可靠反设计精度结论。

## 光谱筛选和可辨识性

默认以反射谷的 prominence 筛选，阈值 0.10；同时记录谷底、半 prominence 线宽、采样点数和吸收最大值。`lambda_over_width` 是谱线特征指标，**不能自动当作模态 Q**。不会通过裁剪 R/T 或逐条 min-max 归一化修饰光谱。T 监视器放在金属下方空气中，A=1−R−T 为完整结构总吸收；不强制设 T=0。

正方形取向每 90° 等效，角度损失使用周期距离，避免 −45° 与 +45° 被当作不同取向。两个角的符号不能分别取绝对值，镜像会交换象限组并改变符号。一个强度反射谱仍可能对应多个结构，最终评价应包含预测结构重新仿真的光谱误差。Au 和 Ag 应分别建数据集和训练模型；若混合，则须显式加入金属条件输入，这已超出当前七参数标签方案。

## 运行

在开发工作树中使用 `D:\Anaconda3\envs\transformer_env\python.exe`：

```powershell
& D:\Anaconda3\envs\transformer_env\python.exe generate_dataset.py doctor --config configs/as2se3_au_pilot.json
& D:\Anaconda3\envs\transformer_env\python.exe explore_spectra.py plan
& D:\Anaconda3\envs\transformer_env\python.exe explore_spectra.py run --limit 2
& D:\Anaconda3\envs\transformer_env\python.exe explore_spectra.py run
& D:\Anaconda3\envs\transformer_env\python.exe generate_dataset.py generate --config configs/as2se3_au_pilot.json
& D:\Anaconda3\envs\transformer_env\python.exe train_metaformer.py --config configs/train_as2se3_pilot.json
```

继续相同探索命令会校验并跳过已完成样本。探索结果位于 `data/near_ir_study/ranking.json`、`spectra.png` 和独立 `.fsp`；与用于训练的随机样本计划分开。已有目标曲线时，保存 CSV 表头 `wavelength_nm,reflectance` 并传 `--target-csv`，以原始反射率 RMSE 排序。

Ansys 材料导入依据：https://optics.ansys.com/hc/en-us/articles/360034409654-setmaterial-Script-command

材料测量论文：https://doi.org/10.1021/acsami.0c12525

## 当前验证范围

软件测试覆盖旧五/六参数流程及新七参数流程，包括几何碰撞、象限映射、金属层与监视器位置、数据续跑、混合单位、周期角度误差、真实优化器更新和七参数结构预览。测试替代求解器只在测试内部使用，研究光谱来自本机真实 FDTD。尚未用大规模七参数数据训练出可宣称精度的反设计模型。

`validate_materials.py` 可独立导出材料数据库值与 FDTD 拟合值。`configs/as2se3_au_refit.json` 将 As₂Se₃ 的拟合 tolerance 设为 0，imaginary weight 设为 2；在此次本机检查中，k 最大误差由约 0.0215 降至约 0.00171，n 最大误差约 0.00681。金属仍采用相同 Palik 库及其原生拟合，Au 拟合误差仍需在后续损耗敏感设计中复核。

```powershell
& D:\Anaconda3\envs\transformer_env\python.exe validate_materials.py --config configs/as2se3_au_refit.json --output data/material_refit
& D:\Anaconda3\envs\transformer_env\python.exe explore_spectra.py run --config configs/as2se3_au_refit.json --output data/near_ir_refit --limit 5
& D:\Anaconda3\envs\transformer_env\python.exe verify_candidate.py --case Au_opposite_20 --mesh 10 5
& D:\Anaconda3\envs\transformer_env\python.exe pycharm_demo.py --dataset data/as2se3_au_pilot/dataset.npz
```

`verify_candidate.py` 保存每一级网格的独立结果，按最接近原主谷的位置配对，比较波长、谷底及半 prominence 线宽，支持已完成网格的复用。仅进行 20→10 nm 一次比较时，不会标为最终网格收敛。材料拟合与网格收敛分别记录，互不替代。
