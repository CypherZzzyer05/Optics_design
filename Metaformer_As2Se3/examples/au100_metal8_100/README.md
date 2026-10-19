# 正向模型首轮验证数据

复制自本机已完成的真实 FDTD 数据 `D:/仿真/au100_metal8_100/dataset.npz`，没有新增仿真或更改样本。SHA256：`591ca0f4d9f4fc0be9803f3a566f72eee1ee3ac871ad9d03472b93a732aed0e3`。

- 100 条，波长 700–1700 nm、1000 点；固定划分 80/10/10。
- 七参数：`p,l,h,alpha_x,alpha_y,theta_odd_deg,theta_even_deg`。
- 历史 v1 结构，自下而上为 Au 100 nm、有限厚 350 nm SiO2、As2Se3 四聚体。不是 v2 的中间金层/厚基板结构。
- 网格策略为 metal_only，金属 z 网格请求 8 nm；不是全域 8 nm。
- 原始 R/T/A 和来源元数据均保留；本轮正向目标只读取 `spectra` 中的 R，未训练透射率。
- 数据质量为 `pipeline_validation`，Au 材料拟合损耗存在历史限制。训练入口需要显式配置 `allow_validation_data=true`；通过数值训练不代表实验器件性能已认证。
