# 8 nm正式计算的原生网格预检

`mesh_grid.json`来自当前真实Lumerical FDTD项目在求解前读取的全部x/y/z坐标，逐单元检查了步长。整个计算域最大步长低于8 nm，金属z最小2.5 nm，符合用户严格低于10 nm要求。

对应第3号双谷候选、Au背板、3000个700–1700 nm光谱点、改进As₂Se₃损耗拟合。正式第一个光谱已完成，证据见`../double_dip_controls8/`；这份文件本身只记录预检，不代表网格收敛。原生项目位于本机`data/double_dip_controls8/projects/Au_full_difference.fsp`，不提交大型原生文件。
