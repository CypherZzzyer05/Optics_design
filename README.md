# Optics_design
Optic design

本分支 `codex/metaformer-forward-pilot` 新增七参数结构到反射光谱的正向 Metaformer 首轮验证。

在 PyCharm 中打开 `Metaformer_As2Se3/`。使用说明及实验范围见 [正向模型说明](Metaformer_As2Se3/docs/forward_pilot.md)。训练、看图、单结构预测分别使用运行配置 06、07、08。

首轮已完成：598,026 参数、100 条数据、固定 80/10/10 划分。模型能预测部分宽带趋势，尖锐谷仍明显遗漏或变浅。[全部测试图与数值结果](Metaformer_As2Se3/results/forward_au100_seed42/README.md)。这是正向反射谱验证，尚未接入逆向闭环或透射谱训练。
