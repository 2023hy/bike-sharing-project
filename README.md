# 共享单车：从原始数据到仓库、OLAP 和数据挖掘

课程项目主题：**共享单车需求的数据仓库、多维分析与数据挖掘**。

## 先看什么

1. `reports/report.html`：中文实验报告，图片已嵌入，单文件可离线打开。
2. `bike_sharing_full_project.ipynb`：按阶段组织的 Jupyter Notebook，交付版本保留执行输出。
3. `warehouse.sqlite`：已建好的 SQLite 数据仓库，可用 DB Browser for SQLite 或 Python 打开。
4. `sql/`：建表与七组分析 SQL；`results/`：实测指标、预测和 OLAP 结果。

这是基于已有共享单车滞后预测练习扩展的独立项目。为保留真实日期和小时，采用 UCI 原始 CSV，未修改桌面原 Notebook。项目可整体复制到桌面 Jupyter 文件夹，无绝对路径依赖。

## 环境与运行

建议 Python 3.11 或更高，创建独立环境：

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell 可用：.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python project.py
python -m unittest discover -s tests -v
```

打开 Notebook：

```bash
python -m ipykernel install --user --name bike-project --display-name "Bike Sharing Project"
jupyter lab bike_sharing_full_project.ipynb
```

在 Jupyter 中选择刚注册的 **Bike Sharing Project** 内核，依次运行全部单元格。首格会检查工作目录；请从项目目录启动。已有 Notebook 输出可以直接阅读，不必重跑模型。

命令行和 Notebook 均执行完整流程；只需选择一种。已包含官方压缩包和 CSV，运行分析无需下载数据。安装依赖需要可用的软件源。首次运行/重跑会更新项目自己的 `results/`、`reports/`、`data/processed/` 和 `warehouse.sqlite`；请将自行修改的结果另存。

只重新生成报告：`python project.py --report-only`。报告生成要求所有分析结果已经存在。

`results/environment.json` 记录实际验证环境及库版本；`requirements.txt` 给出兼容范围，跨版本的浮点结果可能略有差异。模型 CPU 运行，默认随机种子 42、树模型并行线程 2，无需显卡或付费 API。

## 目录

```text
bike_sharing_project/
  bike_sharing_full_project.ipynb  # 全流程 Notebook（附输出）
  project.py                     # ETL、OLAP、特征、模型、Apriori、CLI
  reporting.py                   # 按真实结果生成中文报告
  requirements.txt
  DATA_DICTIONARY.md             # 原始字段、仓库粒度与防泄漏说明
  data/raw/                      # 官方 ZIP、hour.csv、day.csv、Readme、来源清单
  data/processed/                # 清洗表、缺失小时清单、模型特征表
  warehouse.sqlite               # 星型仓库、预测事实表、ETL 日志
  sql/                           # 可独立运行的建表及 OLAP SQL
  results/                       # 指标、逐条预测、规则、聚类与质量检查
  reports/report.html            # 独立可读报告（内嵌图片）
  reports/report.md              # 可编辑报告文本
  reports/figures/               # 图表 PNG
  tests/test_project.py          # 时间泄漏、Apriori、仓库集成检查
```

## 设计要点

- 事实粒度为全系统每个观测小时，不是单笔订单或站点。
- 仓库使用 SQLite；ROLLUP、CUBE 通过 `GROUP BY + UNION ALL` 实现同义分组集合。
- 缺失小时不补为零；时序滞后按完整小时网格计算，不按“上一行”冒充“上一小时”。
- 需求预测不使用当前小时天气，也不使用相加等于目标的 `casual`、`registered`。
- 训练截至 2012 年 6 月；7—9 月验证；10—12 月测试。回归按验证 MAE、分类按验证 F1 选择。
- 测试是滚动单步，允许下一步使用新观测，模型参数在测试期固定。
- 聚类只用完整日期的 24 维小时份额；关联规则只在训练期挖掘并在后续时期复核。
- 季节编码和温度保留原值，报告明确披露官网与压缩包说明差异。

## 汇报建议

按“问题→数据来源→质量审计→星型模型→六类 OLAP→四类挖掘→评价→局限”讲述。
SQL 的输出既证明仓库能查询，也回答需求如何随日类型和时间变化。预测、分类、聚类和规则分别回答不同问题，不应只罗列算法名称。

最终提交前可按课程模板添加姓名、学号、课程名和教师要求的封面。报告如实包含实际指标和局限，不保证特定成绩。

## 数据与方法引用

数据：Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5W894 （数据集页面标注 CC BY 4.0）。

原论文：Fanaee-T & Gama, *Event labeling combining ensemble detectors and background knowledge*.
https://doi.org/10.1007/s13748-013-0040-3 。保留随数据分发的 `data/raw/Readme.txt`。

方法参考：scikit-learn 官方 *Lagged features for time series forecasting* 示例。
https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html 。

## 课程专项分析：下一小时租赁量
在项目目录运行 `python analyze_demand_errors.py`，可使用已保存的逐小时预测重新计算分组 MAE、平均有符号误差（预测减实际）、低估比例，并提取所选模型低估最多的一天。输出为 results/demand_group_errors.csv 与 results/largest_underprediction_day.csv。分组沿用训练集租赁量的第 75 百分位数；该标签仅用于事后评价。需要重新训练时先按上文运行 project.py。
