# TRG Pre-only Radiomics Baseline

胃癌新辅助治疗后肿瘤退缩分级（TRG）预测研究的重新起点。

当前只建立一个最简单、可检查的基线：

- 输入：治疗前（Pre）增强 CT；
- ROI：`labelsTr` 中的肿瘤 mask；
- 任务：TRG 0/1 与 TRG 2/3 的二分类；
- 标签：TRG 0/1 记为 responder（`binary_label=1`），TRG 2/3 记为
  non-responder（`binary_label=0`）。

旧仓库 [GC-TRG](https://github.com/CminMaj9/GC-TRG) 只用于追溯历史和核对数据
命名规则。本仓库中的代码从数据审计开始重新建立，不直接继承旧实验结果。

## 当前阶段：Step 00 数据终审

原始数据固定保存在仓库之外：

```text
C:\Users\16129\Desktop\My-TRG\data
├── imagesTr
├── labelsTr
└── labelsTr_old
```

创建环境并安装项目：

```powershell
conda activate trg
python -m pip install -e ".[dev]"
```

运行数据终审：

```powershell
python .\scripts\00_build_manifest.py `
  --data-root "C:\Users\16129\Desktop\My-TRG\data" `
  --strict
```

输出写入 `outputs/step00_manifest/`。其中最重要的是：

- `manifest_report.json`：数据数量与严格校验结果；
- `timepoint_manifest.csv`：150 位患者的 300 个 Pre/Post 时间点；
- `patient_manifest.csv`：150 位完整纵向患者；
- `preonly_labeled_cohort.csv`：后续影像组学 baseline 唯一允许使用的 146 位
  有标签患者的 Pre CT 与当前肿瘤 mask；
- `unassigned_timepoints.csv`：无法由明确 `_000` / `_001` 后缀判定时间点的
  71 对文件；
- `manifest_issues.csv`：所有结构问题和警告。

脚本不读取旧 `manifest.csv`、旧影像组学特征或旧模型结果。原始 NIfTI、本地绝对
路径和生成的 manifest 均不会提交到 Git。

## 预期数据冻结值

| 项目 | 预期值 |
|---|---:|
| CT 文件 | 371 |
| 当前 mask 文件 | 371 |
| 旧 mask 文件 | 370 |
| CT-mask 配对 | 371 |
| 已分配 Pre/Post 时间点 | 300 |
| 未分配时间点 | 71 |
| 完整 Pre/Post 患者 | 150 |
| 有 TRG 标签患者 | 146 |
| 无 TRG 标签患者 | 4 |
| TRG 0/1/2/3 | 25 / 17 / 54 / 50 |
| responder / non-responder | 42 / 104 |

只有 Step 00 严格通过并人工核对输出后，才进入 PyRadiomics 特征提取。
