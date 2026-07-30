---
name: pdf-scan-to-text
description: >-
  将扫描版PDF转换成文字版Markdown文档。通过OCR识别扫描图片中的文字，
  提取图片并单独保存，生成结构化的Markdown文件。当用户提到：扫描版PDF转文字、
  PDF OCR识别、扫描件转电子版、图片PDF转文本、PDF文字提取、扫描书转文字版等请求时使用此技能。
---

# 扫描版PDF转文字版

将扫描版PDF（纯图片、无可提取文字层的PDF）通过OCR识别为文字，提取页面中的图片，输出为Markdown格式文档。

## 概述

三阶段流程：环境准备 → 逐页OCR+图片提取 → 生成Markdown文档。

核心工具链：`poppler`(PDF转图片) + `RapidOCR`(OCR识别) + `OpenCV`(图片预处理)。

## 工作流程

```
用户请求 → 阶段1:环境检查 → 阶段2:逐页处理 → 阶段3:生成Markdown
```

## 阶段1：环境检查与准备

### 1.1 检查依赖

确认以下工具和Python包已安装：

**系统工具**（通过scoop安装）：
- `poppler`：提供 `pdftoppm`、`pdfinfo` 等PDF处理命令
- `tesseract`（可选，备选OCR引擎）

**Python包**：
- `rapidocr_onnxruntime`：OCR识别引擎
- `opencv-python`：图片预处理
- `numpy`：数值计算
- `opencc-python-reimplemented`：繁简转换（如需要）

检查并安装命令：

```bash
# 检查系统工具
where.exe pdftoppm
where.exe pdfinfo

# 如未安装，通过scoop安装
scoop install poppler

# 检查Python包
python3 -c "import rapidocr_onnxruntime; print('OK')"
python3 -c "import cv2; print('OK')"
python3 -c "import numpy; print('OK')"

# 如未安装
python3 -m pip install rapidocr_onnxruntime opencv-python numpy
```

### 1.2 获取PDF信息

```bash
pdfinfo "<PDF文件路径>"
```

关注输出中的 `Pages`（总页数）和 `Page size`（页面尺寸）。

### 1.3 创建输出目录

在PDF文件同目录下创建以PDF文件名命名的输出目录：

```
<PDF所在目录>/<PDF文件名（不含扩展名）>/
├── images/          ← 提取的图片
│   ├── p001.png
│   ├── p002.png
│   └── ...
└── <PDF文件名>.md   ← 最终输出的Markdown文件
```

## 阶段2：逐页处理

对PDF的每一页执行以下操作，使用 `scripts/convert_page.py` 脚本：

```bash
python3 scripts/convert_page.py --pdf "<PDF路径>" --output "<输出目录>" --page <页码> [--dpi 300]
```

脚本对每一页执行：

### 2.1 PDF页面转图片

使用 `pdftoppm` 将单页PDF转为PNG图片：

```bash
pdftoppm -png -r 300 -f <页码> -l <页码> "<PDF路径>" "<临时前缀>"
```

- DPI建议300，如文字较小可提高至400或600
- 生成临时PNG文件供后续处理

### 2.2 图片预处理

对图片进行预处理以提高OCR准确率：

1. **灰度化**：转为灰度图
2. **自适应二值化**：使用高斯自适应阈值，增强文字对比度
3. **去噪**（可选）：对质量较差的扫描件可加中值滤波

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 10)
```

### 2.3 OCR文字识别

使用RapidOCR识别文字：

```python
from rapidocr_onnxruntime import RapidOCR
ocr = RapidOCR()
result, elapse = ocr(preprocessed_image)
```

识别结果为文本区域列表，每个区域包含：
- `box`：四角坐标 `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]`
- `text`：识别出的文字
- `confidence`：置信度

### 2.4 文字排序

按阅读顺序排列识别结果：

1. 计算每个文本区域的中心坐标 `(cx, cy)`、宽高
2. 判断文本方向：`height > width * 1.2` 为竖排，否则为横排
3. **横排文本**：按 `cy` 从小到大排序（从上到下），同一行内按 `cx` 从小到大排序
4. **竖排文本**：按 `cx` 从大到小排序（从右到左），同一列内按 `cy` 从小到大排序
5. 对于同一行/列的判断，使用聚类方法：相邻区域间距小于平均字高的0.8倍归为同一行/列

### 2.5 图片检测与提取

检测页面中非文字的图片区域：

1. 在灰度图上先擦除文字区域（用白色填充），防止文字边缘干扰图片检测
2. 对擦除文字后的图像进行Canny边缘检测
3. 膨胀边缘使图片碎片连成片，用非文字掩码过滤，闭运算填充空洞
4. 连通域分析 + 相邻区域合并（仅合并没有文字行隔开的相邻区域）
5. 按最小尺寸过滤（宽度≥150px 且 高度≥150px 或 面积≥40000px²）
6. 裁剪图片区域，自动裁白边，保存为单独的PNG文件
7. 命名规则：`p{页码}_img{序号}.png`（如 `p047_img01.png`）

### 2.6 段落合并与Markdown生成

同一段落内的文本行合并为连续文字（去除行末换行），段落之间用空行分隔。

**段落判断逻辑**：
1. **间距判断**：计算平均行高，以 `avg_height × 2.5` 作为段落分隔阈值。相邻文本行cy间距超过阈值视为不同段落
2. **图片标题分段**：以"图+数字"开头的文本行（如"图 3.国王主题。"）结尾强制分段，后续正文另起段落
3. **图片分段**：图片前后自动分段

每页生成Markdown片段，格式：

```markdown
---

## 第X页

同一段落的连续文字，不会因为原书换行而被截断。

![图片描述](images/p047_img01.png)

图 3.国王主题。

图片标题后的正文另起一段。
```

## 阶段3：生成最终Markdown

### 3.1 合并所有页面

将所有页面的Markdown片段合并为一个完整的Markdown文件，包含：

```markdown
# <书名/文档名>

> 由扫描版PDF转换生成，原始文件：<PDF文件名>
> 转换时间：<时间戳>
> 总页数：<N>

---

（各页内容）
```

### 3.2 图片链接

所有图片使用相对路径引用：

```markdown
![p047_img01](images/p047_img01.png)
```

### 3.3 保存输出

将最终Markdown文件保存到输出目录：

```
<输出目录>/<PDF文件名>.md
```

## 脚本说明

| 文件 | 用途 |
|------|------|
| `scripts/convert_page.py` | 单页处理：PDF转图片→预处理→OCR→排序→图片提取→生成Markdown片段 |
| `scripts/convert_all.py` | 批量处理：遍历所有页面，调用convert_page.py，合并生成最终Markdown |
| `references/ocr_quality_guide.md` | OCR质量优化指南：不同扫描质量的预处理策略 |

## 注意事项

- **中文路径**：OpenCV的 `imread` 不支持中文路径，需使用 `numpy.fromfile` + `cv2.imdecode` 读取，`cv2.imencode` + `tofile` 保存
- **内存控制**：大PDF（数百页）应逐页处理，避免一次性加载所有图片
- **DPI选择**：300DPI适合大多数扫描件；字小或模糊时可提高至400-600，但处理时间会增加
- **置信度过滤**：可设置置信度阈值（如0.3），过滤低质量识别结果
- **图片区域判断**：纯文字页面通常无需提取图片；含图表、插图的页面才需要
- **页码偏移**：PDF页码可能与书籍页码不同（如有封面、目录等），需在Markdown中注明

## 扩展能力（暂不实现）

以下能力留待后续版本扩展：

- **竖排文字处理**：自动检测竖排页面，按从右到左、从上到下重排文字顺序
- **繁简转换**：集成OpenCC，将繁体中文自动转换为简体中文
- **表格识别**：检测并还原表格结构
- **版面分析**：区分标题、正文、脚注等版面元素
- **批量PDF处理**：一次处理多个PDF文件
