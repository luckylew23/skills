# OCR质量优化指南

针对不同扫描质量的PDF，提供预处理策略和参数调优建议。

## 扫描质量分级

| 等级 | 特征 | 推荐DPI | 预处理策略 |
|------|------|---------|-----------|
| **优** | 字迹清晰、对比度高、无噪点 | 300 | 仅灰度化+二值化 |
| **良** | 轻微模糊或浅色背景 | 300-400 | 灰度化+自适应二值化+对比度增强 |
| **中** | 明显模糊、有噪点、字迹浅 | 400-600 | 灰度化+去噪+自适应二值化+形态学处理 |
| **差** | 严重模糊、大量噪点、字迹断裂 | 600 | 灰度化+强去噪+二值化+形态学修复+多次尝试 |

## 预处理技术详解

### 1. 灰度化

将彩色图片转为灰度，减少干扰：

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

### 2. 自适应二值化

将灰度图转为黑白图，增强文字对比度。关键参数：

- `blockSize`：邻域大小，必须为奇数。建议值31，大字号可增大至51
- `C`：常数，从均值中减去。建议值10，字迹浅时减小至5

```python
# 标准参数
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 10)

# 字迹浅的文档
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 5)

# 大字号文档
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 51, 10)
```

### 3. 去噪

```python
# 中值滤波（适合椒盐噪声）
denoised = cv2.medianBlur(gray, 3)

# 高斯滤波（适合高斯噪声）
denoised = cv2.GaussianBlur(gray, (3, 3), 0)
```

### 4. 对比度增强

```python
# CLAHE（限制对比度自适应直方图均衡化）
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
```

### 5. 形态学处理

修复断裂文字：

```python
kernel = np.ones((2, 2), np.uint8)
# 膨胀（修复断裂）
dilated = cv2.dilate(binary, kernel, iterations=1)
# 闭运算（填充小孔）
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```

## 常见问题与解决方案

### 问题1：文字断裂

**现象**：OCR识别出大量单字，缺少连贯性

**解决方案**：
1. 降低二值化的C值（如从10降至5）
2. 使用形态学闭运算修复
3. 提高DPI

### 问题2：背景噪点过多

**现象**：OCR识别出大量无意义字符

**解决方案**：
1. 增大二值化的C值（如从10增至15）
2. 预先进行中值滤波去噪
3. 增大置信度过滤阈值

### 问题3：字迹太浅

**现象**：大量文字未被识别

**解决方案**：
1. 降低二值化的C值
2. 使用CLAHE增强对比度
3. 提高DPI

### 问题4：页面倾斜

**现象**：文字行不水平，识别率低

**解决方案**：
1. 使用霍夫变换检测倾斜角度
2. 旋转校正后再OCR

```python
# 检测倾斜角度并旋转校正
coords = np.column_stack(np.where(binary > 0))
angle = cv2.minAreaRect(coords)[-1]
if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
```

### 问题5：中文路径问题

**现象**：OpenCV无法读取/保存含中文路径的文件

**解决方案**：使用numpy中转

```python
# 读取
img_data = np.fromfile(path, dtype=np.uint8)
img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

# 保存
cv2.imencode('.png', img)[1].tofile(path)
```

## RapidOCR参数调优

RapidOCR默认参数对大多数中文文档效果良好，以下为可选调优：

```python
ocr = RapidOCR()

# 可通过调整置信度阈值过滤低质量结果
# 在生成Markdown时设置 conf_threshold 参数
```

## 特殊文档处理建议

### 竖排中文

当前版本暂不完整支持竖排文字的阅读顺序重排。临时方案：
1. 使用 `is_vertical` 标记识别竖排文本区域
2. 按 `cx` 从大到小（从右到左）、`cy` 从小到大（从上到下）排序
3. 后续版本将增加完整的竖排处理支持

### 繁体中文

RapidOCR的中文模型同时支持繁体和简体识别。如需转换为简体：

```python
from opencc import OpenCC
cc = OpenCC('t2s')
simplified = cc.convert(traditional_text)
```

### 双栏排版

对于双栏排版的文档：
1. 检测页面中线区域（非文字的纵向空白带）
2. 将页面分为左右两半
3. 分别对左右两半进行文字排序
4. 先左栏后右栏合并
