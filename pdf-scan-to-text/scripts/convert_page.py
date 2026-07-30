"""
扫描版PDF单页转换脚本
功能：将PDF单页转为图片 → 预处理 → OCR识别 → 文字排序 → 图片提取 → 生成Markdown片段
"""
import argparse
import os
import sys
import subprocess

sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR


def read_image_chinese(path):
    """读取含中文路径的图片"""
    img_data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    return img


def save_image_chinese(img, path):
    """保存图片到含中文路径"""
    ext = os.path.splitext(path)[1]
    cv2.imencode(ext, img)[1].tofile(path)


def pdf_page_to_image(pdf_path, page_num, output_dir, dpi=300):
    """将PDF单页转为PNG图片，返回图片路径"""
    prefix = os.path.join(output_dir, f"temp_page_{page_num}")
    cmd = [
        'pdftoppm', '-png', '-r', str(dpi),
        '-f', str(page_num), '-l', str(page_num),
        pdf_path, prefix
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    padded = f"{page_num:03d}"
    img_path = f"{prefix}-{padded}.png"
    if not os.path.exists(img_path):
        for f in os.listdir(output_dir):
            if f.startswith(f"temp_page_{page_num}") and f.endswith('.png'):
                img_path = os.path.join(output_dir, f)
                break
    return img_path


def preprocess_image(img):
    """图片预处理：灰度化 + 自适应二值化"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    return binary


def ocr_image(img, ocr_engine=None):
    """OCR识别，返回文本区域列表"""
    if ocr_engine is None:
        ocr_engine = RapidOCR()

    result, elapse = ocr_engine(img)

    if not result:
        return []

    items = []
    for item in result:
        box = item[0]
        text = item[1]
        confidence = float(item[2]) if not isinstance(item[2], float) else item[2]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        items.append({
            'box': box,
            'text': text,
            'conf': confidence,
            'cx': cx,
            'cy': cy,
            'width': width,
            'height': height,
            'is_vertical': height > width * 1.2,
        })
    return items


def sort_text_items(items):
    """按阅读顺序排列文本区域（横排从左到右、从上到下）"""
    if not items:
        return []

    vertical_items = [i for i in items if i['is_vertical']]
    horizontal_items = [i for i in items if not i['is_vertical']]

    sorted_items = []

    if horizontal_items:
        horizontal_items.sort(key=lambda x: x['cy'])
        avg_height = sum(i['height'] for i in horizontal_items) / len(horizontal_items)
        row_threshold = max(avg_height * 0.5, 20)

        rows = []
        current_row = [horizontal_items[0]]
        for i in range(1, len(horizontal_items)):
            if horizontal_items[i]['cy'] - current_row[-1]['cy'] < row_threshold:
                current_row.append(horizontal_items[i])
            else:
                rows.append(current_row)
                current_row = [horizontal_items[i]]
        rows.append(current_row)

        for row in rows:
            row.sort(key=lambda x: x['cx'])
            sorted_items.extend(row)

    if vertical_items:
        vertical_items.sort(key=lambda x: (-x['cx'], x['cy']))
        sorted_items.extend(vertical_items)

    return sorted_items


def detect_and_extract_images(orig_img, text_items, page_num, images_dir,
                              min_img_width=150, min_img_height=150,
                              min_img_pixels=40000, margin_ratio=0.02):
    """
    检测页面中的图片区域（非文字区域），裁剪并保存。
    
    策略：
    1. Canny边缘检测 + 文字掩码排除 → 找到边缘丰富的非文字区域
    2. 形态学合并 → 让相邻碎片连成片
    3. 相邻区域合并 → 同列、垂直间距小的区域合为一体
    4. 擦除文字 → 合并后图片中夹带的文字行用白色填掉
    5. 裁白边 → 去掉四周白色边距
    6. 过滤碎片 → 去掉过小和纯色区域
    
    返回图片信息列表 [{path, filename, box, y_top, cy, cx}]
    """
    if not os.path.exists(images_dir):
        os.makedirs(images_dir, exist_ok=True)

    h, w = orig_img.shape[:2]
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    
    # 创建文字区域掩码
    text_mask = np.zeros((h, w), dtype=np.uint8)
    for item in text_items:
        box = np.array(item['box'], dtype=np.int32)
        cv2.fillPoly(text_mask, [box], 255)

    # 扩展文字区域
    kernel = np.ones((15, 15), np.uint8)
    text_mask_dilated = cv2.dilate(text_mask, kernel, iterations=1)
    non_text_mask = cv2.bitwise_not(text_mask_dilated)

    # 去除页面边距
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    inner_gray = gray[margin_y:h-margin_y, margin_x:w-margin_x]
    inner_mask = non_text_mask[margin_y:h-margin_y, margin_x:w-margin_x]
    
    # 在灰度图上先擦除文字区域（用白色填充），防止文字边缘干扰图片检测
    inner_gray_clean = inner_gray.copy()
    text_mask_inner = text_mask_dilated[margin_y:h-margin_y, margin_x:w-margin_x]
    inner_gray_clean[text_mask_inner > 0] = 255
    
    # Canny边缘检测（在擦除文字后的图上）
    edges = cv2.Canny(inner_gray_clean, 50, 150)
    
    # 膨胀边缘，使图片区域连成片
    edge_kernel = np.ones((60, 60), np.uint8)
    edges_dilated = cv2.dilate(edges, edge_kernel, iterations=1)
    
    # 只保留非文字区域内的边缘
    edges_in_nontext = cv2.bitwise_and(edges_dilated, edges_dilated, mask=inner_mask)
    
    # 闭运算填充内部空洞
    close_kernel = np.ones((30, 30), np.uint8)
    edges_closed = cv2.morphologyEx(edges_in_nontext, cv2.MORPH_CLOSE, close_kernel)
    
    # 查找连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        edges_closed, connectivity=8)

    # 收集候选区域
    candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT] + margin_x
        y = stats[i, cv2.CC_STAT_TOP] + margin_y
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if bw > w * 0.9 and bh > h * 0.9:
            continue
        aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
        if aspect_ratio > 20:
            continue
        candidates.append({'x1': x, 'y1': y, 'x2': x + bw, 'y2': y + bh,
                           'area': area, 'bw': bw, 'bh': bh})

    # ---- 相邻区域合并 ----
    # 只合并没有被文字行隔开的相邻区域
    # 判断依据：两个区域之间的垂直间距内，如果存在文字行，则不合并
    merge_gap_y = 80
    merge_overlap_x = 0.3

    merged = True
    while merged:
        merged = False
        new_candidates = []
        used = [False] * len(candidates)
        for i in range(len(candidates)):
            if used[i]:
                continue
            cur = candidates[i].copy()
            for j in range(i + 1, len(candidates)):
                if used[j]:
                    continue
                other = candidates[j]
                overlap_x1 = max(cur['x1'], other['x1'])
                overlap_x2 = min(cur['x2'], other['x2'])
                overlap_w = max(0, overlap_x2 - overlap_x1)
                min_w = min(cur['x2'] - cur['x1'], other['x2'] - other['x1'])
                if min_w <= 0:
                    continue
                overlap_ratio = overlap_w / min_w
                gap_y = max(0, max(cur['y1'], other['y1']) - min(cur['y2'], other['y2']))
                if overlap_ratio >= merge_overlap_x and gap_y <= merge_gap_y:
                    # 检查两个区域之间是否有文字行隔开
                    # 上方区域 = y较小的那个，取其底部y2
                    # 下方区域 = y较大的那个，取其顶部y1
                    if cur['y1'] <= other['y1']:
                        upper_y2 = cur['y2']
                        lower_y1 = other['y1']
                    else:
                        upper_y2 = other['y2']
                        lower_y1 = cur['y1']
                    has_text_between = False
                    if lower_y1 > upper_y2:
                        for item in text_items:
                            text_cy = item['cy']
                            text_cx = item['cx']
                            # 文字中心在两个区域之间，且x范围有重叠
                            if upper_y2 < text_cy < lower_y1:
                                if overlap_x1 < text_cx < overlap_x2:
                                    has_text_between = True
                                    break
                    if has_text_between:
                        continue  # 有文字隔开，不合并
                    # 合并：取外接矩形
                    cur['x1'] = min(cur['x1'], other['x1'])
                    cur['y1'] = min(cur['y1'], other['y1'])
                    cur['x2'] = max(cur['x2'], other['x2'])
                    cur['y2'] = max(cur['y2'], other['y2'])
                    cur['bw'] = cur['x2'] - cur['x1']
                    cur['bh'] = cur['y2'] - cur['y1']
                    cur['area'] += other['area']
                    used[j] = True
                    merged = True
            new_candidates.append(cur)
            used[i] = True
        candidates = new_candidates

    # ---- 过滤、裁剪、擦文字、裁白边、保存 ----
    images_info = []
    img_idx = 1

    for c in candidates:
        bw, bh, area = c['bw'], c['bh'], c['area']
        if bw < min_img_width and bh < min_img_height and area < min_img_pixels:
            continue

        # 裁剪区域（整页坐标）
        crop_x1 = max(0, c['x1'] - 5)
        crop_y1 = max(0, c['y1'] - 5)
        crop_x2 = min(w, c['x2'] + 5)
        crop_y2 = min(h, c['y2'] + 5)
        cropped = orig_img[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        # 检查非纯色
        gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        std_dev = gray_crop.std()
        if std_dev < 10:
            continue

        # 自动裁白边：从四边向内收缩，去掉大面积白色边距
        _, binary_crop = cv2.threshold(gray_crop, 240, 255, cv2.THRESH_BINARY_INV)
        rows_nz = np.any(binary_crop, axis=1)
        cols_nz = np.any(binary_crop, axis=0)
        if rows_nz.any() and cols_nz.any():
            rmin, rmax = np.where(rows_nz)[0][[0, -1]]
            cmin, cmax = np.where(cols_nz)[0][[0, -1]]
            ch, cw = cropped.shape[:2]
            pad = 3
            rmin = max(0, rmin - pad)
            rmax = min(ch - 1, rmax + pad)
            cmin = max(0, cmin - pad)
            cmax = min(cw - 1, cmax + pad)
            cropped = cropped[rmin:rmax + 1, cmin:cmax + 1]

        # 保存图片
        img_filename = f"p{page_num:03d}_img{img_idx:02d}.png"
        img_path = os.path.join(images_dir, img_filename)
        save_image_chinese(cropped, img_path)

        images_info.append({
            'path': img_path,
            'filename': img_filename,
            'box': [[crop_x1, crop_y1], [crop_x2, crop_y1], [crop_x2, crop_y2], [crop_x1, crop_y2]],
            'y_top': crop_y1,  # 图片顶部y坐标，用于排序
            'cy': (crop_y1 + crop_y2) / 2,
            'cx': (crop_x1 + crop_x2) / 2,
        })
        img_idx += 1

    return images_info


def is_figure_caption(text):
    """判断文本是否为图片标题（如 '图 3.国王主题。'、'图2.无忧中的长笛音乐会'）"""
    import re
    return bool(re.match(r'^图\s*\d', text))


def generate_page_markdown(page_num, text_items, images_info, conf_threshold=0.3):
    """
    生成单页Markdown内容。
    
    同一段落内的文本行合并为一行（不加换行），段落之间用空行分隔。
    判断逻辑：
    1. 相邻文本行的cy间距超过段落阈值 → 不同段落
    2. 图片标题行（以"图+数字"开头）结尾处强制分段
    """
    # 过滤低置信度文本
    filtered_items = [i for i in text_items if i['conf'] >= conf_threshold]

    # 排序文字
    sorted_items = sort_text_items(filtered_items)

    # 合并文字和图片，按位置排序
    all_elements = []
    for item in sorted_items:
        all_elements.append({
            'type': 'text',
            'cy': item['cy'],
            'cx': item['cx'],
            'height': item['height'],
            'content': item['text'],
        })
    for img in images_info:
        all_elements.append({
            'type': 'image',
            # 图片用顶部y坐标排序，这样图片会排在它下方的标题文字之前
            'cy': img['y_top'],
            'cx': img['cx'],
            'height': 0,
            'content': f"![{img['filename']}](images/{img['filename']})",
        })

    # 按cy排序（从上到下），同cy按cx排序（从左到右）
    all_elements.sort(key=lambda x: (x['cy'], x['cx']))

    # 计算平均行高，用于判断段落分隔
    text_heights = [e['height'] for e in all_elements if e['type'] == 'text' and e['height'] > 0]
    avg_height = sum(text_heights) / len(text_heights) if text_heights else 60
    # 段落阈值：cy间距超过此值视为不同段落
    # 同段行间距约1.7倍行高，段落间距约3.5倍行高，取中间值2.5倍
    paragraph_gap_threshold = avg_height * 2.5

    # 生成Markdown：合并同段落文本行
    paragraphs = []  # 每个元素是一个段落（字符串）或图片标记
    current_para = []  # 当前段落中的文本片段

    def flush_para():
        """将当前段落中的文本合并为一个字符串"""
        if current_para:
            paragraphs.append(''.join(current_para))
            current_para.clear()

    for i, elem in enumerate(all_elements):
        if elem['type'] == 'image':
            flush_para()
            paragraphs.append('')
            paragraphs.append(elem['content'])
            paragraphs.append('')
        else:
            # 检查与上一个文本元素之间是否有段落间隔
            if current_para:
                prev_elem = None
                # 向前查找最近的文本元素
                for j in range(i - 1, -1, -1):
                    if all_elements[j]['type'] == 'text':
                        prev_elem = all_elements[j]
                        break
                if prev_elem:
                    gap = elem['cy'] - prev_elem['cy']
                    if gap > paragraph_gap_threshold:
                        # 段落分隔：先输出当前段落，再开始新段落
                        flush_para()
            current_para.append(elem['content'])
            # 图片标题行结尾强制分段
            if is_figure_caption(elem['content']):
                flush_para()

    flush_para()

    # 组装页面Markdown
    md = f"\n---\n\n## 第{page_num}页\n\n"
    md += '\n\n'.join(paragraphs)
    return md


def process_page(pdf_path, page_num, output_dir, dpi=300, ocr_engine=None):
    """
    处理PDF单页，返回Markdown片段。
    """
    images_dir = os.path.join(output_dir, 'images')

    # 1. PDF页面转图片
    img_path = pdf_page_to_image(pdf_path, page_num, output_dir, dpi)

    # 2. 读取原始图片
    orig_img = read_image_chinese(img_path)
    if orig_img is None:
        return f"\n---\n\n## 第{page_num}页\n\n> 无法读取页面图片\n"

    # 3. 预处理
    preprocessed = preprocess_image(orig_img)

    # 4. OCR识别
    text_items = ocr_image(preprocessed, ocr_engine)

    # 5. 图片检测与提取
    images_info = detect_and_extract_images(orig_img, text_items, page_num, images_dir)

    # 6. 生成Markdown
    md = generate_page_markdown(page_num, text_items, images_info)

    # 7. 清理临时文件
    try:
        os.remove(img_path)
    except:
        pass

    return md


def main():
    parser = argparse.ArgumentParser(description='扫描版PDF单页转换')
    parser.add_argument('--pdf', required=True, help='PDF文件路径')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--page', type=int, required=True, help='页码（1-based）')
    parser.add_argument('--dpi', type=int, default=300, help='图片DPI（默认300）')
    parser.add_argument('--save-temp', action='store_true', help='保留临时图片文件')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'images'), exist_ok=True)

    ocr_engine = RapidOCR()
    md = process_page(args.pdf, args.page, args.output, args.dpi, ocr_engine)

    output_file = os.path.join(args.output, f"page_{args.page:03d}.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"第{args.page}页处理完成，已保存到: {output_file}")


if __name__ == '__main__':
    main()
