"""
扫描版PDF批量转换脚本
功能：遍历PDF所有页面，逐页OCR识别+图片提取，合并生成最终Markdown文档
"""
import argparse
import os
import sys
import subprocess
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

from rapidocr_onnxruntime import RapidOCR

# 导入单页处理模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_page import process_page, read_image_chinese


def get_pdf_info(pdf_path):
    """获取PDF信息，返回字典 {pages, page_size, etc.}"""
    try:
        result = subprocess.run(
            ['pdfinfo', pdf_path],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        info = {}
        for line in result.stdout.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                info[key.strip()] = value.strip()
        return info
    except Exception as e:
        print(f"获取PDF信息失败: {e}")
        return {}


def generate_final_markdown(pdf_path, output_dir, all_pages_md, pdf_info):
    """生成最终合并的Markdown文件"""
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    total_pages = len(all_pages_md)
    
    # 从pdfinfo获取页数
    info_pages = pdf_info.get('Pages', str(total_pages))
    page_size = pdf_info.get('Page size', 'unknown')
    
    header = f"""# {pdf_name}

> 由扫描版PDF转换生成，原始文件：{os.path.basename(pdf_path)}
> 转换时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 总页数：{info_pages}
> 页面尺寸：{page_size}
> OCR引擎：RapidOCR (ONNX)

"""
    
    full_md = header + '\n'.join(all_pages_md)
    
    # 保存
    output_file = os.path.join(output_dir, f"{pdf_name}.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_md)
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='扫描版PDF批量转换')
    parser.add_argument('--pdf', required=True, help='PDF文件路径')
    parser.add_argument('--output', help='输出目录（默认为PDF同目录下以文件名命名的文件夹）')
    parser.add_argument('--start-page', type=int, default=1, help='起始页码（默认1）')
    parser.add_argument('--end-page', type=int, default=0, help='结束页码（0=全部）')
    parser.add_argument('--dpi', type=int, default=300, help='图片DPI（默认300）')
    parser.add_argument('--page-range', help='页码范围，如 "1-10,15,20-30"')
    args = parser.parse_args()

    # 检查PDF文件
    if not os.path.exists(args.pdf):
        print(f"错误：PDF文件不存在: {args.pdf}")
        sys.exit(1)

    # 设置输出目录
    pdf_name = os.path.splitext(os.path.basename(args.pdf))[0]
    pdf_dir = os.path.dirname(os.path.abspath(args.pdf))
    output_dir = args.output or os.path.join(pdf_dir, pdf_name)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

    # 获取PDF信息
    pdf_info = get_pdf_info(args.pdf)
    total_pages = int(pdf_info.get('Pages', 0))
    if total_pages == 0:
        print("警告：无法获取PDF页数，将尝试处理直到失败")

    # 解析页码范围
    pages_to_process = []
    if args.page_range:
        # 解析 "1-10,15,20-30" 格式
        for part in args.page_range.split(','):
            part = part.strip()
            if '-' in part:
                start, end = part.split('-', 1)
                pages_to_process.extend(range(int(start), int(end) + 1))
            else:
                pages_to_process.append(int(part))
    else:
        start = args.start_page
        end = args.end_page if args.end_page > 0 else total_pages
        if end == 0:
            end = 9999  # 安全上限
        pages_to_process = list(range(start, end + 1))

    # 过滤超出范围的页码
    if total_pages > 0:
        pages_to_process = [p for p in pages_to_process if 1 <= p <= total_pages]

    if not pages_to_process:
        print("错误：没有需要处理的页面")
        sys.exit(1)

    print(f"PDF文件: {args.pdf}")
    print(f"输出目录: {output_dir}")
    print(f"待处理页数: {len(pages_to_process)} (第{pages_to_process[0]}页 - 第{pages_to_process[-1]}页)")
    print(f"DPI: {args.dpi}")
    print()

    # 初始化OCR引擎（复用同一实例）
    ocr_engine = RapidOCR()

    # 逐页处理
    all_pages_md = []
    failed_pages = []

    for idx, page_num in enumerate(pages_to_process):
        print(f"[{idx + 1}/{len(pages_to_process)}] 处理第{page_num}页...", end=' ', flush=True)
        try:
            md = process_page(args.pdf, page_num, output_dir, args.dpi, ocr_engine)
            all_pages_md.append(md)
            print("OK")
        except Exception as e:
            print(f"失败: {e}")
            failed_pages.append(page_num)
            all_pages_md.append(f"\n---\n\n## 第{page_num}页\n\n> 页面处理失败: {e}\n")

    # 生成最终Markdown
    print()
    print("正在生成最终Markdown文件...")
    output_file = generate_final_markdown(args.pdf, output_dir, all_pages_md, pdf_info)

    # 统计
    print()
    print("=" * 50)
    print(f"转换完成！")
    print(f"  输出文件: {output_file}")
    print(f"  成功页数: {len(pages_to_process) - len(failed_pages)}")
    if failed_pages:
        print(f"  失败页数: {len(failed_pages)}")
        print(f"  失败页码: {failed_pages}")
    print(f"  图片目录: {os.path.join(output_dir, 'images')}")
    print("=" * 50)


if __name__ == '__main__':
    main()
