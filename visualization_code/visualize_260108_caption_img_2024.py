import json
import os
import concurrent.futures
from collections import defaultdict
from functools import partial
import time

# ================= 配置区域 =================

# 输入 JSON 路径
JSON_SOURCE_PATH = "/home/v-yuxluo/data/caption_output/v2/captions_from_agg_0000.json"

# 输出 HTML 文件名
OUTPUT_HTML = "visualize_captions_v2_2024.html"

# Blob 基础路径
BLOB_BASE_URL = "https://mcgvisionflowsa.blob.core.windows.net/yuxuanluo/ArXiV_Cleaned_Data_260108/2024/"

# SAS Token
SAS_TOKEN = "sv=2025-07-05&spr=https%2Chttp&st=2026-01-08T13%3A56%3A24Z&se=2026-01-15T14%3A11%3A00Z&skoid=4b98d1ff-397b-40e8-a04e-6d6cbbb1ee35&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2026-01-08T13%3A56%3A24Z&ske=2026-01-15T14%3A11%3A00Z&sks=b&skv=2025-07-05&sr=c&sp=racwdxltf&sig=yvVYd394DOUUus%2B%2BiWo07sVVlR0YY2QVjF360Sf%2Bj5s%3D"

# [新功能] 限制处理的 Paper 数量
# 设置为整数 (e.g., 50) 以仅可视化前 50 个 Paper
# 设置为 None 则处理所有数据
MAX_PAPERS = 200 

# [新功能] 并发进程数 (建议设置为 CPU 核心数)
NUM_WORKERS = 32

# ================= 核心逻辑 =================

def convert_path_to_blob(local_path):
    """路径转换函数"""
    if not SAS_TOKEN.startswith("?"):
        sas = "?" + SAS_TOKEN
    else:
        sas = SAS_TOKEN
        
    parts = local_path.strip().split('/')
    if len(parts) >= 2:
        relative_part = f"{parts[-2]}/{parts[-1]}"
    else:
        relative_part = os.path.basename(local_path)
    
    return f"{BLOB_BASE_URL}{relative_part}{sas}"

def process_single_paper_group(pid, items):
    """
    单个进程的工作函数：接收一个 paper_id 和其对应的 items 列表，
    返回该 Paper 的完整 HTML div 字符串。
    """
    # 预先计算该组所有图片的 HTML，减少主进程负担
    cards_html = []
    
    for item in items:
        # 在子进程中进行路径转换，分散计算压力
        img_url = convert_path_to_blob(item['image_path'])
        filename = os.path.basename(item['image_path'])
        gt_cap = item.get('gt_caption', '')
        gen_desc = item.get('generated_fig_desc', '')
        
        card = f"""
            <div class="card">
                <div class="img-container">
                    <div class="img-title">{filename}</div>
                    <a href="{img_url}" target="_blank">
                        <img src="{img_url}" loading="lazy" alt="{filename}">
                    </a>
                </div>
                <div class="text-container">
                    <div class="caption-box">
                        <span class="label gt">Ground Truth Caption</span>
                        {gt_cap}
                    </div>
                    <div class="caption-box">
                        <span class="label gen">Generated Description</span>
                        {gen_desc}
                    </div>
                </div>
            </div>
        """
        cards_html.append(card)
    
    # 组装该 Paper 的 Section
    paper_html = f"""
    <div class="paper-section">
        <div class="paper-header">📄 Paper ID: {pid} <span style="font-size:0.8em; color:#777; font-weight:normal">({len(items)} images)</span></div>
        <div class="scroll-container">
            {''.join(cards_html)}
        </div>
    </div>
    """
    return paper_html

def load_and_group_data(json_path):
    print(f"Loading JSON from: {json_path} ...")
    start_t = time.time()
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"JSON loaded in {time.time() - start_t:.2f}s. Total items: {len(data)}")
    
    grouped = defaultdict(list)
    for item in data:
        pid = item.get('paper_id', 'Unknown')
        grouped[pid].append(item)
    
    return grouped

# ================= HTML 模板 (保持不变) =================

CSS_STYLE = """
<style>
    body { font-family: 'Segoe UI', sans-serif; background-color: #f0f2f5; margin: 0; padding: 20px; }
    h1 { text-align: center; color: #333; }
    .paper-section { background: white; margin-bottom: 25px; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .paper-header { font-size: 18px; font-weight: bold; color: #1a73e8; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-bottom: 15px; }
    .scroll-container { display: flex; overflow-x: auto; gap: 20px; padding-bottom: 10px; }
    .card { flex: 0 0 500px; background: #fff; border: 1px solid #ddd; border-radius: 6px; display: flex; flex-direction: column; }
    .img-container { background: #e9e9e9; text-align: center; padding: 10px; border-bottom: 1px solid #ddd; min-height: 200px; display: flex; flex-direction: column; justify-content: center; align-items: center; }
    .img-title { font-size: 12px; color: #555; font-family: monospace; margin-bottom: 8px; word-break: break-all; }
    img { max-width: 100%; max-height: 400px; object-fit: contain; cursor: pointer; transition: transform 0.2s; }
    img:hover { transform: scale(1.02); }
    .text-container { padding: 10px; font-size: 13px; line-height: 1.5; flex: 1; display: flex; flex-direction: column; gap: 10px; }
    .caption-box { max-height: 200px; overflow-y: auto; border: 1px solid #eee; padding: 8px; background: #fafafa; border-radius: 4px; }
    .label { font-weight: bold; display: block; margin-bottom: 4px; font-size: 11px; text-transform: uppercase; }
    .label.gt { color: #2e7d32; }
    .label.gen { color: #c62828; }
</style>
"""

HTML_HEADER = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Dataset Visualization</title>
{CSS_STYLE}
</head>
<body>
<h1>Dataset Visualization (Batched by Paper ID)</h1>
"""

# ================= 主程序 =================

if __name__ == '__main__':
    # 1. 加载数据
    grouped_data = load_and_group_data(JSON_SOURCE_PATH)
    sorted_pids = sorted(grouped_data.keys())
    
    # 2. 应用数量限制
    total_papers = len(sorted_pids)
    if MAX_PAPERS is not None and MAX_PAPERS < total_papers:
        sorted_pids = sorted_pids[:MAX_PAPERS]
        print(f"Limiting output to first {MAX_PAPERS} papers (out of {total_papers}).")
    else:
        print(f"Processing all {total_papers} papers.")

    # 准备多进程任务参数
    # 将字典转为 (pid, items) 的元组列表，方便 map
    tasks = [(pid, grouped_data[pid]) for pid in sorted_pids]
    
    print(f"Starting HTML generation with {NUM_WORKERS} workers...")
    start_t = time.time()
    
    html_body_parts = []
    
    # 3. 多进程处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 使用 map 保持顺序，starmap 实际上在 Python 3.3+ 的 executor 中没有直接实现，
        # 所以这里用 lambda 或者简单包装一下，或者直接 submit
        # 为了简单，我们手动 submit 并按顺序收集结果
        
        future_to_pid = {executor.submit(process_single_paper_group, pid, items): pid for pid, items in tasks}
        
        # 按提交顺序（即排序后的 pid）收集结果比较麻烦，as_completed 是乱序的。
        # 更简单的方法是直接 map 一个包装函数，但 map 需要可序列化参数。
        # 我们这里用 map，但需要把参数打包。
        
        results = executor.map(process_single_paper_group, [t[0] for t in tasks], [t[1] for t in tasks])
        
        # executor.map 返回的是一个迭代器，按输入顺序返回结果
        for res in results:
            html_body_parts.append(res)

    print(f"HTML fragments generated in {time.time() - start_t:.2f}s. Assembling file...")

    # 4. 写入文件
    full_html = HTML_HEADER + "".join(html_body_parts) + "</body></html>"
    
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"\nDone! HTML generated at: {os.path.abspath(OUTPUT_HTML)}")