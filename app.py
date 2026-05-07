import os
import io
import zipfile
import tempfile
import traceback
import torch
import gradio as gr
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger


# ── 全域模型（延遲載入） ──────────────────────────────────
ws_model = None
pos_model = None


def get_device():
    """自動偵測可用裝置"""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return 0, f"GPU ({name})"
    return -1, "CPU"


def load_models(device_id):
    """載入 CKIP 模型（僅在首次呼叫時載入）"""
    global ws_model, pos_model
    if ws_model is None or pos_model is None:
        ws_model = CkipWordSegmenter(model="bert-base", device=device_id)
        pos_model = CkipPosTagger(model="bert-base", device=device_id)
    return ws_model, pos_model


def load_user_dictionary(dict_path):
    """載入自訂字典，回傳詞彙集合"""
    words = set()
    try:
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    words.add(word)
    except Exception as e:
        print(f"載入字典時發生錯誤: {e}")
    return words


def merge_tokens_with_dict(tokens, pos_tags, user_dict):
    """使用自訂字典合併斷詞結果與對應的詞性標註"""
    if not user_dict:
        return tokens, pos_tags

    merged_tokens = []
    merged_pos = []
    i = 0
    while i < len(tokens):
        matched = False
        for length in range(min(10, len(tokens) - i), 0, -1):
            candidate = ''.join(tokens[i:i+length])
            if candidate in user_dict:
                merged_tokens.append(candidate)
                merged_pos.append(pos_tags[i + length - 1])
                i += length
                matched = True
                break
        if not matched:
            merged_tokens.append(tokens[i])
            merged_pos.append(pos_tags[i])
            i += 1

    return merged_tokens, merged_pos


def process_files(input_files, dict_file, progress=gr.Progress()):
    """主要處理函式"""
    if not input_files:
        return "請上傳至少一個 .txt 檔案", None

    # 偵測裝置
    device_id, device_name = get_device()
    log_lines = [f"裝置: {device_name}"]

    # 載入模型
    progress(0, desc="載入 CKIP 模型中...")
    log_lines.append("載入 CKIP 模型中...")
    try:
        ws, pos = load_models(device_id)
    except Exception as e:
        return f"模型載入失敗: {e}", None
    log_lines.append("模型載入完成")

    # 載入自訂字典
    user_words = set()
    if dict_file is not None:
        log_lines.append(f"載入自訂字典: {os.path.basename(dict_file)}")
        user_words = load_user_dictionary(dict_file)
        log_lines.append(f"已載入 {len(user_words)} 個自訂詞彙")

    # 建立暫存資料夾存放結果
    tmp_dir = tempfile.mkdtemp()
    total = len(input_files)

    for idx, file_path in enumerate(input_files):
        file_name = os.path.basename(file_path)
        progress((idx) / total, desc=f"處理 {file_name} ({idx+1}/{total})")
        log_lines.append(f"\n[{idx+1}/{total}] 處理: {file_name}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if not text.strip():
                log_lines.append(f"  警告: {file_name} 為空檔案，跳過")
                continue

            lines = text.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]

            if non_empty_lines:
                # 斷詞
                log_lines.append("  執行斷詞...")
                seg_results = ws(non_empty_lines)

                # 詞性標註（在原始斷詞結果上執行）
                log_lines.append("  執行詞性標註...")
                pos_results = pos(seg_results)

                # 套用自訂字典合併
                if user_words:
                    log_lines.append("  套用自訂字典...")
                    merged = [merge_tokens_with_dict(seg, pos_tag, user_words)
                              for seg, pos_tag in zip(seg_results, pos_results)]
                    final_tokens_list = [m[0] for m in merged]
                    pos_results = [m[1] for m in merged]
                else:
                    final_tokens_list = seg_results
            else:
                final_tokens_list = []
                pos_results = []

            # 重建每一行
            result_lines = []
            non_empty_idx = 0
            for line in lines:
                if not line.strip():
                    result_lines.append('')
                else:
                    tokens = final_tokens_list[non_empty_idx]
                    pos_tags = pos_results[non_empty_idx]
                    word_pos_pairs = []
                    for word, pos_tag in zip(tokens, pos_tags):
                        if pos_tag not in ['DASHCATEGORY', 'WHITESPACE']:
                            pos_tag = pos_tag.replace('V_2', 'V2')
                            word_pos_pairs.append(f"{word}_{pos_tag}")
                    result_lines.append(' '.join(word_pos_pairs))
                    non_empty_idx += 1

            seg_text = '\n'.join(result_lines)

            # 儲存結果
            base_name = os.path.splitext(file_name)[0]
            output_file_name = f"{base_name}_seg.txt"
            output_path = os.path.join(tmp_dir, output_file_name)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(seg_text)

            log_lines.append(f"  完成 → {output_file_name}")

        except Exception as e:
            log_lines.append(f"  錯誤: {e}")
            traceback.print_exc()
            continue

    progress(1, desc="打包結果中...")

    # 打包成 zip
    zip_path = os.path.join(tmp_dir, "segmentation_results.zip")
    seg_files = [f for f in os.listdir(tmp_dir) if f.endswith("_seg.txt")]

    if not seg_files:
        return '\n'.join(log_lines) + "\n\n沒有產生任何結果檔案", None

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for seg_file in seg_files:
            zf.write(os.path.join(tmp_dir, seg_file), seg_file)

    log_lines.append(f"\n所有檔案處理完成！共 {len(seg_files)} 個結果檔案已打包")

    return '\n'.join(log_lines), zip_path


# ── Gradio 介面 ──────────────────────────────────────────
device_id, device_name = get_device()

with gr.Blocks(title="CKIP 中文斷詞與詞性標註工具") as app:
    gr.Markdown(
        f"""
        # CKIP 中文斷詞與詞性標註工具
        使用中研院 CKIP Transformers 進行中文斷詞與詞性標註（POS Tagging）。

        **目前裝置：{device_name}**
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_files = gr.File(
                label="上傳文本檔案（可多選 .txt）",
                file_count="multiple",
                file_types=[".txt"],
                type="filepath",
            )
            dict_file = gr.File(
                label="上傳自訂字典（選填，.txt，每行一個詞彙）",
                file_count="single",
                file_types=[".txt"],
                type="filepath",
            )
            run_btn = gr.Button("開始斷詞與標註", variant="primary", size="lg")

        with gr.Column(scale=1):
            log_output = gr.Textbox(
                label="處理紀錄",
                lines=20,
                max_lines=30,
                interactive=False,
            )
            download_output = gr.File(
                label="下載結果（ZIP）",
                interactive=False,
            )

    run_btn.click(
        fn=process_files,
        inputs=[input_files, dict_file],
        outputs=[log_output, download_output],
    )

    gr.Markdown(
        """
        ---
        **輸出格式：** `詞彙_詞性` 以空格分隔，例如：`那_Nep 一陣子_Nd 東京都_Nc`
        **使用方式：** 上傳 .txt 檔案 → 選擇性上傳自訂字典 → 點擊「開始斷詞與標註」→ 下載結果 ZIP
        """
    )

if __name__ == "__main__":
    app.launch(inbrowser=True, theme=gr.themes.Soft())
