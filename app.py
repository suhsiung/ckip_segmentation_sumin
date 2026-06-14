import os
import io
import re
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


# ── 斷詞前：文字前處理 ────────────────────────────────────
# (1) 半形標點符號 → 全形（只轉標點，保留英數字，例如 B.D 的 B/D 不變）
_HALF_PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
_PUNCT_TO_FULL = {c: chr(ord(c) + 0xFEE0) for c in _HALF_PUNCT}

# (2) 異體字 / 錯字修正（變體字 → 標準字）
_CHAR_FIXES = {
    '躱': '躲', '内': '內', '麽': '麼', '爲': '為', '着': '著',
    '眞': '真', '揷': '插', '旣': '既', '羣': '群', '踪': '蹤',
    '脚': '腳', '啓': '啟', '衆': '眾', '参': '參', '靑': '青',
    '盗': '盜', '祇': '只', '祗': '只',
}
_CHAR_FIX_TABLE = str.maketrans(_CHAR_FIXES)


def preprocess_text(text):
    """讀取原始文本後的前處理：半形標點轉全形，並修正異體字 / 錯字"""
    text = ''.join(_PUNCT_TO_FULL.get(ch, ch) for ch in text)
    text = text.translate(_CHAR_FIX_TABLE)
    return text


# ── 斷詞後：詞性 / 斷詞修正規則（規則 6 的 i–liv）─────────────
# 每條為 (pattern, replacement)，於每個段落（以空格分隔的 詞_詞性 串）上套用。
# 先做「單一 token 詞性修正」，再做「多 token 合併 / 拆分」，
# 並讓需要前置結果的合併規則（如 小五郞、明智小五郞）排在後面。
_POST_RULES_RAW = [
    # --- 單一 token 詞性修正 ---
    ('着_FW', '著_Di'),                 # iii
    ('著_FW', '著_Di'),                 # iii（前處理已將 着→著，故併同處理）
    ('ㄚ頭_FW', 'ㄚ頭_Na'),             # ii
    ('巡查_VC', '巡查_Na'),             # ix
    ('明智_VH', '明智_Nb'),             # xii/xx（明智為偵探專名，統一為 Nb）
    ('門野_Na', '門野_Nb'),             # xviii
    ('春木_Na', '春木_Nb'),             # xix
    ('篠崎始_Nd', '篠崎始_Nb'),         # xxx
    ('左門_Na', '左門_Nb'),             # xxxiii
    ('小始_Na', '小始_Nb'),             # xxxiv
    ('幸子_Na', '幸子_Nb'),             # xxxv
    ('嗶波_D', '嗶波_Nb'),              # xxxvii
    ('文代_Na', '文代_Nb'),             # xxxviii
    ('文代_Nd', '文代_Nb'),             # xxxviii
    ('十吉_Na', '十吉_Nb'),             # xl
    ('新治_Na', '新治_Nb'),             # xl
    ('新治_VC', '新治_Nb'),             # xli
    ('照老頭_Na', '照老頭_Nb'),         # xl
    ('中村_Nc', '中村_Nb'),             # xliv
    ('黃金塔_Na', '黃金塔_Nb'),         # xlv
    ('空氣人_Na', '空氣人_Nb'),         # xlvii
    ('大友_Na', '大友_Nb'),             # xvi
    ('黑川_Na', '黑川_Nb'),             # xvi
    ('島田_Na', '島田_Nb'),             # xvi
    ('千代_Nd', '千代_Nb'),             # xxi
    ('日下部_Na', '日下部_Nb'),         # xxxii
    ('真的_Na', '真的_D'),              # liii
    ('真是_Na', '真是_D'),              # liii
    ('真_Na', '真_D'),                  # liii

    # --- 多 token 合併 / 拆分 ---
    ('B_FW ．_PERIODCATEGORY D_FW', 'B.D_FW'),       # i
    ('怎_D [麽麼]_FW', '怎麼_D'),                      # iv
    ('那_Dk [麽麼]_FW', '那麼_Dk'),                    # v
    ('時_Nd 不時_D', '時不時_D'),                      # vi
    ('隊員化裝成_VH', '隊員_Na 化裝成_VG'),            # vii
    ('地道_VH(?= 的_DE 中間)', '地道_Na'),             # viii
    ('智擒竊賊_VA', '智擒_VC 竊賊_Na'),                # x
    ('怪_Dfa(?= 輕氣球_Na)', '怪_VH'),                 # xi
    ('小_VH 五_Neu 郞_Nf', '小五郞_Nb'),               # xiii
    ('小_VH 五_Neu 郞_Na', '小五郞_Nb'),               # xiii
    ('小五_Na 郞_Na', '小五郞_Nb'),                    # xiii
    ('明智_Nb 小五郞_Nb', '明智小五郞_Nb'),            # xv（需先完成 xx、xiii）
    ('明智小五_Nb 郞_Na', '明智小五郞_Nb'),            # xiv
    ('木_Na 下_Ncd', '木下_Nb'),                       # xvii
    ('千面_Na 人_Na', '千面人_Nb'),                    # xxii
    ('千_Neu 面_Na 人_Na', '千面人_Nb'),               # xxii
    ('二十_Neu 面_Nf 相_Na', '二十面相_Nb'),           # xxiii / xxiv
    ('二十_Neu 面_Na 相_D', '二十面相_Nb'),            # xxiv
    ('二十_Neu 面_Nf 相_D', '二十面相_Nb'),            # xxiv
    ('二十_Neu 面相_Na', '二十面相_Nb'),               # xxiv
    ('二十_Neu 面_Na 相_Na', '二十面相_Nb'),           # xxiv
    ('小_VH 林芳雄_Nb', '小林芳雄_Nb'),                # xxv
    ('正_D 一_Neu', '正一_Nb'),                        # xxvi
    ('正_D 一_D', '正一_Nb'),                          # xxvi
    ('桂正_Nb 一_D', '桂正一_Nb'),                     # xxvii
    ('桂正_Nb 一_Neu', '桂正一_Nb'),                   # xxvii
    ('壯_VH 二_Neu', '壯二_Nb'),                       # xxviii
    ('今_Nd 井_Na', '今井_Nb'),                        # xxix
    ('羽柴壯_Nb 二_Neu', '羽柴壯二_Nb'),               # xxxi
    ('日_Nd 下_Nes 部_Nc', '日下部_Nb'),               # xxxii
    ('日_Nd 下_Nes 部_Nf', '日下部_Nb'),               # xxxii
    ('大_VH 鳥_Na', '大鳥_Nb'),                        # xxxvi
    ('戶山_Nc 原_A', '戶山原_Nc'),                     # xxxix
    ('八代_Nc 神社_Nc', '八代神社_Nc'),                # xlii
    ('志摩_Nc 半島_Na', '志摩半島_Nc'),                # xliii
    ('透明_VH 怪人_Na', '透明怪人_Nb'),                # xlvi
    ('本_Nes 堂_Nc', '本堂_Nc'),                       # xlviii
    ('本_Nes 堂_Nf', '本堂_Nc'),                       # xlviii
    ('淺_VH 草塔_Na', '淺草塔_Nc'),                    # xlix
    ('埃及_Nc 菸_Na', '埃及菸_Na'),                    # l
    ('搜索_VC 隊_Na', '搜索隊_Na'),                    # li
    ('真_VH 貨_Na', '真貨_Na'),                        # lii
    ('假_VH 貨_Na', '假貨_Na'),                        # lii
    ('自我_Nh', '自_P 我_Nh'),                         # liv

    # ── 第二批斷詞後修正 ────────────────────────────────
    # (1) 省略號（……）正規化為 ……_ETCCATEGORY（多 token 先合併，再處理單 token）
    ('…_Nb …_FW', '……_ETCCATEGORY'),
    ('…_FW …_FW', '……_ETCCATEGORY'),
    ('…_ETCCATEGORY …_ETCCATEGORY', '……_ETCCATEGORY'),
    ('……_FW', '……_ETCCATEGORY'),
    ('…_FW', '……_ETCCATEGORY'),
    # (2) 壯一
    ('壯_VH 一_Neu', '壯一_Nb'),
    ('壯一_VH', '壯一_Nb'),
    # (3) 二十面相 補充（多 token 合併先，再修單 token 詞性）
    ('二十_Neu 面_Na 相身_D 旁_Ncd', '二十面相_Nb 身旁_Nc'),
    ('二十_Neu 面相吃鱉_Na', '二十面相_Nb 吃鱉_Na'),
    ('二十_Neu 面_Nf 相面_Na 前_Ncd', '二十面相_Nb 面前_Nc'),
    ('二十面相_Na', '二十面相_Nb'),
    ('二十面相_D', '二十面相_Nb'),
    ('二十面相_VH', '二十面相_Nb'),
    ('二十面相_VA', '二十面相_Nb'),
    # (4) 哈哈 拆分
    ('哈哈_D', '哈_D 哈_D'),
    # (5) 明智（同前 xx，再次確保）
    ('明智_VH', '明智_Nb'),
    # (6) 日下部
    ('日下部_Nc', '日下部_Nb'),
    # (7) 左門
    ('左門_Nc', '左門_Nb'),
    # (8) 嗶啵
    ('嗶啵_D', '嗶啵_Nb'),
    ('嗶啵_I', '嗶啵_Nb'),
    ('嗶啵_VA', '嗶啵_Nb'),
    # (9) 中村（同前 xliv，再次確保）
    ('中村_Nc', '中村_Nb'),
    # (10) 警部
    ('警部_Nc', '警部_Na'),
    # (11) 真貨 / 假貨（同前 lii，再次確保）
    ('真_VH 貨_Na', '真貨_Na'),
    ('假_VH 貨_Na', '假貨_Na'),
    # (12) 全國 / 全身
    ('全_Neqa 國_Nc', '全國_Nc'),
    ('全_Neqa 身_Na', '全身_Na'),
    # (13) 反方向的「內」→ 內：依需求刻意略過，不處理
    # (14) 明治神宮
    ('明治_Nd 神宮_Nc', '明治神宮_Nc'),
    # (15) 奈良時代
    ('奈良_Nc 時代_Na', '奈良時代_Nd'),
    # (16) 警視總監
    ('警視_Na 總監_Na', '警視總監_Na'),
]

# 以「token 邊界」錨定每條規則，避免跨 token 的子字串誤命中
# （例：真_Na 不會誤觸 天真_Na）。\s 含換行，故段落首尾也算邊界。
_POST_RULES = [
    (re.compile(r'(?<![^\s])(?:' + pat + r')(?![^\s])'), repl)
    for pat, repl in _POST_RULES_RAW
]


def postprocess_line(line):
    """套用規則 6（i–liv）的詞性 / 斷詞修正於單一段落"""
    for pat, repl in _POST_RULES:
        line = pat.sub(repl, line)
    return line


BATCH_SIZE = 50  # 每批次處理的行數，用於即時回報進度


def render_log(log_lines):
    """將紀錄反轉顯示，使最新訊息出現在最上方、舊訊息往下排"""
    return '\n'.join(reversed(log_lines))


def process_files(input_files, dict_file):
    """主要處理函式（使用 yield 串流即時回報進度）"""
    if not input_files:
        yield "請上傳至少一個 .txt 檔案", None
        return

    # 偵測裝置
    device_id, device_name = get_device()
    log_lines = [f"裝置: {device_name}"]

    # 載入模型
    log_lines.append("載入 CKIP 模型中（首次需下載模型，請稍候）...")
    yield render_log(log_lines), None
    try:
        ws, pos = load_models(device_id)
    except Exception as e:
        yield f"模型載入失敗: {e}", None
        return
    log_lines.append("模型載入完成")
    yield render_log(log_lines), None

    # 載入自訂字典
    user_words = set()
    if dict_file is not None:
        log_lines.append(f"載入自訂字典: {os.path.basename(dict_file)}")
        user_words = load_user_dictionary(dict_file)
        log_lines.append(f"已載入 {len(user_words)} 個自訂詞彙")
        yield render_log(log_lines), None

    # 建立暫存資料夾存放結果
    tmp_dir = tempfile.mkdtemp()
    total = len(input_files)

    for idx, file_path in enumerate(input_files):
        file_name = os.path.basename(file_path)
        log_lines.append(f"[{idx+1}/{total}] 處理: {file_name}")
        yield render_log(log_lines), None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # 步驟 1：斷詞前文字前處理（半形標點轉全形 + 異體字修正）
            text = preprocess_text(text)

            if not text.strip():
                log_lines.append(f"  警告: {file_name} 為空檔案，跳過")
                yield render_log(log_lines), None
                continue

            lines = text.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            total_lines = len(non_empty_lines)

            if non_empty_lines:
                # 分批斷詞與標註
                all_seg = []
                all_pos = []
                num_batches = (total_lines + BATCH_SIZE - 1) // BATCH_SIZE

                for b in range(num_batches):
                    start = b * BATCH_SIZE
                    end = min(start + BATCH_SIZE, total_lines)
                    batch = non_empty_lines[start:end]
                    pct = round(end / total_lines * 100)

                    log_lines.append(f"  斷詞中... {end}/{total_lines} 行 ({pct}%)")
                    yield render_log(log_lines), None

                    seg_batch = ws(batch)
                    all_seg.extend(seg_batch)

                    log_lines.append(f"  詞性標註中... {end}/{total_lines} 行 ({pct}%)")
                    yield render_log(log_lines), None

                    pos_batch = pos(seg_batch)
                    all_pos.extend(pos_batch)

                seg_results = all_seg
                pos_results = all_pos

                # 套用自訂字典合併
                if user_words:
                    log_lines.append("  套用自訂字典...")
                    yield render_log(log_lines), None
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
                        # 規則 6-(1)(2)(4)：刪除所有 _WHITESPACE（含段落開頭）；DASHCATEGORY 維持原本過濾
                        if pos_tag in ('DASHCATEGORY', 'WHITESPACE'):
                            continue
                        pos_tag = pos_tag.replace('V_2', 'V2')
                        token = f"{word}_{pos_tag}"
                        # 規則 6-(5)：＇_FW 都刪除
                        if token == '＇_FW':
                            continue
                        word_pos_pairs.append(token)
                    # 規則 6-(3)：刪除段落開頭的 ^_FW
                    if word_pos_pairs and word_pos_pairs[0].endswith('_FW'):
                        word_pos_pairs.pop(0)
                    # 步驟 2-(6)：套用詞性 / 斷詞修正規則（i–liv）
                    line_text = postprocess_line(' '.join(word_pos_pairs))
                    result_lines.append(line_text)
                    non_empty_idx += 1

            seg_text = '\n'.join(result_lines)

            # 儲存結果
            base_name = os.path.splitext(file_name)[0]
            output_file_name = f"{base_name}_seg.txt"
            output_path = os.path.join(tmp_dir, output_file_name)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(seg_text)

            log_lines.append(f"  完成 → {output_file_name}")
            yield render_log(log_lines), None

        except Exception as e:
            log_lines.append(f"  錯誤: {e}")
            traceback.print_exc()
            yield render_log(log_lines), None
            continue

    # 打包成 zip
    log_lines.append("打包結果中...")
    yield render_log(log_lines), None

    zip_path = os.path.join(tmp_dir, "segmentation_results.zip")
    seg_files = [f for f in os.listdir(tmp_dir) if f.endswith("_seg.txt")]

    if not seg_files:
        log_lines.append("沒有產生任何結果檔案")
        yield render_log(log_lines), None
        return

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for seg_file in seg_files:
            zf.write(os.path.join(tmp_dir, seg_file), seg_file)

    log_lines.append(f"所有檔案處理完成！共 {len(seg_files)} 個結果檔案已打包")

    yield render_log(log_lines), zip_path


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
            download_output = gr.File(
                label="下載結果（ZIP）",
                interactive=False,
            )
            log_output = gr.Textbox(
                label="處理紀錄",
                lines=20,
                max_lines=30,
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
