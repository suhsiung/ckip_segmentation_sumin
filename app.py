import os
import io
import re
import json
import time
import zipfile
import tempfile
import threading
import traceback
import urllib.request
import urllib.error
from collections import Counter, defaultdict
import torch
import gradio as gr
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker


# ── 全域模型（延遲載入） ──────────────────────────────────
ws_model = None
pos_model = None
ner_model = None

# ── 全域互斥鎖：斷詞與專名探勘一次只能執行一項 ──────────────
# （兩者都吃 GPU/CPU 模型，避免同時執行造成顯存不足或互相拖慢）
_RUN_LOCK = threading.Lock()


def _run_exclusive(gen, busy_value):
    """以全域鎖包裝一個產生器：若另一項作業正在執行，立即回報忙碌並結束；
    否則取得鎖、串流原作業，結束時（含例外）必定釋放鎖。"""
    if not _RUN_LOCK.acquire(blocking=False):
        yield busy_value
        return
    try:
        yield from gen
    finally:
        _RUN_LOCK.release()


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


def load_ner_model(device_id=-1):
    """載入 CKIP NER 命名實體模型（僅在首次呼叫時載入，供專名探勘使用）。

    預設固定用 CPU（device=-1），讓專名探勘不與斷詞搶 GPU 顯存，
    避免在小顯存（如 6GB）上同時執行兩個分頁時 VRAM 不足而中斷。
    """
    global ner_model
    if ner_model is None:
        ner_model = CkipNerChunker(model="bert-base", device=device_id)
    return ner_model


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
    """斷詞主流程（互斥包裝）：與專名探勘一次只能執行一項。"""
    busy = ("⚠️ 系統忙碌中：「專名探勘」或另一項斷詞作業正在執行。"
            "一次只能執行一項（以避免顯存不足），請等其完成後再試。", None)
    yield from _run_exclusive(_process_files_impl(input_files, dict_file), busy)


def _process_files_impl(input_files, dict_file):
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


# ══════════════════════════════════════════════════════════
#  專名探勘（OOV 偵測）：NER 抓候選 → LLM 篩選誤判 → 補字典
#  此功能與斷詞主流程完全獨立，不影響既有處理。
# ══════════════════════════════════════════════════════════

# ── LLM 供應商設定 ───────────────────────────────────────
# 各家 LLM 供應商：name -> {端點, 格式, 常用模型清單}
# 除 Anthropic 採 Messages API 外，其餘皆使用 OpenAI 相容的 /chat/completions。
_PROVIDERS = {
    "OpenRouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "format": "openai",
        "models": [
            "google/gemma-4-26b-a4b-it",
            "google/gemini-2.0-flash-001",
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "deepseek/deepseek-chat",
            "meta-llama/llama-3.3-70b-instruct",
        ],
    },
    "OpenAI": {
        "url": "https://api.openai.com/v1/chat/completions",
        "format": "openai",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "o4-mini"],
    },
    "Google Gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "format": "openai",
        "models": ["gemini-2.0-flash", "gemini-2.0-flash-lite",
                   "gemini-1.5-flash", "gemini-1.5-pro"],
    },
    "DeepSeek": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "format": "openai",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "Anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "format": "anthropic",
        "models": ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest",
                   "claude-sonnet-4-20250514"],
    },
    "Groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "format": "openai",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    },
}
_DEFAULT_PROVIDER = "OpenRouter"
# 模型下拉選單最後一個固定選項：讓使用者改為自行輸入模型名稱
_CUSTOM_MODEL = "✏️ 自行輸入模型名稱…"


def _model_choices(provider):
    """回傳某供應商的模型下拉選項：常用模型清單 + 末尾『自行輸入』選項"""
    models = list(_PROVIDERS.get(provider, {}).get("models", []))
    return models + [_CUSTOM_MODEL]

# 只保留「名稱類」實體，丟掉數字 / 日期 / 時間 / 金額 / 數量等雜訊
_NAME_TYPES = {"PERSON", "LOC", "GPE", "ORG", "FAC", "NORP", "WORK_OF_ART", "EVENT"}
_OOV_BATCH_SIZE = 12     # 每次送給 LLM 的候選數
_OOV_MAX_EXAMPLES = 2    # 每個候選附帶幾個例句
_OOV_MAX_RETRY = 4


def _load_dotenv_value(key):
    """從 .env 讀取設定（先找 repo 目錄，再找上層目錄），找不到回 None"""
    here = os.path.dirname(os.path.abspath(__file__))
    for env_path in (os.path.join(here, ".env"),
                     os.path.join(os.path.dirname(here), ".env")):
        if os.path.exists(env_path):
            try:
                with open(env_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        if k.strip() == key:
                            return v.strip()
            except Exception:
                pass
    return None


def _get_llm_config(provider_override, api_key_override, model_override):
    """決定實際使用的供應商 / API key / 模型：UI 欄位優先，其次環境變數 / .env。
    為相容舊設定，仍接受 OPENROUTER_API_KEY / OPENROUTER_MODEL。"""
    provider = (provider_override or "").strip() or \
        os.environ.get("LLM_PROVIDER") or _load_dotenv_value("LLM_PROVIDER") or \
        _DEFAULT_PROVIDER
    api_key = (api_key_override or "").strip() or \
        os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or \
        _load_dotenv_value("LLM_API_KEY") or _load_dotenv_value("OPENROUTER_API_KEY")
    default_model = _PROVIDERS.get(provider, {}).get("models") or ["google/gemma-4-26b-a4b-it"]
    model = (model_override or "").strip() or \
        os.environ.get("LLM_MODEL") or os.environ.get("OPENROUTER_MODEL") or \
        _load_dotenv_value("LLM_MODEL") or _load_dotenv_value("OPENROUTER_MODEL") or \
        default_model[0]
    return provider, api_key, model


def _call_llm(provider, api_key, model, messages, max_retry=_OOV_MAX_RETRY):
    """呼叫所選供應商的 chat/messages 端點，含退避重試。
    支援 OpenAI 相容格式（OpenRouter / OpenAI / Gemini / DeepSeek / Groq）
    與 Anthropic Messages API。"""
    cfg = _PROVIDERS.get(provider, _PROVIDERS[_DEFAULT_PROVIDER])
    url = cfg["url"]
    if cfg["format"] == "anthropic":
        # Anthropic Messages API：system 需獨立成參數、headers 也不同
        sys_txt = "\n".join(m["content"] for m in messages if m["role"] == "system")
        conv = [m for m in messages if m["role"] != "system"]
        payload = {"model": model, "max_tokens": 4096, "temperature": 0,
                   "system": sys_txt, "messages": conv}
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    else:
        payload = {"model": model, "messages": messages, "temperature": 0}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "ckip-oov-filter",
        }
    data = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(1, max_retry + 1):
        try:
            req = urllib.request.Request(url, data=data,
                                         headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if cfg["format"] == "anthropic":
                return body["content"][0]["text"]
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")
            last_err = f"HTTP {e.code}: {detail[:200]}"
            if e.code in (429, 500, 502, 503):
                time.sleep(2.0 * attempt)
                continue
            break
        except Exception as e:  # noqa
            last_err = repr(e)
            time.sleep(1.5 * attempt)
    raise RuntimeError(last_err or "未知錯誤")


def _extract_json_objects(text):
    """從 LLM 回應容錯抽出 JSON 物件（吃 code fence / 陣列 / 逐個物件）"""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [o for o in parsed if isinstance(o, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except Exception:
        pass
    objs = []
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            objs.append(json.loads(m.group(0)))
        except Exception:
            continue
    return objs


def _build_oov_messages(batch):
    """batch: list of (candidate, ner_type, count, [examples])"""
    lines = []
    for cand, ntype, cnt, exs in batch:
        ex_txt = " / ".join(exs) if exs else "（無）"
        lines.append(f'- 候選詞「{cand}」 NER類型={ntype} 出現{cnt}次\n  例句：{ex_txt}')
    items = "\n".join(lines)
    sys_msg = (
        "你是中文語料的命名實體審核員。我會給你一批由 NER 模型抓出的『專有名詞候選詞』，"
        "這些候選來自一部日系偵探小說（怪盜二十面相／明智小五郎系列）的中文譯本，可能含有誤判。"
        "請逐一判斷每個候選詞是否為『值得收進斷詞字典的真正專有名詞』"
        "（人名、地名、機構、設施、作品名、事件名等）。"
        "下列情況請判為 false：神祇或宗教泛稱、慣用語、單字殘片、被切斷的不完整詞、"
        "純數字/日期/時間/數量、一般名詞。"
        "只回傳 JSON 陣列，每個元素格式為："
        '{"candidate":字串,"is_proper_noun":布林,"type":字串或null,'
        '"add_to_dict":布林,"reason":簡短中文理由}。不要輸出 JSON 以外的任何文字。'
    )
    user_msg = f"請審核下列 {len(batch)} 個候選詞：\n{items}"
    return [{"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}]


# Dataframe 欄位
_OOV_HEADERS = ["收錄", "候選詞", "次數", "類型", "LLM建議", "理由"]


# ── 防呆：偵測並還原「已斷詞」格式的輸入 ──────────────────────
# 若使用者誤把斷詞輸出檔（詞_詞性，如「日本_Nc 的_DE 大_Na 飯店_Nc」）
# 拿來做專名探勘，NER 會把詞性標記也吞進去，產生 Nc / Di / 小林_Nb 等垃圾候選。
# 這裡偵測該格式並還原為原始文字，再進行 NER。
_POS_TAGGED_TOKEN = re.compile(r'^.+_[A-Z][A-Za-z]*\d*$')


def _looks_segmented(lines, sample=60):
    """以前 sample 行抽樣，若多數空白分隔 token 形如『詞_詞性』，判定為已斷詞格式。"""
    tagged = total = 0
    for ln in lines[:sample]:
        for tok in ln.split():
            total += 1
            if _POS_TAGGED_TOKEN.match(tok):
                tagged += 1
    return total > 0 and (tagged / total) >= 0.6


def _strip_pos_tags(line):
    """將『詞_詞性 詞_詞性 …』還原為原始文字（去詞性標記與人工空白）。"""
    words = []
    for tok in line.split():
        m = re.match(r'^(.+)_[A-Z][A-Za-z]*\d*$', tok)
        words.append(m.group(1) if m else tok)
    return ''.join(words)


def discover_proper_nouns(input_files, dict_file, provider_override,
                          api_key_override, model_override):
    """專名探勘主流程（互斥包裝）：與斷詞一次只能執行一項。"""
    busy = ("⚠️ 系統忙碌中：「斷詞」或另一項專名探勘作業正在執行。"
            "一次只能執行一項（以避免顯存不足），請等其完成後再試。",
            gr.update(), 0)
    yield from _run_exclusive(
        _discover_proper_nouns_impl(input_files, dict_file, provider_override,
                                    api_key_override, model_override),
        busy)


def _discover_proper_nouns_impl(input_files, dict_file, provider_override,
                                api_key_override, model_override):
    """專名探勘主流程（generator，串流回報進度）。
    產出：(處理紀錄, 結果表格, 進度百分比)。"""
    empty_df = gr.update(value=[], headers=_OOV_HEADERS)
    if not input_files:
        yield "請上傳至少一個 .txt 檔案", empty_df, 0
        return

    if (model_override or "").strip() == _CUSTOM_MODEL:
        yield ("你選擇了『自行輸入模型名稱』，請直接在模型欄位輸入想使用的"
               "模型名稱後再執行。"), empty_df, 0
        return

    provider, api_key, model = _get_llm_config(
        provider_override, api_key_override, model_override)
    if not api_key:
        yield (f"找不到 {provider} 的 API key。請在下方欄位填入，或設定 .env 的 "
               "LLM_API_KEY。"), empty_df, 0
        return

    log_lines = [f"使用供應商: {provider}｜模型: {model}"]
    # NER 自動偵測裝置：有 GPU 就用 GPU 加速（大顯存如 4090 不必擔心與斷詞搶顯存）；
    # 若 GPU 顯存不足載入失敗，會自動退回 CPU。
    ner_device_id, ner_device_name = get_device()
    log_lines.append(f"載入 NER 模型中（{ner_device_name}）...")
    yield render_log(log_lines), empty_df, 3
    try:
        ner = load_ner_model(ner_device_id)
    except Exception as e:
        # GPU 載入失敗（例如顯存不足）→ 退回 CPU 重試
        log_lines.append(f"NER 於 {ner_device_name} 載入失敗（{e}），改用 CPU 重試...")
        yield render_log(log_lines), empty_df, 3
        try:
            global ner_model
            ner_model = None
            ner = load_ner_model(-1)
        except Exception as e2:
            yield f"NER 模型載入失敗: {e2}", empty_df, 0
            return

    # 載入字典（用於排除已收錄詞）
    user_words = set()
    if dict_file is not None:
        user_words = load_user_dictionary(dict_file)
        log_lines.append(f"已載入字典 {len(user_words)} 詞（將排除已收錄者）")
        yield render_log(log_lines), empty_df, 5

    # 讀取所有文本行
    all_lines = []
    for fp in input_files:
        try:
            with open(fp, "r", encoding="utf-8-sig") as f:
                txt = f.read()
            all_lines.extend([ln.strip() for ln in txt.splitlines() if ln.strip()])
        except Exception as e:
            log_lines.append(f"讀取 {os.path.basename(fp)} 失敗: {e}")

    # 防呆：若輸入其實是「已斷詞」格式（詞_詞性），自動還原為原始文字再做 NER
    if _looks_segmented(all_lines):
        all_lines = [_strip_pos_tags(ln) for ln in all_lines]
        log_lines.append("⚠️ 偵測到輸入為『已斷詞』格式（詞_詞性），已自動還原為原始文字。"
                         "建議直接上傳原始文本以獲得最佳結果。")
        yield render_log(log_lines), empty_df, 7

    log_lines.append(f"共 {len(all_lines)} 段文字，開始 NER 抽取（{ner_device_name}）...")
    yield render_log(log_lines), empty_df, 8

    # NER 抽取名稱類實體
    results = ner(all_lines, use_delim=True)
    counter = Counter()
    type_map = {}
    examples = defaultdict(list)
    for line, sent in zip(all_lines, results):
        for ent in sent:
            w = ent.word.strip()
            if not w or ent.ner not in _NAME_TYPES:
                continue
            counter[w] += 1
            type_map.setdefault(w, ent.ner)
            if len(examples[w]) < _OOV_MAX_EXAMPLES and line not in examples[w]:
                examples[w].append(line if len(line) <= 40 else line[:40] + "…")

    candidates = [(w, type_map[w], c, examples[w])
                  for w, c in counter.items() if w not in user_words]
    candidates.sort(key=lambda x: (-x[2], x[0]))
    log_lines.append(f"NER 完成。名稱類、未收錄候選 {len(candidates)} 個，送 LLM 審核中...")
    yield render_log(log_lines), empty_df, 30

    if not candidates:
        log_lines.append("沒有發現新的專名候選。")
        yield render_log(log_lines), empty_df, 100
        return

    # 分批送 LLM 審核（LLM 速度忽快忽慢，故每批前後都更新進度條與用時，讓使用者
    # 確認系統仍在運作；進度由 30% 線性推進到 100%）
    rows = []
    total_batches = (len(candidates) + _OOV_BATCH_SIZE - 1) // _OOV_BATCH_SIZE
    t_start = time.time()
    for bi in range(total_batches):
        batch = candidates[bi * _OOV_BATCH_SIZE:(bi + 1) * _OOV_BATCH_SIZE]
        elapsed = int(time.time() - t_start)
        # 送出前：先把進度條推到「本批起點」，並標明正在等待 LLM 回應
        pct_before = 30 + int(68 * bi / total_batches)
        log_lines.append(f"⏳ LLM 審核中... 批次 {bi+1}/{total_batches}"
                         f"（已用時 {elapsed}s，請稍候，LLM 回應速度不一）")
        yield (render_log(log_lines),
               gr.update(value=rows, headers=_OOV_HEADERS), pct_before)
        try:
            content = _call_llm(provider, api_key, model, _build_oov_messages(batch))
            got = {o.get("candidate"): o for o in _extract_json_objects(content)}
        except Exception as e:
            log_lines.append(f"    批次 {bi+1} 失敗: {e}")
            got = {}
        for cand, ntype, cnt, _ex in batch:
            o = got.get(cand, {})
            add = o.get("add_to_dict")
            suggest = "收錄" if add is True else ("剔除" if add is False else "未判定")
            rows.append([
                bool(add is True),           # 收錄（勾選）：預設依 LLM 建議
                cand, cnt,
                o.get("type") or ntype,
                suggest,
                o.get("reason", "(LLM未回傳)"),
            ])
        # 完成本批：進度條推到「本批終點」，並回報已處理 / 建議收錄數
        pct_after = 30 + int(68 * (bi + 1) / total_batches)
        keep_so_far = sum(1 for r in rows if r[0])
        log_lines.append(f"✓ 批次 {bi+1}/{total_batches} 完成"
                         f"（已審 {len(rows)} 詞，建議收錄 {keep_so_far} 個）")
        yield (render_log(log_lines),
               gr.update(value=rows, headers=_OOV_HEADERS), pct_after)
        time.sleep(0.3)

    keep_n = sum(1 for r in rows if r[0])
    total_time = int(time.time() - t_start)
    log_lines.append(f"🎉 全部完成！候選 {len(rows)} 個，LLM 建議收錄 {keep_n} 個，"
                     f"共用時 {total_time}s。請在表格勾選確認後，按下方按鈕匯出字典。")
    yield render_log(log_lines), gr.update(value=rows, headers=_OOV_HEADERS), 100


def export_selected_dict(table, dict_file):
    """把表格中『收錄』勾選的候選詞，併入原字典，輸出可下載的新字典檔。"""
    # 取得列資料（Gradio 可能傳 pandas.DataFrame 或 list）
    rows = []
    if table is None:
        rows = []
    elif hasattr(table, "values"):          # pandas DataFrame
        rows = table.values.tolist()
    else:
        rows = list(table)

    selected = []
    for r in rows:
        if len(r) < 2:
            continue
        checked = r[0]
        word = str(r[1]).strip()
        if word and (checked is True or str(checked).lower() in ("true", "1", "勾選", "v")):
            selected.append(word)

    # 併入原字典（直接重讀以保留原順序，新詞附在後面，去重）
    base_words = []
    seen = set()
    if dict_file is not None:
        try:
            with open(dict_file, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip()
                    if w and w not in seen:
                        base_words.append(w)
                        seen.add(w)
        except Exception:
            pass

    added = []
    for w in selected:
        if w not in seen:
            base_words.append(w)
            seen.add(w)
            added.append(w)

    tmp_dir = tempfile.mkdtemp()
    out_path = os.path.join(tmp_dir, "user_dict_updated.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(base_words) + "\n")

    status = (f"已輸出新字典：原 {len(base_words) - len(added)} 詞 + 新增 {len(added)} 詞 "
              f"= 共 {len(base_words)} 詞。新增：{'、'.join(added) if added else '（無）'}")
    return out_path, status


# ── Gradio 介面 ──────────────────────────────────────────
device_id, device_name = get_device()

# 設計系統（由 ui-ux-pro-max skill 產生）：
#   信任藍 #2563EB 主視覺 + 翡翠綠 #10B981「執行」鍵（開發工具慣例 run=綠）
#   字體：Poppins（拉丁/數字）→ Noto Sans TC（繁中），等寬用 JetBrains Mono
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    # 注意：font 清單只在第 0 位放 GoogleFont，其餘用字串，
    # 以避開 Gradio launch 比對主題時的 GoogleFont.__eq__(str) 例外。
    # Poppins（標題拉丁字）改於 CSS @import 套用。
    font=[gr.themes.GoogleFont("Noto Sans TC"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#F8FAFC",
    block_background_fill="#FFFFFF",
    block_border_width="1px",
    block_radius="14px",
    block_label_text_weight="600",
    button_large_radius="10px",
    button_primary_background_fill="#10B981",
    button_primary_background_fill_hover="#059669",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#EEF2FF",
    button_secondary_text_color="#1E293B",
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&display=swap');
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
* { -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility; }
.app-header h1, h1, h2, h3 { font-family: 'Poppins', 'Noto Sans TC', sans-serif; }

/* ── 標題橫幅 ── */
.app-header {
  background: linear-gradient(135deg, #0F172A 0%, #1E293B 45%, #2563EB 130%);
  color: #fff; border-radius: 16px; padding: 26px 30px; margin-bottom: 6px;
  box-shadow: 0 8px 24px rgba(37,99,235,.18);
}
.app-header h1 { margin: 0; font-size: 1.55rem; font-weight: 700; letter-spacing: .2px; line-height: 1.25; }
.app-header p  { margin: .45rem 0 0; color: #CBD5E1; font-size: .95rem; line-height: 1.6; }
.app-header .badge {
  display: inline-flex; align-items: center; gap: 7px; margin-top: 15px;
  background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.22);
  padding: 5px 13px; border-radius: 999px; font-size: .82rem; color: #ECFDF5; font-weight: 500;
}
.app-header .dot { width: 8px; height: 8px; border-radius: 50%; background: #34D399;
  box-shadow: 0 0 0 3px rgba(52,211,153,.25); }

/* ── 互動 ── */
button { cursor: pointer; transition: all .2s ease !important; }
.run-btn button, button.run-btn { font-weight: 600 !important; letter-spacing: .3px; }
button:focus-visible, .tab-nav button:focus-visible { outline: 2px solid #2563EB !important; outline-offset: 2px; }

/* ── 分頁 ── */
.tab-nav button { font-weight: 600 !important; }
.tab-nav button.selected { color: #2563EB !important; }

/* ── 紀錄/結果：終端風格等寬 ── */
.log-box textarea {
  font-family: 'JetBrains Mono', ui-monospace, monospace !important;
  font-size: 13px !important; line-height: 1.65 !important;
  background: #0F172A !important; color: #E2E8F0 !important;
  border-radius: 10px !important; border: 1px solid #1E293B !important;
}

/* ── 區塊標籤更清楚 ── */
.block .label-wrap span, label span { letter-spacing: .2px; }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
"""

with gr.Blocks(title="CKIP 中文斷詞與詞性標註工具") as app:
    gr.HTML(
        f"""
        <div class="app-header">
          <h1>CKIP 中文斷詞與詞性標註工具</h1>
          <p>使用中央研究院 CKIP Transformers 進行中文斷詞（Word Segmentation）與詞性標註（POS Tagging），
             並提供以 NER + LLM 擴充字典的「專名探勘」功能。</p>
          <span class="badge"><span class="dot"></span>運算裝置：{device_name}</span>
        </div>
        """
    )

    with gr.Tabs():
        # ── 分頁 1：斷詞與詞性標註（既有功能）──────────────────
        with gr.TabItem("斷詞與詞性標註"):
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
                    run_btn = gr.Button("開始斷詞與標註", variant="primary",
                                        size="lg", elem_classes=["run-btn"])

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
                        elem_classes=["log-box"],
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

        # ── 分頁 2：專名探勘（找出字典未收錄的專有名詞）──────────
        with gr.TabItem("專名探勘（擴充字典）"):
            gr.Markdown(
                "上傳文本，系統會用 **NER** 找出專有名詞候選，再交給 **LLM** "
                "判斷哪些是真正的專名（過濾神祇泛稱、慣用語、切散殘片等誤判）。"
                "你可在表格勾選確認後，匯出併入原字典的新字典檔。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    oov_input_files = gr.File(
                        label="上傳文本檔案（可多選 .txt）",
                        file_count="multiple",
                        file_types=[".txt"],
                        type="filepath",
                    )
                    oov_dict_file = gr.File(
                        label="上傳現有字典（選填，用於排除已收錄詞並做為匯出基底）",
                        file_count="single",
                        file_types=[".txt"],
                        type="filepath",
                    )
                    oov_provider = gr.Dropdown(
                        label="LLM 供應商",
                        choices=list(_PROVIDERS.keys()),
                        value=_DEFAULT_PROVIDER,
                    )
                    oov_api_key = gr.Textbox(
                        label="API Key（留空則使用 .env 設定）",
                        type="password",
                        placeholder="貼上所選供應商的 API Key",
                    )
                    oov_model = gr.Dropdown(
                        label="模型（可從清單選擇，或選『自行輸入模型名稱…』後直接輸入）",
                        choices=_model_choices(_DEFAULT_PROVIDER),
                        value=_PROVIDERS[_DEFAULT_PROVIDER]["models"][0],
                        allow_custom_value=True,
                    )
                    oov_run_btn = gr.Button("開始專名探勘", variant="primary",
                                            size="lg", elem_classes=["run-btn"])

                with gr.Column(scale=2):
                    # 處理紀錄移到右上方，並加上進度條，避免 LLM 忽快忽慢時誤以為當機
                    oov_progress = gr.Slider(
                        label="進度",
                        minimum=0, maximum=100, value=0, step=1,
                        interactive=False,
                    )
                    oov_log = gr.Textbox(
                        label="處理紀錄（最新狀態在最上方）",
                        lines=8,
                        max_lines=12,
                        interactive=False,
                        elem_classes=["log-box"],
                    )
                    oov_table = gr.Dataframe(
                        headers=_OOV_HEADERS,
                        datatype=["bool", "str", "number", "str", "str", "str"],
                        column_count=(6, "fixed"),
                        label="專名候選（可勾選『收錄』欄）",
                        interactive=True,
                        wrap=True,
                    )
                    with gr.Row():
                        oov_export_btn = gr.Button("匯出選取詞典", variant="secondary")
                    oov_export_status = gr.Textbox(label="匯出結果", interactive=False)
                    oov_download = gr.File(label="下載更新後字典", interactive=False)

            def _on_provider_change(provider):
                """切換供應商時，更新模型下拉清單為該供應商的常用模型
                （末尾保留『自行輸入模型名稱…』選項）"""
                models = _PROVIDERS.get(provider, {}).get("models", [])
                return gr.update(choices=_model_choices(provider),
                                 value=(models[0] if models else _CUSTOM_MODEL))

            oov_provider.change(
                fn=_on_provider_change,
                inputs=[oov_provider],
                outputs=[oov_model],
            )

            oov_run_btn.click(
                fn=discover_proper_nouns,
                inputs=[oov_input_files, oov_dict_file, oov_provider,
                        oov_api_key, oov_model],
                outputs=[oov_log, oov_table, oov_progress],
            )
            oov_export_btn.click(
                fn=export_selected_dict,
                inputs=[oov_table, oov_dict_file],
                outputs=[oov_download, oov_export_status],
            )

if __name__ == "__main__":
    app.launch(inbrowser=True, theme=THEME, css=CUSTOM_CSS)
