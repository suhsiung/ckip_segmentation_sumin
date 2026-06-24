# -*- coding: utf-8 -*-
"""產生「譯彩紛呈：重譯文本分析系統」系統架構圖 PNG。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

# 中文字型：微軟正黑體
FONT = r"C:\Windows\Fonts\msjh.ttc"
font_manager.fontManager.addfont(FONT)
_name = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["font.family"] = _name
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(13.2, 10.4), dpi=150)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor("#FFFFFF")


def box(x, y, w, h, text, fc, ec, tc="#0F172A", fs=11, weight="normal", lw=1.6, r=0.025):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={r*100}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, mutation_aspect=0.6)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight=weight, wrap=True, linespacing=1.35)


def arrow(x1, y1, x2, y2, color="#64748B", lw=1.8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=14, color=color, lw=lw, shrinkA=2, shrinkB=2))


# ── 標題 ───────────────────────────────────────────────
box(6, 92.5, 88, 6.2, "", "#0E1530", "#0E1530", r=0.02)
ax.text(50, 96.7, "譯彩紛呈：重譯文本分析系統", ha="center", va="center",
        fontsize=20, color="#A6A6F6", weight="bold")
ax.text(50, 93.7, "TransPrism — Retranslation I  語料淬煉 Corpus Refinery　|　系統架構圖",
        ha="center", va="center", fontsize=10.5, color="#8893C9")

# ── 使用者 / Web UI 層 ─────────────────────────────────
box(36, 85, 28, 4.6, "使用者（瀏覽器）", "#E8EEFF", "#3B5BDB", fs=12, weight="bold")
box(26, 78, 48, 4.8, "Gradio Web 介面　http://127.0.0.1:7860",
    "#DCE7FF", "#2563EB", fs=12, weight="bold")
arrow(50, 85, 50, 82.8)

# ── 兩大分頁 ───────────────────────────────────────────
box(8, 71, 38, 4.8, "分頁 A　斷詞與詞性標註", "#E3F6EE", "#10B981", fs=12.5, weight="bold")
box(54, 71, 38, 4.8, "分頁 B　專名探勘（擴充字典）", "#EFE7FB", "#7C3AED", fs=12.5, weight="bold")
arrow(40, 78, 27, 75.9)
arrow(60, 78, 73, 75.9)

# 全域互斥鎖（置中標註）
box(40, 64.6, 20, 4.2, "全域互斥鎖\n兩流程一次只能執行一項", "#FFECEC", "#E03131",
    tc="#B02020", fs=8.6, weight="bold")

# ── 左：斷詞流程 ───────────────────────────────────────
LX, LW = 9, 36
left_steps = [
    "① 防呆偵測：若輸入已是斷詞格式 → 停止並提示重新上傳原文",
    "② 文字前處理：半形標點轉全形、異體字／錯字修正",
    "③ CKIP 斷詞（WS）＋ 詞性標註（POS）",
    "④ 自訂字典合併（最長匹配）",
    "⑤ 斷詞後修正：規則 i–liv、刪 _ETCCATEGORY 開頭整行等",
    "⑥ 打包輸出結果 ZIP（詞_詞性）",
]
y = 60.5
ys_left = []
for s in left_steps:
    box(LX, y, LW, 5.0, s, "#EAF8F2", "#34D399", fs=8.7)
    ys_left.append(y)
    y -= 6.7
for i in range(len(ys_left) - 1):
    arrow(LX + LW / 2, ys_left[i], LX + LW / 2, ys_left[i + 1] + 5.0)

# ── 右：專名探勘流程 ───────────────────────────────────
RX, RW = 55, 36
right_steps = [
    "① 防呆：若輸入已斷詞 → 自動還原為原始文字",
    "② CKIP NER 命名實體抽取（名稱類）",
    "③ 候選彙整：排除字典已收錄、計次／附例句",
    "④ LLM 逐批審核：建議收錄／剔除（進度條回報）",
    "⑤ 表格勾選確認 → 匯出擴充後字典",
]
y = 60.5
ys_right = []
for s in right_steps:
    box(RX, y, RW, 5.0, s, "#F3ECFB", "#A78BFA", fs=8.7)
    ys_right.append(y)
    y -= 6.7
for i in range(len(ys_right) - 1):
    arrow(RX + RW / 2, ys_right[i], RX + RW / 2, ys_right[i + 1] + 5.0)

# ── 底層：模型與運算 ───────────────────────────────────
box(6, 6, 88, 13.5, "", "#F8FAFF", "#94A3B8", r=0.015, lw=1.4)
ax.text(50, 18.2, "模型與運算層", ha="center", va="center", fontsize=11.5,
        color="#475569", weight="bold")
box(9, 8.2, 25, 6.6, "CKIP Transformers (bert-base)\n斷詞 WS／詞性 POS／命名實體 NER",
    "#E3F6EE", "#10B981", fs=8.6, weight="bold")
box(36, 8.2, 24, 6.6, "PyTorch + CUDA 12.4\n→ GPU (RTX 4090) 加速",
    "#FFF1E0", "#F08C00", tc="#9A5B00", fs=8.8, weight="bold")
box(62, 8.2, 29, 6.6,
    "LLM API 供應商\nOpenRouter／OpenAI／Gemini／\nDeepSeek／Anthropic／Groq",
    "#E7EFFF", "#2563EB", fs=8.4, weight="bold")

# 連到底層
arrow(LX + LW / 2, ys_left[-1], 21, 14.8)     # 斷詞 → CKIP
arrow(RX + RW / 2, ys_right[-1] , 48, 14.8)    # 探勘 NER → CKIP
arrow(RX + RW / 2, ys_right[-1], 76, 14.8, color="#2563EB")  # 探勘 → LLM API

# .env 設定（右下小註）
box(62, 1.6, 29, 3.4, ".env 設定：API Key／模型名稱（Model）", "#FFFDF0", "#CBB24A",
    tc="#7A6A12", fs=8.2)
arrow(76, 8.2, 76, 5.0, color="#CBB24A", lw=1.4)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
out = r"D:\ckip_segmentation_sumin\系統架構圖.png"
fig.savefig(out, dpi=150, facecolor="#FFFFFF", bbox_inches="tight", pad_inches=0.15)
print("已輸出:", out)
