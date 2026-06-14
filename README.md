# CKIP 中文斷詞與詞性標註工具

使用中研院 [CKIP Transformers](https://github.com/ckiplab/ckip-transformers) 進行中文斷詞（Word Segmentation）與詞性標註（POS Tagging）的本地端應用程式。

提供 Gradio Web 介面，支援批次上傳多個文本檔案，並可載入自訂字典進行詞彙合併。

## 介面預覽

![CKIP 斷詞工具介面](screenshot.png)

## 功能特色

- 批次處理多個 `.txt` 文本檔案
- **斷詞前文字前處理**：半形標點轉全形、異體字／錯字修正
- **斷詞後修正**：清除空白標記、依規則修正詞性與斷詞結果
- 自訂字典支援（最長匹配合併）
- 自動偵測 GPU / CPU，有 GPU 時自動加速
- 處理紀錄即時顯示（最新訊息在最上方）
- 結果打包為 ZIP 下載
- 跨平台支援（Windows / macOS / Linux）

## 輸出格式

每個詞彙以 `詞彙_詞性` 格式輸出，詞彙之間以空格分隔：

```
那_Nep 一陣子_Nd ，_COMMACATEGORY 東京都_Nc 家家戶戶_Na 所_D 閒談_VA 的_DE 內容_Na
```

---

## 處理流程

每個檔案的處理依序經過三個階段：**斷詞前處理 → CKIP 斷詞與詞性標註 → 斷詞後修正**。

### 1. 斷詞前文字前處理（`preprocess_text`）

在送入 CKIP 模型前，先對原始文字做正規化：

- **半形標點轉全形**：將半形標點符號（如 `,` `.` `:` `"` `'`）轉為對應全形（`，` `．` `：` `＂` `＇`），但保留英數字（例如 `B.D` 中的 `B`、`D` 不變）。
- **異體字／錯字修正**：將常見異體字統一為標準字，對應如下：

  | 修正前 | 修正後 | 修正前 | 修正後 | 修正前 | 修正後 |
  |--------|--------|--------|--------|--------|--------|
  | 躱 | 躲 | 内 | 內 | 麽 | 麼 |
  | 爲 | 為 | 着 | 著 | 眞 | 真 |
  | 揷 | 插 | 旣 | 既 | 羣 | 群 |
  | 踪 | 蹤 | 脚 | 腳 | 啓 | 啟 |
  | 衆 | 眾 | 参 | 參 | 靑 | 青 |
  | 盗 | 盜 | 祇／祗 | 只 | | |

### 2. 斷詞後修正（`postprocess_line`）

完成斷詞與詞性標註後，逐段套用下列修正：

- **清除空白與雜訊標記**：刪除所有 `_WHITESPACE`、段落開頭的 `_FW`、以及 `＇_FW` 標記。
- **詞性與斷詞修正規則**：以「詞_詞性」為單位，套用一組正規表示式規則，修正特定詞彙的斷詞邊界與詞性標註（例如將人名／地名／專名統一標為 `Nb`／`Nc`、合併被切散的專有名詞、修正省略號 `……` 標記等）。

> **注意：** 後處理的詞性修正規則是針對特定語料（怪盜二十面相／明智小五郎系列文本）所調校，內容定義於 `app.py` 的 `_POST_RULES_RAW`。若用於其他文本，可自行在該清單中增刪規則，格式為 `('比對樣式', '取代結果')`，所有規則皆以詞邊界錨定，不會跨詞誤觸。

---

## 系統需求

- Python 3.9 以上
- 建議至少 8GB RAM
- （選用）NVIDIA GPU + CUDA 驅動程式，可大幅加速處理速度

---

## 安裝步驟

### 1. 下載專案

```bash
git clone https://github.com/suhsiung/ckip_segmentation_sumin.git
cd ckip_segmentation_sumin
```

### 2. 建立 Python 虛擬環境

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安裝 PyTorch

請根據你的平台與硬體選擇對應的安裝指令。

#### 僅使用 CPU（所有平台通用）

```bash
pip install torch torchvision torchaudio
```

#### Windows / Linux（NVIDIA GPU，CUDA 12.4）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### macOS（Apple Silicon M1/M2/M3/M4，自動使用 MPS 加速）

```bash
pip install torch torchvision torchaudio
```

> 其他 CUDA 版本或平台組合，請參考 [PyTorch 官方安裝頁面](https://pytorch.org/get-started/locally/)。

### 4. 安裝其他依賴套件

```bash
pip install -r requirements.txt
```

---

## 使用方式

### 啟動應用程式

```bash
python app.py
```

啟動後瀏覽器會自動開啟，若未開啟請手動前往：

```
http://127.0.0.1:7860
```

### 操作步驟

1. **上傳文本檔案** — 點擊左側上傳區，選擇一個或多個 `.txt` 檔案
2. **上傳自訂字典**（選填）— 上傳一個 `.txt` 檔案，每行一個詞彙
3. **點擊「開始斷詞與標註」** — 程式將自動進行斷詞與詞性標註
4. **下載結果** — 處理完成後，右側會出現 ZIP 下載連結

### 自訂字典格式

自訂字典為純文字檔，每行一個詞彙，例如：

```
人工智慧
機器學習
自然語言處理
深度學習
```

程式會使用最長匹配法，將 CKIP 斷詞結果中被切散的詞彙重新合併為字典中的完整詞彙。

---

## 詞性標記（POS Tag）對照表

以下為 CKIP 常見詞性標記說明：

| 標記 | 說明 | 標記 | 說明 |
|------|------|------|------|
| Na | 普通名詞 | VA | 動作不及物動詞 |
| Nb | 專有名詞 | VC | 動作及物動詞 |
| Nc | 地方名詞 | VH | 狀態不及物動詞 |
| Nd | 時間名詞 | VK | 狀態及物動詞 |
| Nep | 指代詞 | D | 副詞 |
| Nf | 量詞 | P | 介詞 |
| Nh | 代名詞 | Caa | 對等連接詞 |
| SHI | 「是」 | Cbb | 關聯連接詞 |
| DE | 「的」 | T | 語助詞 |

> 完整詞性標記請參考 [CKIP 詞性標記說明](https://github.com/ckiplab/ckip-transformers/wiki/POS-Tags)。

---

## 常見問題

### Q: 啟動時出現 CUDA 相關錯誤？

請確認：
1. 已安裝 NVIDIA GPU 驅動程式
2. PyTorch 安裝時選擇了正確的 CUDA 版本
3. 若無 GPU，程式會自動使用 CPU 模式運行

### Q: 首次執行速度很慢？

首次執行時需從 HuggingFace 下載 CKIP BERT 模型（約 400MB），下載完成後會快取在本機，後續啟動不需重新下載。

### Q: macOS 上沒有 NVIDIA GPU，可以使用嗎？

可以。程式會自動偵測裝置，沒有 NVIDIA GPU 時會使用 CPU 運行。Apple Silicon 的 Mac 也可正常使用。

### Q: 如何更換模型？

在 `app.py` 中修改 `load_models` 函式的 `model` 參數：
- `"bert-base"` — 預設，平衡速度與準確度
- `"bert-tiny"` — 更快但準確度略低
- `"albert-base"` — 較小的模型

---

## 專案結構

```
ckip_segmentation_sumin/
├── app.py               # Gradio 主程式
├── requirements.txt     # Python 套件依賴
└── README.md            # 使用說明（本檔案）
```

---

## 授權

本工具使用 [CKIP Transformers](https://github.com/ckiplab/ckip-transformers)，該套件採用 GPL-3.0 授權。
