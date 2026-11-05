# 智慧化焊道／塗膠路徑產生（第一組 PBL 專題）

用相機拍攝工件，自動辨識工件邊緣或接縫，產生焊道／塗膠路徑，再交給台達六軸手臂執行。目標是取代人工逐點示教：換工件、換擺放位置時，不必重新教點。

> 目前仍在開發中，程式放在 [main/](main/)。已完成「相機內參校正」與「手眼校正（eye-in-hand）」，使用方式請見[手眼校正流程](#手眼校正流程eye-in-hand)。舊的紅藍筆蓋夾取專案說明請見 [legacy/README.md](legacy/README.md)。

## 資料夾內容

```
mid_project/
├── main/
│   ├── config.py               # 共用設定（相機、校正目標、手臂 IP、暫存器），到實驗室先改這裡
│   ├── 01_intrinsic_calib.py   # 步驟 1：相機內參校正
│   ├── 03_handeye_solve.py     # 步驟 3：計算夾爪到相機的轉換 T_gripper_cam
│   ├── 04_verify.py            # 步驟 4：驗證手眼校正（即時顯示標記在基座座標的位置）
│   ├── 05_pen_touch.py         # 步驟 5：筆尖校正與筆尖對點驗證
│   ├── aruco_target.py         # ArUco 標記偵測與位姿估計
│   ├── rs_camera.py            # RealSense 影像串流（彩色／IR 自動切換）
│   ├── robot_modbus.py         # 用 Modbus TCP 讀取手臂位姿
│   └── data/
│       ├── intrinsics.npz          # 自行校正的內參
│       ├── factory_intrinsics.npz  # RealSense 出廠內參
│       └── T_gripper_cam.npz       # 手眼校正結果
├── legacy/
│   ├── mid.py          # 舊專題：紅藍筆蓋分類夾取
│   └── README.md       # 舊專題說明
├── drv_modbus/
│   ├── send.py         # 手臂控制：Go_Position、gripper_ON、gripper_OFF
│   └── request.py      # 讀取手臂狀態（send.py 會用到）
├── requirements.txt
└── README.md
```

## 專題目標

- 利用相機影像量測產生路徑，提高精確度、降低誤差。
- 工件放在不同位置或旋轉角度時，都能產生對應的正確路徑。

**預期效益**：減少示教時間、讓產品品質一致，適合少量多樣的生產。

## 系統流程

```
相機擷取影像 → 邊緣／接縫辨識 → 像素座標轉手臂座標（手眼校正）
           → 路徑點產生 → IKM 計算與撞機檢查 → 手臂沿路徑作業
```

| 模組 | 項目 | 說明 |
|---|---|---|
| 1 感知模組 | 1-1 相機架設與內參校正 | 固定相機高度，校正鏡頭內參與畸變 |
| | 1-2 手眼校正 | 建立影像座標與手臂座標的轉換 |
| | 1-3 邊緣辨識 | 找出工件輪廓或接縫 |
| 2 路徑規劃模組 | 2-1 路徑點產生 | 沿輪廓取樣出手臂可執行的路徑點 |
| | 2-2 焊接品質 | 評估路徑與作業結果（評估標準待定義） |
| 3 手臂控制 | 3-1 IKM 計算與撞機檢查 | 用逆向運動學檢查奇異點與碰撞 |
| | 3-2 實機操作 | 透過 Modbus TCP 控制手臂執行路徑 |

手臂控制沿用 [drv_modbus/](drv_modbus/)：

- `send.Go_Position(c, x, y, z, rx, ry, rz, speed, mov)`：移動到指定點位，`mov` 可選 `send.MovP`（點對點）或 `send.MovL`（直線，適合走焊道）。
- `request.Get_TCP_Pose(c)`：讀取手臂目前的 TCP 座標。

## 環境與執行

需要 Python 3.8 ~ 3.12（pyrealsense2 不一定支援更新的版本）、Intel RealSense 相機（USB 3.0），以及和電腦在同一個網段的台達 DRV 手臂。

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

> pymodbus 必須是 3.6.x。新版（3.7 以後）拿掉了 `drv_modbus` 用到的 `pymodbus.payload`，也改了函式參數寫法，裝新版會直接出錯。

手眼校正的程式要先進到 `main/` 資料夾再執行（程式裡的 `data/` 路徑是相對於 `main/`）：

```bash
cd main
python 01_intrinsic_calib.py
```

## 手眼校正流程（eye-in-hand）

相機裝在手臂夾爪上，校正目標是桌上的四個 ArUco 標記（也可以在 `config.py` 改成棋盤格）。開始前先檢查 [main/config.py](main/config.py)：

- `ROBOT_IP`、`ROBOT_UNIT_ID`：手臂控制器的 IP 和 Unit ID。
- `ARUCO_SIZE_MM`：用尺量黑色正方形外框的邊長，不含外面的白邊。
- `COLOR_WIDTH`／`COLOR_HEIGHT`：內參校正和手眼校正必須用同一個解析度。
- `CAMERA_STREAM`：彩色和 IR 不能混用，換串流就要重做內參和手眼校正。

| 步驟 | 指令 | 說明 |
|---|---|---|
| 1 內參校正 | `python 01_intrinsic_calib.py` | 拍至少 15 張棋盤格照片後按 `c` 計算。沒有棋盤格時可加 `--factory`，直接用 RealSense 出廠內參 |
| 3 手眼校正 | `python 03_handeye_solve.py` | 讀取 `data/handeye/` 裡的照片和 `poses.csv`（每張照片拍攝時的手臂位姿），算出夾爪到相機的 offset，存成 `data/T_gripper_cam.npz`。至少要 6 組樣本 |
| 4 驗證 | `python 04_verify.py` | 用教導器移動手臂，標記在基座座標的位置應該幾乎不變（散佈 < 2~3 mm 才算準） |
| 5 筆尖對點 | `python 05_pen_touch.py --tool`，再執行 `python 05_pen_touch.py` | 先做筆尖校正，再讓筆尖逐步下降去碰標記中心，確認實際誤差 |

> 步驟 4 按 `v`、步驟 5 按 `m`/`n`/`j`/`t` 時手臂會移動（速度 10%）。操作時手要放在教導器的緊急停止鈕旁邊，第一次對點先停在 `j`（離標記 2 mm）確認。
>
> 第 2 步「收集手眼校正資料」目前沒有程式，`data/handeye/` 裡的照片和 `poses.csv` 要先準備好。

## 團隊分工

| 工作項目 | 楊程軒 | 郭樺甄 | 劉昱甫 | 羅守倫 | 余泰澤 |
|---|:-:|:-:|:-:|:-:|:-:|
| 相機架設與內參校正 | **P** | A | C | S | A |
| 手眼校正 | A | **P** | A | C | S |
| 邊緣辨識 | C | A | **P** | A | S |
| 路徑點產生 | S | C | A | **P** | A |
| 焊接品質 | A | S | C | A | **P** |
| IKM 計算與撞機檢查 | C | S | A | **P** | A |

P：全責　A：協助、參與　C：諮詢　S：確認核可

## 時程

| 週次（起始日） | 工作項目 |
|---|---|
| 11/4 | 相機架設與內參校正、手眼校正 |
| 11/11 | 相機架設與內參校正、手眼校正、邊緣辨識 |
| 11/18 | 邊緣辨識 |
| 11/25 | 專題初型報告 ▲查核點、路徑點產生 |
| 12/2 | 路徑點產生、焊接品質 |
| 12/9 | 焊接品質、IKM 計算與撞機檢查 |
| 12/16 | IKM 計算與撞機檢查 |
| 12/23 | 期末展示 ▲查核點 |

## 風險與對策

| 風險 | 影響 | 對策 |
|---|---|---|
| 奇異點無解 | 手臂無法依規劃路徑作業 | 限制工作範圍，確保手臂能到達所需姿態 |
| 路徑生成誤差或缺漏 | 路徑偏差或缺少路徑點 | 固定相機高度、確保光線充足，並確認相機與手臂座標轉換一致 |
| 手臂與工件碰撞 | 作業中撞到工件 | 將加工分為四段，每完成一段就回到安全點 |

## 待確認事項

- 「焊接品質」的評估標準（例如路徑與實際邊緣的誤差上限）。
- 工件範圍：先以圓柱（不同直徑、相同高度）與平板工件為主。
