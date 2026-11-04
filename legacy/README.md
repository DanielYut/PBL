# 紅藍筆蓋分類夾取（舊專題，legacy/mid.py）

用 Intel RealSense 相機辨識工作區上的紅色與藍色筆蓋，換算成手臂座標後，控制台達 DRV 手臂（Modbus TCP）把筆蓋依顏色夾到不同的放置區。

> 環境需求與安裝方式請見根目錄的 [README.md](../README.md)。另外需要夾爪或吸盤，由 `send.gripper_ON` / `send.gripper_OFF` 控制。

## 執行

在 `mid_project/` 根目錄下執行：

```bash
python -m legacy.mid
```

> 不要用 `python legacy/mid.py`。這樣 Python 會從 `legacy/` 找模組，找不到 `drv_modbus`，出現 `ModuleNotFoundError`。

1. 程式連上手臂後，手臂先回到 Safe Home 點位。
2. 跳出預覽視窗，畫面上會框出辨識到的筆蓋和換算後的手臂座標。
3. 確認無誤後按 `s` 開始夾取；按 `q` 直接離開。
4. 紅色筆蓋放到紅色放置區，藍色筆蓋放到藍色放置區，全部完成後手臂回到 Safe Home。

## 執行前要確認的設定（都在 mid.py 開頭）

| 設定 | 說明 |
|---|---|
| `ARM_IP`、`ARM_PORT` | 手臂的 IP（預設 `192.168.1.1`）和 port（`502`） |
| `X_HOME` ~ `RZ_HOME` | 初始安全點位與夾取時的姿態角度 |
| `Z_TRAVEL` | 移動時的抬升高度，要高於所有障礙物 |
| `Z_GRASP` | 夾取與放置的高度 |
| `DROP_ZONE_RED`、`DROP_ZONE_BLUE` | 紅、藍筆蓋的放置位置 |
| `pts_pixel`、`pts_arm` | 手眼校正用的四個 ArUco Marker 中心點：影像像素座標與對應的手臂座標，順序為左上、右上、右下、左下 |
| `roi_mask[20:460, 80:600]` | 辨識範圍（在 `detect_red_and_blue_caps` 裡），只看工作區內 |
| HSV 範圍 | 紅、藍顏色的門檻值，光線改變時可能要重調 |

> 相機或工作檯只要移動過，就要重新量 `pts_pixel` 和 `pts_arm`，否則夾取位置會偏掉。

## 原理簡述

1. **手眼校正**：用四個已知點（像素座標 ↔ 手臂座標）以 `cv2.findHomography` 算出轉換矩陣，把畫面上的點換算成手臂的 X、Y。
2. **顏色辨識**：轉成 HSV 後依紅、藍範圍做遮罩，經過形態學去雜訊，取面積大於 400 的輪廓中心當作筆蓋位置。
3. **夾取流程**：移到目標上方 → 下降 → 夾爪 ON → 抬升 → 移到放置區上方 → 下降 → 夾爪 OFF → 抬升。
