"""
沿邊緣塗膠 (用麥克筆模擬膠槍)

路徑 (PATH_MODE)：
  vertex：找出工件頂點，頂點之間拉直線 (不是多邊形時自動改用 edge)
  edge  ：偵測到的邊緣每個像素都用上 → 平滑 → 沿邊緣每隔 GLUE_STEP_MM 取一點，曲線照實際形狀走
執行 (USE_RL_PATH)：整條路徑交給控制器的 RL 程式 (rl/pbl_path.lua，由 rl_path.py 送進去) 用
  MovL + PASS 連續走完，不會每段停；RL 失敗時改用 Modbus 逐段 MovL (每段會停)

流程：
  [GRIP]    開始時不打開夾爪，筆要先夾好 (程式只送一次夾緊)
  [ARUCO]   Enter：筆尖依序到 4 個角的 Marker，對準後按 Enter，
                   自動算出筆尖偏移並存檔 (data/aruco_xy_calib.json，下次自動載入)
            c    ：換了 Marker / 紙板 / 相機 → 完整重新校正：重新偵測 4 個 Marker，
                   示教器讓筆尖對準每個中心，重建像素→手臂 XY 對應 (data/aruco_xy_calib.json)
            ※ 只校正 XY。Z 一律用程式裡設定的 Z_PAPER / Z_DRAW
  [HOME]    回拍照位置
  [DETECT]  預覽畫面：偵測白紙上的物體，算出邊緣路徑 (藍線)。畫面已轉 180° 跟實際方向一致
              a: 重新做 ArUco 定位
              n: 換下一個物體   s: 鎖定路徑
  [CONFIRM] 終端機印出路徑摘要 → 終端機回答 y 開始塗膠 / n 取消
  [GLUE]    到第 1 點上方 → 下筆 → 沿邊緣一小段一小段直線走一圈 (回到起點再多走 GLUE_OVERLAP_MM) → 抬筆
  [HOME]    回拍照位置，回到 DETECT
  結束時 (q) 問你要不要鬆開夾爪

其他按鍵 (所有按鍵都在「終端機」按，影像視窗只負責顯示)：
  b : 拍底圖 (白紙清空時)，灰色/銀色物體需要底圖才抓得到
  t : 用 i/j/k/l 移動十字選位置，Enter 後在那裡點一個點 (用來檢查/校正偏移量)

執行 (在 mid_project 資料夾)：python main/glue_edge.py

⚠️ 第一次執行前要確認的值 (都在「1. 設定」)：
  Z_DRAW          筆尖碰到要畫的表面時的 Z (預設 None = 紙面 Z_PAPER + 物體高度，自動算)
  Z_PAPER         筆尖碰到「板面」時的 Z (用示教器量一次，板子不動就不用改)
  DRAW_SURFACE    "top" = 畫在工件頂面；"board" = 畫在板面沿工件外圍
  工件高度        AUTO_HEIGHT = True 時用深度自動量 (預覽畫面下方顯示，按 s 鎖定時連量取中位數)
  TOOL_OFFSET_X/Y 筆尖相對夾爪中心的偏移 (先用 t 點一個點檢查)
"""
import collections
import json
import math
import os
import struct
import sys
import time
import warnings

import cv2
import numpy as np
import pyrealsense2 as rs
from pymodbus.client import ModbusTcpClient

# ==========================================
# 1. 設定
# ==========================================
ARM_IP = "192.168.1.1"
ARM_PORT = 502
UNIT_ID = 2

# 拍照位置 (Home)：pts_pixel 是在這裡量的，改了要重量
X_HOME, Y_HOME, Z_HOME = 357.206, -29.763, 729.799
RX_HOME, RY_HOME, RZ_HOME = -179.260, 1.185, -105.234

# --- 高度 (TCP 的 Z) ---
Z_DRAW = None           # 筆尖剛好碰到要畫的表面時的 Z。None = 自動用 Z_PAPER + OBJ_HEIGHT_MM
                        # (想直接指定就填數字，例如用示教器量到的物體頂面 Z)
PEN_PRESS_MM = -2       # 筆尖剛好碰到表面，不再往下壓 (要壓的話填正數 mm)
Z_LIFT = 15.0           # 抬筆高度 (畫之前/之後停在表面上方幾 mm)
Z_TRAVEL = 560.000      # 從 Home 過去 / 回來的安全高度
Z_MIN = 395.0           # 安全下限，低於這個高度不准移動

SPEED_FAST = 40
SPEED_DRAW = 70         # 畫線速度 (%)

# --- 畫線路徑 ---
OBJ_HEIGHT_MM = 0.0     # 沒有深度 (AUTO_HEIGHT 關掉 / 深度量不到) 時用的工件高度 mm

# --- 自動量工件高度 (RealSense 深度) ---
# 高度 = 工件周圍板面的深度 - 工件頂面的深度 (相對量，板面的絕對 Z 還是用 Z_PAPER)
AUTO_HEIGHT = True          # True = 用深度自動量高度；False = 用上面的 OBJ_HEIGHT_MM
DRAW_SURFACE = "top"        # "top"  : 畫在工件頂面   (Z = Z_PAPER + 量到的高度)
                            # "board": 畫在板面沿工件外圍 (Z = Z_PAPER，高度只拿來修正視差)
HEIGHT_INNER_ERODE_PX = 8   # 工件內部往內縮幾 px 再量 (邊緣的深度不準)
HEIGHT_RING_PX = (10, 30)   # 工件外圍第幾 px 到第幾 px 的一圈當作板面
HEIGHT_MAX_MM = 150.0       # 量到超過這個高度 → 當作量錯
SIDE_WALL_FIX = True        # 工件不在相機正下方時，輪廓會多包到「側面」→ 依高度和每個點的位置算透視，只留頂面
SIDE_WALL_MIN_PX = 1.0      # 最寬的側面小於幾 px (640x480 基準) 就不處理 (工件在正下方附近)
SIDE_WALL_DEPTH_RATIO = 0.2  # 要拿掉的那條要比頂面深「高度 × 這個比例」以上才算側面，否則不動
DRAW_OFFSET_PX = 0.0    # 路徑往外 (+) / 往內 (-) 偏幾 px (在影像上沿邊緣法線移，每個邊都一樣)。-2 = 每邊內縮 2 px；0 = 不內縮
DRAW_OFFSET_MM = 0.0    # 另外再偏幾 mm (在手臂座標沿法線移)，跟上面的 px 會相加。0 = 不用

# --- 路徑方式 ---
PATH_MODE = "auto"          # "auto"  : 自動 (建議)。有深度 → 俯視高度圖 (下面「3D 外框」)：
                            #             形狀：多邊形 → 尖角頂點；不是 → 沿曲線
                            #             高度：外框高低差 < FLAT_TOL_MM → 平面 (整條同一個 Z)；否則每點自己的 Z
                            #           深度量不到 → 自動改用下面的 2D 做法 (單一高度)
                            # "vertex": 2D：偵測頂點，頂點之間拉直線，整條同一個 Z (不是多邊形 → "edge")
                            # "edge"  : 2D：沿邊緣走，整條同一個 Z
FLAT_TOL_MM = 2.0           # auto：外框高度變化 (5%~95%) 小於這個 → 當平面 (深度雜訊約 1~1.5 mm，不要設太小)
POLY_FIT_TOL_MM = 1.2       # auto：外框 95% 的點離多邊形要在幾 mm 內才算多邊形 (圓會被當成多邊形 → 調小)
VERTEX_EPS_PX = 4.0         # 找頂點時輪廓簡化的容許誤差 (px)：越大頂點越少
VERTEX_MIN_TURN_DEG = 25.0  # 轉彎小於幾度的點不算頂點 (幾乎直線)
VERTEX_MIN_SIDE_PX = 12.0   # 兩個頂點距離小於幾 px → 合併 (尖角被裁鈍會變成兩個很近的點)
VERTEX_TRIM_RATIO = 0.15   # 擬合每條邊時，頭尾各去掉多少比例的點 (轉角在影像上是圓的)
VERTEX_FIT_TOL_PX = 3.0     # 輪廓 95% 的點離多邊形要在幾 px 內，否則不是多邊形 → 改用 edge (邊緣粗糙就調大)
VERTEX_MAX_SHIFT_PX = 15.0  # 擬合出的頂點離原本的角超過幾 px → 擬合失敗 → 改用 edge
VERTEX_MAX = 12             # 頂點超過幾個 → 當作曲線 → 改用 edge

# --- 3D 外框 (PATH_MODE = "auto" 有深度時) ---
# 做法：深度 → 每個像素的 3D 點 (用自己的高度做視差修正) → 投影到手臂 XY 的方格，每格取最高
#       = 「正上方看下去的高度圖」(沒有透視、側面自動疊在外框上)。外輪廓 = 工件外框，Z = 外框附近最高點
RIM_WIDTH_MM = 0.0          # 往內縮多少再描：0 = 描外緣 (實心工件，例如楔形塊)；
                            # 空心工件描邊框中線 → 填邊框寬度 (例如零件盒 4.0，會往內縮一半)
RIM_GRID_MM = 1.0           # 高度圖方格大小 (mm)
RIM_NEIGH_MM = 1.0          # 外框每一點取「附近幾 mm 內最高」當 Z (太大 → 斜面低邊會量太高；合成測試 1.0 偏高約 0.8 mm)
RIM_MIN_H_MM = 2.0          # 比板面高幾 mm 以上才算工件 (濾掉板面雜訊；太低的薄邊會被切掉)
RIM_ERODE_PX = 1.0          # 工件遮罩先往內縮幾 px 再取深度 (640x480 基準；剪影邊緣的深度會混到板面)
RIM_SMOOTH_MM = 6.0         # 高度沿路徑做中位數平滑的範圍 (mm)，去掉單點跳動
RIM_XY_SMOOTH_MM = 3.0      # 外框 XY 沿路徑平均的範圍 (mm)，去掉方格鋸齒
RIM_EDGE_COMP_MM = None     # 外框往外補幾 mm (None = 自動：遮罩內縮量 + RIM_EDGE_CELL_FRAC 格)
RIM_EDGE_CELL_FRAC = 0.0    # 自動補償再加幾格 (合成測試：0 最準，楔形塊 89.5x50.5 / 真 90x50、圓 D60.1 / 真 60)
RIM_STEP_MM = 2.0           # 沿邊框每隔幾 mm 取一點，再合併成最少段數
RIM_MERGE_TOL_MM = 1.0      # 3D 合併容許誤差 (mm)：直的 (含斜直線) 合成一段，只在轉折處切段
RIM_MIN_VALID = 0.6         # 有量到深度的點至少要佔幾成，否則不畫 (反光 / 黑色)
DRY_RUN_LIFT_MM = 20.0      # 空中試跑：整條路徑抬高幾 mm

# --- 塗膠路徑 (沿邊緣，PATH_MODE = "edge" 或不是多邊形時) ---
GLUE_STEP_MM = 3.0      # 沿邊緣每隔幾 mm 取一點 (每一點 = 一小段 MovL)。越小越貼邊、越多停頓
SMOOTH_PX = 7           # 邊緣平滑：前後幾個像素取平均，去掉像素鋸齒 (1 = 不平滑；要奇數)
MASK_BLUR_PX = 2.0      # 取邊緣前先把工件遮罩模糊 (σ px) 再二值化，磨掉階梯 / 小凸起 (0 = 不做；越大尖角越圓)
LOCK_FRAMES = 15        # 按 s 鎖定時連拍幾張：遮罩平均 (去邊緣雜訊) + 高度取中位數
USE_RL_PATH = True      # True = 塗膠交給控制器的 RL 程式 (rl/pbl_path.lua) 用 MovL + PASS 連續走，不會每段停
RL_SPEED_TRAVEL = 30    # RL 模式移動速度 (%)。注意：RL 的 % 跟 Modbus GO 的 % 不一定一樣 (原本 10)
RL_SPEED_GLUE = 15      # RL 模式塗膠速度 (%)。越快 DEC 模式轉角切得越內側 (原本 5)
RL_OVERLAP_MM = 3.0     # 曲線 (edge 模式) 每個點提早幾 mm 轉彎，連成平順曲線
RL_CORNER_OVERLAP_MM = 3.0  # 多邊形 (vertex 模式) 的轉角：0 = 尖角 (到角才轉)，>= 1 = 轉角磨圓不停頓
GLUE_OVERLAP_MM = 5.0   # 走完一圈回到起點後再多走幾 mm，讓膠線頭尾接起來 (0 = 不重疊)
SPEED_GLUE = SPEED_DRAW # 塗膠速度 (%)
GLUE_PAUSE_S = 0.0      # 每段走到點之後額外停幾秒 (0 = 一到位馬上走下一段，最流暢)
GLUE_EARLY_MM = 1.0     # 提前送點：離這段終點剩幾 mm 就先送下一段 (控制器接受 → 不停直接轉過去；
                        # 控制器忽略 → 程式偵測到會補送，不會卡)。0 = 關閉 (每段都等到位)
                        # 轉角會被切掉最多約這麼多 mm，越大越順、越不貼角
GLUE_MERGE_TOL_MM = 0.5 # 路徑跟邊緣最多差幾 mm。用最少段數走完：直邊 = 一整段，只有彎的地方切段
                        # 越大段數越少、越順 (停頓少)，越小越貼邊。0 = 不合併，改成每 GLUE_STEP_MM 一段

# --- 工具偏移 (筆尖相對夾爪中心) ---
# ArUco 定位時微調對準並按 Enter 記錄，程式會自動算出偏移存到 data/aruco_xy_calib.json，
# 下次啟動自動載入 (會蓋掉下面兩個值)
TOOL_OFFSET_X = 0.0
TOOL_OFFSET_Y = 0.0

# --- 顯示 ---
DISPLAY_SCALE = 0.5         # 監控視窗縮放 (1280x720 → 640x360)；只影響顯示，偵測 / 計算都用原解析度
DISPLAY_ROTATE_180 = True   # 相機畫面跟實際看到的上下左右相反 → 轉 180° 顯示 (只影響顯示，不影響計算)

# --- ArUco 定位 (畫之前先讓筆尖到 4 個角的 Marker 中心，確認位置準不準) ---
Z_PAPER = 440.04# 筆尖碰到「白板板面」時的 Z (2026-11-16 示教器實測；換板子要重量)
                        # ※ 高度只用這裡 (或 Z_DRAW) 的值，校正不會改 Z
ARUCO_HOVER_MM = 10.0   # 停在 Marker 上方幾 mm
ARUCO_DICT = cv2.aruco.DICT_5X5_50
ARUCO_MATCH_PX = 40     # 偵測到的 Marker 跟記錄位置差多少 px 內算同一個
TEACH_MAX_ABOVE_MM = 30.0   # 教點按 Enter 時筆尖最多比板面高幾 mm，超過 = 還沒移過去，不記錄
TEACH_WITH_PENDANT = True   # True = 用示教器教點 (程式只讀座標，不動手臂)；False = 程式移過去，用鍵盤微調

# 工作範圍限制 (程式自己擋)。None = 不限制，由手臂控制器判斷 (奇異點/到不了時控制器會報警、不執行)
# 想加保護時填 (最小, 最大)，例如 X_RANGE = (360.0, 620.0)
X_RANGE = None
Y_RANGE = None
JOG_TIMEOUT = 8.0       # 微調時超過幾秒沒到位 → 視為到不了 (奇異點/超出可達範圍)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BG_PATH = os.path.join(DATA_DIR, "background.png")

# ==========================================
# 2. 手臂控制 (Modbus TCP)
# ==========================================
REG_POSE, REG_DONE = 0x00F0, 0x031F
REG_SPEED, REG_TARGET, REG_CMD, REG_IO = 0x0324, 0x0330, 0x0300, 0x02FE
CMD_MOVP, CMD_MOVL = 301, 302
GRIP_CLOSE, GRIP_OPEN = 128, 64   # 夾爪 DO (寫到 REG_IO)。這台手臂實測：128 = 夾緊、64 = 打開 (舊手臂相反)


class Arm:
    def __init__(self):
        self.c = ModbusTcpClient(host=ARM_IP, port=ARM_PORT, timeout=2)
        if not self.c.connect():
            raise ConnectionError(f"連不到手臂 {ARM_IP}，請檢查網路設定")
        print("✅ 已連線手臂")

    def _retry(self, fn, what):
        """控制器會把閒置太久的連線關掉 → 失敗時重連再試"""
        last = None
        for _ in range(3):
            try:
                if not self.c.connected:
                    self.c.connect()
                rr = fn()
                if not rr.isError():
                    return rr
                last = rr
            except Exception as e:
                last = e
            self.c.close()
            time.sleep(0.3)
        raise RuntimeError(f"{what} 失敗：{last}")

    def _write(self, addr, values):
        if isinstance(values, list):
            return self._retry(lambda: self.c.write_registers(addr, values, UNIT_ID), f"寫入 0x{addr:04X}")
        return self._retry(lambda: self.c.write_register(addr, values, UNIT_ID), f"寫入 0x{addr:04X}")

    def _read(self, addr, n):
        return self._retry(lambda: self.c.read_holding_registers(addr, n, UNIT_ID),
                           f"讀取 0x{addr:04X}").registers

    def pose(self):
        """回傳 [X, Y, Z, RX, RY, RZ] (mm / 度)"""
        r = self._read(REG_POSE, 12)
        return [struct.unpack("<i", struct.pack("<HH", r[i], r[i + 1]))[0] * 0.001 for i in range(0, 12, 2)]

    def send_move(self, x, y, z, rx=RX_HOME, ry=RY_HOME, rz=RZ_HOME, speed=SPEED_FAST, linear=False):
        """送出移動指令就回傳，不等到位 (用 is_idle() 查是否到了)"""
        if z < Z_MIN:
            raise ValueError(f"Z={z:.2f} 低於安全下限 Z_MIN={Z_MIN}")
        regs = []
        for v in (x, y, z, rx, ry, rz):
            raw = int(round(v * 1000)) & 0xFFFFFFFF
            regs += [raw & 0xFFFF, raw >> 16]
        if getattr(self, "speed", None) != int(speed):      # 速度沒變就不重寫，少一次通訊
            self._write(REG_SPEED, int(speed))
            self.speed = int(speed)
        self._write(REG_TARGET, regs)
        self._write(REG_CMD, CMD_MOVL if linear else CMD_MOVP)
        self.target = (x, y, z)
        self.t_cmd = time.time()

    def is_idle(self, tol=1.0):
        """到位了嗎：送出後超過 0.2 秒 (舊的到位旗標已清掉) + 旗標=1 + 位置差 < tol mm"""
        if time.time() - getattr(self, "t_cmd", 0) < 0.2:
            return False
        if self._read(REG_DONE, 1)[0] != 1:
            return False
        p = self.pose()
        return max(abs(a - b) for a, b in zip(p[:3], self.target)) < tol

    def wait_idle(self, timeout=60.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.is_idle():
                return
            cb = getattr(self, "on_tick", None)     # 描邊時的即時追蹤畫面 (LiveTrack)
            if cb:
                cb()
            time.sleep(0.03)
        x, y, z = self.target
        raise RuntimeError(f"移動逾時，目標 ({x:.1f}, {y:.1f}, {z:.1f})，請看教導器是否有警報")

    def move(self, x, y, z, rx=RX_HOME, ry=RY_HOME, rz=RZ_HOME, speed=SPEED_FAST, linear=False,
             timeout=60.0):
        """移到指定位置，等到位 (旗標=1 且 位置差 < 1 mm) 才回傳"""
        self.send_move(x, y, z, rx, ry, rz, speed, linear)
        self.wait_idle(timeout)

    def home(self):
        print("🏠 回 Home")
        p = self.pose()
        if p[2] < Z_TRAVEL:
            self.move(p[0], p[1], Z_TRAVEL)      # 先垂直抬起，避免拖著筆撞到東西
        self.move(X_HOME, Y_HOME, Z_HOME)

    def grip(self, close):
        self._write(REG_IO, GRIP_CLOSE if close else GRIP_OPEN)
        time.sleep(1.0)

    def close(self):
        self.c.close()


# ==========================================
# 3. 像素 → 手臂座標 (校正)
# ==========================================
# 4 個 ArUco Marker：Home 拍照時的像素中心 ↔ 「麥克筆尖」碰 Marker 中心時的 TCP X, Y
# (2026-11-05 舊 Marker 示教器實測。已包含筆尖偏移，所以 TOOL_OFFSET 保持 0)
# 順序 = 相機原始畫面 左上, 右上, 右下, 左下 (顯示轉 180° 後是 右下, 左下, 左上, 右上)
# 手臂在 Home 時：相機畫面往右 = 手臂 -Y，往下 = 手臂 -X
# ※ data/aruco_xy_calib.json 存在時 (做過「完整重新校正」c) 會用那裡的值蓋掉下面兩組
PTS_PIXEL = np.float32([[134.61, 91.71], [529.51, 84.99], [538.75, 420.06], [130.86, 420.98]])
PTS_ARM = np.float32([[653.210, 75.842], [647.689, -209.430], [403.074, -211.327], [402.948, 75.895]])
# 舊值 (舊專案留下的，不準)：[[607.466, 62.870], [607.466, -221.951], [374.376, -221.951], [374.376, 58.249]]
H_MATRIX, _ = cv2.findHomography(PTS_PIXEL, PTS_ARM)
CALIB_PATH = os.path.join(DATA_DIR, "aruco_xy_calib.json")   # 新 Marker 的 XY 校正 + 筆尖偏移

# --- 相機解析度 ---
# 改解析度後：校正 (aruco_xy_calib.json 的像素座標) 要重做、底圖要重拍 (程式啟動時會提示)。
# 程式裡以 px 為單位的參數都是用 640x480 調的，開相機時會依焦距 fx 自動等比例放大 (見 apply_px_scale)
CAM_W, CAM_H = 1280, 720         # 彩色
DEPTH_W, DEPTH_H = 848, 480      # 深度 (848x480 是 D435 深度最準的解析度；會對齊到彩色，所以深度圖 = 彩色大小)
FX_REF = 615.0                   # 640x480 彩色的焦距 fx (px)，px 參數就是在這個解析度下調的
PX_SCALE = 1.0                   # 目前 fx / FX_REF，開相機後設定

# 高度視差：物體頂面比紙面高、離相機近，看起來會往外擴。以相機正下方為中心縮回。
CAM_CENTER_PX = (318.7, 246.8)   # 相機光軸在畫面上的位置 (開相機後由內參 ppx, ppy 覆蓋)
CAM_HEIGHT_MM = 445.0            # 相機到紙面距離 (由 Marker 大小估算)


def pixel_to_arm(u, v, height_mm=0.0):
    """像素 → 夾爪中心要去的 X, Y。height_mm = 這個點離紙面多高"""
    if height_mm:
        k = 1.0 - height_mm / CAM_HEIGHT_MM
        u = CAM_CENTER_PX[0] + (u - CAM_CENTER_PX[0]) * k
        v = CAM_CENTER_PX[1] + (v - CAM_CENTER_PX[1]) * k
    p = H_MATRIX @ np.array([u, v, 1.0])
    return p[0] / p[2] + TOOL_OFFSET_X, p[1] / p[2] + TOOL_OFFSET_Y


def pixels_to_arm(u, v, h):
    """pixel_to_arm 的陣列版：u, v, h 都是陣列 (h = 每點離紙面多高 mm)，回傳 (X, Y) 陣列"""
    k = 1.0 - np.asarray(h, float) / CAM_HEIGHT_MM
    uu = CAM_CENTER_PX[0] + (np.asarray(u, float) - CAM_CENTER_PX[0]) * k
    vv = CAM_CENTER_PX[1] + (np.asarray(v, float) - CAM_CENTER_PX[1]) * k
    H = H_MATRIX
    w = H[2, 0] * uu + H[2, 1] * vv + H[2, 2]
    x = (H[0, 0] * uu + H[0, 1] * vv + H[0, 2]) / w + TOOL_OFFSET_X
    y = (H[1, 0] * uu + H[1, 1] * vv + H[1, 2]) / w + TOOL_OFFSET_Y
    return x, y


def arm_to_pixel(x, y, h=0.0):
    """手臂 X, Y (高度 h) → 相機像素 (pixel_to_arm 的反函數，畫面顯示用)"""
    p = np.linalg.inv(H_MATRIX) @ np.array([x - TOOL_OFFSET_X, y - TOOL_OFFSET_Y, 1.0])
    u, v = p[0] / p[2], p[1] / p[2]
    k = 1.0 - h / CAM_HEIGHT_MM
    return CAM_CENTER_PX[0] + (u - CAM_CENTER_PX[0]) / k, CAM_CENTER_PX[1] + (v - CAM_CENTER_PX[1]) / k


def side_scales(pts_pixel, pts_arm):
    """4 個邊各自的 mm/px"""
    return [float(np.linalg.norm(pts_arm[i] - pts_arm[(i + 1) % 4]) /
                  np.linalg.norm(pts_pixel[i] - pts_pixel[(i + 1) % 4])) for i in range(4)]


MM_PER_PX = float(np.mean(side_scales(PTS_PIXEL, PTS_ARM)))


def set_calibration(pts_pixel, pts_arm):
    global PTS_PIXEL, PTS_ARM, H_MATRIX, MM_PER_PX
    PTS_PIXEL, PTS_ARM = np.float32(pts_pixel), np.float32(pts_arm)
    H_MATRIX, _ = cv2.findHomography(PTS_PIXEL, PTS_ARM)
    MM_PER_PX = float(np.mean(side_scales(PTS_PIXEL, PTS_ARM)))


def h_scale(H, u=None, v=None):
    """單應矩陣在 (u, v) (預設畫面中央) 附近每 px 幾 mm (x、y 方向平均)"""
    if u is None:
        u, v = CAM_CENTER_PX

    def f(a, b):
        p = H @ np.array([a, b, 1.0])
        return p[:2] / p[2]
    p0 = f(u, v)
    return float((np.linalg.norm(f(u + 1, v) - p0) + np.linalg.norm(f(u, v + 1) - p0)) / 2)


def use_h_matrix(H):
    """直接指定 像素 → 手臂 的單應矩陣 (自主校正的結果)；ArUco 的 PTS_PIXEL / PTS_ARM 不動 (a 檢查偏移還要用)"""
    global H_MATRIX, MM_PER_PX
    H_MATRIX = np.array(H, float)
    MM_PER_PX = h_scale(H_MATRIX)


CALIB_SIZE = (640, 480)   # 校正 (PTS_PIXEL) 是在哪個解析度做的；跟相機不一樣 → 像素座標要重抓


def load_calibration():
    """讀 data/aruco_xy_calib.json：4 點 XY 對應 + 筆尖偏移 (Z 不從檔案讀，用程式裡的 Z_PAPER / Z_DRAW)"""
    global ARUCO_DICT
    if not os.path.exists(CALIB_PATH):
        print("📐 Marker 對應用程式裡寫死的值 (還沒做過完整重新校正)")
        return
    global TOOL_OFFSET_X, TOOL_OFFSET_Y, CALIB_SIZE
    with open(CALIB_PATH, encoding="utf-8") as fp:
        d = json.load(fp)
    CALIB_SIZE = tuple(d.get("image_size", (640, 480)))   # 舊檔沒記錄 = 當時都是 640x480
    if d.get("pts_pixel") and d.get("pts_arm"):
        set_calibration(d["pts_pixel"], d["pts_arm"])
        print(f"📐 已載入 Marker XY 對應 {CALIB_PATH} ({d['saved_at']}, {d.get('aruco_dict', '')})")
    if d.get("h_source") == "dots" and d.get("h_matrix"):
        use_h_matrix(d["h_matrix"])                    # 自主校正 (畫點) 的結果優先
        dc = d.get("dotcal", {})
        print(f"📐 使用自主校正的 像素→手臂 對應 ({dc.get('n', '?')} 點，平均誤差 {dc.get('mean_mm', 0):.2f} mm，"
              f"{dc.get('saved_at', d.get('saved_at', ''))})")
    if d.get("aruco_dict"):
        ARUCO_DICT = getattr(cv2.aruco, d["aruco_dict"])
    TOOL_OFFSET_X, TOOL_OFFSET_Y = d.get("tool_offset_x", 0.0), d.get("tool_offset_y", 0.0)
    print(f"📐 筆尖偏移 X={TOOL_OFFSET_X:+.2f}, Y={TOOL_OFFSET_Y:+.2f} mm")


# ==========================================
# 4. 物體偵測 (白紙上跟底圖不同的東西)
# ==========================================
MIN_AREA, MAX_AREA = 400, 60000   # 640x480 基準 px²；60000 ≈ 236x236 mm (1280x720 時自動 ×4)，比整張 A4 小
LAST_REJECTED = None              # 上一次被面積濾掉的最大物體 (面積 px², "big"/"small")，畫面提示用
PAPER_S_MAX, PAPER_MARGIN = 40, 15
OBJ_S_DIFF, OBJ_V_DIFF = 35, 90                                # 顏色模式 (沒有底圖)
BG_CHROMA_DIFF, BG_BRIGHT_DIFF, BG_DARK_DIFF = 12, 25, 60      # 背景相減模式
EDGE_LOW, EDGE_HIGH = 25, 70


def find_paper_mask(hsv):
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    low_sat = S < PAPER_S_MAX
    if not np.any(low_sat):
        return None
    v_thr, _ = cv2.threshold(V[low_sat].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white = (low_sat & (V > v_thr)).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((kpx(15), kpx(15)), np.uint8))
    cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    mask = np.zeros(hsv.shape[:2], np.uint8)
    cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    k = 2 * PAPER_MARGIN + 1
    return cv2.erode(mask, np.ones((k, k), np.uint8))


def object_mask(img, bg):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    paper = find_paper_mask(hsv if bg is None else cv2.cvtColor(bg, cv2.COLOR_BGR2HSV))
    if paper is None:
        return np.zeros(img.shape[:2], np.uint8)

    if bg is None:   # 顏色模式：有顏色或很暗
        S, V = hsv[:, :, 1], hsv[:, :, 2]
        ps, pv = np.median(S[paper > 0]), np.median(V[paper > 0])
        mask = ((S > ps + OBJ_S_DIFF) | (V < pv - OBJ_V_DIFF)).astype(np.uint8) * 255
    else:            # 背景相減：顏色變、變亮、很暗、或多了清楚邊緣
        gk = (kpx(5), kpx(5))
        a, b = cv2.GaussianBlur(img, gk, 0), cv2.GaussianBlur(bg, gk, 0)
        lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_bg = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)
        dL = lab[:, :, 0] - lab_bg[:, :, 0]
        dC = np.hypot(lab[:, :, 1] - lab_bg[:, :, 1], lab[:, :, 2] - lab_bg[:, :, 2])
        mask = ((dC > BG_CHROMA_DIFF) | (dL > BG_BRIGHT_DIFF) | (dL < -BG_DARK_DIFF)).astype(np.uint8) * 255
        e = cv2.Canny(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), EDGE_LOW, EDGE_HIGH)
        eb = cv2.dilate(cv2.Canny(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), EDGE_LOW, EDGE_HIGH),
                        np.ones((kpx(5), kpx(5)), np.uint8))
        new_e = cv2.morphologyEx(cv2.bitwise_and(e, cv2.bitwise_not(eb)), cv2.MORPH_CLOSE,
                                 np.ones((kpx(7), kpx(7)), np.uint8))
        cnts, _ = cv2.findContours(new_e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, cnts, -1, 255, -1)

    mask &= paper
    if ARUCO_EXCLUDE is not None and ARUCO_EXCLUDE.shape == mask.shape:
        mask[ARUCO_EXCLUDE > 0] = 0          # Marker 不算物體
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((kpx(5), kpx(5)), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kpx(7), kpx(7)), np.uint8))


def detect_objects(img, bg):
    """回傳 (物體輪廓 list (由大到小), 物體遮罩)"""
    global LAST_REJECTED
    mask = object_mask(img, bg)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in cnts]
    keep = [c for c, a in zip(cnts, areas) if MIN_AREA < a < MAX_AREA]
    big = [a for a in areas if a >= MAX_AREA]
    small = [a for a in areas if MIN_AREA * 0.3 < a <= MIN_AREA]
    LAST_REJECTED = (max(big), "big") if big else (max(small), "small") if small and not keep else None
    return sorted(keep, key=cv2.contourArea, reverse=True), mask


def measure_height(cnt, obj_mask, depth):
    """
    工件高度 mm = 外圍一圈板面深度的中位數 - 工件內部深度的中位數。量不到回傳 None
    obj_mask = 所有物體的遮罩 (外圍那圈要扣掉別的物體)
    """
    if depth is None or depth.shape[:2] != obj_mask.shape[:2]:
        return None
    fill = np.zeros(obj_mask.shape[:2], np.uint8)
    cv2.drawContours(fill, [cnt], -1, 255, -1)

    def ell(r):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    inner = cv2.erode(fill, ell(HEIGHT_INNER_ERODE_PX))
    if cv2.countNonZero(inner) < 20 * PX_SCALE ** 2:      # 工件太小 → 少縮一點
        inner = cv2.erode(fill, ell(max(1, int(round(2 * PX_SCALE)))))
    r_in, r_out = HEIGHT_RING_PX
    ring = cv2.dilate(fill, ell(r_out)) & ~cv2.dilate(fill, ell(r_in)) & ~cv2.dilate(obj_mask, ell(r_in))
    if ARUCO_EXCLUDE is not None and ARUCO_EXCLUDE.shape == ring.shape:
        ring &= ~ARUCO_EXCLUDE
    d_in = depth[(inner > 0) & (depth > 0)]
    d_ring = depth[(ring > 0) & (depth > 0)]
    if len(d_in) < 15 or len(d_ring) < 30:
        return None
    h = float(np.median(d_ring) - np.median(d_in))
    if SIDE_WALL_FIX and 1.0 < h <= HEIGHT_MAX_MM:
        # 輪廓裡含側面時 (側面比頂面深)，內部的中位數會偏深 → 拿掉側面後再量一次頂面
        top = top_face(cnt, obj_mask.shape, h, depth)
        if top is not cnt:
            tf = np.zeros(obj_mask.shape[:2], np.uint8)
            cv2.drawContours(tf, [top], -1, 255, -1)
            ti = cv2.erode(tf, ell(HEIGHT_INNER_ERODE_PX))
            d_top = depth[(ti > 0) & (depth > 0)]
            if len(d_top) >= 15:
                h = float(np.median(d_ring) - np.median(d_top))
    if not -5.0 <= h <= HEIGHT_MAX_MM:
        return None
    return max(h, 0.0)


def top_face(cnt, shape, h, depth=None):
    """
    拿掉透視看到的側面，只留工件「頂面」的輪廓。
    工件不在相機正下方時，相機會斜斜看到朝向畫面中心那一側的側面，偵測到的輪廓 = 頂面 + 側面。
    透視：高度 z 的點在畫面上 = 光軸中心 c + (底面位置 - c) × Zc / (Zc - z)，
    所以頂面 T 往中心縮 k = 1 - t·h/Zc (t: 0 → 1) 一路掃到底面，掃過的範圍就是偵測到的輪廓 S。
    反過來：x 屬於頂面 ⇔ 對所有 t，c + (x - c)·k 都還在 S 裡 → 把 S 縮放幾次取交集。
    每個點依自己離光軸的距離算位移 (越靠畫面邊緣側面越寬)，不是整個物體用同一個位移。
    有深度時先確認：要拿掉的那條真的比頂面深 (是側面)；不是就不動，避免側面沒被偵測到時削掉頂面。
    回傳頂面輪廓；沒有修正時回傳原本的 cnt (同一個物件)
    (想法來自組員 feature/detect-tuning 的 remove_side_wall，改成逐點透視縮放)
    """
    if not SIDE_WALL_FIX or h is None or h <= 0.5:
        return cnt
    f = h / CAM_HEIGHT_MM
    cx, cy = CAM_CENTER_PX
    pts = cnt.reshape(-1, 2).astype(np.float32)
    dmax = float(np.max(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy))) * f   # 最寬的側面 (px)
    if dmax < SIDE_WALL_MIN_PX * PX_SCALE:
        return cnt
    # 只處理工件附近的區域 (整張 1280x720 縮放很多次太慢)
    x, y, w, hh = cv2.boundingRect(cnt)
    pad = int(math.ceil(dmax)) + 2
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, shape[1]), min(y + hh + pad, shape[0])
    sil = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.drawContours(sil, [cnt], -1, 255, -1, offset=(-x0, -y0))
    top = sil.copy()
    steps = int(math.ceil(dmax)) + 1                    # 每步位移 < 1 px
    rcx, rcy = cx - x0, cy - y0                         # 光軸中心在 ROI 裡的座標
    for i in range(1, steps + 1):
        k = 1.0 - f * i / steps
        M = np.float32([[k, 0, rcx * (1 - k)], [0, k, rcy * (1 - k)]])
        top &= cv2.warpAffine(sil, M, (sil.shape[1], sil.shape[0]),
                              flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    if depth is not None and depth.shape[:2] == shape[:2]:
        d = depth[y0:y1, x0:x1]
        strip = (sil > 0) & (top == 0)                  # 要拿掉的那條 (應該是側面)
        core = cv2.erode(top, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kpx(7), kpx(7)))) > 0
        ds, dt = d[strip & (d > 0)], d[core & (d > 0)]
        if len(ds) > 30 and len(dt) > 30 and float(np.median(ds) - np.median(dt)) < SIDE_WALL_DEPTH_RATIO * h:
            return cnt                                  # 那條跟頂面一樣高 → 不是側面 (側面沒被偵測到)，不動
    cs, _ = cv2.findContours(top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(x0, y0))
    if not cs:
        return cnt
    best = max(cs, key=cv2.contourArea)
    return best if cv2.contourArea(best) > 0.3 * cv2.contourArea(cnt) else cnt


def imwrite_u(path, img):
    """cv2.imwrite 遇到中文路徑會默默失敗 → 先編碼再用 numpy 寫檔"""
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if ok:
        buf.tofile(path)
    else:
        print(f"⚠️ 存檔失敗 {path}")
    return ok


def imread_u(path):
    """cv2.imread 遇到中文路徑會回傳 None → 用 numpy 讀再解碼"""
    try:
        return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None


def lock_object(pipeline, bg, cnt):
    """
    按 s 鎖定時連拍 LOCK_FRAMES 張，找同一個工件：
      - 邊緣：每張的工件遮罩加起來平均，超過一半的張數都算工件的像素才保留
              → 每張邊界亂跳的 1~3 px 雜訊會抵消，路徑的鋸齒少很多
      - 高度 (有深度時)：每張量一次取中位數
    回傳 (平均後的輪廓 或 None, 高度 或 None, 高度跳動 = 90% - 10% 分位 或 None)
    """
    m = cv2.moments(cnt)
    if not m["m00"]:
        return None, None, None
    global LOCKED_DEPTH
    c0 = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
    acc, n_ok, hs, depths = None, 0, [], []
    for _ in range(LOCK_FRAMES):
        f = grab(pipeline)
        if DEPTH_ON and LAST_DEPTH is not None:
            depths.append(LAST_DEPTH.copy())
        cs, msk = detect_objects(f, bg)
        best = None
        for c in cs:                                   # 找同一個工件 (中心最近的)
            mm = cv2.moments(c)
            if mm["m00"]:
                dd = float(np.hypot(*(np.array([mm["m10"] / mm["m00"], mm["m01"] / mm["m00"]]) - c0)))
                if dd < 20 * PX_SCALE and (best is None or dd < best[0]):
                    best = (dd, c, msk)
        if not best:
            continue
        fill = np.zeros(f.shape[:2], np.uint8)
        cv2.drawContours(fill, [best[1]], -1, 1, -1)
        acc = fill.astype(np.float32) if acc is None else acc + fill
        n_ok += 1
        if DEPTH_ON:
            h = measure_height(best[1], best[2], LAST_DEPTH)
            if h is not None:
                hs.append(h)

    LOCKED_DEPTH = None
    if len(depths) >= 3:                               # 深度逐像素取中位數 (0 = 量不到，不算)
        st = np.stack(depths)
        st[st <= 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # 某些像素每張都量不到 → 全 nan
            med = np.nanmedian(st, axis=0)
        LOCKED_DEPTH = np.nan_to_num(med, nan=0.0).astype(np.float32)

    avg_cnt = None
    if n_ok >= max(3, LOCK_FRAMES // 3):
        avg = ((acc / n_ok) >= 0.5).astype(np.uint8) * 255
        cs, _ = cv2.findContours(avg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cs:
            avg_cnt = max(cs, key=cv2.contourArea)
    if len(hs) < max(3, LOCK_FRAMES // 3):
        return avg_cnt, None, None
    return avg_cnt, float(np.median(hs)), float(np.percentile(hs, 90) - np.percentile(hs, 10))


DEBUG_DIR = os.path.join(DATA_DIR, "debug")


def save_debug(frame, mask, cnts, sel, path_px, bg):
    """p：存下目前畫面 / 物體遮罩 / 疊圖 / 深度，用來檢查偵測問題"""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(DEBUG_DIR, tag)
    imwrite_u(base + "_frame.png", frame)
    imwrite_u(base + "_mask.png", mask)
    ov = frame.copy()
    for i, c in enumerate(cnts):
        cv2.drawContours(ov, [c], -1, (0, 255, 255) if i != sel else (0, 0, 255), 1)
    if len(path_px) > 1:
        cv2.polylines(ov, [np.int32(path_px)], False, (255, 0, 0), 1)
    imwrite_u(base + "_overlay.png", ov)
    if bg is not None:
        imwrite_u(base + "_bg.png", bg)
    if LAST_DEPTH is not None:
        np.save(base + "_depth.npy", LAST_DEPTH)
    print(f"💾 已存偵測快照 {base}_*.png  (模式：{'背景相減' if bg is not None else '顏色'})")
    if cnts:
        x, y, w, h = cv2.boundingRect(cnts[sel])
        print(f"   選中物體外框 {w} x {h} px ≈ {w * MM_PER_PX:.1f} x {h * MM_PER_PX:.1f} mm (板面高度換算)，"
              f"高度 {obj_height():.1f} mm")


def load_background():
    if os.path.exists(BG_PATH):
        bg = imread_u(BG_PATH)
        if bg is not None and (bg.shape[1], bg.shape[0]) != CALIB_SIZE_NOW:
            print(f"⚠️ 底圖是 {bg.shape[1]}x{bg.shape[0]}，相機現在是 {CALIB_SIZE_NOW[0]}x{CALIB_SIZE_NOW[1]}："
                  "不能用，先用顏色模式。板子清空後按 b 重拍")
            return None
        if bg is not None:
            print(f"📂 已載入底圖 {BG_PATH}")
            return bg
    print("⚠️ 沒有底圖，使用顏色模式。白紙清空後按 b 拍底圖")
    return None


BG_SHIFT_WARN_PX = 3.0     # 底圖跟目前畫面的 Marker 位置差超過幾 px → 提醒重拍底圖 (相機 / 白板動過)


def aruco_centers(img):
    """單張影像裡的 Marker 中心 {id: (u, v)}"""
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=cv2.aruco.DetectorParameters_create())
    if ids is None:
        return {}
    return {int(i): tuple(c.reshape(4, 2).mean(axis=0)) for i, c in zip(ids.flatten(), corners)}


def check_background(bg, frame):
    """底圖跟目前畫面比 Marker 位置：位移太大 = 底圖過期 (相機 / 白板 / 拍照位置動過)，印出警告"""
    if bg is None:
        return
    a, b = aruco_centers(bg), aruco_centers(frame)
    common = set(a) & set(b)
    if not common:
        print("⚠️ 底圖或目前畫面找不到 Marker，沒辦法確認底圖有沒有過期")
        return
    shift = max(math.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1]) for i in common)
    if shift > BG_SHIFT_WARN_PX:
        print(f"⚠️ 底圖可能過期：Marker 跟拍底圖時差了 {shift:.1f} px (相機或白板動過)。"
              f"板子清空後按 b 重拍，不然邊緣會被誤判成物體")
    else:
        print(f"  底圖位置檢查 OK (Marker 位移 {shift:.1f} px)")


def capture_background(pipeline, old_bg=None, n=15):
    acc = sum(grab(pipeline).astype(np.float32) for _ in range(n))
    bg = (acc / n).astype(np.uint8)
    # 防呆：用顏色模式看板子上還有沒有東西 (底圖拍到物體 → 之後那個位置永遠被當成物體)
    left, _ = detect_objects(bg, None)
    if left:
        areas = ", ".join(f"{cv2.contourArea(c):.0f}" for c in left[:5])
        print(f"⚠️ 板子上好像還有 {len(left)} 個東西 (面積 px²：{areas})，底圖要拍「空的板子」")
        if ask("  還是要用這張當底圖嗎？ y: 要   Enter: 不要 (先把東西拿開再按 b) > ", ("", "y"),
               {"y": "Use anyway (y)", "": "Cancel (Enter)"}) != "y":
            print("  沒有存，保留原本的底圖")
            return old_bg
    os.makedirs(os.path.dirname(BG_PATH), exist_ok=True)
    imwrite_u(BG_PATH, bg)
    print(f"📷 已拍底圖 {BG_PATH}")
    return bg


# ==========================================
# 5. 邊緣 → 畫線路徑
# ==========================================
LAST_PATH_MODE = ""   # 上一次 make_path 實際的形狀 ("vertex" 多邊形 / "edge" 曲線)，畫面顯示 + RL 轉角設定用
LOCKED_DEPTH = None   # 按 s 鎖定時連拍的深度中位數 (比單張穩)；算完路徑就清掉，預覽時用 LAST_DEPTH
LAST_HMAP = None      # 上一次的俯視高度圖 (heightmap_outline)，2-1 幾何量測用
LAST_RIM = {}         # 上一次 rim 路徑的資訊 (高度範圍、有效比例…)，印出 / 顯示用


def make_path(cnt, shape):
    """
    物體邊緣 → (路徑像素點, 路徑手臂點)
    PATH_MODE = "vertex"：偵測頂點，頂點之間拉直線 (不是多邊形 → 自動改用沿邊緣)
    PATH_MODE = "edge"  ：沿邊緣走
    PATH_MODE = "auto"  ：有深度 → 俯視高度圖，手臂點是 (x, y, 表面高度 mm)；量不到 → 下面的 2D 做法
    2D 的手臂點是 (x, y)，Z 用 draw_z()
    """
    global LAST_PATH_MODE
    global LAST_RIM
    if PATH_MODE == "auto" and DEPTH_ON:
        r = make_path_3d(cnt, shape)
        if r is not None:
            return r                                    # LAST_PATH_MODE 在 make_path_3d 裡設定
    elif PATH_MODE == "auto":
        LAST_RIM = {"error": "深度沒開"}
    # 拿掉透視看到的側面，只留頂面。畫頂面、畫板面都用頂面輪廓：側面是直的 → 底面 XY = 頂面 XY，
    # 而底面邊緣常被頂面擋住看不到；頂面 + 高度視差修正 (pixel_to_arm) 換到手臂座標最準
    cnt = top_face(cnt, shape, obj_height(), LAST_DEPTH if DEPTH_ON else None)
    if PATH_MODE in ("vertex", "auto"):
        r = make_path_vertex(cnt, shape)
        if r is not None:
            LAST_PATH_MODE = "vertex"
            return r
    LAST_PATH_MODE = "edge"
    return make_path_edge(cnt, shape)


def _line_intersect(l1, l2):
    """兩條直線 (點 p, 方向 d) 的交點；幾乎平行回傳 None"""
    (p1, d1), (p2, d2) = l1, l2
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-6:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
    return p1 + t * d1


def find_vertices(cnt):
    """
    工件輪廓 → 頂點 (px，浮點數，順著輪廓方向)。不是多邊形回傳 None
      1. approxPolyDP 找大概的頂點，拿掉幾乎沒轉彎的
      2. 每個邊：拿兩個頂點中間的輪廓點 (頭尾各去掉一段，轉角在影像上是圓的) 擬合直線
      3. 相鄰兩條直線的交點 = 頂點 (亞像素、轉角是尖的)
      4. 檢查：整條輪廓離這個多邊形都不能太遠，否則不是多邊形 (例如圓、大圓角)
    回傳 (頂點 Nx2, 每條邊的直線 [(點, 單位方向)], 逆時針?)
    """
    pts = cnt.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 20:
        return None
    approx = cv2.approxPolyDP(cnt, VERTEX_EPS_PX, True).reshape(-1, 2)
    # approxPolyDP 的點都是輪廓上的點 → 找回 index
    idx = sorted({int(np.argmin(np.hypot(*(pts - a).T))) for a in approx})

    # 拿掉轉角太小 (幾乎直線) 的頂點、合併太近的頂點 (尖角在影像上被裁鈍，會被拆成兩個點)，直到穩定
    changed = True
    while changed and len(idx) > 3:
        changed = False
        for k in range(len(idx)):
            a, b, c = pts[idx[k - 1]], pts[idx[k]], pts[idx[(k + 1) % len(idx)]]
            v1, v2 = b - a, c - b
            ang = abs(math.degrees(math.atan2(v1[0] * v2[1] - v1[1] * v2[0], v1 @ v2)))
            if ang < VERTEX_MIN_TURN_DEG or np.hypot(*v2) < VERTEX_MIN_SIDE_PX:
                idx.pop(k)          # 太近的兩點留一個就好，真正的頂點由兩側長邊的直線交點求出
                changed = True
                break
    m = len(idx)
    if m < 3 or m > VERTEX_MAX:
        return None

    # 每條邊擬合直線
    lines = []
    for k in range(m):
        i0, i1 = idx[k], idx[(k + 1) % m]
        seg = pts[i0:i1 + 1] if i1 > i0 else np.vstack([pts[i0:], pts[:i1 + 1]])
        trim = max(3, int(len(seg) * VERTEX_TRIM_RATIO))
        core = seg[trim:-trim] if len(seg) > 2 * trim + 5 else seg
        if len(core) < 5:
            return None
        vx, vy, x0, y0 = cv2.fitLine(np.float32(core), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        d = np.array([vx, vy], float)
        if d @ (seg[-1] - seg[0]) < 0:                    # fitLine 的方向是任意的 → 統一成沿輪廓走的方向
            d = -d
        lines.append((np.array([x0, y0], float), d))

    verts = []
    for k in range(m):
        v = _line_intersect(lines[k - 1], lines[k])       # 第 k-1 邊和第 k 邊的交點 = 第 k 個頂點
        if v is None:
            return None
        verts.append(v)
    verts = np.array(verts)

    # 交點不能跑離原本的頂點太遠 (擬合失敗)
    if np.max(np.hypot(*(verts - pts[idx]).T)) > VERTEX_MAX_SHIFT_PX:
        return None
    # 整條輪廓離多邊形的距離：95% 的點都要在 VERTEX_FIT_TOL_PX 內
    poly = np.float32(verts).reshape(-1, 1, 2)
    d = np.abs([cv2.pointPolygonTest(poly, (float(x), float(y)), True) for x, y in pts])
    if np.percentile(d, 95) > VERTEX_FIT_TOL_PX:
        return None
    area = 0.5 * np.sum(verts[:, 0] * np.roll(verts[:, 1], -1) - np.roll(verts[:, 0], -1) * verts[:, 1])
    return verts, lines, area > 0


def make_path_vertex(cnt, shape):
    """
    頂點路徑：每個邊一段 MovL (一條邊從頭到尾速度穩定)，回到起點後沿第一條邊多走 GLUE_OVERLAP_MM
    內縮：每條邊的直線往內平移 DRAW_OFFSET_PX (+ DRAW_OFFSET_MM 換成 px)，再重新求交點 → 轉角保持尖的
    不是多邊形回傳 None
    """
    r = find_vertices(cnt)
    if r is None:
        return None
    verts, lines, ccw = r
    off = DRAW_OFFSET_PX + DRAW_OFFSET_MM / MM_PER_PX         # + 往外 / - 往內 (px)
    if off:
        moved = []
        for p, d in lines:
            nrm = np.array([d[1], -d[0]])                     # 逆時針時是往外
            if not ccw:
                nrm = -nrm
            moved.append((p + off * nrm, d))
        new = [_line_intersect(moved[k - 1], moved[k]) for k in range(len(moved))]
        if any(v is None for v in new):
            return None
        verts = np.array(new)
        # 內縮太多 (邊縮到沒了 / 翻轉) → 不做
        a2 = 0.5 * np.sum(verts[:, 0] * np.roll(verts[:, 1], -1) - np.roll(verts[:, 0], -1) * verts[:, 1])
        if (a2 > 0) != ccw:
            return None

    h = obj_height()
    arm_v = np.array([pixel_to_arm(u, v, h) for u, v in verts])
    pts_px = [tuple(v) for v in verts] + [tuple(verts[0])]
    pts_arm = [tuple(a) for a in arm_v] + [tuple(arm_v[0])]
    # 重疊段：回到起點後沿第一條邊再走 GLUE_OVERLAP_MM (不超過第一條邊長)
    if GLUE_OVERLAP_MM > 0:
        e = arm_v[1] - arm_v[0]
        L = float(np.hypot(*e))
        if L > 1e-6:
            t = min(GLUE_OVERLAP_MM, L) / L
            pts_arm.append(tuple(arm_v[0] + t * e))
            pts_px.append(tuple(verts[0] + t * (verts[1] - verts[0])))
    px = [(int(round(u)), int(round(v))) for u, v in pts_px]
    arm = [(float(x), float(y)) for x, y in pts_arm]
    return px, arm


def make_path_edge(cnt, shape):
    """
    物體邊緣 → (路徑像素點, 路徑手臂點)。沿邊緣等距取點，走一圈回到起點 (+ 重疊段)
    """
    edge = smooth_edge(cnt, shape)
    if edge is None:
        return [], []

    # 3a. 內縮 / 外擴 (px)：在影像上沿邊緣法線移 DRAW_OFFSET_PX，每個邊都一樣
    if DRAW_OFFSET_PX:
        edge = offset_closed(edge, DRAW_OFFSET_PX)

    # 3. 每個邊緣點換成手臂座標 (含高度視差修正)
    arm_all = np.array([pixel_to_arm(u, v, obj_height()) for u, v in edge])

    # 4. 內縮 / 外擴：在手臂座標 (mm) 沿邊緣法線移，比遮罩侵蝕 (一次至少 1 px ≈ 0.7 mm) 精準
    if DRAW_OFFSET_MM:
        arm_all = offset_closed(arm_all, DRAW_OFFSET_MM)

    # 5. 封閉一圈 + 回到起點後沿原路多走 GLUE_OVERLAP_MM (膠線頭尾接上)
    loop_arm = np.vstack([arm_all, arm_all[:1]])
    loop_px = np.vstack([edge, edge[:1]])
    seg = np.hypot(*np.diff(loop_arm, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])            # 每個邊緣點的累積長度
    total = s[-1]
    ov = min(GLUE_OVERLAP_MM, total)

    if GLUE_MERGE_TOL_MM > 0:
        # 6a. 段數越少越順：直接拿「全部」邊緣點，找出誤差 < GLUE_MERGE_TOL_MM 的最少段數
        #     (直邊 = 一整段；只有彎的地方才切段，彎越急段越短)
        k = int(np.searchsorted(s, ov)) if ov > 0 else 0
        tail_a = [loop_arm[1:k]] if k > 1 else []
        tail_p = [loop_px[1:k]] if k > 1 else []
        if ov > 0:                                          # 重疊段的終點 (內插)
            tail_a.append([[np.interp(ov, s, loop_arm[:, 0]), np.interp(ov, s, loop_arm[:, 1])]])
            tail_p.append([[np.interp(ov, s, loop_px[:, 0]), np.interp(ov, s, loop_px[:, 1])]])
        all_a = np.vstack([loop_arm] + tail_a)
        all_p = np.vstack([loop_px] + tail_p)
        keep = merge_straight(all_a, GLUE_MERGE_TOL_MM)
        ax, ay = all_a[keep, 0], all_a[keep, 1]
        pu, pv = all_p[keep, 0], all_p[keep, 1]
    else:
        # 6b. 不合併：每 GLUE_STEP_MM 一段 (舊做法，段很多、停頓多)
        n = max(3, int(round(total / GLUE_STEP_MM)))
        t = list(np.linspace(0.0, total, n + 1))
        if ov > 0:
            t += list(np.arange(GLUE_STEP_MM, ov, GLUE_STEP_MM)) + [ov]
        ax, ay = np.interp(t, s, loop_arm[:, 0]), np.interp(t, s, loop_arm[:, 1])
        pu, pv = np.interp(t, s, loop_px[:, 0]), np.interp(t, s, loop_px[:, 1])

    arm = [(float(x), float(y)) for x, y in zip(ax, ay)]
    px = [(int(round(u)), int(round(v))) for u, v in zip(pu, pv)]
    return px, arm


def smooth_edge(cnt, shape):
    """工件輪廓 → 平滑後的封閉邊緣點 (N x 2 浮點 px)；太短回傳 None"""
    if MASK_BLUR_PX > 0:
        m = np.zeros(shape[:2], np.uint8)
        cv2.drawContours(m, [cnt], -1, 255, -1)
        # 遮罩先模糊再取一半 → 階梯、小凸起被磨平，形狀大小不變 (直邊不動，只有尖角會稍微變圓)
        k = 2 * int(round(3 * MASK_BLUR_PX)) + 1
        m = ((cv2.GaussianBlur(m, (k, k), MASK_BLUR_PX)) >= 128).astype(np.uint8) * 255
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            return None
        cnt = max(cs, key=cv2.contourArea)

    # 邊緣上每一個像素 (CHAIN_APPROX_NONE)，不做特徵點簡化
    edge = cnt.reshape(-1, 2).astype(float)
    if len(edge) < 10:
        return None

    # 平滑：封閉曲線，前後 SMOOTH_PX 個點取平均 (頭尾接起來算)，去掉像素階梯
    if SMOOTH_PX > 1:
        k = SMOOTH_PX // 2
        ext = np.vstack([edge[-k:], edge, edge[:k]])
        kernel = np.ones(2 * k + 1) / (2 * k + 1)
        edge = np.column_stack([np.convolve(ext[:, 0], kernel, "valid"),
                                np.convolve(ext[:, 1], kernel, "valid")])
    return edge


def normals_closed(pts):
    """封閉曲線每一點往外的單位法線 (N x 2)"""
    nxt, prv = np.roll(pts, -1, axis=0), np.roll(pts, 1, axis=0)
    t = nxt - prv                                           # 切線
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    area = 0.5 * np.sum(pts[:, 0] * nxt[:, 1] - nxt[:, 0] * pts[:, 1])
    n = np.column_stack([t[:, 1], -t[:, 0]])                # 逆時針時是往外
    return -n if area < 0 else n


def offset_closed(pts, d):
    """封閉曲線每一點沿法線移 d mm (+ 往外 / - 往內)"""
    return pts + d * normals_closed(pts)


def board_plane(cnt, shape, depth):
    """
    工件周圍板面的深度平面。平面在相機裡：1/深度 跟像素 (u, v) 是線性關係 → 擬合 1/d = a·u + b·v + c，
    去掉離群點 (別的物體、Marker、雜訊) 再擬合一次。回傳 f(u, v) → 板面深度 mm；量不到回傳 None
    """
    fill = np.zeros(shape[:2], np.uint8)
    cv2.drawContours(fill, [cnt], -1, 255, -1)

    def ell(r):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    r_in, r_out = HEIGHT_RING_PX
    ring = cv2.dilate(fill, ell(2 * r_out)) & ~cv2.dilate(fill, ell(r_in))
    if ARUCO_EXCLUDE is not None and ARUCO_EXCLUDE.shape == ring.shape:
        ring &= ~ARUCO_EXCLUDE
    vs, us = np.nonzero((ring > 0) & (depth > 0))
    if len(us) < 200:
        return None
    if len(us) > 20000:                                     # 點太多 → 抽樣 (擬合 3 個參數用不到那麼多)
        sel = np.random.default_rng(0).choice(len(us), 20000, replace=False)
        us, vs = us[sel], vs[sel]
    A = np.column_stack([us, vs, np.ones(len(us))]).astype(np.float64)
    y = 1.0 / depth[vs, us].astype(np.float64)
    ok = np.ones(len(us), bool)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
        res = np.abs(A @ coef - y)
        mad = np.median(res[ok]) + 1e-12
        ok = res < 4.0 * 1.4826 * mad
        if ok.sum() < 100:
            return None
    a, b, c = coef
    return lambda u, v: 1.0 / (a * u + b * v + c)


def heightmap_outline(cnt, shape):
    """
    俯視高度圖的外框：
      1. 工件範圍內每個有深度的像素 → 高度 (板面平面 - 深度) → 用「自己的高度」做視差修正換成手臂 XY
      2. 投影到手臂 XY 的 RIM_GRID_MM 方格，每格取最高 = 正上方看下去的高度圖
         (沒有透視問題；垂直的側面在 XY 上疊在外框上，取最高就是頂邊)
      3. 高度圖的外輪廓 = 工件外框 (手臂 XY)，每一點的高度 = 附近 RIM_NEIGH_MM 內最高，沿外框中位數平滑
    回傳 (外框 XY (N x 2 mm), 高度 (N,), 深度有效比例)；量不到回傳 None (原因在 LAST_RIM["error"])
    """
    global LAST_RIM, LAST_HMAP
    LAST_HMAP = None
    depth = LOCKED_DEPTH if LOCKED_DEPTH is not None else LAST_DEPTH
    if not DEPTH_ON or depth is None or depth.shape[:2] != shape[:2]:
        LAST_RIM = {"error": "沒有深度"}
        return None
    plane = board_plane(cnt, shape, depth)
    if plane is None:
        LAST_RIM = {"error": "量不到周圍板面"}
        return None

    # 1. 工件範圍內的像素 → 高度 → 手臂 XY
    fill = np.zeros(shape[:2], np.uint8)
    cv2.drawContours(fill, [cnt], -1, 255, -1)
    r = int(round(RIM_ERODE_PX))
    if r > 0:
        fill = cv2.erode(fill, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n_in = int(np.count_nonzero(fill))
    vs, us = np.nonzero((fill > 0) & (depth > 0))
    ratio = len(us) / max(n_in, 1)
    if ratio < RIM_MIN_VALID:
        LAST_RIM = {"error": f"工件深度只量到 {ratio:.0%} (反光 / 黑色？)", "valid": ratio}
        return None
    hh = plane(us.astype(float), vs.astype(float)) - depth[vs, us]
    ok = (hh > RIM_MIN_H_MM) & (hh < HEIGHT_MAX_MM)
    us, vs, hh = us[ok], vs[ok], hh[ok]
    if len(us) < 50:
        LAST_RIM = {"error": f"比板面高 {RIM_MIN_H_MM} mm 以上的點太少 (工件太薄？)"}
        return None
    X, Y = pixels_to_arm(us, vs, hh)

    # 2. 手臂 XY 方格，每格取最高
    g = RIM_GRID_MM
    pad = 3
    x0, y0 = X.min() - pad * g, Y.min() - pad * g
    ix = ((X - x0) / g).astype(int)
    iy = ((Y - y0) / g).astype(int)
    hmap = np.full((iy.max() + pad + 1, ix.max() + pad + 1), -1.0, np.float32)
    np.maximum.at(hmap, (iy, ix), hh.astype(np.float32))
    occ = (hmap > 0).astype(np.uint8) * 255
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))     # 補方格之間的小洞
    occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))      # 去掉零星雜點
    cs, _ = cv2.findContours(occ, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        LAST_RIM = {"error": "高度圖找不到外框"}
        return None
    outline = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)       # (col, row)
    if len(outline) < 12:
        LAST_RIM = {"error": "外框太小"}
        return None

    # 3. 外框每一點的高度 = 附近 RIM_NEIGH_MM 內最高 (灰階膨脹 = 鄰域最大)，沿外框中位數平滑
    rn = max(1, int(round(RIM_NEIGH_MM / g)))
    hmax = cv2.dilate(np.maximum(hmap, 0), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rn + 1, 2 * rn + 1)))
    h = hmax[outline[:, 1].astype(int), outline[:, 0].astype(int)].astype(float)
    xy = np.column_stack([x0 + (outline[:, 0] + 0.5) * g, y0 + (outline[:, 1] + 0.5) * g])
    # 外框補償：取深度前遮罩內縮了 RIM_ERODE_PX、外框是邊緣那一格的「中心」(比真正邊緣內縮約半格)
    comp = RIM_EDGE_COMP_MM if RIM_EDGE_COMP_MM is not None else RIM_ERODE_PX * MM_PER_PX + RIM_EDGE_CELL_FRAC * g
    if comp:
        xy = offset_closed(xy, comp)
    w = max(1, int(round(RIM_SMOOTH_MM / g / 2)))
    ext = np.concatenate([h[-w:], h, h[:w]])
    h = np.median(np.lib.stride_tricks.sliding_window_view(ext, 2 * w + 1), axis=1)
    LAST_HMAP = {"hmap": hmap, "occ": occ, "x0": x0, "y0": y0, "g": g, "outline": xy, "h": h,
                 "verts": None}                   # 2-1 幾何量測用
    return xy, h, ratio


def _loop_resample(pts, step):
    """封閉折線 (最後一點 = 第一點) 每 step 取一點 (保留原本的頂點)"""
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        L = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(L / step)))
        for t in np.arange(n) / n:
            out.append(a + t * (b - a))
    out.append(pts[-1])
    return np.array(out)


def make_path_3d(cnt, shape):
    """
    auto 模式 (有深度)：俯視高度圖外框 → 路徑
      形狀：外框能擬合成多邊形 (find_vertices) → 尖角頂點；不是 → 沿外框曲線
      高度：每一點取最近外框點的高度；整條高低差 (5%~95%) < FLAT_TOL_MM → 平面，整條用中位數
            DRAW_SURFACE = "board" → 高度全部 0 (畫在板面沿工件外圍)
      往內縮 RIM_WIDTH_MM / 2 (空心工件描邊框中線)，封閉 + 重疊段，3D 合併 (斜直線一段，只在轉折切段)
    回傳 (路徑 px (顯示用), [(x, y, 表面高度 mm)])；量不到回傳 None (呼叫端改用 2D 做法)
    """
    global LAST_RIM, LAST_PATH_MODE
    LAST_RIM = {}
    r = heightmap_outline(cnt, shape)
    if r is None:
        return None
    xy, h_out, ratio = r
    shift = -RIM_WIDTH_MM / 2.0 + DRAW_OFFSET_MM + DRAW_OFFSET_PX * MM_PER_PX     # + 往外 / - 往內 (mm)

    # 形狀：多邊形? (find_vertices 用 px 為單位的參數 → 外框換成「板面 px」再找)
    loop = None
    fv = find_vertices(np.float32(xy / MM_PER_PX).reshape(-1, 1, 2))
    if fv is not None:                                  # 高度圖是乾淨的 mm 資料 → 用更嚴的標準確認真的是多邊形
        poly = np.float32(fv[0] * MM_PER_PX).reshape(-1, 1, 2)
        dev = np.abs([cv2.pointPolygonTest(poly, (float(x), float(y)), True) for x, y in xy])
        if np.percentile(dev, 95) > POLY_FIT_TOL_MM:
            fv = None
    if fv is not None:
        verts_px, lines, ccw = fv
        lines = [(p * MM_PER_PX, d) for p, d in lines]
        if shift:
            moved = []
            for p, d in lines:
                nrm = np.array([d[1], -d[0]])                   # 逆時針時是往外
                moved.append((p + shift * (nrm if ccw else -nrm), d))
            verts = [_line_intersect(moved[k - 1], moved[k]) for k in range(len(moved))]
        else:
            verts = list(verts_px * MM_PER_PX)
        if all(v is not None for v in verts):
            verts = np.array(verts)
            LAST_HMAP["verts"] = verts_px * MM_PER_PX         # 外框的頂點 (內縮前)，幾何量測用
            loop = _loop_resample(np.vstack([verts, verts[:1]]), RIM_STEP_MM)
            LAST_PATH_MODE = "vertex"
    if loop is None:
        w = max(1, int(round(RIM_XY_SMOOTH_MM / RIM_GRID_MM / 2)))
        ext = np.vstack([xy[-w:], xy, xy[:w]])
        kern = np.ones(2 * w + 1) / (2 * w + 1)
        cur = np.column_stack([np.convolve(ext[:, 0], kern, "valid"), np.convolve(ext[:, 1], kern, "valid")])
        if shift:
            cur = offset_closed(cur, shift)
        loop = _loop_resample(np.vstack([cur, cur[:1]]), RIM_STEP_MM)
        LAST_PATH_MODE = "edge"

    # 高度：最近的外框點
    d2 = ((loop[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
    h = h_out[np.argmin(d2, axis=1)]
    spread = float(np.percentile(h_out, 95) - np.percentile(h_out, 5))
    flat = spread < FLAT_TOL_MM
    hmed = float(np.median(h_out))
    if DRAW_SURFACE != "top":
        h = np.zeros_like(h)
    elif flat:
        h = np.full_like(h, hmed)
    pts3 = np.column_stack([loop, h])

    # 重疊段：回到起點後沿路徑再走 GLUE_OVERLAP_MM
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts3, axis=0), axis=1))])
    total = float(s[-1])
    ov = min(GLUE_OVERLAP_MM, total)
    if ov > 0:
        tt = np.linspace(0.0, ov, max(2, int(ov / RIM_STEP_MM) + 1))[1:]
        tail = np.column_stack([np.interp(tt, s, pts3[:, k]) for k in range(3)])
        pts3 = np.vstack([pts3, tail])

    # 合併：多邊形 / 斜面用 3D 容許誤差；平面的曲線保留跟 2D edge 一樣的貼合度
    tol = GLUE_MERGE_TOL_MM if (flat and LAST_PATH_MODE == "edge" and GLUE_MERGE_TOL_MM > 0) else RIM_MERGE_TOL_MM
    pts3 = pts3[merge_straight(pts3, tol)]

    LAST_RIM = {"hmin": float(h_out.min()), "hmax": float(h_out.max()), "hmed": hmed, "spread": spread,
                "flat": flat, "valid": ratio, "segments": len(pts3) - 1, "length": total}
    arm = [(float(x), float(y), float(z)) for x, y, z in pts3]
    px = [tuple(int(round(c)) for c in arm_to_pixel(x, y, z)) for x, y, z in arm]
    return px, arm


def merge_straight(pts, tol):
    """Douglas-Peucker (2D 或 3D 都可以)：回傳要保留的點 index (頭尾一定保留)。tol <= 0 → 全部保留"""
    pts = np.asarray(pts, float)
    n = len(pts)
    if tol <= 0 or n < 3:
        return list(range(n))
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        p, q = pts[a], pts[b]
        d = q - p
        L2 = float(d @ d)
        rel = pts[a + 1:b] - p
        if L2 < 1e-18:
            dist = np.linalg.norm(rel, axis=1)
        else:                                               # 點到直線的距離 (任意維度)
            dist = np.linalg.norm(rel - np.outer(rel @ d / L2, d), axis=1)
        k = int(np.argmax(dist))
        if dist[k] > tol:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return list(np.flatnonzero(keep))


def path_ok(arm_pts):
    """X_RANGE / Y_RANGE 是 None 就不檢查"""
    def inside(val, rng):
        return rng is None or rng[0] <= val <= rng[1]
    return all(inside(p[0], X_RANGE) and inside(p[1], Y_RANGE) for p in arm_pts)   # 點可能是 (x, y) 或 (x, y, h)


def path_length(arm_pts):
    return sum(math.hypot(arm_pts[i + 1][0] - arm_pts[i][0], arm_pts[i + 1][1] - arm_pts[i][1])
               for i in range(len(arm_pts) - 1))


# ==========================================
# 6. 動作
# ==========================================
CUR_HEIGHT = None   # 目前選中工件量到的高度 mm (None = 量不到 → 用 OBJ_HEIGHT_MM)


def obj_height():
    """工件高度：有量到就用量到的，否則用 OBJ_HEIGHT_MM"""
    if AUTO_HEIGHT and CUR_HEIGHT is not None:
        return CUR_HEIGHT
    return OBJ_HEIGHT_MM


def draw_z():
    """筆尖碰到要畫的表面時的 Z"""
    if Z_DRAW is not None:
        return Z_DRAW
    return Z_PAPER + (obj_height() if DRAW_SURFACE == "top" else 0.0)


def run_glue_segments(arm, pts, z, tol=1.0, timeout=30.0, on_tick=None):
    """
    已經下筆在 pts[0]，沿 pts[1:] 一段一段 MovL 走完。盡量不停：
      - 到位判斷：看到旗標變 0 (真的開始動了) 後，旗標=1 + 位置差 < tol 就算到 (不固定等 0.2 秒)
      - 提前送點：離終點 < GLUE_EARLY_MM 就先送下一段。
        之後如果發現手臂停在「舊的終點」(= 控制器忽略了)，馬上補送
    回傳統計 {"early": 提前送幾次, "resent": 被忽略補送幾次}
    """
    def dist(p, q):
        return math.hypot(p[0] - q[0], p[1] - q[1])

    st = {"early": 0, "resent": 0}
    n = len(pts) - 1
    i = 1
    old = None                # 提前送出時還沒走到的那個點 (用來偵測被忽略)
    use_early = GLUE_EARLY_MM > 0

    zs = list(z) if isinstance(z, (list, tuple, np.ndarray)) else [z] * len(pts)

    def send(k):
        arm.send_move(pts[k][0], pts[k][1], zs[k], speed=SPEED_GLUE, linear=True)

    send(1)
    seen_moving, t_wait = False, time.time()
    while i <= n:
        if time.time() - t_wait > timeout:
            raise RuntimeError(f"塗膠第 {i}/{n} 段逾時，請看教導器是否有警報")
        flag = arm._read(REG_DONE, 1)[0]
        p = arm.pose()
        if flag != 1:
            seen_moving = True
        tgt = pts[i]
        seg_len = dist(pts[i], pts[i - 1])
        fresh = (seen_moving and seg_len >= 2 * tol) or time.time() - arm.t_cmd >= 0.2

        # 提前送的指令被忽略：手臂停在舊點、沒往新點走 → 補送
        if old is not None and flag == 1 and dist(p, old) < tol and dist(p, tgt) >= tol \
                and time.time() - arm.t_cmd > 0.05:
            send(i)
            st["resent"] += 1
            old, seen_moving, t_wait = None, False, time.time()
            if st["resent"] >= 2 and st["resent"] == st["early"]:
                use_early = False          # 控制器不吃提前送點 → 這次之後都不提前，免得反而變慢
                st["auto_off"] = True
            continue

        arrived = flag == 1 and fresh and dist(p, tgt) < tol
        early = (use_early and i < n and flag != 1 and seen_moving
                 and dist(p, tgt) < GLUE_EARLY_MM)
        if arrived or early:
            i += 1
            if i <= n:
                if GLUE_PAUSE_S > 0 and arrived:
                    time.sleep(GLUE_PAUSE_S)
                send(i)
                old = tgt if early else None
                if early:
                    st["early"] += 1
                seen_moving, t_wait = False, time.time()
            continue
        if on_tick:
            on_tick(i, n)
        time.sleep(0.003)

    # 最後一段一定要確實到位 (上面最後一段只會用 arrived 判斷，不會提前)
    return st


def path_z(pts, lift=0.0):
    """
    每個路徑點的塗膠 Z (TCP)。點是 (x, y) → 全部用 draw_z()；點是 (x, y, 表面高度) (rim 模式) → 各自的 Z
    lift = 空中試跑時整條抬高幾 mm
    """
    return [(Z_PAPER + p[2] if len(p) > 2 else draw_z()) - PEN_PRESS_MM + lift for p in pts]


def build_rl_points(pts, z_draw, z_up):
    """
    整段塗膠 (移過去、下筆、沿邊、抬筆) 轉成 RL 點列：(x, y, z, 速度%, 種類, 轉角混合 mm)
    z_draw：一個數字 (整條同高) 或每點一個 Z 的 list (3D 邊框)；z_up：抬筆比塗膠 Z 高幾 mm 的那個 Z
    (list 時改用 Z_LIFT 加在每點自己的 Z 上)
    """
    import rl_path
    corner = RL_CORNER_OVERLAP_MM if LAST_PATH_MODE == "vertex" else RL_OVERLAP_MM
    zs = list(z_draw) if isinstance(z_draw, (list, tuple, np.ndarray)) else [z_draw] * len(pts)
    x0, y0 = pts[0][:2]
    xe, ye = pts[-1][:2]
    z_up0 = z_up if not isinstance(z_draw, (list, tuple, np.ndarray)) else zs[0] + Z_LIFT
    z_upe = z_up if not isinstance(z_draw, (list, tuple, np.ndarray)) else zs[-1] + Z_LIFT
    if max(zs) + Z_LIFT > Z_TRAVEL:
        raise ValueError(f"最高塗膠 Z {max(zs):.1f} + 抬筆 {Z_LIFT} 超過移動高度 Z_TRAVEL={Z_TRAVEL}，"
                         "工件太高：請調高 Z_TRAVEL")
    out = [(x0, y0, Z_TRAVEL, RL_SPEED_TRAVEL, rl_path.KIND_MOVP, 0),
           (x0, y0, z_up0, RL_SPEED_TRAVEL, rl_path.KIND_MOVL, 0),
           (x0, y0, zs[0], RL_SPEED_GLUE, rl_path.KIND_MOVL, 0)]           # 下筆，到位才開始畫
    n = len(pts)
    for i in range(1, n):
        ovl = 0.0
        if i < n - 1 and corner > 0 and rl_path.PASS_MODE == rl_path.PASS_DEC:
            ovl = 1.0                                                      # DEC 模式：只當「要 PASS」的旗標
        elif i < n - 1 and corner > 0:                                     # 最後一點要到位 (接著抬筆)
            a, b, c = pts[i - 1], pts[i], pts[i + 1]
            shortest = min(math.hypot(b[0] - a[0], b[1] - a[1]), math.hypot(c[0] - b[0], c[1] - b[1]))
            ovl = min(corner, 0.45 * shortest)
            if ovl < 1.0:                                                  # 控制器下限 1 mm，太短就到位再走
                ovl = 0.0
        out.append((pts[i][0], pts[i][1], zs[i], RL_SPEED_GLUE, rl_path.KIND_MOVL, ovl))
    out += [(xe, ye, z_upe, RL_SPEED_GLUE, rl_path.KIND_MOVL, 0),
            (xe, ye, Z_TRAVEL, RL_SPEED_TRAVEL, rl_path.KIND_MOVL, 0)]
    for p in out:
        if p[2] < Z_MIN:
            raise ValueError(f"Z={p[2]:.2f} 低於安全下限 Z_MIN={Z_MIN}")
    return out


def draw_path(arm, pts, lift=0.0, track=None):
    """
    沿邊緣塗膠：USE_RL_PATH 時交給控制器 RL 程式連續走；否則每一小段用 Modbus MovL 走
    pts 是 (x, y) 或 (x, y, 表面高度) (每點自己的 Z)。lift > 0 = 空中試跑 (整條抬高，不碰工件)
    track：LiveTrack → 走的時候視窗上即時標出筆尖位置
    """
    arm.on_tick = track.update if track else None       # arm.move 等到位時也會更新畫面
    try:
        _draw_path(arm, pts, lift, track)
    finally:
        arm.on_tick = None


def _draw_path(arm, pts, lift, track):
    zs = path_z(pts, lift)
    per_point = len(pts[0]) > 2
    z_draw = zs if per_point else zs[0]
    z_up = (zs[0] + Z_LIFT) if not per_point else None
    tag = f"空中試跑 (+{lift:.0f} mm)" if lift else "塗膠"
    if USE_RL_PATH:
        import rl_path
        rl_pts = build_rl_points(pts, z_draw, z_up)
        print(f"🧴 開始{tag} (RL 連續路徑)：{len(pts) - 1} 段，總長 {path_length(pts):.0f} mm，"
              f"速度 {RL_SPEED_GLUE}%" + (f"，Z {min(zs):.1f} ~ {max(zs):.1f}" if per_point else ""))
        while True:
            try:
                rl_path.RLPath(arm).run(rl_pts, (RX_HOME, RY_HOME, RZ_HOME),
                                        on_tick=(lambda i: track.update(i, len(rl_pts))) if track else None)
                return
            except rl_path.RLPathError as e:
                if arm._read(rl_path.BUF + 3, 1)[0] > 0:      # 已經開始動了 → 不要自動重來
                    raise
                print(f"⚠️ RL 路徑無法執行：{e}")
            # T2 手動模式下 Modbus 啟動不了 RL，要在 DRAStudio 手動按 Start (每次塗膠前都要)
            ans = ask("  Enter: 在 DRAStudio 按 Start 之後重試   s: 改用逐段 MovL (每段會停) > ", ("", "s"),
                      {"": "Retry (Enter)", "s": "Step MovL (s)"})
            if ans == "s":
                print("   改用逐段 MovL")
                break
    x0, y0 = pts[0][:2]
    n = len(pts) - 1
    if max(zs) + Z_LIFT > Z_TRAVEL:
        raise ValueError(f"最高塗膠 Z {max(zs):.1f} + 抬筆 {Z_LIFT} 超過 Z_TRAVEL={Z_TRAVEL}，請調高 Z_TRAVEL")
    if min(zs) < Z_MIN:
        raise ValueError(f"Z={min(zs):.2f} 低於安全下限 Z_MIN={Z_MIN}")
    print(f"🧴 開始{tag}：{n} 小段，總長 {path_length(pts):.0f} mm")
    arm.move(x0, y0, Z_TRAVEL)
    arm.move(x0, y0, zs[0] + Z_LIFT)
    arm.move(x0, y0, zs[0], speed=SPEED_GLUE)                  # 下筆
    t0 = time.time()
    st = run_glue_segments(arm, pts, zs, on_tick=track.update if track else None)
    print(f"  ✅ {n} 段走完，塗膠 {time.time() - t0:.1f} s")
    if st["early"]:
        print(f"  提前送點 {st['early']} 次：控制器接受 {st['early'] - st['resent']} 次、"
              f"忽略後補送 {st['resent']} 次"
              + ("  → 控制器不吃提前送點，已自動關閉；可以把 GLUE_EARLY_MM 設 0" if st.get("auto_off") else ""))
    xe, ye = pts[-1][:2]
    arm.move(xe, ye, zs[-1] + Z_LIFT, speed=SPEED_GLUE)        # 抬筆 (在終點原地抬起)
    arm.move(xe, ye, Z_TRAVEL)


ARUCO_EXCLUDE_PAD_PX = 15   # Marker 外框再往外多排除幾 px (白邊 + 偵測誤差)
ARUCO_EXCLUDE = None        # 排除遮罩 (相機像素，255 = 不偵測物體)


def build_aruco_exclude(pipeline, n_frames=10):
    """
    手臂在 Home 時呼叫：偵測 Marker 外框做成排除遮罩，物體偵測就不會把 Marker 當成物體。
    Marker 跟 Home 都固定，所以做一次就好 (a / c 校正完會自動重做)。
    沒偵測到的 Marker (例如被東西擋住)：用校正記錄的中心 PTS_PIXEL + 其他 Marker 的平均大小補一個方框
    """
    global ARUCO_EXCLUDE
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    acc, frame = {}, None
    for _ in range(n_frames):
        frame = grab(pipeline)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if hasattr(cv2.aruco, "ArucoDetector"):
            corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=cv2.aruco.DetectorParameters_create())
        if ids is not None:
            for i, c in zip(ids.flatten(), corners):
                acc.setdefault(int(i), []).append(c.reshape(4, 2))
    polys = [np.mean(v, axis=0) for v in acc.values()]
    n_found = len(polys)
    # 邊長 ≈ 對角線 / √2；都沒偵測到就假設 50 px
    side = float(np.mean([np.linalg.norm(q[0] - q[2]) for q in polys])) / math.sqrt(2) if polys else 50.0
    for u, v in PTS_PIXEL:
        if all(math.hypot(*(q.mean(axis=0) - (u, v))) > ARUCO_MATCH_PX for q in polys):
            h = side / 2
            polys.append(np.float32([[u - h, v - h], [u + h, v - h], [u + h, v + h], [u - h, v + h]]))
    mask = np.zeros(frame.shape[:2], np.uint8)
    for q in polys:
        cv2.fillConvexPoly(mask, np.int32(np.round(q)), 255)
    k = 2 * ARUCO_EXCLUDE_PAD_PX + 1
    ARUCO_EXCLUDE = cv2.dilate(mask, np.ones((k, k), np.uint8))
    extra = len(polys) - n_found
    print(f"🔲 物體偵測排除 Marker：偵測到 {n_found} 個" +
          (f"，另外 {extra} 個沒偵測到、用校正位置補" if extra else "") +
          f" (外擴 {ARUCO_EXCLUDE_PAD_PX} px，畫面上灰框)")


def detect_aruco(pipeline, n_frames=10, dict_id=None):
    """連拍 n 張平均，回傳 {id: (u, v)} Marker 中心"""
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT if dict_id is None else dict_id)
    acc = {}
    for _ in range(n_frames):
        gray = cv2.cvtColor(grab(pipeline), cv2.COLOR_BGR2GRAY)
        if hasattr(cv2.aruco, "ArucoDetector"):
            corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=cv2.aruco.DetectorParameters_create())
        if ids is not None:
            for i, c in zip(ids.flatten(), corners):
                acc.setdefault(int(i), []).append(c.reshape(4, 2).mean(axis=0))
    return {i: tuple(np.mean(v, axis=0)) for i, v in acc.items()}


ARUCO_DICT_CANDIDATES = ["DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_50", "DICT_7X7_50",
                         "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_36h11"]


PICK_STEPS = {ord('1'): 1, ord('2'): 5, ord('3'): 20}   # 十字游標每步幾 px


def pick_point(pipeline, win, start, title, marks=(), frame=None, live_overlay=None):
    """
    用終端機選畫面上的一個點 (取代滑鼠點選)：
      i / k / j / l : 十字往 上 / 下 / 左 / 右 (= 轉正後畫面上看到的方向)
      1 / 2 / 3     : 每步 1 / 5 / 20 px     Enter: 確定     Esc: 取消
    start = 起始位置 (相機像素)；marks = 要一起畫出來的參考圈；frame = 固定畫面 (None = 即時畫面)
    回傳 (u, v) 相機像素，取消回傳 None
    """
    u, v = float(start[0]), float(start[1])
    step = 5
    print(f"  👉 {title}：i/k/j/l 移動十字、1/2/3 步距 1/5/20 px、Enter 確定、Esc 取消 (都在終端機按)")
    flush_keys()
    while True:
        img = (frame if frame is not None else grab(pipeline)).copy()
        h, w = img.shape[:2]
        for mu, mv in marks:
            cv2.circle(img, (int(mu), int(mv)), 10, (0, 255, 0), 2)
        cv2.drawMarker(img, (int(round(u)), int(round(v))), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
        out = to_display(img)
        put_text(out, f"{title}  ijkl: move  1/2/3: step ({step}px)  Enter: OK  Esc: cancel",
                 (10, 25), (0, 0, 255), 0.5)
        if live_overlay:
            put_text(out, live_overlay(u, v), (10, out.shape[0] - 10), (0, 255, 0), 0.55)
        cv2.imshow(win, out)
        key = poll_key(30)
        if key in JOG_KEYS:
            sdx, sdy = JOG_KEYS[key]
            if DISPLAY_ROTATE_180:            # 畫面轉 180°：顯示方向跟相機像素方向相反
                sdx, sdy = -sdx, -sdy
            u = min(max(u + sdx * step, 0), w - 1)
            v = min(max(v + sdy * step, 0), h - 1)
        elif key in PICK_STEPS:
            step = PICK_STEPS[key]
        elif key in (13, 10):
            return u, v
        elif key == 27:
            return None


def click_marker(pipeline, win, pts):
    """偵測不到的那個 Marker：用十字游標指出它的中心 (起始位置 = 用另外 3 個推算的第 4 角)"""
    p = np.array([[u, v] for _, u, v in pts], float)
    # 3 個角裡距離最遠的兩個是對角線，剩下那個的對角就是缺的那個角 (平行四邊形補點)
    pairs = [(0, 1), (0, 2), (1, 2)]
    a, c = max(pairs, key=lambda ij: np.linalg.norm(p[ij[0]] - p[ij[1]]))
    b = 3 - a - c
    guess = p[a] + p[c] - p[b]
    frame = grab(pipeline)
    h, w = frame.shape[:2]
    guess = (min(max(guess[0], 0), w - 1), min(max(guess[1], 0), h - 1))
    return pick_point(pipeline, win, guess, "MISSING marker center",
                      marks=[(u, v) for _, u, v in pts], frame=frame)


def detect_four(pipeline, win):
    """
    完整重新校正用：直接抓畫面上的 4 個 Marker (不跟舊位置比對)。
    各種字典都試，合起來剛好 4 個就用；只抓到 3 個 → 讓你點第 4 個
    回傳 (字典名稱, [(id, u, v)] 依相機原始畫面 左上, 右上, 右下, 左下)；失敗回傳 None
    """
    global ARUCO_DICT
    found, best = {}, (0, None)
    for n in ARUCO_DICT_CANDIDATES:
        f = detect_aruco(pipeline, 10, getattr(cv2.aruco, n))
        print(f"  {n}: 偵測到 {len(f)} 個 {sorted(f)}")
        for i, uv in f.items():
            # 同一個 Marker 被不同字典抓到 → 位置一樣，只留一個
            if all(math.hypot(uv[0] - q[0], uv[1] - q[1]) > 20 * PX_SCALE for q in found.values()):
                found[f"{n}:{i}"] = uv
        if len(f) > best[0]:
            best = (len(f), n)
    dict_name = best[1]
    if len(found) == 3:
        print("⚠️ 只偵測到 3 個 Marker")
        pts = [(k, u, v) for k, (u, v) in found.items()]
        uv = click_marker(pipeline, win, pts)
        if uv is None:
            return None
        found["點選"] = uv
    if len(found) != 4:
        print(f"⚠️ 偵測到 {len(found)} 個 Marker，要剛好 4 個。確認 4 個都在畫面內、沒被擋住/反光，"
              "畫面裡沒有多的 Marker")
        return None
    ARUCO_DICT = getattr(cv2.aruco, dict_name)
    print(f"  → 用 {dict_name}")
    pts = [(i, u, v) for i, (u, v) in found.items()]
    cu, cv_ = np.mean([p[1] for p in pts]), np.mean([p[2] for p in pts])
    order = {(True, True): 0, (True, False): 1, (False, False): 2, (False, True): 3}  # (上?, 左?)
    pts.sort(key=lambda p: order[(p[2] < cv_, p[1] < cu)])
    if len({order[(p[2] < cv_, p[1] < cu)] for p in pts}) != 4:
        print("⚠️ 4 個 Marker 沒有分在 4 個角，請擺成矩形四角")
        return None
    return dict_name, pts


try:
    import msvcrt
except ImportError:          # 非 Windows：只能用影像視窗按鍵
    msvcrt = None


# ---------- 按鈕：影像視窗下方一列按鈕，滑鼠點 = 按那個鍵 (鍵盤照樣能用) ----------
GUI_BUTTONS = True          # False = 回到只用終端機按鍵 / input()
BTN_H = 30                  # 按鈕高度 (顯示 px)
_BTN = []                   # 目前畫面上的按鈕 [(x0, y0, x1, y1, 按鍵碼)] (顯示座標)
_KEYQ = collections.deque() # 點按鈕產生的按鍵，poll_key 會先讀這裡
_BTN_WINS = set()           # 已經掛上滑鼠事件的視窗
LAST_SHOWN = None           # (視窗, 最後顯示的影像 (不含按鈕列))：ask() 在上面加按鈕用

PREVIEW_BUTTONS = [("Next (n)", "n"), ("Lock & glue (s)", "s"), ("Mode (m)", "m"), ("Check offset (a)", "a"),
                   ("Recalib (c)", "c"), ("Auto calib (k)", "k"), ("Measure (g)", "g"), ("Verify (v)", "v"), ("Background (b)", "b"), ("Dot (t)", "t"), ("Snapshot (p)", "p"),
                   ("Quit (q)", "q")]


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for x0, y0, x1, y1, code in _BTN:
            if x0 <= x < x1 and y0 <= y < y1:
                _KEYQ.append(code)
                break


def _key_code(k):
    """按鈕的鍵：'' = Enter (13)、'esc' = 27、單一字元 = 它的碼"""
    if k == "":
        return 13
    if k == "esc":
        return 27
    return ord(k.lower()) & 0xFF


def show(win, img, buttons=(), msg=None):
    """
    imshow + 下方加一列按鈕 [(標籤, 鍵)]。標籤只能英文 (OpenCV 畫不出中文)，問題本身印在終端機
    msg：按鈕上面一行提示 (英文)
    """
    global _BTN, LAST_SHOWN
    LAST_SHOWN = (win, img)
    if not GUI_BUTTONS or not buttons:
        _BTN = []
        cv2.imshow(win, img)
        return
    if win not in _BTN_WINS:
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, _on_mouse)
        _BTN_WINS.add(win)
    w = img.shape[1]
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    # 排版：一列放不下就換行
    rows, row, x = [], [], 4
    for item in buttons:
        label, key = item[0], item[1]
        tw = cv2.getTextSize(label, font, fs, th)[0][0] + 16
        if x + tw > w - 4 and row:
            rows.append(row)
            row, x = [], 4
        row.append((label, key, x, tw, item[2] if len(item) > 2 else None))
        x += tw + 4
    if row:
        rows.append(row)
    top = 22 if msg else 0
    bar = np.full((top + len(rows) * (BTN_H + 4) + 4, w, 3), 45, np.uint8)
    if msg:
        cv2.putText(bar, msg, (6, 16), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    _BTN = []
    y0 = img.shape[0]
    for r, row in enumerate(rows):
        by = top + 4 + r * (BTN_H + 4)
        for label, key, bx, tw, col in row:
            col = col or (90, 90, 90)
            cv2.rectangle(bar, (bx, by), (bx + tw, by + BTN_H), col, -1)
            cv2.rectangle(bar, (bx, by), (bx + tw, by + BTN_H), (200, 200, 200), 1)
            cv2.putText(bar, label, (bx + 8, by + BTN_H // 2 + 5), font, fs, (255, 255, 255), th, cv2.LINE_AA)
            _BTN.append((bx, y0 + by, bx + tw, y0 + by + BTN_H, _key_code(key)))
    out = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imshow(win, np.vstack([out, bar]))


def flush_keys():
    """清掉終端機鍵盤緩衝區和還沒處理的按鈕 (手臂在動時亂按的，不要被下一個步驟讀到)"""
    while msvcrt and msvcrt.kbhit():
        msvcrt.getwch()
    _KEYQ.clear()


def poll_key(ms=1):
    """
    按鍵可以在「終端機」按，或點影像視窗下方的按鈕：
    一定要呼叫 cv2.waitKey 影像視窗才會更新 (滑鼠事件也是這時候處理)，所以照呼叫；
    順序：按鈕 → 終端機 → 影像視窗的按鍵 (焦點不小心在視窗上時也能用)。
    回傳跟 cv2.waitKey(...) & 0xFF 一樣的碼 (沒按 = 255，Enter = 13，Esc = 27)
    """
    win_key = cv2.waitKey(ms) & 0xFF
    if _KEYQ:
        return _KEYQ.popleft()
    if msvcrt and msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):       # 方向鍵等特殊鍵：丟掉
            msvcrt.getwch()
            return 255
        if ch == "\x03":                 # Ctrl+C
            raise KeyboardInterrupt
        if ch in ("\n", "\r"):
            return 13
        return ord(ch.lower()) & 0xFF
    return win_key


def ask(prompt, choices=None, labels=None):
    """
    問問題：終端機印出問題，影像視窗下方出現每個選項的按鈕 (點按鈕或在終端機按那個鍵都可以，不用 Enter)
    choices = 可以回答的鍵 ("" = Enter)；None = 只等 Enter 繼續
    labels = {鍵: 按鈕上的英文標籤}；沒給就用鍵本身
    GUI_BUTTONS = False → 跟以前一樣用 input()
    """
    flush_keys()
    if not GUI_BUTTONS:
        while True:
            ans = input(prompt).strip().lower()
            if choices is None or ans in choices:
                return ans
            print(f"  請輸入 {' / '.join(choices)}")
    labels = labels or {}
    keys = list(choices) if choices is not None else [""]
    buttons = [(labels.get(k, "Enter" if k == "" else k), k,
                (40, 120, 40) if k in ("y", "") else (40, 40, 160) if k in ("n", "q", "x") else None)
               for k in keys]
    if LAST_SHOWN is not None:
        win, img = LAST_SHOWN
    else:
        win, img = "Glue Edge", np.zeros((120, 640, 3), np.uint8)
    show(win, img, buttons, "Answer (see terminal): click a button or press its key")
    LAST_SHOWN_KEEP = (win, img)
    print(prompt, end="", flush=True)
    codes = {_key_code(k): k for k in keys}
    while True:
        c = poll_key(30)
        if c in codes:
            ans = codes[c]
            print(ans if ans else "(Enter)")
            show(*LAST_SHOWN_KEEP)                  # 拿掉回答用的按鈕
            return ans


def write_calib_json(**fields):
    """更新 CALIB_PATH 裡的欄位 (其他欄位保留)"""
    d = {}
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH, encoding="utf-8") as fp:
            d = json.load(fp)
    d.update(fields, saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CALIB_PATH, "w", encoding="utf-8") as fp:
        json.dump(d, fp, ensure_ascii=False, indent=2)


def save_calibration(taught, dict_name):
    """
    taught = [(u, v, X, Y)]：Marker 像素中心 + 筆尖對準 Marker 中心時的 TCP X, Y (只做 XY，不管 Z)
    重建 像素 → 手臂 的對應。教的就是「筆尖」位置，所以筆尖偏移歸 0
    """
    global TOOL_OFFSET_X, TOOL_OFFSET_Y
    cu, cv_ = np.mean([t[0] for t in taught]), np.mean([t[1] for t in taught])
    order = {(True, True): 0, (True, False): 1, (False, False): 2, (False, True): 3}  # 原始畫面 (上?, 左?)
    taught = sorted(taught, key=lambda t: order[(t[1] < cv_, t[0] < cu)])
    px = np.float32([[t[0], t[1]] for t in taught])
    am = np.float32([[t[2], t[3]] for t in taught])
    sc = side_scales(px, am)
    print("\n========== 完整重新校正結果 (XY) ==========")
    for (u, v, x, y) in taught:
        print(f"  像素 ({u:6.1f}, {v:6.1f}) → X={x:.3f}, Y={y:.3f}")
    print(f"  4 邊的 mm/px：{', '.join(f'{s:.3f}' for s in sc)}  (舊的平均 {MM_PER_PX:.3f})")
    ratio = (max(sc) - min(sc)) / np.mean(sc)
    if ratio > 0.30:
        print(f"  ❌ 4 邊比例差 {ratio:.0%} (> 30%)，一定有點教錯 (沒移過去就按 Enter / 順序錯)，不存。請重做")
        return False
    if ratio > 0.15:   # 相機有點斜，舊校正本身就差 ~8%
        print("  ⚠️ 4 邊比例差超過 15%，可能有點對歪或順序教錯了，建議重做")
    if ask("  要套用並存檔嗎？(y/n) > ", ("y", "n"), {"y": "Save (y)", "n": "Discard (n)"}) != "y":
        print("  沒有存")
        return False
    set_calibration(px, am)
    TOOL_OFFSET_X = TOOL_OFFSET_Y = 0.0
    write_calib_json(pts_pixel=px.tolist(), pts_arm=am.tolist(), aruco_dict=dict_name,
                     home=[X_HOME, Y_HOME, Z_HOME], tool_offset_x=0.0, tool_offset_y=0.0, corrections=[],
                     image_size=list(CALIB_SIZE_NOW), h_source="aruco", h_matrix=None)
    global CALIB_SIZE
    CALIB_SIZE = CALIB_SIZE_NOW
    print(f"  💾 已存到 {CALIB_PATH}，筆尖偏移歸 0")
    return True


JOG_KEYS = {ord('i'): (0, -1), ord('k'): (0, 1), ord('j'): (-1, 0), ord('l'): (1, 0)}  # 畫面上/下/左/右
JOG_STEPS = {ord('1'): 0.5, ord('2'): 2.0, ord('3'): 5.0}
SPEED_JOG = 20          # 微調速度 (%)


def screen_dir_to_arm(u, v, sdx, sdy):
    """顯示畫面上的方向 (sdx, sdy) → 手臂 X, Y 的單位方向 (在像素 (u, v) 附近)"""
    if DISPLAY_ROTATE_180:
        sdx, sdy = -sdx, -sdy
    x0, y0 = pixel_to_arm(u, v)
    x1, y1 = pixel_to_arm(u + 10 * sdx, v + 10 * sdy)
    n = math.hypot(x1 - x0, y1 - y0)
    return (x1 - x0) / n, (y1 - y0) / n


def save_tool_offset(corrections):
    """corrections = [(名稱, dX, dY)]：實際對準位置 - 程式算的位置 (只做 XY)"""
    global TOOL_OFFSET_X, TOOL_OFFSET_Y
    d = np.array([[c[1], c[2]] for c in corrections])
    mean = d.mean(axis=0)
    spread = np.hypot(*(d - mean).T).max() if len(d) > 1 else 0.0
    print("\n========== 筆尖偏移計算 ==========")
    for name, dx, dy in corrections:
        print(f"  {name}: 實際 - 程式算的 = ΔX {dx:+6.2f}, ΔY {dy:+6.2f} mm")
    print(f"  平均 ΔX {mean[0]:+.2f}, ΔY {mean[1]:+.2f} mm，各點跟平均最多差 {spread:.2f} mm")
    if spread > 3.0:
        print("  ⚠️ 各點偏的不一樣 (> 3 mm)，不只是筆夾歪，校正本身可能有問題，先不要存\n"
              "     → 紙板 / Marker / 相機動過的話，重跑程式在 [ArUco] 按 c 做完整重新校正")
    new_x, new_y = TOOL_OFFSET_X + mean[0], TOOL_OFFSET_Y + mean[1]
    print(f"  新的筆尖偏移：X={new_x:+.2f}, Y={new_y:+.2f} (原本 {TOOL_OFFSET_X:+.2f}, {TOOL_OFFSET_Y:+.2f})")
    if ask("  要套用並存檔嗎？(y/n) > ", ("y", "n"), {"y": "Save (y)", "n": "Discard (n)"}) != "y":
        print("  沒有存")
        return
    TOOL_OFFSET_X, TOOL_OFFSET_Y = float(new_x), float(new_y)
    write_calib_json(tool_offset_x=TOOL_OFFSET_X, tool_offset_y=TOOL_OFFSET_Y,
                     corrections=[{"name": n, "dx": float(dx), "dy": float(dy)} for n, dx, dy in corrections])
    print(f"  💾 已套用並存到 {CALIB_PATH}")


def teach_with_pendant(arm, pipeline, win, targets, recalib=None):
    """
    用示教器教點：程式不動手臂，只告訴你順序、在你按 Enter 時讀手臂座標。
    順序依「轉正後的畫面」(= 實際看到的方向)：左上 → 右上 → 右下 → 左下
    recalib = 字典名稱 → 完整重新校正 (4 點都要教，重建像素 → 手臂對應)；None → 只算筆尖偏移
    """
    snapshot = grab(pipeline)                 # 手臂在 Home 拍的畫面，教點時當參考
    shape = snapshot.shape

    # 依顯示畫面位置重新命名、排序
    def disp_name(u, v):
        sx, sy = flip_pt((u, v), shape)
        return ("上" if sy < shape[0] / 2 else "下"), ("左" if sx < shape[1] / 2 else "右")
    order = {("上", "左"): 0, ("上", "右"): 1, ("下", "右"): 2, ("下", "左"): 3}
    items = []
    for _, src, u, v, x, y, _ in targets:
        ud, lr = disp_name(u, v)
        items.append((order[(ud, lr)], f"{lr}{ud}", src, u, v, x, y))
    items.sort()
    cx = np.mean([t[5] for t in items])
    cy = np.mean([t[6] for t in items])

    print("\n========== 示教器教點順序 (照轉正後的畫面 = 你實際看到的方向) ==========")
    for k, (_, name, src, u, v, x, y) in enumerate(items, 1):
        side = f"X {'大' if x > cx else '小'}、Y {'大' if y > cy else '小'}"
        print(f"  第 {k} 點：畫面{name} 的 Marker [{src}]  ({side})  預估 X={x:.2f}, Y={y:.2f}")
    if recalib:
        print("※ 完整重新校正：預估座標是用舊的對應算的，可能差很多，以畫面上 Marker 的實際位置為準。4 點都要教")
    print("用示教器把「筆尖」移到 Marker 正中心 (只看 XY，高度隨意)，在終端機按 Enter 記錄")
    print("  (終端機) Enter: 記錄   空白: 跳過   Esc: 結束")

    corrections, taught = [], []
    stopped = False
    last_poll, pose = 0.0, None
    for k, (_, name, src, u, v, x, y) in enumerate(items, 1):
        print(f"\n📍 第 {k}/4 點：畫面{name} [{src}]  預估 X={x:.2f}, Y={y:.2f}")
        flush_keys()                     # 上一點多按的 Enter 不要直接記錄這一點
        while True:
            # 左：Home 拍的參考畫面 (紅圈 = 這次要教的)；右：手臂上的即時畫面
            ref = snapshot.copy()
            for j, (_, _, _, uu, vv, _, _) in enumerate(items, 1):
                cv2.circle(ref, (int(uu), int(vv)), 22 if j == k else 10,
                           (0, 0, 255) if j == k else (0, 255, 255), 3 if j == k else 1)
            ref = to_display(ref)
            for j, (_, _, _, uu, vv, _, _) in enumerate(items, 1):
                sx, sy = disp_pt((uu, vv), shape)
                put_text(ref, str(j), (sx + 24, sy + 6), (0, 0, 255) if j == k else (0, 255, 255), 0.8)
            en = {"左上": "TOP-LEFT", "右上": "TOP-RIGHT", "右下": "BOTTOM-RIGHT", "左下": "BOTTOM-LEFT"}
            put_text(ref, f"TEACH {k}/4 : {en.get(name, name)}", (10, 25), (0, 0, 255), 0.7)

            live = to_display(grab(pipeline))
            if time.time() - last_poll > 0.3:
                last_poll = time.time()
                try:
                    pose = arm.pose()
                except Exception:
                    pose = None
            if pose:
                put_text(live, f"TCP X={pose[0]:.2f} Y={pose[1]:.2f} Z={pose[2]:.2f}", (10, 25), (0, 255, 0), 0.6)
            put_text(live, "Enter: record  Space: skip  Esc: stop", (10, live.shape[0] - 10), (0, 255, 0))
            cv2.imshow(win, np.hstack([ref, live]))      # 兩張都已經縮成 DISPLAY_SCALE
            key = poll_key(30)

            if key in (13, 10):
                p = arm.pose()
                # 防呆：筆尖離板子太高 = 手臂根本還沒移到 Marker (例如還在 Home)，不記錄
                if p[2] > Z_PAPER + TEACH_MAX_ABOVE_MM:
                    print(f"   ⚠️ 現在 Z={p[2]:.1f}，比板面高 {p[2] - Z_PAPER:.0f} mm，手臂還沒移到 Marker？"
                          f"沒有記錄。筆尖碰到 Marker 中心後再按 Enter")
                    continue
                # 防呆：跟已經教過的點太近 = 同一個位置按了兩次
                if any(math.hypot(p[0] - t[2], p[1] - t[3]) < 20 for t in taught):
                    print(f"   ⚠️ X={p[0]:.1f}, Y={p[1]:.1f} 跟前面教過的點幾乎一樣，沒有記錄。移到這一點的 Marker 再按 Enter")
                    continue
                dx, dy = p[0] - x, p[1] - y
                corrections.append((name, dx, dy))
                taught.append((u, v, p[0], p[1]))
                print(f"   ✅ 記錄 X={p[0]:.3f}, Y={p[1]:.3f}, Z={p[2]:.3f}  (跟預估差 ΔX {dx:+.2f}, ΔY {dy:+.2f})")
                break
            if key == 32:
                print("   跳過")
                break
            if key == 27:
                stopped = True
                break
        if stopped:
            break

    if recalib:
        if len(taught) == 4:
            save_calibration(taught, recalib)
        else:
            print(f"⚠️ 完整重新校正要 4 點都教 (只教了 {len(taught)} 點)，沒有存")
    elif corrections:
        save_tool_offset(corrections)
    ask("\n教點結束。用示教器把筆抬高、控制器切回遠端/自動模式後按 Enter，手臂會回 Home > ", None, {"": "Continue (Enter)"})
    arm.home()
    time.sleep(1.0)
    return not stopped


def recalibrate(arm, pipeline, win):
    """
    換了 Marker / 移了紙板 / 動了相機：在 Home 重新抓 4 個 Marker 的像素位置，
    用示教器讓筆尖碰每個 Marker 中心，重建 像素 → 手臂 的對應
    """
    arm.home()
    time.sleep(1.0)
    print("\n========== 完整重新校正：偵測 Marker ==========")
    r = detect_four(pipeline, win)
    if r is None:
        return False
    dict_name, pts = r
    names = ["左上", "右上", "右下", "左下"]
    targets = []
    for name, (mid, u, v) in zip(names, pts):
        x, y = pixel_to_arm(u, v)
        targets.append((name, f"ID {mid}", u, v, x, y, None))
        print(f"  {name} [ID {mid}] 像素 ({u:.1f}, {v:.1f})")
    return teach_with_pendant(arm, pipeline, win, targets, recalib=dict_name)


CALIB_SIZE_NOW = (CAM_W, CAM_H)   # 目前相機實際的解析度 (開相機後設定)


def repixel_calibration(arm, pipeline, win):
    """
    換了相機解析度：Marker 的手臂座標 (PTS_ARM) 沒變，只要在 Home 重新抓 4 個 Marker 的像素位置。
    前提是白板 / Marker / 相機都沒動過 (動過就要用 c 完整重新校正)
    """
    global CALIB_SIZE
    arm.home()
    time.sleep(1.0)
    print("\n========== 換解析度：重新抓 Marker 像素位置 ==========")
    r = detect_four(pipeline, win)
    if r is None:
        return False
    dict_name, pts = r
    px = np.float32([[u, v] for _, u, v in pts])     # 已排成 左上, 右上, 右下, 左下 (跟 PTS_ARM 同順序)
    sc = side_scales(px, PTS_ARM)
    print(f"  4 邊的 mm/px：{', '.join(f'{s:.3f}' for s in sc)}")
    if (max(sc) - min(sc)) / np.mean(sc) > 0.15:
        print("  ❌ 4 邊比例差太多：白板或 Marker 可能動過，請改用 c 完整重新校正")
        return False
    set_calibration(px, PTS_ARM)
    write_calib_json(pts_pixel=px.tolist(), pts_arm=PTS_ARM.tolist(), aruco_dict=dict_name,
                     image_size=list(CALIB_SIZE_NOW), h_source="aruco", h_matrix=None)
    CALIB_SIZE = CALIB_SIZE_NOW
    print(f"  💾 已更新 {CALIB_PATH} ({CALIB_SIZE_NOW[0]}x{CALIB_SIZE_NOW[1]}，平均 {MM_PER_PX:.3f} mm/px)。"
          "筆尖偏移保留，下一步建議按 Enter 檢查筆尖偏移")
    return True


def locate_aruco(arm, pipeline, win):
    """
    筆尖依序移到 4 個角的 Marker 中心上方，用鍵盤微調到正中心後按 Enter 記錄，
    4 點做完自動算出筆尖偏移。順序：畫面 左上 → 右上 → 右下 → 左下 (相機原始畫面)
    停住時 (方向是「顯示畫面」上的方向，跟你看到的實際方向一致)：
      i / k / j / l : 筆往 上 / 下 / 左 / 右 移     1 / 2 / 3 : 每步 0.5 / 2 / 5 mm
      Enter : 已對準，記錄並去下一個     空白 : 跳過這個 (不記錄)
      d : 下筆點一下     Esc : 結束定位
    """
    arm.home()
    time.sleep(1.0)
    found = detect_aruco(pipeline)
    names = ["左上", "右上", "右下", "左下"]
    targets = []
    for name, ref, ref_arm in zip(names, PTS_PIXEL, PTS_ARM):
        # 找離記錄位置最近的 Marker；找不到 (例如左下那個辨識不到) 就用記錄的像素
        near = [(math.hypot(u - ref[0], v - ref[1]), i, (u, v)) for i, (u, v) in found.items()]
        near = [n for n in near if n[0] < ARUCO_MATCH_PX]
        if near:
            _, mid, (u, v) = min(near)
            src = f"ID {mid}"
        else:
            u, v = float(ref[0]), float(ref[1])
            src = "記錄值 (沒偵測到)"
        x, y = pixel_to_arm(u, v)          # Marker 在紙面高度，不用視差修正
        targets.append((name, src, u, v, x, y, ref_arm))

    print("\n========== ArUco 定位 (畫面 左上 → 右上 → 右下 → 左下) ==========")
    for name, src, u, v, x, y, ref_arm in targets:
        print(f"  {name} [{src}] 像素 ({u:.1f}, {v:.1f}) → X={x:.2f}, Y={y:.2f}  "
              f"(記錄的 Marker 座標 {ref_arm[0]:.2f}, {ref_arm[1]:.2f}，差 "
              f"{math.hypot(x - TOOL_OFFSET_X - ref_arm[0], y - TOOL_OFFSET_Y - ref_arm[1]):.1f} mm)")
    if not path_ok([(t[4], t[5]) for t in targets]):
        print("⚠️ 有 Marker 超出工作範圍 X_RANGE / Y_RANGE，不執行定位")
        return False

    if TEACH_WITH_PENDANT:
        return teach_with_pendant(arm, pipeline, win, targets)

    z_hover = Z_PAPER + ARUCO_HOVER_MM
    corrections = []
    stopped = False
    for k, (name, src, u, v, x, y, _) in enumerate(targets, 1):
        print(f"\n📍 {k}/4 {name}：移到 Marker 中心上方 {ARUCO_HOVER_MM} mm (X={x:.2f}, Y={y:.2f})")
        arm.move(x, y, Z_TRAVEL)
        arm.move(x, y, z_hover, speed=SPEED_DRAW)
        print("   i/k/j/l: 筆往畫面上/下/左/右   1/2/3: 每步 0.5/2/5 mm   "
              "(終端機) Enter: 對準了記錄   空白: 跳過   d: 點一下   Esc: 結束")
        flush_keys()                       # 手臂移過來時多按的鍵不要算
        step = 2.0
        cx, cy = x, y                      # 已送出的目標位置
        pend = [0.0, 0.0]                  # 手臂還在動時按的鍵先累加，停下後一次走完
        moving = False

        def flush():
            """把累加的移動送出 (不等)；回傳是否有送"""
            nonlocal cx, cy, moving
            if pend == [0.0, 0.0]:
                return False
            nx, ny = cx + pend[0], cy + pend[1]
            pend[0] = pend[1] = 0.0
            if not path_ok([(nx, ny)]):
                print("   ⚠️ 超出工作範圍")
                return False
            arm.send_move(nx, ny, z_hover, speed=SPEED_JOG)
            cx, cy, moving = nx, ny, True
            return True

        while True:
            if moving and arm.is_idle():
                moving = False
            elif moving and time.time() - arm.t_cmd > JOG_TIMEOUT:
                # 到不了 (奇異點 / 超出可達範圍)：以手臂實際位置為準，繼續微調
                p = arm.pose()
                print(f"   ⚠️ {JOG_TIMEOUT:.0f} 秒沒到位，可能碰到奇異點或到不了，請看教導器是否有警報。"
                      f"目前位置 X={p[0]:.2f}, Y={p[1]:.2f}")
                cx, cy, moving = p[0], p[1], False
                pend[0] = pend[1] = 0.0
                arm.target = (p[0], p[1], p[2])
            if not moving:
                flush()

            live = to_display(grab(pipeline))
            state = "MOVING" if moving else "ready"
            put_text(live, f"ArUco {k}/4 {name}  dX={cx + pend[0] - x:+.1f} dY={cy + pend[1] - y:+.1f}  "
                           f"step {step}mm  {state}", (10, 25), (0, 255, 255), 0.55)
            put_text(live, "ijkl: move  1/2/3: step  Enter: record  Space: skip  d: dot  Esc: stop",
                     (10, live.shape[0] - 10), (0, 255, 0), 0.45)
            cv2.imshow(win, live)
            key = poll_key(1)

            if key in JOG_KEYS:
                ax, ay = screen_dir_to_arm(u, v, *JOG_KEYS[key])
                pend[0] += ax * step
                pend[1] += ay * step
            elif key in JOG_STEPS:
                step = JOG_STEPS[key]
            elif key in (13, 10, 32, ord('d'), 27):
                # 先把還沒走完的移動做完，位置才準
                flush()
                arm.wait_idle()
                moving = False
                if key in (13, 10):
                    corrections.append((name, cx - x, cy - y))
                    print(f"   ✅ 記錄：要再移 ΔX {cx - x:+.2f}, ΔY {cy - y:+.2f} mm 才對準")
                    break
                if key == 32:
                    print("   跳過")
                    break
                if key == ord('d'):
                    arm.move(cx, cy, Z_PAPER - PEN_PRESS_MM, speed=SPEED_DRAW)
                    time.sleep(0.3)
                    arm.move(cx, cy, z_hover, speed=SPEED_DRAW)
                if key == 27:
                    stopped = True
                    break
        arm.move(cx, cy, Z_TRAVEL)
        if stopped:
            break

    arm.home()
    time.sleep(1.0)
    if corrections:
        save_tool_offset(corrections)
    print("\n✅ ArUco 定位結束")
    return not stopped


# ==========================================
# 6b. 自主校正 (畫點)：不用示教器、不用對準 ArUco
#     手臂在板上自己畫十字 (位置 = 已知的手臂座標) → 回 Home 拍照找十字 (像素) →
#     很多組 像素 ↔ 筆尖手臂座標 用最小平方法重算 像素→手臂 的對應 (筆尖偏移自動包含在裡面)
# ==========================================
DOTCAL_GRID = (5, 4)            # 排幾行 x 幾列十字
DOTCAL_INSET = 0.12             # 在 4 個 Marker 圍成的範圍內排點，每邊往內縮幾成 (避開 Marker)
DOTCAL_CROSS_MM = 4.0           # 十字每一劃 (中心到端點) 幾 mm；每一劃來回畫一次，抵消毛筆拖尾
DOTCAL_DIFF = 35                # 畫完比畫之前暗多少 (灰階 0~255) 算墨跡
DOTCAL_SEARCH_MM = 15.0         # 在目前校正預測的位置附近幾 mm 內找十字 (舊校正偏太多就調大)
DOTCAL_MIN_PTS = 8              # 至少找到幾個十字才算
DOTCAL_WARN_MEAN_MM, DOTCAL_WARN_MAX_MM = 1.0, 2.5   # 誤差超過就警告


def grab_avg(pipeline, n=10):
    """連拍 n 張平均 (去雜訊)"""
    acc = sum(grab(pipeline).astype(np.float32) for _ in range(n))
    return (acc / n).astype(np.uint8)


DOTCAL_PAPER_MARGIN_MM = 5.0   # 十字離白紙邊緣至少幾 mm (再加上十字本身的長度)


def dotcal_targets(img):
    """
    排十字位置：[(預測像素 u, v, 手臂 X, Y)]，蛇形順序 (少走路)
    範圍 = 白紙 (find_paper_mask) 往內縮 (十字長度 + DOTCAL_PAPER_MARGIN_MM)，
    在它的外框內排 DOTCAL_GRID，落在白紙外的點丟掉 (白紙比 Marker 圍的範圍小時不會畫出去)
    """
    paper = find_paper_mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    if paper is None:
        return []
    r = int(round((DOTCAL_CROSS_MM + DOTCAL_PAPER_MARGIN_MM) / MM_PER_PX))
    safe = cv2.erode(paper, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    if ARUCO_EXCLUDE is not None and ARUCO_EXCLUDE.shape == safe.shape:
        safe[cv2.dilate(ARUCO_EXCLUDE, np.ones((r + 1, r + 1), np.uint8)) > 0] = 0
    ys, xs = np.nonzero(safe)
    if len(xs) == 0:
        return []
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    nx, ny = DOTCAL_GRID
    out = []
    for j in range(ny):
        v = y0 + (y1 - y0) * j / max(ny - 1, 1)
        cols = range(nx) if j % 2 == 0 else range(nx - 1, -1, -1)
        for i in cols:
            u = x0 + (x1 - x0) * i / max(nx - 1, 1)
            if not safe[int(round(v)), int(round(u))]:
                continue
            x, y = pixel_to_arm(u, v, 0.0)
            out.append((float(u), float(v), float(x), float(y)))
    return out


def label_points(img, pts, labels, color, radius=0):
    """在顯示用畫面 (已轉向、縮放) 上標記：相機像素 pts → 顯示座標，字不會倒過來"""
    out = to_display(img)
    for (u, v), lab in zip(pts, labels):
        sx, sy = disp_pt((u, v), img.shape)
        if radius:
            cv2.circle(out, (sx, sy), radius, color, 2)
        else:
            cv2.drawMarker(out, (sx, sy), color, cv2.MARKER_CROSS, 14, 2)
        put_text(out, lab, (sx + 6, sy - 6), color, 0.45)
    return out


def draw_cross(arm, x, y, z_draw, z_up):
    """畫一個十字：X 方向、Y 方向各一劃，每一劃 去 → 回 (毛筆往前拖的偏差互相抵消)"""
    a = DOTCAL_CROSS_MM
    for dx, dy in ((1, 0), (0, 1)):
        x0, y0, x1, y1 = x - a * dx, y - a * dy, x + a * dx, y + a * dy
        arm.move(x0, y0, z_up)
        arm.move(x0, y0, z_draw, speed=SPEED_DRAW)
        arm.move(x1, y1, z_draw, speed=SPEED_DRAW, linear=True)
        arm.move(x0, y0, z_draw, speed=SPEED_DRAW, linear=True)
        arm.move(x0, y0, z_up, speed=SPEED_DRAW)


def find_marks(before, after, targets):
    """
    畫之前 / 之後相減找墨跡，每個目標在預測位置 DOTCAL_SEARCH_MM 內找最近的一塊，
    中心 = 用「變暗程度」加權的重心 (亞像素)。回傳 ([(u, v, X, Y)], 墨跡遮罩)
    """
    g0 = cv2.GaussianBlur(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (kpx(3), kpx(3)), 0).astype(np.int16)
    g1 = cv2.GaussianBlur(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), (kpx(3), kpx(3)), 0).astype(np.int16)
    diff = np.clip(g0 - g1, 0, 255).astype(np.uint8)
    m = (diff > DOTCAL_DIFF).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((kpx(3), kpx(3)), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((kpx(5), kpx(5)), np.uint8))   # 十字中間斷掉補起來
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
    r = DOTCAL_SEARCH_MM / MM_PER_PX
    min_area = 8 * PX_SCALE ** 2
    found, used = [], set()
    for u, v, x, y in targets:
        best = None
        for k in range(1, n):
            if k in used or stats[k, cv2.CC_STAT_AREA] < min_area:
                continue
            dd = math.hypot(cents[k][0] - u, cents[k][1] - v)
            if dd < r and (best is None or dd < best[0]):
                best = (dd, k)
        if best is None:
            continue
        k = best[1]
        used.add(k)
        x0, y0 = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        sub = lab[y0:y0 + h, x0:x0 + w] == k
        wts = diff[y0:y0 + h, x0:x0 + w].astype(np.float64) * sub
        ys, xs = np.mgrid[y0:y0 + h, x0:x0 + w]
        sw = wts.sum()
        found.append((float((xs * wts).sum() / sw), float((ys * wts).sum() / sw), x, y))
    return found, m


def fit_dotcal(found):
    """[(u, v, X, Y)] → (H, 每點誤差 mm, 有用到的點 (RANSAC inlier))"""
    px = np.float32([[f[0], f[1]] for f in found])
    am = np.float32([[f[2], f[3]] for f in found])
    H, inl = cv2.findHomography(px, am, cv2.RANSAC, 1.5)
    if H is None:
        return None, None, None
    use = inl.ravel().astype(bool)
    H, _ = cv2.findHomography(px[use], am[use])                  # 用 inlier 再做一次最小平方法
    pred = cv2.perspectiveTransform(px[None], H)[0]
    err = np.linalg.norm(pred - am, axis=1)
    return H, err, use


def dot_calibrate(arm, pipeline, win):
    """
    自主校正：手臂畫十字 → 相機找 → 重算 像素→手臂。成功並存檔回傳 True
    板面要清空 (工件拿開)，畫完要把十字擦掉 (不然會被當成物體)
    """
    global TOOL_OFFSET_X, TOOL_OFFSET_Y
    z_draw, z_up = Z_PAPER - PEN_PRESS_MM, Z_PAPER + Z_LIFT
    arm.home()
    time.sleep(1.0)
    before = grab_avg(pipeline)
    targets = dotcal_targets(before)
    if len(targets) < DOTCAL_MIN_PTS:
        print(f"⚠️ 白紙範圍內只排得下 {len(targets)} 個十字 (至少 {DOTCAL_MIN_PTS})：白紙太小或沒偵測到白紙")
        return False
    if not path_ok([(t[2], t[3]) for t in targets]):
        print("⚠️ 有十字超出工作範圍 X_RANGE / Y_RANGE，不執行")
        return False
    print(f"\n========== 自主校正 (畫點) ==========\n"
          f"  手臂會在白紙上畫 {len(targets)} 個十字 (每劃 ±{DOTCAL_CROSS_MM} mm，Z={z_draw:.2f})，"
          f"再回 Home 拍照算出新的 像素→手臂 對應 (筆尖偏移一起包含)")
    show(win, label_points(before, [(t[0], t[1]) for t in targets],
                           [str(i) for i in range(1, len(targets) + 1)], (0, 0, 255)))
    if ask("  紅色十字 = 要畫的位置。白板清空 (工件拿開) 了嗎？ y: 開始畫   n: 取消 > ", ("y", "n"),
           {"y": "Start (y)", "n": "Cancel (n)"}) != "y":
        print("  已取消")
        return False

    try:
        for i, (_, _, x, y) in enumerate(targets, 1):
            print(f"\r  畫十字 {i}/{len(targets)} @ ({x:.1f}, {y:.1f})   (Esc 中止)", end="", flush=True)
            draw_cross(arm, x, y, z_draw, z_up)
            if poll_key(1) == 27:
                raise UserStop("中止")
        print()
    except UserStop:
        print("\n  已中止")
        arm.move(*arm.pose()[:2], Z_TRAVEL)
        arm.home()
        return False
    arm.move(*arm.pose()[:2], Z_TRAVEL)
    arm.home()
    time.sleep(1.0)
    after = grab_avg(pipeline)

    found, ink = find_marks(before, after, targets)
    print(f"  找到 {len(found)}/{len(targets)} 個十字")
    if len(found) < DOTCAL_MIN_PTS:
        print(f"  ❌ 太少 (至少 {DOTCAL_MIN_PTS} 個)：墨跡太淡 → 調低 DOTCAL_DIFF；"
              f"位置偏太多 → 調大 DOTCAL_SEARCH_MM (現在 {DOTCAL_SEARCH_MM} mm)")
        show(win, to_display(cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR)))
        ask("  (畫面是找到的墨跡) 按 Enter 繼續 > ", None, {"": "Continue (Enter)"})
        return False
    H, err, use = fit_dotcal(found)
    if H is None:
        print("  ❌ 算不出對應")
        return False
    # 舊校正的誤差 (畫的位置 vs 舊對應預測的位置)：看改善多少
    old_err = [math.hypot(*(np.array(pixel_to_arm(f[0], f[1], 0.0)) - (f[2], f[3]))) for f in found]
    mean_e, max_e = float(err[use].mean()), float(err[use].max())
    print(f"  使用 {int(use.sum())}/{len(found)} 點，新對應誤差：平均 {mean_e:.2f} mm、最大 {max_e:.2f} mm "
          f"(舊的：平均 {np.mean(old_err):.2f} mm、最大 {np.max(old_err):.2f} mm)")
    ov = to_display(after)
    for (u, v, _, _), e, ok in zip(found, err, use):
        col = (128, 128, 128) if not ok else (0, 200, 0) if e < 1.0 else (0, 200, 255) if e < 2.0 else (0, 0, 255)
        sx, sy = disp_pt((u, v), after.shape)
        cv2.circle(ov, (sx, sy), 8, col, 2)
        put_text(ov, f"{e:.1f}", (sx + 10, sy + 4), col, 0.45)
    show(win, ov)
    if mean_e > DOTCAL_WARN_MEAN_MM or max_e > DOTCAL_WARN_MAX_MM:
        print(f"  ⚠️ 誤差偏大 (平均 > {DOTCAL_WARN_MEAN_MM} 或最大 > {DOTCAL_WARN_MAX_MM} mm)：筆有沒有夾穩？白板有沒有動？")
    if ask("  (畫面：綠 < 1 mm、黃 < 2 mm、紅 = 偏大、灰 = 沒用到) 要套用並存檔嗎？ y: 存   n: 不存 > ",
           ("y", "n"), {"y": "Save (y)", "n": "Discard (n)"}) != "y":
        print("  沒有存")
        return False
    use_h_matrix(H)
    TOOL_OFFSET_X = TOOL_OFFSET_Y = 0.0
    write_calib_json(h_matrix=H.tolist(), h_source="dots", tool_offset_x=0.0, tool_offset_y=0.0,
                     image_size=list(CALIB_SIZE_NOW),
                     dotcal={"n": int(use.sum()), "mean_mm": mean_e, "max_mm": max_e,
                             "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "pts_pixel": [[f[0], f[1]] for f in found], "pts_arm": [[f[2], f[3]] for f in found]})
    print(f"  💾 已存到 {CALIB_PATH} (自主校正，筆尖偏移歸 0)。⚠️ 請把白板上的十字擦掉，再繼續偵測工件")
    return True


# ==========================================
# 6c. 2-1 工件幾何資訊 + 2-4 畫線品質驗證
# ==========================================
GEOM_DIR = os.path.join(DATA_DIR, "geometry")
QUALITY_DIR = os.path.join(DATA_DIR, "quality")
QUALITY_LOG = os.path.join(QUALITY_DIR, "log.csv")    # 每次描邊一列 (3-4 整合測試用)
CIRCLE_TOL_MM = 1.0         # 外框 95% 的點離擬合圓多少 mm 內才算「圓」
FLAT_SLOPE_DEG = 3.0        # 頂面斜角小於幾度算平的
VERIFY_STEP_MM = 1.0        # 沿路徑每幾 mm 量一次
VERIFY_HALF_MM = 8.0        # 橫切面往兩側各量幾 mm (要比偏差 + 半個線寬大)
VERIFY_RES_MM = 0.25        # 橫切面取樣間距
VERIFY_DIFF = 30            # 描完比描之前暗多少 (灰階) 算墨跡
VERIFY_OK_MM, VERIFY_WARN_MM = 1.0, 2.0     # 上色：綠 < 1 mm、黃 < 2 mm、紅
VERIFY_BIAS_MIN_MM = 0.3    # 整體偏移超過這個才建議修正筆尖偏移
LAST_GEOM = None            # 上一次量的幾何資訊
LAST_RUN = None             # 上一次描邊 {"before", "path_arm", "geom", "mode"}，v 重新驗證用


def arms_to_pixels(x, y, h):
    """arm_to_pixel 的陣列版：手臂 X, Y (高度 h，陣列) → 相機像素 (u, v)"""
    Hi = np.linalg.inv(H_MATRIX)
    X = np.asarray(x, float) - TOOL_OFFSET_X
    Y = np.asarray(y, float) - TOOL_OFFSET_Y
    w = Hi[2, 0] * X + Hi[2, 1] * Y + Hi[2, 2]
    u = (Hi[0, 0] * X + Hi[0, 1] * Y + Hi[0, 2]) / w
    v = (Hi[1, 0] * X + Hi[1, 1] * Y + Hi[1, 2]) / w
    k = 1.0 - np.asarray(h, float) / CAM_HEIGHT_MM
    return CAM_CENTER_PX[0] + (u - CAM_CENTER_PX[0]) / k, CAM_CENTER_PX[1] + (v - CAM_CENTER_PX[1]) / k


def _poly_area(p):
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


# ---------- 2-1 幾何資訊 ----------
def measure_geometry():
    """
    用上一次的俯視高度圖 (LAST_HMAP，auto 模式有深度時才有) 算工件幾何。回傳 dict；沒有高度圖回傳 None
      位置：中心 X/Y、擺放角度 (最小外接矩形)
      尺寸：長、寬、周長、面積
      形狀：多邊形 → 邊數、邊長、內角；圓 → 半徑、圓度；其他 → 曲線
      高度：最高 / 最低 / 平均；頂面擬合平面 → 斜角、往哪邊變高；體積
    """
    global LAST_GEOM
    if not LAST_HMAP:
        return None
    hm, g = LAST_HMAP["hmap"], LAST_HMAP["g"]
    xy, h_out = LAST_HMAP["outline"], LAST_HMAP["h"]
    occ = LAST_HMAP["occ"] > 0
    cells = occ & (hm > 0)
    rows, cols = np.nonzero(cells)
    X = LAST_HMAP["x0"] + (cols + 0.5) * g
    Y = LAST_HMAP["y0"] + (rows + 0.5) * g
    Hc = hm[rows, cols].astype(float)

    geo = {"time": time.strftime("%Y-%m-%d %H:%M:%S")}
    area = abs(_poly_area(xy))
    perim = float(np.sum(np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)))
    m = cv2.moments(np.float32(xy).reshape(-1, 1, 2))
    cx, cy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else tuple(xy.mean(axis=0))
    (_, _), (rw, rh), ang = cv2.minAreaRect(np.float32(xy))
    if rw < rh:
        rw, rh, ang = rh, rw, ang + 90.0
    geo.update(center_x=float(cx), center_y=float(cy), length=float(rw), width=float(rh),
               angle_deg=float((ang + 90) % 180 - 90), area_mm2=float(area), perimeter=perim)

    verts = LAST_HMAP.get("verts")
    if verts is not None and len(verts) >= 3:
        n = len(verts)
        sides = [float(np.linalg.norm(verts[(k + 1) % n] - verts[k])) for k in range(n)]
        angs = []
        for k in range(n):
            a, b = verts[k - 1] - verts[k], verts[(k + 1) % n] - verts[k]
            cosv = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))
            angs.append(math.degrees(math.acos(max(-1.0, min(1.0, cosv)))))
        geo.update(shape="polygon", n_vertices=n, sides=sides, angles=angs,
                   vertices=[[float(v[0]), float(v[1])] for v in verts])
    else:
        A = np.column_stack([2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))])
        c, *_ = np.linalg.lstsq(A, xy[:, 0] ** 2 + xy[:, 1] ** 2, rcond=None)
        r = math.sqrt(max(c[2] + c[0] ** 2 + c[1] ** 2, 0.0))
        dev = np.abs(np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1]) - r)
        if np.percentile(dev, 95) < CIRCLE_TOL_MM:
            geo.update(shape="circle", radius=float(r), diameter=float(2 * r), circle_x=float(c[0]),
                       circle_y=float(c[1]), roundness_mm=float(np.percentile(dev, 95)))
        else:
            geo.update(shape="curve")

    if len(Hc):
        geo.update(h_max=float(np.percentile(Hc, 98)), h_min=float(np.percentile(h_out, 2)),
                   h_mean=float(Hc.mean()), volume_cm3=float(Hc.sum() * g * g / 1000.0))
        # 頂面平面：去掉外框附近 (側面 / 邊緣深度不準)，擬合 h = a·X + b·Y + c，丟離群再擬合
        inner = cv2.erode(occ.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        sel = inner[rows, cols]
        if sel.sum() >= 30:
            P = np.column_stack([X[sel], Y[sel], np.ones(sel.sum())])
            z = Hc[sel]
            ok = np.ones(len(z), bool)
            for _ in range(3):
                coef, *_ = np.linalg.lstsq(P[ok], z[ok], rcond=None)
                res = np.abs(P @ coef - z)
                ok = res < max(3.0 * np.median(res[ok]) * 1.4826, 1.0)
            a, b = coef[0], coef[1]
            slope = math.degrees(math.atan(math.hypot(a, b)))
            geo.update(top_slope_deg=float(slope), top_rms_mm=float(np.sqrt(np.mean(res[ok] ** 2))),
                       top_flat=bool(slope < FLAT_SLOPE_DEG),
                       top_rise_dir_deg=float(math.degrees(math.atan2(b, a))) if slope >= FLAT_SLOPE_DEG else None)
    LAST_GEOM = geo
    return geo


def print_geometry(geo):
    if not geo:
        print("  (沒有高度圖，量不到幾何資訊：要 auto 模式 + 深度)")
        return
    print(f"\n========== 工件幾何 (2-1) ==========")
    print(f"  中心 X={geo['center_x']:.1f}, Y={geo['center_y']:.1f}，擺放角度 {geo['angle_deg']:+.1f}°")
    print(f"  外框 長 {geo['length']:.1f} × 寬 {geo['width']:.1f} mm，周長 {geo['perimeter']:.1f} mm，"
          f"面積 {geo['area_mm2']:.0f} mm²")
    if geo.get("shape") == "polygon":
        print(f"  形狀：{geo['n_vertices']} 邊形")
        for k, (sd, an) in enumerate(zip(geo["sides"], geo["angles"]), 1):
            print(f"    邊 {k}: {sd:.1f} mm   頂點 {k} 內角 {an:.1f}°")
    elif geo.get("shape") == "circle":
        print(f"  形狀：圓，直徑 {geo['diameter']:.1f} mm (半徑 {geo['radius']:.1f})，圓度 {geo['roundness_mm']:.2f} mm")
    else:
        print("  形狀：曲線 (不是多邊形也不是圓)")
    if "h_max" in geo:
        print(f"  高度：最高 {geo['h_max']:.1f}、外框最低 {geo['h_min']:.1f}、平均 {geo['h_mean']:.1f} mm，"
              f"體積約 {geo['volume_cm3']:.1f} cm³")
    if "top_slope_deg" in geo:
        if geo["top_flat"]:
            print(f"  頂面：平面 (斜角 {geo['top_slope_deg']:.1f}°)")
        else:
            print(f"  頂面：斜面 {geo['top_slope_deg']:.1f}°，往手臂座標 {geo['top_rise_dir_deg']:.0f}° 方向變高"
                  + (f"  (平面擬合誤差 {geo['top_rms_mm']:.1f} mm，頂面可能不是單一平面)" if geo["top_rms_mm"] > 3 else ""))


def draw_geometry(frame, geo):
    """幾何標註圖 (顯示方向)：外框、邊長、中心，左上角摘要"""
    out = to_display(frame)
    if not geo or not LAST_HMAP:
        return out
    xy, h = LAST_HMAP["outline"], LAST_HMAP["h"]
    U, V = arms_to_pixels(xy[:, 0], xy[:, 1], h)
    pts = np.int32([disp_pt((u, v), frame.shape) for u, v in zip(U, V)])
    cv2.polylines(out, [pts], True, (0, 255, 255), 2)
    hmean = float(np.median(h))
    if geo.get("shape") == "polygon":
        vs = np.array(geo["vertices"])
        for k in range(len(vs)):
            a, b = vs[k], vs[(k + 1) % len(vs)]
            mid = (a + b) / 2
            u, v = arm_to_pixel(mid[0], mid[1], hmean)
            put_text(out, f"{geo['sides'][k]:.1f}", disp_pt((u, v), frame.shape), (0, 255, 255), 0.5)
    u, v = arm_to_pixel(geo["center_x"], geo["center_y"], hmean)
    cv2.drawMarker(out, disp_pt((u, v), frame.shape), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
    lines = [f"L {geo['length']:.1f} x W {geo['width']:.1f} mm  angle {geo['angle_deg']:+.1f}"]
    if geo.get("shape") == "circle":
        lines.append(f"circle D {geo['diameter']:.1f} mm  roundness {geo['roundness_mm']:.2f}")
    elif geo.get("shape") == "polygon":
        lines.append(f"{geo['n_vertices']}-gon")
    if "h_max" in geo:
        lines.append(f"H {geo['h_min']:.1f}~{geo['h_max']:.1f} mm  vol {geo['volume_cm3']:.1f} cm3")
    if "top_slope_deg" in geo:
        lines.append("top flat" if geo["top_flat"] else f"top slope {geo['top_slope_deg']:.1f} deg")
    for i, t in enumerate(lines):
        put_text(out, t, (10, 45 + 20 * i), (0, 255, 255), 0.5)
    return out


def save_geometry(frame, geo):
    os.makedirs(GEOM_DIR, exist_ok=True)
    base = os.path.join(GEOM_DIR, time.strftime("%Y%m%d_%H%M%S"))
    with open(base + ".json", "w", encoding="utf-8") as fp:
        json.dump(geo, fp, ensure_ascii=False, indent=2)
    imwrite_u(base + ".png", draw_geometry(frame, geo))
    print(f"  💾 幾何資訊已存到 {base}.json / .png")


# ---------- 2-4 畫線品質 ----------
def verify_drawing(before, after, path_arm, overlap_mm=None):
    """
    描之前 / 之後相減找墨跡，沿規劃路徑每 VERIFY_STEP_MM 做一條橫切面 (法線方向 ±VERIFY_HALF_MM)：
      墨跡中心 (用變暗程度加權) → 偏差 (+ 往外 / - 往內)；墨跡寬 → 線寬；整條沒墨跡 → 漏畫
    整體偏移 = 用最小平方法找一個 XY 平移最能解釋所有偏差 (= 校正 / 筆尖偏移的誤差)
    回傳 (結果 dict, 每個取樣點的資料)
    """
    overlap_mm = GLUE_OVERLAP_MM if overlap_mm is None else overlap_mm
    pts = np.array([[p[0], p[1]] for p in path_arm], float)
    hs = np.array([p[2] if len(p) > 2 else (obj_height() if DRAW_SURFACE == "top" else 0.0) for p in path_arm])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    s_end = max(total - overlap_mm, 0.5 * total)            # 頭尾重疊段不重複算
    ss = np.arange(0.0, s_end, VERIFY_STEP_MM)
    px, py = np.interp(ss, s, pts[:, 0]), np.interp(ss, s, pts[:, 1])
    ph = np.interp(ss, s, hs)
    idx = np.clip(np.searchsorted(s, ss, side="right") - 1, 0, len(seg) - 1)
    t = (pts[idx + 1] - pts[idx]) / np.maximum(seg[idx], 1e-9)[:, None]
    nrm = np.column_stack([t[:, 1], -t[:, 0]])               # 逆時針時是往外
    if _poly_area(pts) < 0:
        nrm = -nrm
    offs = np.arange(-VERIFY_HALF_MM, VERIFY_HALF_MM + 1e-9, VERIFY_RES_MM)
    QX = px[:, None] + offs[None, :] * nrm[:, :1]
    QY = py[:, None] + offs[None, :] * nrm[:, 1:]
    U, V = arms_to_pixels(QX, QY, np.repeat(ph[:, None], len(offs), axis=1))
    k = (kpx(3), kpx(3))
    g0 = cv2.GaussianBlur(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), k, 0).astype(np.float32)
    g1 = cv2.GaussianBlur(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), k, 0).astype(np.float32)
    dark = np.clip(g0 - g1, 0, 255).astype(np.float32)
    samp = cv2.remap(dark, U.astype(np.float32), V.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    off = np.full(len(ss), np.nan)
    width = np.full(len(ss), np.nan)
    for i in range(len(ss)):
        row = samp[i]
        j = int(np.argmax(row))
        if row[j] <= VERIFY_DIFF:
            continue                                         # 漏畫
        a = j
        while a > 0 and row[a - 1] > VERIFY_DIFF:
            a -= 1
        b = j
        while b < len(row) - 1 and row[b + 1] > VERIFY_DIFF:
            b += 1
        w = row[a:b + 1]
        off[i] = float((offs[a:b + 1] * w).sum() / w.sum())
        width[i] = (b - a + 1) * VERIFY_RES_MM

    found = np.isfinite(off)
    res = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "samples": int(len(ss)), "found": int(found.sum()),
           "coverage": float(found.mean()) if len(ss) else 0.0, "path_length_mm": float(s_end)}
    if found.sum() >= 5:
        o = off[found]
        n_f = nrm[found]
        bias, *_ = np.linalg.lstsq(n_f, o, rcond=None)      # 最能解釋偏差的 XY 平移
        resid = o - n_f @ bias
        res.update(mean_abs_mm=float(np.mean(np.abs(o))), rms_mm=float(np.sqrt(np.mean(o ** 2))),
                   max_abs_mm=float(np.max(np.abs(o))), p95_abs_mm=float(np.percentile(np.abs(o), 95)),
                   mean_signed_mm=float(np.mean(o)), bias_x_mm=float(bias[0]), bias_y_mm=float(bias[1]),
                   rms_after_bias_mm=float(np.sqrt(np.mean(resid ** 2))),
                   width_mean_mm=float(np.median(width[found])), width_std_mm=float(np.std(width[found])),
                   width_min_mm=float(np.min(width[found])), width_max_mm=float(np.max(width[found])))
    # 漏畫的連續段
    gaps, run = [], 0
    for f in list(found) + [True]:
        if not f:
            run += 1
        elif run:
            gaps.append(run * VERIFY_STEP_MM)
            run = 0
    res["gaps_mm"] = gaps
    detail = {"x": px, "y": py, "h": ph, "nrm": nrm, "off": off, "width": width}
    return res, detail


def print_quality(res):
    print(f"\n========== 畫線品質 (2-4) ==========")
    print(f"  量了 {res['samples']} 處 (每 {VERIFY_STEP_MM} mm)，有墨跡 {res['found']} 處，覆蓋率 {res['coverage']:.0%}"
          + (f"，漏畫 {len(res['gaps_mm'])} 段 (最長 {max(res['gaps_mm']):.0f} mm)" if res["gaps_mm"] else ""))
    if "mean_abs_mm" not in res:
        print("  ⚠️ 幾乎找不到墨跡：墨太淡 → 調低 VERIFY_DIFF；或描之前 / 之後工件動過")
        return
    print(f"  位置偏差：平均 {res['mean_abs_mm']:.2f}、RMS {res['rms_mm']:.2f}、95% {res['p95_abs_mm']:.2f}、"
          f"最大 {res['max_abs_mm']:.2f} mm (平均帶正負 {res['mean_signed_mm']:+.2f}：+ 往外 / - 往內)")
    print(f"  整體偏移 X {res['bias_x_mm']:+.2f}, Y {res['bias_y_mm']:+.2f} mm → 扣掉後剩 RMS {res['rms_after_bias_mm']:.2f} mm")
    print(f"  線寬：中位數 {res['width_mean_mm']:.2f} ± {res['width_std_mm']:.2f} mm "
          f"(最細 {res['width_min_mm']:.2f}、最粗 {res['width_max_mm']:.2f})")


def draw_quality(after, res, det, shape):
    """品質標註圖 (顯示方向)：每個取樣點畫在墨跡中心，依偏差上色；漏畫 = 灰色 (畫在規劃位置)"""
    out = to_display(after)
    for i in range(len(det["x"])):
        o = det["off"][i]
        if np.isfinite(o):
            x, y = det["x"][i] + o * det["nrm"][i, 0], det["y"][i] + o * det["nrm"][i, 1]
            a = abs(o)
            col = (0, 200, 0) if a < VERIFY_OK_MM else (0, 200, 255) if a < VERIFY_WARN_MM else (0, 0, 255)
        else:
            x, y, col = det["x"][i], det["y"][i], (160, 160, 160)
        u, v = arm_to_pixel(x, y, det["h"][i])
        cv2.circle(out, disp_pt((u, v), shape), 2, col, -1)
    lines = [f"coverage {res['coverage']:.0%}"]
    if "mean_abs_mm" in res:
        lines += [f"error mean {res['mean_abs_mm']:.2f}  max {res['max_abs_mm']:.2f} mm",
                  f"bias X {res['bias_x_mm']:+.2f} Y {res['bias_y_mm']:+.2f} mm",
                  f"width {res['width_mean_mm']:.2f} +- {res['width_std_mm']:.2f} mm"]
    for i, t in enumerate(lines):
        put_text(out, t, (10, 45 + 20 * i), (0, 255, 255), 0.5)
    put_text(out, "green<1mm  yellow<2mm  red>2mm  gray=missing", (10, out.shape[0] - 10), (255, 255, 255), 0.45)
    return out


def log_run(res, geo, mode):
    """3-4 整合測試：每次描邊附加一列到 data/quality/log.csv"""
    import csv
    os.makedirs(QUALITY_DIR, exist_ok=True)
    geo = geo or {}
    row = {"time": res["time"], "path_mode": mode, "shape": geo.get("shape", ""),
           "length": geo.get("length"), "width": geo.get("width"), "h_max": geo.get("h_max"),
           "top_slope_deg": geo.get("top_slope_deg"), "coverage": res.get("coverage"),
           "err_mean_mm": res.get("mean_abs_mm"), "err_max_mm": res.get("max_abs_mm"), "err_rms_mm": res.get("rms_mm"),
           "bias_x_mm": res.get("bias_x_mm"), "bias_y_mm": res.get("bias_y_mm"),
           "line_width_mm": res.get("width_mean_mm"), "line_width_std_mm": res.get("width_std_mm"),
           "speed_pct": RL_SPEED_GLUE if USE_RL_PATH else SPEED_GLUE, "pen_press_mm": PEN_PRESS_MM}
    new = not os.path.exists(QUALITY_LOG)
    with open(QUALITY_LOG, "a", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in row.items()})


def run_verify(pipeline, win, run):
    """描完 (手臂已回 Home)：拍照 → 驗證 → 存檔 → 有整體偏移就問要不要修正筆尖偏移"""
    global TOOL_OFFSET_X, TOOL_OFFSET_Y
    after = grab_avg(pipeline)
    res, det = verify_drawing(run["before"], after, run["path_arm"])
    print_quality(res)
    img = draw_quality(after, res, det, after.shape)
    show(win, img)
    folder = os.path.join(QUALITY_DIR, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    imwrite_u(os.path.join(folder, "before.png"), run["before"])
    imwrite_u(os.path.join(folder, "after.png"), after)
    imwrite_u(os.path.join(folder, "quality.png"), img)
    with open(os.path.join(folder, "result.json"), "w", encoding="utf-8") as fp:
        json.dump({"quality": res, "geometry": run.get("geom"), "path_mode": run.get("mode"),
                   "path_arm": [list(p) for p in run["path_arm"]]}, fp, ensure_ascii=False, indent=2)
    log_run(res, run.get("geom"), run.get("mode"))
    print(f"  💾 已存到 {folder} (並加一列到 {QUALITY_LOG})")
    if "bias_x_mm" in res and math.hypot(res["bias_x_mm"], res["bias_y_mm"]) >= VERIFY_BIAS_MIN_MM:
        nx_, ny_ = TOOL_OFFSET_X - res["bias_x_mm"], TOOL_OFFSET_Y - res["bias_y_mm"]
        if ask(f"  整體偏了 X {res['bias_x_mm']:+.2f}, Y {res['bias_y_mm']:+.2f} mm。要修正筆尖偏移嗎？"
               f"(X {TOOL_OFFSET_X:+.2f} → {nx_:+.2f}，Y {TOOL_OFFSET_Y:+.2f} → {ny_:+.2f}) y: 修正   n: 不要 > ",
               ("y", "n"), {"y": "Apply offset (y)", "n": "No (n)"}) == "y":
            TOOL_OFFSET_X, TOOL_OFFSET_Y = float(nx_), float(ny_)
            write_calib_json(tool_offset_x=TOOL_OFFSET_X, tool_offset_y=TOOL_OFFSET_Y)
            print(f"  已修正並存檔：筆尖偏移 X={TOOL_OFFSET_X:+.2f}, Y={TOOL_OFFSET_Y:+.2f} mm")
    else:
        ask("  按 Enter 回預覽 > ", None, {"": "Back (Enter)"})
    return res


def dot_at(arm, x, y):
    """在 (x, y) 點一個點，用來檢查偏移量"""
    print(f"• 點一個點 @ ({x:.2f}, {y:.2f})")
    arm.move(x, y, Z_TRAVEL)
    arm.move(x, y, Z_PAPER + Z_LIFT)                              # 點在板面上
    arm.move(x, y, Z_PAPER - PEN_PRESS_MM, speed=SPEED_DRAW)
    time.sleep(0.3)
    arm.move(x, y, Z_PAPER + Z_LIFT, speed=SPEED_DRAW)
    arm.move(x, y, Z_TRAVEL)


# ==========================================
# 7. 畫面
# ==========================================
CAM_CFG = None      # main() 開相機時存起來，斷流時用來重新啟動
DEPTH_ON = False    # 深度串流有開成功
ALIGN = None        # 深度對齊到彩色 (同一個像素 = 同一個點)
DEPTH_SCALE = 0.001 # 深度單位 → 公尺
LAST_DEPTH = None   # 最近一次 grab 的深度圖 (mm，float32，0 = 量不到)


# 以 640x480 調好的 px 參數：開相機後依 PX_SCALE 放大 (線性的乘一次，面積的乘平方)
_PX_LINEAR = ["HEIGHT_INNER_ERODE_PX", "HEIGHT_RING_PX", "DRAW_OFFSET_PX", "VERTEX_EPS_PX",
              "VERTEX_MIN_SIDE_PX", "VERTEX_FIT_TOL_PX", "VERTEX_MAX_SHIFT_PX", "SMOOTH_PX",
              "MASK_BLUR_PX", "ARUCO_MATCH_PX", "PAPER_MARGIN", "ARUCO_EXCLUDE_PAD_PX", "BG_SHIFT_WARN_PX",
              "RIM_ERODE_PX"]
_PX_AREA = ["MIN_AREA", "MAX_AREA"]
_PX_INT = {"HEIGHT_INNER_ERODE_PX", "HEIGHT_RING_PX", "SMOOTH_PX", "PAPER_MARGIN", "ARUCO_EXCLUDE_PAD_PX"}
_PX_BASE = {}


def apply_px_scale(scale):
    """把 px 參數換成目前解析度 (以 640x480 的原始值為準，重複呼叫不會一直放大)"""
    global PX_SCALE
    PX_SCALE = scale
    g = globals()
    for name in _PX_LINEAR + _PX_AREA:
        _PX_BASE.setdefault(name, g[name])
        base = _PX_BASE[name]
        f = scale * scale if name in _PX_AREA else scale

        def conv(v):
            v = v * f
            return int(round(v)) if name in _PX_INT else v
        g[name] = tuple(conv(v) for v in base) if isinstance(base, tuple) else conv(base)
    if SMOOTH_PX > 1 and SMOOTH_PX % 2 == 0:           # 平滑視窗要奇數
        g["SMOOTH_PX"] = SMOOTH_PX + 1


def kpx(n):
    """函式裡寫死的 kernel 大小 (640x480 時是 n)：換成目前解析度的奇數大小"""
    m = max(1, int(round(n * PX_SCALE)))
    return m if m % 2 else m + 1


def start_camera(pipeline):
    """
    開相機 (彩色；AUTO_HEIGHT 時加開深度)，印出 USB 版本 (USB 2.x 頻寬不夠，容易 Frame didn't arrive)
    深度開不起來 → 只用彩色，高度改用 OBJ_HEIGHT_MM
    """
    global CAM_CFG, DEPTH_ON, ALIGN, DEPTH_SCALE, CAM_CENTER_PX, CALIB_SIZE_NOW
    profile = None
    if AUTO_HEIGHT:
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, 30)
        try:
            profile = pipeline.start(cfg)
            DEPTH_ON = True
            ALIGN = rs.align(rs.stream.color)
            DEPTH_SCALE = profile.get_device().first_depth_sensor().get_depth_scale()
            print("📏 深度已開啟：自動量工件高度")
        except Exception as e:
            print(f"⚠️ 深度開不起來 ({e})，只用彩色，工件高度用 OBJ_HEIGHT_MM = {OBJ_HEIGHT_MM}")
    if profile is None:
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, 30)
        profile = pipeline.start(cfg)
        DEPTH_ON = False
    CAM_CFG = cfg
    # 內參：光軸中心 (高度視差修正用) + 焦距 (px 參數換算用)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    CAM_CENTER_PX = (intr.ppx, intr.ppy)
    CALIB_SIZE_NOW = (intr.width, intr.height)
    apply_px_scale(intr.fx / FX_REF)
    print(f"📷 彩色 {intr.width}x{intr.height}" + (f"、深度 {DEPTH_W}x{DEPTH_H} (對齊到彩色)" if DEPTH_ON else "") +
          f"，fx {intr.fx:.1f}、光軸 ({intr.ppx:.1f}, {intr.ppy:.1f})，px 參數 ×{PX_SCALE:.2f}")
    try:
        usb = profile.get_device().get_info(rs.camera_info.usb_type_descriptor)
        print(f"📷 RealSense 連線：USB {usb}" + ("  ⚠️ 不是 USB 3，容易掉幀，請換 USB 3 孔 / 線" if usb.startswith("2") else ""))
    except Exception:
        pass
    return profile


def grab(pipeline, timeout_ms=2000, retries=3):
    """
    拿一張彩色影像。相機斷流 (USB 鬆動 / 頻寬不夠) 時不要直接整個程式掛掉：
    先重試，還是不行就重新啟動相機串流，都失敗才丟錯誤
    """
    global LAST_DEPTH
    for attempt in range(retries + 1):
        try:
            for _ in range(10):
                frames = pipeline.wait_for_frames(timeout_ms)
                if DEPTH_ON:
                    frames = ALIGN.process(frames)
                f = frames.get_color_frame()
                if f:
                    if DEPTH_ON:
                        d = frames.get_depth_frame()
                        LAST_DEPTH = (np.asanyarray(d.get_data()).astype(np.float32) * (DEPTH_SCALE * 1000.0)
                                      if d else None)
                    return np.asanyarray(f.get_data())
        except RuntimeError as e:
            print(f"⚠️ 相機 {timeout_ms / 1000:.0f} 秒沒有影像 ({e})")
        if attempt == retries or CAM_CFG is None:
            break
        print(f"   第 {attempt + 1}/{retries} 次重新啟動相機...")
        try:
            pipeline.stop()
        except Exception:
            pass
        time.sleep(1.0)
        try:
            pipeline.start(CAM_CFG)
        except Exception as e:
            print(f"   重新啟動失敗：{e}")
    raise RuntimeError("相機一直收不到影像：檢查 USB 線有沒有鬆、是不是插在 USB 3 孔，"
                       "或用 realsense-viewer 確認相機正常")


def put_text(img, text, org, color=(255, 255, 255), scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def flip_pt(pt, shape):
    """相機像素 ↔ 顯示畫面座標 (轉 180° 時兩個方向的換算一樣)"""
    if not DISPLAY_ROTATE_180:
        return int(pt[0]), int(pt[1])
    h, w = shape[:2]
    return int(w - 1 - pt[0]), int(h - 1 - pt[1])


def disp_pt(pt, shape):
    """相機像素 → 縮小後顯示畫面上的座標 (寫字標記用)"""
    sx, sy = flip_pt(pt, shape)
    return int(sx * DISPLAY_SCALE), int(sy * DISPLAY_SCALE)


def to_display(img):
    """轉成顯示方向 + 縮成 DISPLAY_SCALE (字要在這之後寫，才不會跟著縮小)"""
    out = cv2.rotate(img, cv2.ROTATE_180) if DISPLAY_ROTATE_180 else img
    if DISPLAY_SCALE != 1.0:
        out = cv2.resize(out, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                         interpolation=cv2.INTER_AREA if out.ndim == 3 else cv2.INTER_NEAREST)
    return out


def height_color(h, lo, hi):
    """高度 → 顏色 (BGR)：低 = 藍，高 = 紅"""
    t = 0.0 if hi - lo < 1e-6 else min(max((h - lo) / (hi - lo), 0.0), 1.0)
    return (int(255 * (1 - t)), int(80 * (1 - abs(2 * t - 1))), int(255 * t))


def render(frame, cnts, sel, path_px, click, status, path_h=None):
    """
    先在相機畫面上畫線，轉成顯示方向後再寫字 (字才不會倒過來)
    path_h：每個路徑點的表面高度 (rim 模式)，路徑依高度上色 (藍 = 低，紅 = 高)
    """
    out = frame.copy()
    lt = max(1, int(round(1 / DISPLAY_SCALE)))          # 線條粗細：縮小顯示後還看得清楚
    for i, c in enumerate(cnts):
        cv2.drawContours(out, [c], -1, (0, 255, 255) if i != sel else (0, 0, 255), lt)
    if len(path_px) > 1 and path_h is not None and len(path_h) == len(path_px):
        lo, hi = min(path_h), max(path_h)
        for i in range(len(path_px) - 1):
            cv2.line(out, path_px[i], path_px[i + 1], height_color((path_h[i] + path_h[i + 1]) / 2, lo, hi), 2 * lt)
    elif len(path_px) > 1:
        cv2.polylines(out, [np.int32(path_px)], False, (255, 0, 0), 2 * lt)   # 塗膠路徑
        for p in (path_px if len(path_px) <= 30 else path_px[::5]):   # 點少 (頂點路徑) → 每點都標
            cv2.circle(out, p, 2 * lt, (255, 255, 0), -1)     # 每 5 點標一下，看取點密度
        cv2.circle(out, path_px[0], 6 * lt, (0, 255, 0), -1)  # 起點
    if click:
        cv2.drawMarker(out, click, (255, 0, 255), cv2.MARKER_CROSS, 18 * lt, 2 * lt)
    if ARUCO_EXCLUDE is not None:
        ex, _ = cv2.findContours(ARUCO_EXCLUDE, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, ex, -1, (160, 160, 160), lt)  # 不偵測物體的 Marker 區域
    out = to_display(out)
    if len(path_px) > 1:
        sx, sy = disp_pt(path_px[0], frame.shape)
        put_text(out, "START", (sx + 8, sy - 8), (0, 255, 0))
    put_text(out, status, (10, out.shape[0] - 10), (0, 255, 0))
    put_text(out, "click a button below, or press its key in the terminal", (10, 20))
    return out


class UserStop(Exception):
    """描邊時按了 STOP"""


def draw_with_track(arm, win, frame, path_px, path_arm, lift=0.0, title="GLUE"):
    """
    描邊 + 即時追蹤畫面。回傳 True = 走完；False = 按了 STOP (手臂已停，問要不要回 Home)
    ※ 軟體 STOP 不能取代示教器的緊急停止鈕
    """
    try:
        draw_path(arm, path_arm, lift=lift, track=LiveTrack(arm, win, frame, path_px, path_arm, lift, title))
        return True
    except UserStop:
        print("\n⏹ 已按 STOP：手臂已停止")
        if ask("  y: 回 Home (先垂直抬起)   n: 留在原地 > ", ("y", "n"),
               {"y": "Go Home (y)", "n": "Stay here (n)"}) == "y":
            arm.home()
            time.sleep(1.0)
        return False


class LiveTrack:
    """
    描邊時的即時追蹤畫面：鎖定時的那張相機畫面 + 規劃路徑，
    即時讀手臂 TCP → 換回相機像素 (用筆尖離板面的高度做視差修正)，紅點 = 筆尖現在位置，
    黃線 = 走過的軌跡 (只畫筆尖在規劃路徑上的部分；移過去、下筆、抬筆、回 Home 不畫)。
    用鎖定時的畫面當底圖：描邊時手臂會擋住相機，即時影像反而看不到路徑。
    下方有 STOP 按鈕 (或按 Esc)：送 Motion Stop 並丟 UserStop (RL 模式會一起把 RL 停掉)。
    update() 最多每 TRACK_INTERVAL_S 秒真的更新一次 (讀位置 + 重畫)，其他呼叫直接跳過。
    """
    TRACK_INTERVAL_S = 0.05
    ON_PATH_MM = 4.0              # 筆尖離規劃路徑 (3D，含 Z) 幾 mm 內才算「在路徑上」，畫進軌跡

    def __init__(self, arm, win, frame, path_px, path_arm, lift=0.0, title="GLUE"):
        self.arm, self.win, self.title = arm, win, title
        self.shape = frame.shape
        self.lt = max(1, int(round(1 / DISPLAY_SCALE)))
        path_h = [p[2] for p in path_arm] if path_arm and len(path_arm[0]) > 2 else None
        base = frame.copy()
        if len(path_px) > 1:
            lo, hi = (min(path_h), max(path_h)) if path_h else (0.0, 0.0)
            for i in range(len(path_px) - 1):
                col = height_color((path_h[i] + path_h[i + 1]) / 2, lo, hi) if path_h else (255, 0, 0)
                cv2.line(base, path_px[i], path_px[i + 1], col, 2 * self.lt)
            cv2.circle(base, path_px[0], 6 * self.lt, (0, 255, 0), -1)
        self.base = base
        # 規劃路徑的 TCP 3D 點 (跟 draw_path 用同一個 Z 算法)，判斷筆尖在不在路徑上
        zs = path_z(path_arm, lift) if path_arm else []
        self.plan = np.array([[p[0], p[1], z] for p, z in zip(path_arm, zs)], float) if path_arm else None
        self.trails = [[]]            # 好幾段 (離開路徑就斷開，回到路徑再接新的一段)
        self.t_last = 0.0
        self.progress = ""

    def _dist_to_plan(self, p):
        if self.plan is None or len(self.plan) < 2:
            return 0.0
        a, b = self.plan[:-1], self.plan[1:]
        d = b - a
        L2 = np.maximum((d * d).sum(axis=1), 1e-12)
        t = np.clip(((p - a) * d).sum(axis=1) / L2, 0.0, 1.0)
        return float(np.min(np.linalg.norm(a + t[:, None] * d - p, axis=1)))

    def update(self, i=None, total=None):
        if time.time() - self.t_last < self.TRACK_INTERVAL_S:
            if poll_key(1) == 27:
                self._stop()
            return
        self.t_last = time.time()
        try:
            x, y, z = self.arm.pose()[:3]
        except Exception:
            return
        h = max(z - Z_PAPER, 0.0)                      # 筆尖離板面多高 (視差修正用)
        u, v = arm_to_pixel(x, y, min(h, 0.9 * CAM_HEIGHT_MM))
        H_, W_ = self.shape[:2]
        inside = 0 <= u < W_ and 0 <= v < H_
        on_path = self._dist_to_plan(np.array([x, y, z])) < self.ON_PATH_MM
        if inside and on_path:
            self.trails[-1].append((int(round(u)), int(round(v))))
        elif self.trails[-1]:
            self.trails.append([])
        img = self.base.copy()
        for tr in self.trails:
            if len(tr) > 1:
                cv2.polylines(img, [np.int32(tr)], False, (0, 255, 255), 2 * self.lt)
        if inside:
            p = (int(round(u)), int(round(v)))
            cv2.circle(img, p, 7 * self.lt, (0, 0, 255), -1)
            cv2.circle(img, p, 7 * self.lt, (255, 255, 255), self.lt)
        out = to_display(img)
        if i is not None:
            self.progress = f"  point {i}" + (f"/{total}" if total else "")
        put_text(out, f"{self.title}  X {x:.1f}  Y {y:.1f}  Z {z:.1f}  (pen {z - Z_PAPER:+.1f} mm above board)"
                 + self.progress + ("  [on path]" if on_path else "  [moving]") + ("" if inside else "  [out of view]"),
                 (10, out.shape[0] - 10), (0, 255, 255))
        put_text(out, "red = pen now, yellow = drawn on path", (10, 20))
        show(self.win, out, [("STOP (Esc)", "esc", (0, 0, 200))])
        if poll_key(1) == 27:
            self._stop()

    def _stop(self):
        try:
            self.arm._write(REG_CMD, 1000)              # Motion Stop
        finally:
            raise UserStop("按下 STOP")


# ==========================================
# 8. 主程式
# ==========================================
def main():
    global CUR_HEIGHT, LOCKED_DEPTH, PATH_MODE, LAST_RUN
    arm = Arm()
    pipeline = rs.pipeline()
    start_camera(pipeline)
    win = "Glue Edge"
    cv2.namedWindow(win)
    click = {"pt": None}   # 上次用 t 點點的位置 (相機像素)，預覽畫面上畫紫色十字
    load_calibration()

    try:
        # --- GRIP：筆已經在夾爪上，開始時不打開夾爪 (只確保夾緊) ---
        arm.grip(close=True)
        print("\n✊ 夾爪保持夾緊 (開始時不會打開)")

        # --- 校正是在別的解析度做的 → 像素座標要重抓 ---
        if CALIB_SIZE != CALIB_SIZE_NOW:
            print(f"\n⚠️ 校正是在 {CALIB_SIZE[0]}x{CALIB_SIZE[1]} 做的，相機現在是 "
                  f"{CALIB_SIZE_NOW[0]}x{CALIB_SIZE_NOW[1]}，Marker 像素位置要重抓")
            while CALIB_SIZE != CALIB_SIZE_NOW:
                a2 = ask("  Enter: 重抓 Marker 像素位置 (白板 / Marker 沒動過)   "
                         "c: 完整重新校正 (動過，用示教器重教 4 點) > ", ("", "c"),
                         {"": "Re-detect markers (Enter)", "c": "Full recalib (c)"})
                if a2 == "c":
                    recalibrate(arm, pipeline, win)
                else:
                    repixel_calibration(arm, pipeline, win)

        # --- 先定位 4 個角的 ArUco ---
        ans = ask("\n[ArUco] Enter: 檢查筆尖偏移   c: 換了 Marker/紙板/相機 → 完整重新校正   "
                  "k: 自主校正 (手臂畫十字，不用示教器)   x: 跳過 > ", ("", "c", "k", "x"),
                  {"": "Check offset (Enter)", "c": "Full recalib (c)", "k": "Auto calib (k)", "x": "Skip (x)"})
        if ans == "c":
            recalibrate(arm, pipeline, win)
        elif ans == "k":
            dot_calibrate(arm, pipeline, win)
        elif ans != "x":
            locate_aruco(arm, pipeline, win)
        else:
            arm.home()
            time.sleep(1.0)

        bg = load_background()
        build_aruco_exclude(pipeline)
        check_background(bg, grab(pipeline))
        def preview_help():
            print("\n[預覽] (在終端機按，不用按 Enter) n: 換物體 / s: 鎖定路徑 / a: 檢查筆尖偏移 / "
                  "c: 完整重新校正 / k: 自主校正 (畫十字) / g: 量工件幾何 / v: 重新驗證上次畫線 / b: 拍底圖 / t: 選位置點一個點 / p: 存偵測快照 / "
                  f"m: 切換路徑模式 (目前 {PATH_MODE}) / q: 離開")
        preview_help()
        flush_keys()
        sel = 0
        h_ema = None      # 即時量到的高度 (平滑後)

        while True:
            # --- DETECT ---
            frame = grab(pipeline)
            cnts, mask = detect_objects(frame, bg)
            sel = sel % len(cnts) if cnts else 0
            if DEPTH_ON and cnts:
                h = measure_height(cnts[sel], mask, LAST_DEPTH)
                if h is not None:                      # 平滑，畫面上數字/路徑才不會一直跳
                    h_ema = h if h_ema is None else 0.7 * h_ema + 0.3 * h
                CUR_HEIGHT = h_ema
            path_px, path_arm = make_path(cnts[sel], frame.shape) if cnts else ([], [])
            path_h = [p[2] for p in path_arm] if path_arm and len(path_arm[0]) > 2 else None
            status = (f"object {sel + 1}/{len(cnts)}  [{LAST_PATH_MODE}] segments {max(len(path_arm) - 1, 0)}  "
                      f"length {path_length(path_arm):.0f}mm" if path_arm else "no object")
            if LAST_REJECTED and LAST_REJECTED[1] == "big":
                status += f"  [too big {LAST_REJECTED[0]:.0f}px2 > MAX_AREA {MAX_AREA:.0f}]"
            elif not cnts and LAST_REJECTED:
                status += f"  [too small {LAST_REJECTED[0]:.0f}px2 < MIN_AREA {MIN_AREA:.0f}]"
            if path_h:
                status += (f"  flat h {LAST_RIM.get('hmed', path_h[0]):.1f}mm" if LAST_RIM.get("flat") else
                           f"  3D h {min(path_h):.1f}~{max(path_h):.1f}mm")
            elif PATH_MODE == "auto" and cnts and LAST_RIM.get("error"):
                status += "  2D (depth failed, see terminal on s)"
            if not path_h and DEPTH_ON and cnts:
                status += f"  height {CUR_HEIGHT:.1f}mm" if CUR_HEIGHT is not None else "  height ?"
            show(win, render(frame, cnts, sel, path_px, click["pt"], status, path_h), PREVIEW_BUTTONS)
            cv2.imshow("Object Mask", to_display(mask))
            key = poll_key(1)

            if key == 255:
                continue
            if key == ord('q'):
                break
            if key == ord('n'):
                sel += 1
                h_ema = None
                print(f"  → 物體 {sel % len(cnts) + 1}/{len(cnts)}" if cnts else "  ⚠️ 沒有偵測到物體")
            elif key == ord('a'):
                locate_aruco(arm, pipeline, win)
                build_aruco_exclude(pipeline)
                check_background(bg, grab(pipeline))
            elif key == ord('c'):
                recalibrate(arm, pipeline, win)
                build_aruco_exclude(pipeline)
                check_background(bg, grab(pipeline))
            elif key == ord('k'):
                dot_calibrate(arm, pipeline, win)
                build_aruco_exclude(pipeline)
            elif key == ord('g'):
                if not cnts:
                    print("⚠️ 沒有偵測到物體")
                    continue
                lock_cnt, _, _ = lock_object(pipeline, bg, cnts[sel])     # 連拍平均，比單張準
                make_path(lock_cnt if lock_cnt is not None else cnts[sel], frame.shape)
                LOCKED_DEPTH = None
                geom = measure_geometry()
                print_geometry(geom)
                if geom:
                    show(win, draw_geometry(frame, geom))
                    save_geometry(frame, geom)
                    ask("  按 Enter 回預覽 > ", None, {"": "Back (Enter)"})
            elif key == ord('v'):
                if LAST_RUN is None:
                    print("⚠️ 還沒描過，沒有可以驗證的")
                    continue
                arm.home()
                time.sleep(1.0)
                run_verify(pipeline, win, LAST_RUN)
            elif key == ord('b'):
                bg = capture_background(pipeline, bg)
                flush_keys()
            elif key == ord('p'):
                save_debug(frame, mask, cnts, sel, path_px, bg)
            elif key == ord('m'):
                modes = ["auto", "vertex", "edge"]
                PATH_MODE = modes[(modes.index(PATH_MODE) + 1) % len(modes)] if PATH_MODE in modes else "auto"
                print(f"  → 路徑模式 {PATH_MODE}" + ("  (自動判斷形狀和平面 / 3D)" if PATH_MODE == "auto"
                                                    else "  (2D，整條同一個 Z)"))
            elif key == ord('t'):
                # 起始位置：上次點的地方 → 選中物體的中心 → 畫面中央
                start = click["pt"]
                if start is None and cnts:
                    m = cv2.moments(cnts[sel])
                    start = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else None
                if start is None:
                    start = (frame.shape[1] / 2, frame.shape[0] / 2)
                pt = pick_point(pipeline, win, start, "DOT position",
                                live_overlay=lambda u, v: "arm X=%.2f Y=%.2f" % pixel_to_arm(u, v, 0.0))
                if pt is None:
                    print("  已取消")
                    flush_keys()
                    preview_help()
                    continue
                click["pt"] = (int(round(pt[0])), int(round(pt[1])))
                x, y = pixel_to_arm(*pt, 0.0)
                print(f"  點點位置 X={x:.2f}, Y={y:.2f}")
                if path_ok([(x, y)]):
                    dot_at(arm, x, y)
                    arm.home()
                    time.sleep(1.0)
                    print("  看筆點和十字的位置差多少，調整 TOOL_OFFSET_X / Y")
                else:
                    print("⚠️ 超出工作範圍")
            elif key == ord('s'):
                # --- CONFIRM ---
                if not path_arm:
                    print("⚠️ 沒有偵測到物體")
                    continue
                # 連拍 LOCK_FRAMES 張：遮罩平均 (邊緣雜訊抵消) + 高度取中位數
                lock_cnt, h, spread = lock_object(pipeline, bg, cnts[sel])
                if DEPTH_ON:                 # 2D 路徑用的單一高度 (auto 量得到高度圖時改用每點高度，不印這個)
                    CUR_HEIGHT = h
                    if PATH_MODE != "auto":
                        if h is None:
                            print(f"⚠️ 深度量不到這個工件的高度 (黑色/反光/太小？)，改用 OBJ_HEIGHT_MM = {OBJ_HEIGHT_MM} mm")
                        else:
                            print(f"\n📏 工件高度 {h:.1f} mm (連量 {LOCK_FRAMES} 張，跳動 {spread:.1f} mm)"
                                  + ("  ⚠️ 跳動偏大，量測可能不穩" if spread > 3 else ""))
                if lock_cnt is None:
                    print("  (多張平均失敗，用目前這一張的邊緣)")
                    lock_cnt = cnts[sel]
                path_px, path_arm = make_path(lock_cnt, frame.shape)   # 用平均後的邊緣 + 鎖定的高度重算
                LOCKED_DEPTH = None                                    # 預覽回到用每張的深度
                if not path_arm:
                    print("⚠️ 路徑算不出來" + (f"：{LAST_RIM['error']}" if LAST_RIM.get("error") else ""))
                    continue
                cnts[sel] = lock_cnt
                xs, ys = [p[0] for p in path_arm], [p[1] for p in path_arm]
                path_h = [p[2] for p in path_arm] if len(path_arm[0]) > 2 else None
                print(f"\n========== 塗膠路徑 ==========")
                if path_h:
                    zs = path_z(path_arm)
                    shape_s = "多邊形 (尖角)" if LAST_PATH_MODE == "vertex" else "曲線"
                    surf_s = (f"平面 (高低差 {LAST_RIM['spread']:.1f} mm < {FLAT_TOL_MM}，整條同一個 Z)"
                              if LAST_RIM.get("flat") else f"3D (高低差 {LAST_RIM['spread']:.1f} mm，每點自己的 Z)")
                    print(f"  自動判斷：{shape_s}，{surf_s}" + ("，畫在板面" if DRAW_SURFACE != "top" else ""))
                    print(f"  {len(path_arm) - 1} 段 (含頭尾重疊 {GLUE_OVERLAP_MM} mm)，總長 {LAST_RIM.get('length', 0):.0f} mm"
                          + (f"，外緣往內 {RIM_WIDTH_MM / 2:.1f} mm" if RIM_WIDTH_MM else "，描外緣"))
                    print(f"  外框高度 {LAST_RIM['hmin']:.1f} ~ {LAST_RIM['hmax']:.1f} mm，深度有效 {LAST_RIM['valid']:.0%}")
                    for i, ((x, y, hh), z) in enumerate(zip(path_arm, zs), 1):
                        print(f"    點 {i:2d}: X={x:.2f}, Y={y:.2f}, Z={z:.2f}  (表面高 {hh:.1f} mm)")
                elif PATH_MODE == "auto":
                    print(f"  ⚠️ 高度圖算不出來 ({LAST_RIM.get('error', '?')})，改用 2D："
                          f"{'頂點' if LAST_PATH_MODE == 'vertex' else '沿邊緣'}，整條同一個高度 {obj_height():.1f} mm")
                    print(f"  {len(path_arm) - 1} 段，總長 {path_length(path_arm):.0f} mm")
                elif LAST_PATH_MODE == "vertex":
                    nv = len(path_arm) - (2 if GLUE_OVERLAP_MM > 0 else 1)
                    print(f"  頂點路徑：{nv} 個頂點，{len(path_arm) - 1} 段 (每個邊一段 + 重疊 {GLUE_OVERLAP_MM} mm)，"
                          f"總長 {path_length(path_arm):.0f} mm")
                    for i, (x, y) in enumerate(path_arm[:nv], 1):
                        print(f"    頂點 {i}: X={x:.2f}, Y={y:.2f}")
                else:
                    if PATH_MODE == "vertex":
                        print("  ⚠️ 這個工件不是多邊形 (圓 / 大圓角)，改用沿邊緣走")
                    print(f"  {len(path_arm) - 1} 小段，總長 {path_length(path_arm):.0f} mm (含頭尾重疊 {GLUE_OVERLAP_MM} mm)")
                print(f"  起點 X={xs[0]:.2f}, Y={ys[0]:.2f}")
                print(f"  範圍 X {min(xs):.2f} ~ {max(xs):.2f}，Y {min(ys):.2f} ~ {max(ys):.2f}")
                if path_h:
                    zs = path_z(path_arm)
                    print(f"  塗膠高度 Z={min(zs):.2f} ~ {max(zs):.2f} (紙面 {Z_PAPER:.3f} + 每點表面高度 - 下壓 {PEN_PRESS_MM} mm)")
                else:
                    print(f"  塗膠高度 Z={draw_z() - PEN_PRESS_MM:.3f} (紙面 {Z_PAPER:.3f} + 物體高度 "
                          f"{obj_height():.1f} mm" + (" (畫在板面，不加)" if DRAW_SURFACE != "top" else "") + f" - 下壓 {PEN_PRESS_MM} mm)")
                geom = measure_geometry() if LAST_HMAP else None
                if geom:
                    print_geometry(geom)
                if not path_ok(path_arm):
                    print("⚠️ 路徑超出工作範圍，不執行")
                    continue
                show(win, render(frame, cnts, sel, path_px, None, "LOCKED - answer below / in terminal", path_h))
                cv2.waitKey(1)
                go = ask(f"開始塗膠？ y: 開始   d: 空中試跑 (整條 +{DRY_RUN_LIFT_MM:.0f} mm，不碰工件)   n: 取消 > ",
                         ("y", "d", "n"), {"y": "Glue (y)", "d": "Dry run (d)", "n": "Cancel (n)"})
                if go == "d":
                    if draw_with_track(arm, win, frame, path_px, path_arm, DRY_RUN_LIFT_MM, "DRY RUN"):
                        arm.home()
                        time.sleep(1.0)
                        go = ask("空中試跑結束。路線對嗎？ y: 正式塗膠   n: 取消 > ", ("y", "n"),
                                 {"y": "Glue now (y)", "n": "Cancel (n)"})
                    else:
                        go = "n"
                if go != "y":
                    print("已取消")
                    preview_help()
                    continue
                # --- GLUE ---
                before = grab_avg(pipeline)                    # 2-4 驗證用 (手臂在 Home，不擋鏡頭)
                if draw_with_track(arm, win, frame, path_px, path_arm, 0.0, "GLUE"):
                    arm.home()
                    time.sleep(1.0)
                    LAST_RUN = {"before": before, "path_arm": path_arm, "geom": geom,
                                "mode": f"{LAST_PATH_MODE}{'-3D' if path_h and not LAST_RIM.get('flat') else ''}"}
                    run_verify(pipeline, win, LAST_RUN)
                    print("\n✅ 塗膠完成，回到預覽")
            else:
                continue
            # 做完一個動作 (手臂可能動了很久)：期間亂按的鍵丟掉，再印一次按鍵說明
            if key != ord('n'):
                flush_keys()
                preview_help()

        arm.home()
        if ask("\n要鬆開夾爪放下麥克筆嗎？(y/n) > ", ("y", "n"), {"y": "Open gripper (y)", "n": "Keep (n)"}) == "y":
            arm.grip(close=False)
            print("✋ 已鬆開")
    finally:
        pipeline.stop()
        arm.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
