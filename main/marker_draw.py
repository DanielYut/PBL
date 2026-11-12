"""
麥克筆畫邊緣 (模擬焊槍沿工件邊緣走)  —— 單一檔案版，不需要 main/ 其他檔案

流程：
  [GRIP]    夾爪打開 → 你把麥克筆放進夾爪 (筆尖朝下) → 按 Enter → 夾緊
  [ARUCO]   Enter：筆尖依序到 4 個角的 Marker，對準後按 Enter，
                   自動算出筆尖偏移並存檔 (data/aruco_xy_calib.json，下次自動載入)
            c    ：換了 Marker / 紙板 / 相機 → 完整重新校正：重新偵測 4 個 Marker，
                   示教器讓筆尖對準每個中心，重建像素→手臂 XY 對應 (data/aruco_xy_calib.json)
            ※ 只校正 XY。Z 一律用程式裡設定的 Z_PAPER / Z_DRAW
  [HOME]    回拍照位置
  [DETECT]  預覽畫面：偵測白紙上的物體，算出邊緣路徑 (藍線)。畫面已轉 180° 跟實際方向一致
              a: 重新做 ArUco 定位
              n: 換下一個物體   s: 鎖定路徑
  [CONFIRM] 終端機印出路徑點 → y 開始畫 / 其他鍵取消
  [DRAW]    到第 1 點上方 → 下筆 → 沿邊緣直線移動一圈 → 抬筆
  [HOME]    回拍照位置，回到 DETECT
  結束時 (q) 問你要不要鬆開夾爪

其他按鍵：
  b : 拍底圖 (白紙清空時)，灰色/銀色物體需要底圖才抓得到
  滑鼠點畫面 + t : 在點的位置點一個點 (用來檢查/校正偏移量)

執行 (在 mid_project 資料夾)：python main/marker_draw.py

⚠️ 第一次執行前要確認的值 (都在「1. 設定」)：
  Z_DRAW          筆尖碰到要畫的表面時的 Z (預設 None = 紙面 Z_PAPER + 物體高度，自動算)
  OBJ_HEIGHT_MM   要畫在物體頂面 → 填物體高度；畫在紙上 (物體周圍) → 0
  TOOL_OFFSET_X/Y 筆尖相對夾爪中心的偏移 (先用 t 點一個點檢查)
"""
import json
import math
import os
import struct
import sys
import time

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
PEN_PRESS_MM = 0.5      # 再往下壓多少，讓筆尖確實接觸 (麥克筆筆頭軟，0.5~1 mm)
Z_LIFT = 15.0           # 抬筆高度 (畫之前/之後停在表面上方幾 mm)
Z_TRAVEL = 560.000      # 從 Home 過去 / 回來的安全高度
Z_MIN = 395.0           # 安全下限，低於這個高度不准移動

SPEED_FAST = 40
SPEED_DRAW = 10         # 畫線速度 (%)

# --- 畫線路徑 ---
OBJ_HEIGHT_MM = 0.0     # TODO: 畫在物體頂面 → 物體高度 mm；畫在紙上 → 0
DRAW_OFFSET_MM = 0.0    # 路徑往外 (+) / 往內 (-) 偏多少 mm (例如畫在紙上沿物體外圍 → +3)
SIMPLIFY_PX = 2.0       # 輪廓簡化程度 (越大點越少；方塊只剩 4 個角)
MIN_SEG_MM = 2.0        # 相鄰路徑點至少隔多少 mm

# --- 工具偏移 (筆尖相對夾爪中心) ---
# ArUco 定位時微調對準並按 Enter 記錄，程式會自動算出偏移存到 data/aruco_xy_calib.json，
# 下次啟動自動載入 (會蓋掉下面兩個值)
TOOL_OFFSET_X = 0.0
TOOL_OFFSET_Y = 0.0

# --- 顯示 ---
DISPLAY_ROTATE_180 = True   # 相機畫面跟實際看到的上下左右相反 → 轉 180° 顯示 (只影響顯示，不影響計算)

# --- ArUco 定位 (畫之前先讓筆尖到 4 個角的 Marker 中心，確認位置準不準) ---
Z_PAPER = 436.157       # 筆尖碰到「紙面」時的 Z (2026-11-05 示教器實測)
                        # ※ 高度只用這裡 (或 Z_DRAW) 的值，校正不會改 Z
ARUCO_HOVER_MM = 10.0   # 停在 Marker 上方幾 mm
ARUCO_DICT = cv2.aruco.DICT_5X5_50
ARUCO_MATCH_PX = 40     # 偵測到的 Marker 跟記錄位置差多少 px 內算同一個
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
GRIP_CLOSE, GRIP_OPEN = 64, 128


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
        self._write(REG_SPEED, int(speed))
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

# 高度視差：物體頂面比紙面高、離相機近，看起來會往外擴。以相機正下方為中心縮回。
CAM_CENTER_PX = (318.7, 246.8)   # 相機光軸在畫面上的位置 (RealSense 640x480 內參)
CAM_HEIGHT_MM = 445.0            # 相機到紙面距離 (由 Marker 大小估算)


def pixel_to_arm(u, v, height_mm=0.0):
    """像素 → 夾爪中心要去的 X, Y。height_mm = 這個點離紙面多高"""
    if height_mm:
        k = 1.0 - height_mm / CAM_HEIGHT_MM
        u = CAM_CENTER_PX[0] + (u - CAM_CENTER_PX[0]) * k
        v = CAM_CENTER_PX[1] + (v - CAM_CENTER_PX[1]) * k
    p = H_MATRIX @ np.array([u, v, 1.0])
    return p[0] / p[2] + TOOL_OFFSET_X, p[1] / p[2] + TOOL_OFFSET_Y


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


def load_calibration():
    """讀 data/aruco_xy_calib.json：4 點 XY 對應 + 筆尖偏移 (Z 不從檔案讀，用程式裡的 Z_PAPER / Z_DRAW)"""
    global ARUCO_DICT
    if not os.path.exists(CALIB_PATH):
        print("📐 Marker 對應用程式裡寫死的值 (還沒做過完整重新校正)")
        return
    global TOOL_OFFSET_X, TOOL_OFFSET_Y
    with open(CALIB_PATH, encoding="utf-8") as fp:
        d = json.load(fp)
    if d.get("pts_pixel") and d.get("pts_arm"):
        set_calibration(d["pts_pixel"], d["pts_arm"])
        print(f"📐 已載入 Marker XY 對應 {CALIB_PATH} ({d['saved_at']}, {d.get('aruco_dict', '')})")
    if d.get("aruco_dict"):
        ARUCO_DICT = getattr(cv2.aruco, d["aruco_dict"])
    TOOL_OFFSET_X, TOOL_OFFSET_Y = d.get("tool_offset_x", 0.0), d.get("tool_offset_y", 0.0)
    print(f"📐 筆尖偏移 X={TOOL_OFFSET_X:+.2f}, Y={TOOL_OFFSET_Y:+.2f} mm")


# ==========================================
# 4. 物體偵測 (白紙上跟底圖不同的東西)
# ==========================================
MIN_AREA, MAX_AREA = 400, 30000
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
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
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
        a, b = cv2.GaussianBlur(img, (5, 5), 0), cv2.GaussianBlur(bg, (5, 5), 0)
        lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_bg = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)
        dL = lab[:, :, 0] - lab_bg[:, :, 0]
        dC = np.hypot(lab[:, :, 1] - lab_bg[:, :, 1], lab[:, :, 2] - lab_bg[:, :, 2])
        mask = ((dC > BG_CHROMA_DIFF) | (dL > BG_BRIGHT_DIFF) | (dL < -BG_DARK_DIFF)).astype(np.uint8) * 255
        e = cv2.Canny(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), EDGE_LOW, EDGE_HIGH)
        eb = cv2.dilate(cv2.Canny(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), EDGE_LOW, EDGE_HIGH), np.ones((5, 5), np.uint8))
        new_e = cv2.morphologyEx(cv2.bitwise_and(e, cv2.bitwise_not(eb)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cnts, _ = cv2.findContours(new_e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, cnts, -1, 255, -1)

    mask &= paper
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def detect_objects(img, bg):
    """回傳 (物體輪廓 list (由大到小), 物體遮罩)"""
    mask = object_mask(img, bg)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c for c in cnts if MIN_AREA < cv2.contourArea(c) < MAX_AREA]
    return sorted(cnts, key=cv2.contourArea, reverse=True), mask


def load_background():
    if os.path.exists(BG_PATH):
        bg = cv2.imread(BG_PATH)
        if bg is not None:
            print(f"📂 已載入底圖 {BG_PATH}")
            return bg
    print("⚠️ 沒有底圖，使用顏色模式。白紙清空後按 b 拍底圖")
    return None


def capture_background(pipeline, n=15):
    acc = sum(grab(pipeline).astype(np.float32) for _ in range(n))
    bg = (acc / n).astype(np.uint8)
    os.makedirs(os.path.dirname(BG_PATH), exist_ok=True)
    cv2.imwrite(BG_PATH, bg)
    print(f"📷 已拍底圖 {BG_PATH}")
    return bg


# ==========================================
# 5. 邊緣 → 畫線路徑
# ==========================================
def make_path(cnt, shape):
    """
    物體輪廓 → (路徑像素點, 路徑手臂點)。路徑是封閉的，最後一點 = 第一點
    DRAW_OFFSET_MM 用遮罩膨脹/侵蝕做出外擴/內縮
    """
    if DRAW_OFFSET_MM:
        m = np.zeros(shape[:2], np.uint8)
        cv2.drawContours(m, [cnt], -1, 255, -1)
        r = max(1, int(round(abs(DRAW_OFFSET_MM) / MM_PER_PX)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        m = cv2.dilate(m, k) if DRAW_OFFSET_MM > 0 else cv2.erode(m, k)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            return [], []
        cnt = max(cs, key=cv2.contourArea)

    poly = cv2.approxPolyDP(cnt, SIMPLIFY_PX, True).reshape(-1, 2).astype(float)
    px, arm = [], []
    for u, v in poly:
        x, y = pixel_to_arm(u, v, OBJ_HEIGHT_MM)
        if arm and math.hypot(x - arm[-1][0], y - arm[-1][1]) < MIN_SEG_MM:
            continue
        px.append((int(u), int(v)))
        arm.append((x, y))
    if len(arm) < 3:
        return [], []
    px.append(px[0])
    arm.append(arm[0])
    return px, arm


def path_ok(arm_pts):
    """X_RANGE / Y_RANGE 是 None 就不檢查"""
    def inside(val, rng):
        return rng is None or rng[0] <= val <= rng[1]
    return all(inside(x, X_RANGE) and inside(y, Y_RANGE) for x, y in arm_pts)


def path_length(arm_pts):
    return sum(math.hypot(arm_pts[i + 1][0] - arm_pts[i][0], arm_pts[i + 1][1] - arm_pts[i][1])
               for i in range(len(arm_pts) - 1))


# ==========================================
# 6. 動作
# ==========================================
def draw_z():
    """筆尖碰到要畫的表面時的 Z"""
    return Z_DRAW if Z_DRAW is not None else Z_PAPER + OBJ_HEIGHT_MM


def draw_path(arm, pts):
    z_draw = draw_z() - PEN_PRESS_MM
    z_up = draw_z() + Z_LIFT
    x0, y0 = pts[0]
    print(f"✏️ 開始畫：{len(pts) - 1} 段，總長 {path_length(pts):.0f} mm")
    arm.move(x0, y0, Z_TRAVEL)
    arm.move(x0, y0, z_up)
    arm.move(x0, y0, z_draw, speed=SPEED_DRAW)                 # 下筆
    for i, (x, y) in enumerate(pts[1:], 1):
        print(f"  → 第 {i}/{len(pts) - 1} 點 ({x:.2f}, {y:.2f})")
        arm.move(x, y, z_draw, speed=SPEED_DRAW, linear=True)   # 直線畫過去
    arm.move(x0, y0, z_up, speed=SPEED_DRAW)                   # 抬筆
    arm.move(x0, y0, Z_TRAVEL)


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


MAIN_MOUSE = None   # main() 的滑鼠 callback，借用視窗後要換回去


def click_marker(pipeline, win, pts):
    """偵測不到的那個 Marker：讓使用者在畫面上點它的中心。回傳 (u, v) 相機像素或 None"""
    frame = grab(pipeline)
    clicked = {"pt": None}

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["pt"] = flip_pt((x, y), frame.shape)
    cv2.setMouseCallback(win, cb)
    print("  👉 在影像視窗點「沒偵測到的那個 Marker」正中心，Enter 確定，Esc 取消")
    try:
        while True:
            out = frame.copy()
            for _, u, v in pts:
                cv2.circle(out, (int(u), int(v)), 10, (0, 255, 0), 2)
            if clicked["pt"]:
                cv2.drawMarker(out, clicked["pt"], (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            out = to_display(out)
            put_text(out, "Click the MISSING marker center, Enter: OK, Esc: cancel", (10, 25), (0, 0, 255), 0.55)
            cv2.imshow(win, out)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10) and clicked["pt"]:
                return float(clicked["pt"][0]), float(clicked["pt"][1])
            if key == 27:
                return None
    finally:
        if MAIN_MOUSE:
            cv2.setMouseCallback(win, MAIN_MOUSE)   # 換回預覽畫面的點選


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
            if all(math.hypot(uv[0] - q[0], uv[1] - q[1]) > 20 for q in found.values()):
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
    if (max(sc) - min(sc)) / np.mean(sc) > 0.15:   # 相機有點斜，舊校正本身就差 ~8%
        print("  ⚠️ 4 邊比例差超過 15%，可能有點對歪或順序教錯了，建議重做")
    if input("  要套用並存檔嗎？(y/n) > ").strip().lower() != "y":
        print("  沒有存")
        return False
    set_calibration(px, am)
    TOOL_OFFSET_X = TOOL_OFFSET_Y = 0.0
    write_calib_json(pts_pixel=px.tolist(), pts_arm=am.tolist(), aruco_dict=dict_name,
                     home=[X_HOME, Y_HOME, Z_HOME], tool_offset_x=0.0, tool_offset_y=0.0, corrections=[])
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
        print("  ⚠️ 各點偏的不一樣 (> 3 mm)，不只是筆夾歪，校正本身可能有問題，先不要存")
    new_x, new_y = TOOL_OFFSET_X + mean[0], TOOL_OFFSET_Y + mean[1]
    print(f"  新的筆尖偏移：X={new_x:+.2f}, Y={new_y:+.2f} (原本 {TOOL_OFFSET_X:+.2f}, {TOOL_OFFSET_Y:+.2f})")
    if input("  要套用並存檔嗎？(y/n) > ").strip().lower() != "y":
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
    print("用示教器把「筆尖」移到 Marker 正中心 (只看 XY，高度隨意)，回到影像視窗按 Enter 記錄")
    print("  Enter: 記錄   空白: 跳過   Esc: 結束")

    corrections, taught = [], []
    stopped = False
    last_poll, pose = 0.0, None
    for k, (_, name, src, u, v, x, y) in enumerate(items, 1):
        print(f"\n📍 第 {k}/4 點：畫面{name} [{src}]  預估 X={x:.2f}, Y={y:.2f}")
        while True:
            # 左：Home 拍的參考畫面 (紅圈 = 這次要教的)；右：手臂上的即時畫面
            ref = snapshot.copy()
            for j, (_, _, _, uu, vv, _, _) in enumerate(items, 1):
                cv2.circle(ref, (int(uu), int(vv)), 22 if j == k else 10,
                           (0, 0, 255) if j == k else (0, 255, 255), 3 if j == k else 1)
            ref = to_display(ref)
            for j, (_, _, _, uu, vv, _, _) in enumerate(items, 1):
                sx, sy = flip_pt((uu, vv), shape)
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
            put_text(live, "Enter: record  Space: skip  Esc: stop", (10, 470), (0, 255, 0))
            cv2.imshow(win, cv2.resize(np.hstack([ref, live]), None, fx=0.75, fy=0.75))
            key = cv2.waitKey(30) & 0xFF

            if key in (13, 10):
                p = arm.pose()
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
    input("\n教點結束。用示教器把筆抬高、控制器切回遠端/自動模式後按 Enter，手臂會回 Home > ")
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
              "Enter: 對準了記錄   空白: 跳過   d: 點一下   Esc: 結束")
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
                     (10, 470), (0, 255, 0), 0.45)
            cv2.imshow(win, live)
            key = cv2.waitKey(1) & 0xFF

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


def dot_at(arm, x, y):
    """在 (x, y) 點一個點，用來檢查偏移量"""
    print(f"• 點一個點 @ ({x:.2f}, {y:.2f})")
    arm.move(x, y, Z_TRAVEL)
    arm.move(x, y, draw_z() + Z_LIFT)
    arm.move(x, y, draw_z() - PEN_PRESS_MM, speed=SPEED_DRAW)
    time.sleep(0.3)
    arm.move(x, y, draw_z() + Z_LIFT, speed=SPEED_DRAW)
    arm.move(x, y, Z_TRAVEL)


# ==========================================
# 7. 畫面
# ==========================================
def grab(pipeline):
    while True:
        f = pipeline.wait_for_frames().get_color_frame()
        if f:
            return np.asanyarray(f.get_data())


def put_text(img, text, org, color=(255, 255, 255), scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def flip_pt(pt, shape):
    """相機像素 ↔ 顯示畫面座標 (轉 180° 時兩個方向的換算一樣)"""
    if not DISPLAY_ROTATE_180:
        return int(pt[0]), int(pt[1])
    h, w = shape[:2]
    return int(w - 1 - pt[0]), int(h - 1 - pt[1])


def to_display(img):
    return cv2.rotate(img, cv2.ROTATE_180) if DISPLAY_ROTATE_180 else img


def render(frame, cnts, sel, path_px, click, status):
    """先在相機畫面上畫線，轉成顯示方向後再寫字 (字才不會倒過來)"""
    out = frame.copy()
    for i, c in enumerate(cnts):
        cv2.drawContours(out, [c], -1, (0, 255, 255) if i != sel else (0, 0, 255), 1)
    if len(path_px) > 1:
        cv2.polylines(out, [np.int32(path_px)], True, (255, 0, 0), 2)
        cv2.circle(out, path_px[0], 6, (0, 255, 0), -1)       # 起點
    if click:
        cv2.drawMarker(out, click, (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
    out = to_display(out)
    if len(path_px) > 1:
        sx, sy = flip_pt(path_px[0], frame.shape)
        put_text(out, "START", (sx + 8, sy - 8), (0, 255, 0))
    put_text(out, status, (10, 470), (0, 255, 0))
    put_text(out, "n:next s:lock a:aruco c:recalib b:bg click+t:dot q:quit", (10, 20))
    return out


# ==========================================
# 8. 主程式
# ==========================================
def main():
    arm = Arm()
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)
    win = "Marker Draw"
    cv2.namedWindow(win)
    click = {"pt": None}   # 存「相機像素」座標

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["pt"] = flip_pt((x, y), (480, 640))   # 顯示畫面 → 相機像素

    global MAIN_MOUSE
    MAIN_MOUSE = on_mouse
    cv2.setMouseCallback(win, on_mouse)
    load_calibration()

    try:
        # --- GRIP：直接夾麥克筆 ---
        print("\n[夾筆] 夾爪打開")
        arm.grip(close=False)
        input("把麥克筆放進夾爪 (筆尖朝下、對正)，放好後按 Enter 夾緊 > ")
        arm.grip(close=True)
        print("✊ 已夾緊")

        # --- 先定位 4 個角的 ArUco ---
        ans = input("\n[ArUco] Enter: 檢查筆尖偏移   c: 換了 Marker/紙板/相機 → 完整重新校正   "
                    "x: 跳過 > ").strip().lower()
        if ans == "c":
            recalibrate(arm, pipeline, win)
        elif ans != "x":
            locate_aruco(arm, pipeline, win)
        else:
            arm.home()
            time.sleep(1.0)

        bg = load_background()
        print("\n[預覽] n: 換物體 / s: 鎖定路徑 / a: 檢查筆尖偏移 / c: 完整重新校正 / b: 拍底圖 / "
              "點畫面+t: 點一個點 / q: 離開")
        sel = 0

        while True:
            # --- DETECT ---
            frame = grab(pipeline)
            cnts, mask = detect_objects(frame, bg)
            sel = sel % len(cnts) if cnts else 0
            path_px, path_arm = make_path(cnts[sel], frame.shape) if cnts else ([], [])
            status = (f"object {sel + 1}/{len(cnts)}  points {max(len(path_arm) - 1, 0)}  "
                      f"length {path_length(path_arm):.0f}mm" if path_arm else "no object")
            cv2.imshow(win, render(frame, cnts, sel, path_px, click["pt"], status))
            cv2.imshow("Object Mask", to_display(mask))
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            if key == ord('n'):
                sel += 1
            elif key == ord('a'):
                locate_aruco(arm, pipeline, win)
            elif key == ord('c'):
                recalibrate(arm, pipeline, win)
            elif key == ord('b'):
                bg = capture_background(pipeline)
            elif key == ord('t') and click["pt"]:
                x, y = pixel_to_arm(*click["pt"], OBJ_HEIGHT_MM)
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
                print(f"\n========== 路徑 ({len(path_arm) - 1} 點，{path_length(path_arm):.0f} mm) ==========")
                for i, ((u, v), (x, y)) in enumerate(zip(path_px[:-1], path_arm[:-1]), 1):
                    print(f"  [{i}] 像素 ({u}, {v}) → X={x:.2f}, Y={y:.2f}")
                print(f"  畫線高度 Z={draw_z() - PEN_PRESS_MM:.3f} (紙面 {Z_PAPER:.3f} + 物體高度 "
                      f"{OBJ_HEIGHT_MM} mm - 下壓 {PEN_PRESS_MM} mm)")
                if not path_ok(path_arm):
                    print("⚠️ 路徑超出工作範圍，不執行")
                    continue
                cv2.imshow(win, render(frame, cnts, sel, path_px, None, "LOCKED - y: draw / other: cancel"))
                print("按 y 開始畫，其他鍵取消")
                if (cv2.waitKey(0) & 0xFF) != ord('y'):
                    print("已取消")
                    continue
                # --- DRAW ---
                draw_path(arm, path_arm)
                arm.home()
                time.sleep(1.0)
                print("\n✅ 畫完，回到預覽")

        arm.home()
        if input("\n要鬆開夾爪放下麥克筆嗎？(y/n) > ").strip().lower() == "y":
            arm.grip(close=False)
            print("✋ 已鬆開")
    finally:
        pipeline.stop()
        arm.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
