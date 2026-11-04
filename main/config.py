# -*- coding: utf-8 -*-
"""共用設定：所有校正程式都從這裡讀參數。到實驗室先改這個檔案。"""

# ---------------- 相機 (RealSense) ----------------
# "auto" = 有彩色鏡頭用彩色，沒有就用左 IR；"color" / "ir" = 強制指定
# 換串流（彩色 <-> IR）就要重做內參和手眼校正，兩者不能混用
CAMERA_STREAM = "auto"
COLOR_WIDTH = 1280      # 內參校正與手眼校正必須用「同一個解析度」（IR 也用這組設定）
COLOR_HEIGHT = 720
COLOR_FPS = 30

# ---------------- 校正目標 ----------------
# "aruco" = 用桌上四個 ArUco 標記；"chessboard" = 用印出來的棋盤格
TARGET = "aruco"
ARUCO_SIZE_MM = 50.0     # 用尺量「黑色正方形外框」的邊長(mm)，不含外面白邊！
ARUCO_DICT = "DICT_5X5_50"  # 2026-11-04 實機判斷結果；換標記時改回 "auto" 重新判斷

# ---------------- 棋盤格校正板（TARGET="chessboard" 才用到） ----------------
# 「內角點」數量 = 格子數 - 1。例如 10x7 格的棋盤 -> (9, 6)
BOARD_INNER_CORNERS = (9, 6)
SQUARE_SIZE_MM = 25.0   # 印出來後用尺量實際一格邊長(mm)，量得越準越好

# ---------------- 台達手臂 Modbus TCP ----------------
ROBOT_IP = "192.168.1.1"   # 改成手臂控制器 IP
ROBOT_PORT = 502
ROBOT_UNIT_ID = 2          # 台達 DRV 範例常用 2；若讀不到可試 1

# 讀取目前位姿的暫存器（請用課堂範例程式或手冊核對！）
POSE_REG_START = 0x00F0    # X,Y,Z,RX,RY,RZ 起始位址
POSE_REG_COUNT = 12        # 6 個值 x 每個 32-bit(2 個暫存器)
POSE_SCALE = 0.001         # 原始整數 x 0.001 -> mm / 度
WORD_ORDER_LOW_FIRST = True  # 32-bit 值是低位字在前(True)還是高位字在前(False)

# 手臂姿態角 RX,RY,RZ 的旋轉順序："xyz" 代表 R = Rz·Ry·Rx，"zyx" 代表 R = Rx·Ry·Rz
# 不確定的話保持 "xyz"，03 程式會自動比較兩種順序並告訴你哪一種才對
EULER_ORDER = "xyz"

# ---------------- 檔案位置 ----------------
INTRINSIC_IMG_DIR = "data/intrinsic_imgs"
INTRINSIC_FILE = "data/intrinsics.npz"
HANDEYE_DIR = "data/handeye"
HANDEYE_RESULT_FILE = "data/T_gripper_cam.npz"
