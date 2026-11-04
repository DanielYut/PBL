import cv2
import numpy as np
import time
import pyrealsense2 as rs
from pymodbus.client import ModbusTcpClient
from drv_modbus import send, request

# ==========================================
# 1. 初始化台達手臂 (Modbus TCP)
# ==========================================
ARM_IP = "192.168.1.1"
ARM_PORT = 502
client = ModbusTcpClient(host=ARM_IP, port=ARM_PORT, unit_id=2)

if not client.connect():
    print("❌ 無法連線至台達手臂，請檢查網路設定！")
    exit()
print("✅ 成功連線至台達手臂！")

# 初始點位 (Safe Home)
X_HOME, Y_HOME, Z_HOME = 343.176, -51.628, 704.34
RX_HOME, RY_HOME, RZ_HOME = -179.26, 1.185, -99.119

Z_TRAVEL = 520.000   # 移動抬升高度 (高於筆蓋與障礙物)
Z_GRASP = 410.243    # 筆蓋夾取與放置高度

# 分流放置目標點 [X, Y, Z]
DROP_ZONE_RED = [374.376, 58.249, Z_GRASP]    # 紅色筆蓋放置區
DROP_ZONE_BLUE = [374.376, -221.951, Z_GRASP]  # 藍色筆蓋放置區

# ==========================================
# 2. 手眼校正轉換 (Pixel -> Arm Coordinates)
# ==========================================
# 現場 4 個 ArUco Marker 的像素中心點 (u, v)
# 依序對應：[左上 P1, 右上 P4, 右下 P3, 左下 P2]
pts_pixel = np.float32([
    [157, 35],   # 左上 (ID 141)
    [545, 35],   # 右上
    [545, 390],  # 右下
    [92,  390]   # 左下 (ID 103)
])

# 對應 4 個點的手臂實體 X, Y (mm) 座標
pts_arm = np.float32([
    [374.376,  58.249],   # 左上 P1 手臂座標
    [607.466,  62.870],   # 右上 P4 手臂座標
    [607.466, -221.951],  # 右下 P3 手臂座標
    [374.376, -221.951]   # 左下 P2 手臂座標
])

# 計算單應性透視變換矩陣 H
H_matrix, _ = cv2.findHomography(pts_pixel, pts_arm)

def pixel_to_arm(u, v):
    """將像素座標 (u, v) 轉換為手臂 (X, Y) 座標"""
    pt = np.array([u, v, 1.0], dtype=np.float32).reshape(3, 1)
    arm_pt = np.dot(H_matrix, pt)
    arm_x = arm_pt[0][0] / arm_pt[2][0]
    arm_y = arm_pt[1][0] / arm_pt[2][0]
    return arm_x, arm_y

# ==========================================
# 3. 藍紅顏色辨識 (帶 ROI 濾除邊框雜訊)
# ==========================================
def detect_red_and_blue_caps(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    detected_targets = []

    # 建立 ROI 區域：涵蓋工作白紙區域，同時濾除四周邊框
    h, w, _ = img.shape
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[20:460, 80:600] = 255

    # --- 紅色 HSV 範圍 ---
    lower_red1 = np.array([0, 90, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([165, 90, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = (cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)) & roi_mask

    # --- 藍色 HSV 範圍 (提高 S, V 下限避開陰影) ---
    lower_blue = np.array([95, 90, 60])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue) & roi_mask

    color_masks = [('Red', mask_red), ('Blue', mask_blue)]

    for color_name, mask in color_masks:
        # 降噪形態學處理
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 400:  # 過濾小雜訊
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    u = int(M["m10"] / M["m00"])
                    v = int(M["m01"] / M["m00"])
                    arm_x, arm_y = pixel_to_arm(u, v)
                    
                    detected_targets.append({
                        'color': color_name,
                        'pixel': (u, v),
                        'arm_pos': (arm_x, arm_y),
                        'contour': cnt
                    })

    return detected_targets

# ==========================================
# 4. 手臂夾取與放置完整動作流程
# ==========================================
def pick_and_place(pick_x, pick_y, drop_x, drop_y):
    """
    執行單次夾取與放置動作
    """
    print(f"  └─ 1. 移動至夾取點上方 (Z={Z_TRAVEL})")
    send.Go_Position(client, pick_x, pick_y, Z_TRAVEL, RX_HOME, RY_HOME, RZ_HOME, 50)
    
    print(f"  └─ 2. 下降至夾取高度 (Z={Z_GRASP})")
    send.Go_Position(client, pick_x, pick_y, Z_GRASP, RX_HOME, RY_HOME, RZ_HOME, 30)
    time.sleep(0.5)
    
    print("  └─ 3. 夾爪/吸盤 閉合 ON")
    send.gripper_ON(client)
    time.sleep(1.0)
    
    print(f"  └─ 4. 抬升至安全高度 (Z={Z_TRAVEL})")
    send.Go_Position(client, pick_x, pick_y, Z_TRAVEL, RX_HOME, RY_HOME, RZ_HOME, 50)

    print(f"  └─ 5. 平移至目標放置點上方 (X={drop_x:.1f}, Y={drop_y:.1f})")
    send.Go_Position(client, drop_x, drop_y, Z_TRAVEL, RX_HOME, RY_HOME, RZ_HOME, 50)
    
    print(f"  └─ 6. 下降至放置高度 (Z={Z_GRASP})")
    send.Go_Position(client, drop_x, drop_y, Z_GRASP, RX_HOME, RY_HOME, RZ_HOME, 30)
    time.sleep(0.5)
    
    print("  └─ 7. 夾爪/吸盤 釋放 OFF")
    send.gripper_OFF(client)
    time.sleep(1.0)
    
    print(f"  └─ 8. 抬升離開放置區 (Z={Z_TRAVEL})")
    send.Go_Position(client, drop_x, drop_y, Z_TRAVEL, RX_HOME, RY_HOME, RZ_HOME, 50)

# ==========================================
# 5. 主流程
# ==========================================
def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    print("\n--- 手臂回初始點位 (Safe Home) ---")
    send.gripper_OFF(client)
    send.Go_Position(client, X_HOME, Y_HOME, Z_HOME, RX_HOME, RY_HOME, RZ_HOME, 50)
    time.sleep(2)

    try:
        print("\n[預覽模式] 請確認畫面中的標記是否正確，按下 's' 鍵開始夾取，按 'q' 離開...")
        targets = []
        
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            targets = detect_red_and_blue_caps(frame)

            debug_img = frame.copy()
            for t in targets:
                u, v = t['pixel']
                arm_x, arm_y = t['arm_pos']
                color = t['color']
                
                # 繪製辨識框與座標資訊
                draw_color = (0, 0, 255) if color == 'Red' else (255, 0, 0)
                cv2.drawContours(debug_img, [t['contour']], -1, draw_color, 2)
                cv2.circle(debug_img, (u, v), 5, (0, 255, 0), -1)
                cv2.putText(debug_img, f"{color}: ({arm_x:.1f}, {arm_y:.1f})", 
                            (u - 40, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            cv2.imshow("Preview - Press 's' to Start Picking", debug_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break
            elif key == ord('q'):
                return

        # 輸出最終判斷的標籤與點位
        print("\n================ 辨識結果輸出 ================")
        print(f"共辨識出 {len(targets)} 個筆蓋：")
        for idx, t in enumerate(targets, 1):
            u, v = t['pixel']
            arm_x, arm_y = t['arm_pos']
            print(f"[{idx}] 顏色: {t['color']:4s} | 像素: ({u}, {v}) -> 手臂座標: (X={arm_x:.2f}, Y={arm_y:.2f})")

        # 依顏色分別執行夾取與放置
        print("\n================ 開始自動夾取流程 ================")
        for idx, t in enumerate(targets, 1):
            color = t['color']
            pick_x, pick_y = t['arm_pos']
            
            # 依據顏色指派目標放置區
            if color == 'Red':
                drop_x, drop_y = DROP_ZONE_RED[0], DROP_ZONE_RED[1]
            else:
                drop_x, drop_y = DROP_ZONE_BLUE[0], DROP_ZONE_BLUE[1]

            print(f"\n🚀 [任務 {idx}/{len(targets)}] 夾取 [{color}] 筆蓋 @ (X={pick_x:.2f}, Y={pick_y:.2f}) ──> 放置於 @ (X={drop_x:.2f}, Y={drop_y:.2f})")
            pick_and_place(pick_x, pick_y, drop_x, drop_y)

        # 任務完成回到 Safe Home
        print("\n✅ 所有筆蓋夾取完成，手臂返回初始點位...")
        send.Go_Position(client, X_HOME, Y_HOME, Z_HOME, RX_HOME, RY_HOME, RZ_HOME, 50)

    finally:
        pipeline.stop()
        client.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
