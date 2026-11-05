# -*- coding: utf-8 -*-
"""步驟 4：驗證手眼校正（只讀取，不會讓手臂移動）。

  python 04_verify.py

畫面上會即時顯示每個 ArUco 中心在「手臂基座座標」的位置 (X, Y, Z)。

驗證 A（不用碰東西）：用教導器把手臂移到不同位置、傾斜、旋轉，
    標記座標應該幾乎不變。按 空白鍵 記錄一次，程式會統計每個標記的散佈。
    → 散佈 < 2~3 mm 代表 offset 正確；隨手臂轉動而漂移很多代表 offset 有問題。

驗證 B（筆尖對點）：先在拍照點按空白鍵記錄標記座標，再按 v 讓手臂「原地」轉成
    RX=180、RY=0（筆垂直朝下，只有這一步是程式在動手臂，要按兩次 v 確認）。
    之後用教導器在「基座座標」只做平移，讓筆尖碰到標記正中心，按 p 比對 dX、dY。

q 離開時會印出總結。
"""
import time

import cv2
import numpy as np

import config
import aruco_target as at
from importlib import import_module
from rs_camera import RSCamera
from robot_modbus import DeltaRobot

he = import_module("03_handeye_solve")
VERT_SPEED = 10      # v 鍵轉垂直時的速度 %


def main():
    res = np.load(config.HANDEYE_RESULT_FILE)
    T_gc, order = res["T_gripper_cam"], str(res["euler_order"])
    intr = np.load(config.INTRINSIC_FILE)
    K, dist = intr["K"], intr["dist"]
    cam = RSCamera()
    robot = DeltaRobot()
    print(f"offset: {np.round(T_gc[:3, 3], 2)} mm，旋轉順序 {order}")
    print("空白鍵 = 記錄一次標記座標；v = 原地轉垂直；p = 印出目前 TCP；q = 離開")

    log = {}                     # id -> list of base xyz
    pose, t_pose = None, 0.0
    while True:
        img = cam.read()
        if time.time() - t_pose > 0.3:          # 約 3 Hz 讀手臂，避免太頻繁
            pose, t_pose = robot.read_pose(), time.time()
        det = at.detect(img)
        T_bg = he.pose_to_T(pose, order)
        centers = {}
        for i, c in det.items():
            T_cm = at.single_pose(c, K, dist)
            p = (T_bg @ T_gc @ T_cm)[:3, 3]
            centers[i] = p
        show = at.draw(img.copy(), det)
        for i, c in det.items():
            p = centers[i]
            org = tuple(c.mean(axis=0).astype(int) + np.array([-80, 40]))
            cv2.putText(show, f"{p[0]:.1f},{p[1]:.1f},{p[2]:.1f}", org,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(show, "TCP " + " ".join(f"{v:.1f}" for v in pose), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("verify (space=record, p=TCP, q=quit)", show)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            pose = robot.read_pose()             # 按下當下重新讀一次，確保是停穩的位姿
            print(f"\n記錄 #{len(next(iter(log.values()), [])) + 1}  TCP={np.round(pose, 1)}")
            for i, p in sorted(centers.items()):
                log.setdefault(i, []).append(p)
                print(f"  id {i}: X={p[0]:8.1f}  Y={p[1]:8.1f}  Z={p[2]:8.1f}")
        elif k == ord("p"):
            p = robot.read_pose()
            print(f"\n目前 TCP: X={p[0]:.1f} Y={p[1]:.1f} Z={p[2]:.1f}  RX={p[3]:.1f} RY={p[4]:.1f} RZ={p[5]:.1f}")
            for i, pts in sorted(log.items()):
                m = np.mean(pts, axis=0)
                d = np.array(p[:3]) - m
                print(f"  與 id {i} 平均座標差: dX={d[0]:+.1f} dY={d[1]:+.1f} dZ={d[2]:+.1f} "
                      f"（水平距離 {np.hypot(d[0], d[1]):.1f} mm）")
        elif k == ord("v"):
            p = robot.read_pose()
            goal = [p[0], p[1], p[2], 180.0, 0.0, p[5]]
            print(f"\n原地轉成垂直：RX {p[3]:.2f}->180, RY {p[4]:.2f}->0（XYZ、RZ 不變，速度 {VERT_SPEED}%）")
            print("  ※ 筆尖會小幅擺動，請確認筆尖離桌面還有距離。再按一次 v 確認，其他鍵取消")
            if (cv2.waitKey(0) & 0xFF) == ord("v"):
                try:
                    robot.move_to(goal, speed=VERT_SPEED, linear=True)
                    print("  完成。之後用教導器在『基座座標』只做平移，姿態就會保持垂直")
                except RuntimeError as e:
                    print("  ※", e)
            else:
                print("  已取消")
        elif k == ord("q"):
            break
    cam.stop(); robot.close()

    if log:
        print("\n=== 驗證 A 總結：各標記在不同姿態下的座標散佈 ===")
        for i, pts in sorted(log.items()):
            P = np.array(pts); m = P.mean(axis=0)
            e = np.linalg.norm(P - m, axis=1)
            print(f"  id {i}: 平均 ({m[0]:.1f}, {m[1]:.1f}, {m[2]:.1f})，{len(P)} 次，"
                  f"散佈 平均 {e.mean():.2f} mm / 最大 {e.max():.2f} mm")
        print("  散佈 < 2~3 mm：offset 正確；若隨傾斜方向系統性漂移，代表 offset 的旋轉或 Z 有誤差")


if __name__ == "__main__":
    main()
