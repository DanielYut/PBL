# -*- coding: utf-8 -*-
"""步驟 5：用筆尖驗證手眼校正。

(1) 筆尖校正（先做一次，換筆或重新夾筆就要重做）
      python 05_pen_touch.py --tool
    在桌上固定一個尖點（例如貼一張紙、畫一個小點）。用教導器讓「筆尖」碰到這個點，
    每次換一個姿態（RX、RY 傾斜 ±15~30 度、RZ 也轉），停穩在終端機按 Enter，收 4~6 次後輸入 c 計算。
    → 算出筆尖在 TCP 座標的位置，存到 data/pen_tool.npz。誤差 < 1 mm 才算好。
    ※ 教導器的工具座標(Tool)要跟手眼校正時一樣，不要改。

(2) 筆尖對點
      python 05_pen_touch.py
    手臂先停在俯拍點（看得到 ArUco）。畫面選目標標記（按 1~4），程式算出標記中心座標。
    之後每一步都要你按鍵，手臂才會動（速度 10%）：
      m : 移到目標正上方 50 mm（筆垂直朝下）
      n : 下降到上方 10 mm
      j : 下降到上方 2 mm   ← 先在這裡看筆尖是不是對準中心
      t : 碰到（0 mm）
      h : 回到上方 50 mm      b : 回到拍照點      q : 離開（會先回到上方 50 mm）
    ※ 手放在教導器緊急停止鈕旁邊。第一次請在 j（2 mm）就停下來看。
"""
import os
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import config
import aruco_target as at
from importlib import import_module
from robot_modbus import DeltaRobot

he = import_module("03_handeye_solve")
TOOL_FILE = "data/pen_tool.npz"
SPEED = 10                      # 控制器速度 %
HOVER, NEAR, JUST = 50.0, 10.0, 2.0


# ---------------- (1) 筆尖校正：pivot calibration ----------------
def tool_calib():
    robot = DeltaRobot()
    order = config.EULER_ORDER
    print("讓筆尖碰到同一個點，換姿態後按 Enter 記錄；輸入 c 計算；q 離開")
    poses = []
    while True:
        s = input(f"[已記錄 {len(poses)} 次] Enter=記錄 / c=計算 / q=離開 > ").strip().lower()
        if s == "q":
            return
        if s == "c":
            if len(poses) < 4:
                print("至少要 4 次"); continue
            break
        p = robot.read_pose()
        poses.append(p)
        print("  TCP =", " ".join(f"{v:.2f}" for v in p))
    robot.close()
    # p_i + R_i·t = P  ->  [R_i  -I] [t; P] = -p_i
    A, b = [], []
    for p in poses:
        T = he.pose_to_T(p, order)
        A.append(np.hstack([T[:3, :3], -np.eye(3)])); b.append(-T[:3, 3])
    A, b = np.vstack(A), np.concatenate(b)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    t_tool, P = x[:3], x[3:]
    res = (A @ x - b).reshape(-1, 3)
    err = np.linalg.norm(res, axis=1)
    rots = [R.from_matrix(he.pose_to_T(p, order)[:3, :3]) for p in poses]
    spread = max((a.inv() * c).magnitude() for a in rots for c in rots) * 180 / np.pi
    print(f"\n姿態最大角度差 {spread:.1f} 度（建議 > 30 度，太小會不準）")
    print(f"筆尖在 TCP 座標: X={t_tool[0]:.2f} Y={t_tool[1]:.2f} Z={t_tool[2]:.2f} mm "
          f"（長度 {np.linalg.norm(t_tool):.1f} mm）")
    print(f"碰到的點在基座座標: X={P[0]:.2f} Y={P[1]:.2f} Z={P[2]:.2f} mm")
    print(f"每次誤差: {np.round(err, 2)} mm，平均 {err.mean():.2f} mm（< 1 mm 很好）")
    os.makedirs("data", exist_ok=True)
    np.savez(TOOL_FILE, t_tool=t_tool, pivot_point=P, err=err.mean())
    print(f"已存檔 {TOOL_FILE}")


# ---------------- (2) 筆尖對點 ----------------
def load_layout(K, dist):
    import csv
    rows = list(csv.reader(open(os.path.join(config.HANDEYE_DIR, "poses.csv"), newline="")))[1:]
    dets = [d for d in (at.detect(cv2.imread(os.path.join(config.HANDEYE_DIR, r[0]))) for r in rows)
            if len(d) >= 2]
    return at.estimate_layout(dets, K, dist)


def marker_centers_base(cam, robot, layout, T_gc, K, dist, order, n=15):
    """平均 n 張影像，用整組標記一起算位姿（比單一標記準），回傳 {id: 基座座標中心}。"""
    pose = robot.read_pose()
    T_bg = he.pose_to_T(pose, order)
    Ts, last = [], None
    for _ in range(n):
        img = cam.read(); det = at.detect(img); last = (img, det)
        T = at.board_pose(det, layout, K, dist)
        if T is not None:
            Ts.append(T)
    if not Ts:
        return {}, last, pose
    T_cr = np.eye(4)
    T_cr[:3, :3] = R.from_matrix([T[:3, :3] for T in Ts]).mean().as_matrix()
    T_cr[:3, 3] = np.median([T[:3, 3] for T in Ts], axis=0)
    T_br = T_bg @ T_gc @ T_cr
    return {i: (T_br @ Tm)[:3, 3] for i, Tm in layout.items()}, last, pose


def tcp_for_tip(P, rz, t_tool, order):
    """筆尖要在 P、筆垂直朝下 (RX=180, RY=0, RZ=rz) 時的 TCP 位姿。"""
    Rm = he.euler_to_R(order, 180.0, 0.0, rz).as_matrix()
    p = np.asarray(P) - Rm @ t_tool
    return [float(p[0]), float(p[1]), float(p[2]), 180.0, 0.0, float(rz)]


def touch():
    from rs_camera import RSCamera
    if not os.path.exists(TOOL_FILE):
        sys.exit("找不到 data/pen_tool.npz，請先執行  python 05_pen_touch.py --tool")
    t_tool = np.load(TOOL_FILE)["t_tool"]
    res = np.load(config.HANDEYE_RESULT_FILE)
    T_gc, order = res["T_gripper_cam"], str(res["euler_order"])
    intr = np.load(config.INTRINSIC_FILE); K, dist = intr["K"], intr["dist"]
    print("載入標記相對位置…")
    ref, layout = load_layout(K, dist)
    cam, robot = RSCamera(), DeltaRobot()
    photo_pose = robot.read_pose()
    print("拍照點 =", " ".join(f"{v:.1f}" for v in photo_pose))

    centers, (img, det), _ = marker_centers_base(cam, robot, layout, T_gc, K, dist, order)
    if not centers:
        sys.exit("看不到標記，請把手臂移到看得到 ArUco 的拍照點")
    for i, p in sorted(centers.items()):
        print(f"  id {i} 中心: X={p[0]:.1f} Y={p[1]:.1f} Z={p[2]:.1f}")
    target = ref
    rz = photo_pose[5]
    level = None
    print(__doc__.split("(2) 筆尖對點")[1])

    def go(h):
        nonlocal level
        P = centers[target] + np.array([0, 0, h])
        goal = tcp_for_tip(P, rz, t_tool, order)
        print(f"→ 筆尖到 id {target} 上方 {h:.0f} mm：TCP 目標 " + " ".join(f"{v:.1f}" for v in goal))
        robot.move_to(goal, speed=SPEED, linear=True)
        level = h

    while True:
        show = at.draw(img.copy(), det)
        cv2.putText(show, f"target id {target}  level {level}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("pen touch", show)
        k = cv2.waitKey(30) & 0xFF
        if k == 255:
            continue
        ch = chr(k)
        try:
            if ch in "123456789" and int(ch) in centers:
                if level is not None and level < HOVER:
                    print("請先按 h 回到上方 50 mm 再換目標"); continue
                target = int(ch); print(f"目標改為 id {target}")
            elif ch == "m":
                go(HOVER)
            elif ch == "n":
                if level is None:
                    print("請先按 m"); continue
                go(NEAR)
            elif ch == "j":
                if level is None or level > NEAR:
                    print("請先按 n 到 10 mm"); continue
                go(JUST)
            elif ch == "t":
                if level != JUST:
                    print("請先按 j 到 2 mm，確認對準再碰"); continue
                go(0.0)
                p = robot.read_pose()
                print("  碰到時 TCP =", " ".join(f"{v:.2f}" for v in p),
                      "\n  → 看筆尖離標記中心多遠（用尺量偏移的方向和距離）")
            elif ch == "h":
                if level is not None:
                    go(HOVER)
            elif ch == "b":
                if level is not None and level < HOVER:
                    go(HOVER)
                print("→ 回到拍照點"); robot.move_to(photo_pose, speed=SPEED, linear=True); level = None
                centers, (img, det), _ = marker_centers_base(cam, robot, layout, T_gc, K, dist, order)
            elif ch == "q":
                if level is not None and level < HOVER:
                    go(HOVER)
                break
        except RuntimeError as e:
            print("※", e)
    cam.stop(); robot.close()


if __name__ == "__main__":
    tool_calib() if "--tool" in sys.argv else touch()
