# -*- coding: utf-8 -*-
"""步驟 3：計算夾爪(TCP)到相機的轉換 T_gripper_cam（也就是夾爪與鏡頭的 offset）。

  python 03_handeye_solve.py

會做的事：
  1. 每張照片用 solvePnP 算出棋盤相對相機的位姿。
  2. 用 OpenCV calibrateHandEye 的 5 種方法計算，挑最一致的。
  3. 自動比較幾種 RX/RY/RZ 旋轉順序，找出台達手臂用的是哪一種。
  4. 自我檢查：棋盤是固定的，所以每一組算出的「棋盤在基座座標的位置」應該幾乎一樣，
     散佈越小代表 offset 越準。
"""
import csv
import os
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import config
from importlib import import_module

intr = import_module("01_intrinsic_calib")
import aruco_target as at

METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}
# 兩種常見定義（角度永遠依名稱對應：RX 繞 X、RY 繞 Y、RZ 繞 Z）
#   "xyz": R = Rz·Ry·Rx（固定軸依序 X→Y→Z，多數手臂用這種）
#   "zyx": R = Rx·Ry·Rz
EULER_CANDIDATES = ["xyz", "zyx"]


def euler_to_R(order, rx, ry, rz):
    ang = {"x": rx, "y": ry, "z": rz}
    return R.from_euler(order, [ang[c] for c in order], degrees=True)


def R_to_euler(order, rot):
    a = dict(zip(order, rot.as_euler(order, degrees=True)))
    return a["x"], a["y"], a["z"]


def to_T(Rm, t):
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = np.ravel(t)
    return T


def pose_to_T(pose, order):
    x, y, z, rx, ry, rz = pose
    return to_T(euler_to_R(order, rx, ry, rz).as_matrix(), [x, y, z])


def load_samples(K, dist):
    csv_path = os.path.join(config.HANDEYE_DIR, "poses.csv")
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))[1:]
    if config.TARGET == "aruco":
        return load_samples_aruco(rows, K, dist)
    global ARUCO
    ARUCO = None
    objp = intr.board_object_points()
    poses, T_cam_board = [], []
    for r in rows:
        img = cv2.imread(os.path.join(config.HANDEYE_DIR, r[0]))
        ok, corners = intr.find_corners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if not ok:
            print(f"  {r[0]} 找不到棋盤，略過")
            continue
        ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        T_cam_board.append(to_T(cv2.Rodrigues(rvec)[0], tvec))
        poses.append([float(v) for v in r[1:7]])
    return poses, T_cam_board


def load_samples_aruco(rows, K, dist):
    dets, keep = [], []
    for r in rows:
        d = at.detect(cv2.imread(os.path.join(config.HANDEYE_DIR, r[0])))
        if len(d) >= 2:
            dets.append(d); keep.append(r)
        else:
            print(f"  {r[0]} 標記少於 2 個，略過")
    ref, layout = at.estimate_layout(dets, K, dist)
    print(f"ArUco 參考標記 id={ref}；估計出的其他標記相對位置 (mm):")
    for i, T in sorted(layout.items()):
        print(f"  id {i}: ({T[0,3]:7.1f}, {T[1,3]:7.1f}, {T[2,3]:6.1f})")
    ids = sorted(layout)
    for a in ids:
        for b in ids:
            if a < b:
                dd = np.linalg.norm(layout[a][:3, 3] - layout[b][:3, 3])
                print(f"  中心距離 {a}-{b}: {dd:6.1f} mm")
    print("  ↑ 可以用尺量標記中心之間的距離比對，差很多代表 ARUCO_SIZE_MM 填錯")
    poses, T_cam_board, used = [], [], []
    for d, r in zip(dets, keep):
        T = at.board_pose(d, layout, K, dist)
        if T is not None:
            T_cam_board.append(T)
            poses.append([float(v) for v in r[1:7]])
            used.append(d)
    global ARUCO
    ARUCO = {"dets": used, "ref": ref, "layout": layout, "K": K, "dist": dist}
    return poses, T_cam_board


ARUCO = None   # ArUco 模式時存每組的偵測結果，給 refine_joint 用


def refine_joint(poses, T_grip_cam, order, aru):
    """整體微調：同時最佳化 T_gripper_cam、標記在基座的位姿、四個標記的平面位置，
    最小化所有照片所有角點的重投影誤差。比 calibrateHandEye 的封閉解準（它只用位姿、不看像素）。"""
    from scipy.optimize import least_squares
    K, dist, ref, dets = aru["K"], aru["dist"], aru["ref"], aru["dets"]
    ids = sorted(i for i in aru["layout"] if i != ref)
    Tbg = [pose_to_T(p, order) for p in poses]
    Tbr = [Tg @ T_grip_cam @ at.board_pose(d, aru["layout"], K, dist) for Tg, d in zip(Tbg, dets)]
    Tbr0 = to_T(R.from_matrix([T[:3, :3] for T in Tbr]).mean().as_matrix(),
                np.mean([T[:3, 3] for T in Tbr], axis=0))
    x0 = np.r_[at._T_rv(T_grip_cam), at._T_rv(Tbr0), at.layout_params(aru["layout"], ids)]

    def f(x):
        inv_gc = np.linalg.inv(at._rv_T(x[:6]))
        Tb = at._rv_T(x[6:12])
        L = at.layout_from_params(x[12:], ref, ids)
        return np.concatenate([at.reproj_residual(inv_gc @ np.linalg.inv(Tg) @ Tb, L, d, K, dist)
                               for Tg, d in zip(Tbg, dets)])

    s = least_squares(f, x0, loss="huber", f_scale=1.0)
    rms0, rms1 = np.sqrt(np.mean(f(x0) ** 2)), np.sqrt(np.mean(f(s.x) ** 2))
    T_gc = at._rv_T(s.x[:6])
    L = at.layout_from_params(s.x[12:], ref, ids)
    # 用跟前面一樣的指標（目標在基座座標的散佈）評估
    T_base_board = [Tg @ T_gc @ at.board_pose(d, L, K, dist) for Tg, d in zip(Tbg, dets)]
    P = np.array([T[:3, 3] for T in T_base_board])
    pe = np.linalg.norm(P - P.mean(axis=0), axis=1)
    Rmean = R.from_matrix([T[:3, :3] for T in T_base_board]).mean()
    re = np.array([(Rmean.inv() * R.from_matrix(T[:3, :3])).magnitude()
                   for T in T_base_board]) * 180 / np.pi
    return T_gc, pe, re, at._rv_T(s.x[6:12])[:3, 3], rms0, rms1


def solve(poses, T_cam_board, order, method):
    T_base_grip = [pose_to_T(p, order) for p in poses]
    Rg = [T[:3, :3] for T in T_base_grip]
    tg = [T[:3, 3] for T in T_base_grip]
    Rt = [T[:3, :3] for T in T_cam_board]
    tt = [T[:3, 3] for T in T_cam_board]
    Rc, tc = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=method)
    T_grip_cam = to_T(Rc, tc)
    # 一致性：棋盤在基座座標的位姿應固定
    T_base_board = [Tg @ T_grip_cam @ Tb for Tg, Tb in zip(T_base_grip, T_cam_board)]
    P = np.array([T[:3, 3] for T in T_base_board])
    pos_err = np.linalg.norm(P - P.mean(axis=0), axis=1)
    Rmean = R.from_matrix([T[:3, :3] for T in T_base_board]).mean()
    rot_err = np.array([(Rmean.inv() * R.from_matrix(T[:3, :3])).magnitude()
                        for T in T_base_board]) * 180 / np.pi
    return T_grip_cam, pos_err, rot_err, P.mean(axis=0)


def main():
    if not os.path.exists(config.INTRINSIC_FILE) and "--factory" not in sys.argv:
        sys.exit("找不到內參檔，請先執行 01_intrinsic_calib.py")
    # 預設用自行校正的內參；加 --factory 改用 RealSense 出廠內參（兩者都跑一次比較）
    src = "data/factory_intrinsics.npz" if "--factory" in sys.argv else config.INTRINSIC_FILE
    d = np.load(src)
    K, dist = d["K"], d["dist"]
    print(f"使用內參: {src}")
    poses, T_cam_board = load_samples(K, dist)
    n = len(poses)
    print(f"可用樣本 {n} 組")
    if n < 6:
        sys.exit("樣本太少（至少 6 組，建議 15~20 組）")

    # 姿態多樣性檢查
    rots = [euler_to_R(config.EULER_ORDER, *p[3:]) for p in poses]
    spread = max((rots[i].inv() * rots[j]).magnitude() for i in range(n) for j in range(n))
    print(f"樣本間最大旋轉差 = {np.degrees(spread):.1f} 度（建議 > 40 度）")

    results = []
    for order in EULER_CANDIDATES:
        for mname, m in METHODS.items():
            try:
                T, pe, re, pb = solve(poses, T_cam_board, order, m)
                results.append((pe.mean(), order, mname, T, pe, re, pb))
            except (cv2.error, ValueError):
                pass
    results.sort(key=lambda x: x[0])

    print("\n各旋轉順序的最佳一致性（目標位置散佈平均值，越小越好）:")
    seen = set()
    for r in results:
        if r[1] not in seen:
            seen.add(r[1])
            print(f"  {r[1]:>4s}: {r[0]:8.2f} mm  ({r[2]})")

    err, order, mname, T, pe, re, pb = results[0]

    # 自動剔除離群樣本（散佈 > 3 倍中位數且 > 3 mm），再重算一次
    bad = [i for i in range(len(pe)) if pe[i] > max(3 * np.median(pe), 3.0)]
    if bad and len(poses) - len(bad) >= 6:
        print(f"\n剔除離群樣本 index {bad} 後重算")
        poses = [p for i, p in enumerate(poses) if i not in bad]
        T_cam_board = [t for i, t in enumerate(T_cam_board) if i not in bad]
        if ARUCO is not None:
            ARUCO["dets"] = [d for i, d in enumerate(ARUCO["dets"]) if i not in bad]
        best = None
        for mname2, m in METHODS.items():
            try:
                r2 = solve(poses, T_cam_board, order, m)
                if best is None or r2[1].mean() < best[1][1].mean():
                    best = (mname2, r2)
            except (cv2.error, ValueError):
                pass
        mname, (T, pe, re, pb) = best
    if order != config.EULER_ORDER:
        print(f"\n※ 最一致的旋轉順序是 '{order}'，請把 config.py 的 EULER_ORDER 改成 \"{order}\"")

    if ARUCO is not None and "--no-refine" not in sys.argv:
        print(f"\n封閉解（{mname}）：散佈 平均 {pe.mean():.2f} mm, 最大 {pe.max():.2f} mm")
        T2, pe2, re2, pb2, rms0, rms1 = refine_joint(poses, T, order, ARUCO)
        print(f"整體微調（重投影 RMS {rms0:.2f} -> {rms1:.2f} px）：散佈 平均 {pe2.mean():.2f} mm, "
              f"最大 {pe2.max():.2f} mm")
        if pe2.mean() < pe.mean():
            T, pe, re, pb, mname = T2, pe2, re2, pb2, mname + "+refine"
        else:
            print("  微調沒有比較好，保留封閉解")

    print(f"\n=== 結果（順序 {order}，方法 {mname}）===")
    print(f"目標位置散佈: 平均 {pe.mean():.2f} mm, 最大 {pe.max():.2f} mm")
    print(f"目標角度散佈: 平均 {re.mean():.2f} 度, 最大 {re.max():.2f} 度")
    print("（平均 < 2 mm 很好；> 5 mm 要檢查樣本或手臂位姿讀取）")
    worst = np.argsort(-pe)[:3]
    print("散佈最大的樣本 index:", ", ".join(f"{i}({pe[i]:.1f}mm)" for i in worst))

    t = T[:3, 3]
    eul = R_to_euler(order, R.from_matrix(T[:3, :3]))
    print("\nT_gripper_cam（相機座標 -> 夾爪 TCP 座標）=\n",
          np.array2string(T, precision=4, suppress_small=True))
    print(f"\n夾爪到鏡頭 offset: X={t[0]:.2f}  Y={t[1]:.2f}  Z={t[2]:.2f} mm")
    print(f"                   RX={eul[0]:.2f} RY={eul[1]:.2f} RZ={eul[2]:.2f} 度 (順序 {order})")
    print("\n" + ("參考 ArUco 標記中心" if config.TARGET == "aruco" else "棋盤原點(第一個內角點)") + "在手臂基座座標約為: "
          f"X={pb[0]:.1f} Y={pb[1]:.1f} Z={pb[2]:.1f} mm")
    print("→ 可以讓手臂夾麥克筆移到這個點，看筆尖是否落在該點（ArUco 為參考標記的中心），當作快速驗證。")

    np.savez(config.HANDEYE_RESULT_FILE, T_gripper_cam=T, euler_order=order,
             method=mname, pos_err_mean=pe.mean())
    print(f"\n已存檔 {config.HANDEYE_RESULT_FILE}")


if __name__ == "__main__":
    main()
