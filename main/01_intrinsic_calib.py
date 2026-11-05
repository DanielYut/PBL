# -*- coding: utf-8 -*-
"""步驟 1：RealSense 彩色相機內參校正。

操作：
  python 01_intrinsic_calib.py          # 開相機拍照
  python 01_intrinsic_calib.py --offline  # 只用 data/intrinsic_imgs 裡已存的照片計算
  python 01_intrinsic_calib.py --factory  # 沒有棋盤格時：直接存 RealSense 出廠內參當作內參

視窗按鍵：
  空白鍵 = 偵測到棋盤時存一張（畫面角點變彩色才算偵測到）
  c      = 開始計算（建議至少 15 張）
  q      = 離開不計算

拍照要領：棋盤要出現在畫面各個位置（四角、中間），並且有傾斜 20~40 度的角度，
不要每張都正對；棋盤要平、不反光、整張都在畫面內。
"""
import glob
import os
import sys

import cv2
import numpy as np

import config

CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def board_object_points():
    cols, rows = config.BOARD_INNER_CORNERS
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * config.SQUARE_SIZE_MM
    return objp


def find_corners(gray):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, config.BOARD_INNER_CORNERS, flags)
    if ok:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRIT)
    return ok, corners


def capture():
    from rs_camera import RSCamera
    os.makedirs(config.INTRINSIC_IMG_DIR, exist_ok=True)
    cam = RSCamera()
    K_f, d_f = cam.factory_intrinsics()
    np.savez("data/factory_intrinsics.npz", K=K_f, dist=d_f)
    n = len(glob.glob(os.path.join(config.INTRINSIC_IMG_DIR, "*.png")))
    print(f"已有 {n} 張。空白鍵存檔，c 計算，q 離開")
    go = False
    while True:
        img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = find_corners(gray)
        show = img.copy()
        if ok:
            cv2.drawChessboardCorners(show, config.BOARD_INNER_CORNERS, corners, ok)
        cv2.putText(show, f"saved: {n}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.imshow("intrinsic calib", show)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" ") and ok:
            cv2.imwrite(os.path.join(config.INTRINSIC_IMG_DIR, f"{n:03d}.png"), img)
            n += 1
            print(f"存第 {n} 張")
        elif k == ord("c"):
            go = True
            break
        elif k == ord("q"):
            break
    cam.stop()
    cv2.destroyAllWindows()
    return go


def calibrate():
    files = sorted(glob.glob(os.path.join(config.INTRINSIC_IMG_DIR, "*.png")))
    objp = board_object_points()
    obj_pts, img_pts, used = [], [], []
    size = None
    for f in files:
        gray = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ok, corners = find_corners(gray)
        if ok:
            obj_pts.append(objp)
            img_pts.append(corners)
            used.append(f)
    print(f"{len(used)}/{len(files)} 張找到棋盤")
    if len(used) < 10:
        sys.exit("可用照片太少（<10），請多拍幾張")

    # 固定 k3=0：RealSense 鏡頭畸變很小，放開 k3 容易過度擬合造成深度誤差
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None,
                                                     flags=cv2.CALIB_FIX_K3)

    # 每張照片的重投影誤差，找出壞照片
    per_img = []
    for i in range(len(used)):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
        per_img.append(np.sqrt(np.mean(np.sum((proj - img_pts[i]) ** 2, axis=2))))
    print(f"\n整體重投影誤差 RMS = {rms:.3f} px  (< 0.5 很好，< 1.0 可用)")
    for f, e in sorted(zip(used, per_img), key=lambda x: -x[1])[:5]:
        print(f"  誤差最大的照片: {os.path.basename(f)}  {e:.3f} px")
    print("若某張明顯偏大（例如 > 2 倍平均），刪掉它後用 --offline 重算。")

    print("\n自行校正 K =\n", np.array2string(K, precision=2, suppress_small=True))
    print("畸變 dist =", np.array2string(dist.ravel(), precision=4))

    if os.path.exists("data/factory_intrinsics.npz"):
        fac = np.load("data/factory_intrinsics.npz")
        Kf = fac["K"]
        print("\n出廠內參 K =\n", np.array2string(Kf, precision=2, suppress_small=True))
        print(f"差異: fx {K[0,0]-Kf[0,0]:+.2f}, fy {K[1,1]-Kf[1,1]:+.2f}, "
              f"cx {K[0,2]-Kf[0,2]:+.2f}, cy {K[1,2]-Kf[1,2]:+.2f} px")
        print("fx/fy 差在 1~2% 內、cx/cy 差在 ~10 px 內，代表校正合理。")

    np.savez(config.INTRINSIC_FILE, K=K, dist=dist, rms=rms,
             width=size[0], height=size[1])
    print(f"\n已存檔 {config.INTRINSIC_FILE}")


def save_factory():
    from rs_camera import RSCamera
    cam = RSCamera()
    K, dist = cam.factory_intrinsics()
    w, h, mode = cam.width, cam.height, cam.mode
    cam.stop()
    np.savez("data/factory_intrinsics.npz", K=K, dist=dist, width=w, height=h, stream=mode)
    np.savez(config.INTRINSIC_FILE, K=K, dist=dist, rms=-1, width=w, height=h, stream=mode)
    print(f"串流 {mode}，解析度 {w}x{h}（之後手眼校正要用同一個）")
    print("RealSense 出廠內參 K =\n", np.array2string(K, precision=2, suppress_small=True))
    print("dist =", np.array2string(dist, precision=4))
    print(f"已存成 {config.INTRINSIC_FILE}，可直接進行手眼校正")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    if "--factory" in sys.argv:
        save_factory()
        sys.exit()
    if "--offline" in sys.argv or capture():
        calibrate()
