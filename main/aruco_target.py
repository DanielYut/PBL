# -*- coding: utf-8 -*-
"""用桌上四個 ArUco 當作手眼校正目標。

* 相容 OpenCV 4.6（舊 API）與 4.7+（ArucoDetector）。
* ARUCO_DICT = "auto" 時會自動試常見字典，找出你們標記用的是哪一種。
* 四個標記彼此的相對位置不用量：程式會從「同時看到多個標記」的照片自動估計，
  之後把所有看得到的角點一起拿來算位姿（比只用一個標記準很多）。

直接執行可即時檢查偵測：  python aruco_target.py
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import config

_DICT_CANDIDATES = ["DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_50", "DICT_7X7_50",
                    "DICT_ARUCO_ORIGINAL", "DICT_4X4_1000", "DICT_5X5_1000", "DICT_6X6_1000"]
_chosen = None


def _make_detector(name):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    if hasattr(cv2.aruco, "ArucoDetector"):            # OpenCV 4.7+
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        det = cv2.aruco.ArucoDetector(d, p)
        return lambda g: det.detectMarkers(g)[:2]
    p = cv2.aruco.DetectorParameters_create()           # OpenCV 4.6
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return lambda g: cv2.aruco.detectMarkers(g, d, parameters=p)[:2]


def _detect_with(name, gray):
    corners, ids = _make_detector(name)(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float32) for i, c in zip(ids.ravel(), corners)}


def detect(img):
    """回傳 {marker_id: 4x2 角點像素座標}。"""
    global _chosen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if config.ARUCO_DICT != "auto":
        return _detect_with(config.ARUCO_DICT, gray)
    if _chosen:
        return _detect_with(_chosen, gray)
    best, best_n = {}, 0
    for name in _DICT_CANDIDATES:
        res = _detect_with(name, gray)
        if len(res) > best_n:
            best, best_n, cand = res, len(res), name
    if best_n >= 2:
        _chosen = cand
        print(f"[ArUco] 偵測到字典 {cand}（可寫進 config.ARUCO_DICT 加速）")
    return best


def draw(img, det):
    for i, c in det.items():
        cv2.polylines(img, [c.astype(int)], True, (0, 255, 0), 2)
        cv2.putText(img, str(i), tuple(c[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)
    return img


def marker_local_corners():
    """單一標記四角在自己座標系 (mm)，順序與 OpenCV 偵測一致：左上、右上、右下、左下。"""
    h = config.ARUCO_SIZE_MM / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def _T(Rm, t):
    T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = np.ravel(t); return T


def single_pose(c, K, dist):
    ok, rv, tv = cv2.solvePnP(marker_local_corners(), c.astype(np.float64), K, dist,
                              flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return _T(cv2.Rodrigues(rv)[0], tv)


def estimate_layout(dets, K, dist, planar=True):
    """從多張照片估計各標記相對參考標記的位姿，回傳 (ref_id, {id: T_ref_m})。
    planar=True：假設所有標記在同一平面，用所有照片一起最佳化（建議）。"""
    count = {}
    for d in dets:
        for i in d:
            count[i] = count.get(i, 0) + 1
    ref = min(i for i in count if count[i] == max(count.values()))
    rel = {}
    for d in dets:
        if ref not in d:
            continue
        Tr = single_pose(d[ref], K, dist)
        for i, c in d.items():
            if i != ref:
                rel.setdefault(i, []).append(np.linalg.inv(Tr) @ single_pose(c, K, dist))
    layout = {ref: np.eye(4)}
    for i, Ts in rel.items():
        t = np.median([T[:3, 3] for T in Ts], axis=0)
        Rm = R.from_matrix([T[:3, :3] for T in Ts]).mean().as_matrix()
        layout[i] = _T(Rm, t)
    if planar and len(layout) > 1:
        layout = refine_layout_planar(dets, ref, layout, K, dist)
    return ref, layout


# ---------------- 平面 layout 最佳化 ----------------
# 標記都平貼在桌上 -> 限制在參考標記的平面 (z=0)，每個標記只剩 x, y, yaw 三個參數，
# 再用所有照片的角點一起最小化重投影誤差。比用單一標記位姿去推準很多（單一標記的傾角誤差很大）。

def _planar_T(x, y, yaw):
    T = np.eye(4); T[:3, :3] = R.from_euler("z", yaw).as_matrix(); T[:2, 3] = [x, y]; return T


def _rv_T(p):
    return _T(R.from_rotvec(p[:3]).as_matrix(), p[3:6])


def _T_rv(T):
    return np.r_[R.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def layout_params(layout, ids):
    return np.concatenate([[layout[i][0, 3], layout[i][1, 3],
                            R.from_matrix(layout[i][:3, :3]).as_euler("zyx")[0]] for i in ids])


def layout_from_params(p, ref, ids):
    L = {ref: np.eye(4)}
    for k, i in enumerate(ids):
        L[i] = _planar_T(*p[3 * k:3 * k + 3])
    return L


def reproj_residual(T_cam_ref, layout, det, K, dist):
    """det 中在 layout 內的所有角點，投影到影像後與偵測位置的差 (px)。"""
    loc = np.c_[marker_local_corners(), np.ones(4)]
    obj = [(layout[i] @ loc.T).T[:, :3] for i in det if i in layout]
    img = [det[i] for i in det if i in layout]
    rv = cv2.Rodrigues(T_cam_ref[:3, :3])[0]
    pp = cv2.projectPoints(np.vstack(obj), rv, T_cam_ref[:3, 3], K, dist)[0].reshape(-1, 2)
    return (pp - np.vstack(img)).ravel()


def refine_layout_planar(dets, ref, layout, K, dist):
    from scipy.optimize import least_squares
    ids = sorted(i for i in layout if i != ref)
    cams = [board_pose(d, layout, K, dist) for d in dets]
    use = [k for k, T in enumerate(cams) if T is not None]
    nL = 3 * len(ids)
    x0 = np.r_[layout_params(layout, ids), np.concatenate([_T_rv(cams[k]) for k in use])]

    def f(x):
        L = layout_from_params(x[:nL], ref, ids)
        return np.concatenate([reproj_residual(_rv_T(x[nL + 6 * j:nL + 6 * j + 6]), L, dets[k], K, dist)
                               for j, k in enumerate(use)])

    s = least_squares(f, x0, loss="huber", f_scale=1.0)
    rms0, rms1 = np.sqrt(np.mean(f(x0) ** 2)), np.sqrt(np.mean(f(s.x) ** 2))
    print(f"[ArUco] 平面 layout 最佳化：重投影 RMS {rms0:.2f} -> {rms1:.2f} px")
    return layout_from_params(s.x[:nL], ref, ids)


def board_pose(det, layout, K, dist):
    """用所有看得到、且在 layout 內的標記角點一起 solvePnP，回傳 T_cam_ref 或 None。"""
    obj, img = [], []
    loc = np.c_[marker_local_corners(), np.ones(4)]
    for i, c in det.items():
        if i in layout:
            obj.append((layout[i] @ loc.T).T[:, :3])
            img.append(c)
    if not obj:
        return None
    obj = np.vstack(obj); img = np.vstack(img).astype(np.float64)
    if len(obj) == 4:
        return single_pose(img, K, dist)
    ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    return _T(cv2.Rodrigues(rv)[0], tv) if ok else None


if __name__ == "__main__":
    from rs_camera import RSCamera
    cam = RSCamera()
    print("q 離開")
    while True:
        img = cam.read()
        det = detect(img)
        cv2.imshow("aruco check", draw(img.copy(), det))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.stop()
