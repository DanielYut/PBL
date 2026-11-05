# -*- coding: utf-8 -*-
"""RealSense 影像串流的小工具。

config.CAMERA_STREAM：
  "auto"  = 有彩色鏡頭就用彩色，抓不到（例如被辨識成 D430i）就自動改用左 IR
  "color" = 強制彩色
  "ir"    = 強制左 IR（會關掉紅外線投射器，避免光點干擾 ArUco）

read() 一律回傳 3 通道 BGR（IR 灰階會複製成 3 通道），其他程式不用改。
若 config 設定的解析度相機不支援（常見原因：插在 USB 2.0 孔），會自動改用支援的解析度。
內參與手眼校正都會用同一個串流和解析度，不影響結果；但「換串流就要重做內參和手眼校正」。
"""
import time

import cv2
import numpy as np
import pyrealsense2 as rs

import config

_FALLBACKS = [(1280, 720, 30), (1280, 720, 15), (1280, 720, 6), (848, 480, 30),
              (640, 480, 30), (640, 480, 15), (640, 360, 30), (480, 270, 30), (424, 240, 30)]


def _device():
    devs = rs.context().query_devices()
    if len(devs) == 0:
        raise RuntimeError("找不到 RealSense：檢查 USB 線是否插好（建議插藍色 USB 3.0 孔）")
    d = devs[0]

    def info(k):
        return d.get_info(k) if d.supports(k) else "?"
    name, usb = info(rs.camera_info.name), info(rs.camera_info.usb_type_descriptor)
    fw, rec = info(rs.camera_info.firmware_version), info(rs.camera_info.recommended_firmware_version)
    print(f"[RealSense] {name}，USB {usb}，韌體 {fw}（建議 {rec}）")
    if usb.startswith("2"):
        print("  ※ 目前是 USB 2.x：換藍色 USB 3.0 孔、用原廠線，Type-C 頭也試著翻面插")
    has_rgb = any(s.get_info(rs.camera_info.name) == "RGB Camera" for s in d.query_sensors())
    if not has_rgb:
        print("  ※ 抓不到 RGB Camera（D435i 被辨識成 D430i 時就是這樣）")
    return d, has_rgb


class RSCamera:
    def __init__(self, stream=None):
        dev, has_rgb = _device()
        mode = (stream or getattr(config, "CAMERA_STREAM", "auto")).lower()
        if mode == "auto":
            mode = "color" if has_rgb else "ir"
        if mode == "color" and not has_rgb:
            raise RuntimeError("config.CAMERA_STREAM='color'，但相機沒有 RGB Camera；改成 'auto' 或 'ir'")
        self.mode = mode
        if mode == "color":
            self._stream, self._index, fmt = rs.stream.color, -1, rs.format.bgr8
        else:
            self._stream, self._index, fmt = rs.stream.infrared, 1, rs.format.y8

        self.pipeline = rs.pipeline()
        tries = [(config.COLOR_WIDTH, config.COLOR_HEIGHT, config.COLOR_FPS)] + _FALLBACKS
        last = None
        for w, h, fps in tries:
            cfg = rs.config()
            cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
            if mode == "color":
                cfg.enable_stream(self._stream, w, h, fmt, fps)
            else:
                cfg.enable_stream(self._stream, self._index, w, h, fmt, fps)
            try:
                self.profile = self.pipeline.start(cfg)
                self._cfg = cfg
                break
            except RuntimeError as e:
                last = e
        else:
            raise RuntimeError(f"相機不支援任何常用解析度（{mode}）: {last}")

        if mode == "ir":
            self._emitter_off()
        vsp = self._vsp()
        self.width, self.height = vsp.width(), vsp.height()
        print(f"[RealSense] 使用 {'彩色' if mode == 'color' else '左 IR'} "
              f"{self.width}x{self.height} @ {vsp.fps()} fps")
        for _ in range(30):            # 丟掉前幾張讓自動曝光穩定
            self.pipeline.wait_for_frames()

    def _emitter_off(self):
        ds = self.profile.get_device().first_depth_sensor()
        if ds.supports(rs.option.emitter_enabled):
            ds.set_option(rs.option.emitter_enabled, 0)
            print("[RealSense] 已關閉紅外線投射器（IR 影像不會有光點）")

    def _vsp(self):
        if self.mode == "color":
            return self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        return self.profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()

    def factory_intrinsics(self):
        """讀取目前串流的 RealSense 出廠內參，回傳 (K, dist)。"""
        i = self._vsp().get_intrinsics()
        K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]], dtype=np.float64)
        dist = np.array(i.coeffs[:5], dtype=np.float64)
        return K, dist

    def _wait(self):
        """等一組影像；逾時會重試，連續失敗就重啟串流（USB 短暫斷線時不會整支程式當掉）。"""
        for attempt in range(3):
            try:
                return self.pipeline.wait_for_frames(3000)
            except RuntimeError as e:
                print(f"[RealSense] 影像逾時（第 {attempt + 1} 次）：{e}")
                if attempt == 0:
                    continue
                print("[RealSense] 重新啟動串流…（若常發生：檢查 USB 線是否被手臂拉扯、接頭是否鬆動）")
                try:
                    self.pipeline.stop()
                except RuntimeError:
                    pass
                time.sleep(1.0)
                self.profile = self.pipeline.start(self._cfg)
                if self.mode == "ir":
                    self._emitter_off()
        raise RuntimeError("相機連續逾時，請重新插拔 USB 後再執行")

    def read(self):
        frames = self._wait()
        if self.mode == "color":
            return np.asanyarray(frames.get_color_frame().get_data()).copy()
        gray = np.asanyarray(frames.get_infrared_frame(1).get_data())
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def stop(self):
        self.pipeline.stop()


if __name__ == "__main__":
    cam = RSCamera()
    K, dist = cam.factory_intrinsics()
    print("K =\n", K, "\ndist =", dist)
    print("q 離開")
    while True:
        cv2.imshow(f"RealSense ({cam.mode})", cam.read())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.stop()
