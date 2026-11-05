# -*- coding: utf-8 -*-
"""用 pymodbus 讀取台達手臂目前位姿 (X,Y,Z mm, RX,RY,RZ deg)。

只做「讀取」，不會讓手臂移動。手眼校正時手臂用教導器(Teach Pendant)手動移動。
直接執行本檔可測試連線：  python robot_modbus.py
"""
import struct
import time

import config


def _words_to_int32(lo_or_hi_first, w0, w1):
    lo, hi = (w0, w1) if lo_or_hi_first else (w1, w0)
    return struct.unpack("<i", struct.pack("<HH", lo, hi))[0]


class DeltaRobot:
    def __init__(self, ip=config.ROBOT_IP, port=config.ROBOT_PORT, unit=config.ROBOT_UNIT_ID):
        from pymodbus.client import ModbusTcpClient  # pymodbus 3.x
        self.client = ModbusTcpClient(ip, port=port)
        self.unit = unit
        if not self.client.connect():
            raise ConnectionError(f"連不到手臂 {ip}:{port}，檢查網路線、IP 與控制器 Modbus 是否開啟")

    def _read_once(self, addr, count):
        # pymodbus 不同版本參數名稱不同：slave= (3.0~3.8) / device_id= (3.9+) / unit= (2.x)
        for kw in ("slave", "device_id", "unit"):
            try:
                return self.client.read_holding_registers(addr, count=count, **{kw: self.unit})
            except TypeError:
                continue
        raise RuntimeError("無法呼叫 read_holding_registers，請確認 pymodbus 版本")

    def _read_holding(self, addr, count, retries=3):
        """讀暫存器；控制器會把閒置太久的連線關掉，所以失敗時自動重連再讀。"""
        last = None
        for attempt in range(retries):
            try:
                if not self.client.connected:
                    self.client.connect()
                rr = self._read_once(addr, count)
                if not rr.isError():
                    return rr.registers
                last = rr
            except Exception as e:      # ConnectionException / ModbusIOException 等
                if isinstance(e, RuntimeError) and "pymodbus 版本" in str(e):
                    raise
                last = e
            print(f"[Robot] 讀取失敗，重新連線（第 {attempt + 1} 次）")
            self.client.close()
            time.sleep(0.5)
        raise RuntimeError(f"讀取暫存器 0x{addr:04X} 失敗（已重連 {retries} 次）: {last}")

    def read_pose(self):
        """回傳 [X, Y, Z, RX, RY, RZ]，單位 mm 與度。"""
        regs = self._read_holding(config.POSE_REG_START, config.POSE_REG_COUNT)
        vals = []
        for i in range(0, 12, 2):
            raw = _words_to_int32(config.WORD_ORDER_LOW_FIRST, regs[i], regs[i + 1])
            vals.append(raw * config.POSE_SCALE)
        return vals

    # ---------------- 移動（暫存器依組員 PBL/drv_modbus/send.py） ----------------
    REG_SPEED, REG_TARGET, REG_CMD, REG_DONE = 0x0324, 0x0330, 0x0300, 0x031F
    CMD_MOVP, CMD_MOVL = 301, 302

    def _write(self, addr, values):
        for attempt in range(3):
            try:
                if not self.client.connected:
                    self.client.connect()
                if len(values) == 1:
                    rr = self.client.write_register(addr, values[0], slave=self.unit)
                else:
                    rr = self.client.write_registers(addr, values, slave=self.unit)
                if not rr.isError():
                    return
                last = rr
            except Exception as e:
                last = e
            self.client.close(); time.sleep(0.3)
        raise RuntimeError(f"寫入暫存器 0x{addr:04X} 失敗: {last}")

    def move_to(self, pose, speed=10, linear=True, timeout=60.0):
        """移到 [X,Y,Z,RX,RY,RZ]（mm/度），等到位才回傳。speed 是控制器的速度百分比。"""
        regs = []
        for v in pose:
            raw = int(round(v / config.POSE_SCALE)) & 0xFFFFFFFF
            lo, hi = raw & 0xFFFF, raw >> 16
            regs += [lo, hi] if config.WORD_ORDER_LOW_FIRST else [hi, lo]
        self._write(self.REG_SPEED, [int(speed)])
        self._write(self.REG_TARGET, regs)
        self._write(self.REG_CMD, [self.CMD_MOVL if linear else self.CMD_MOVP])
        time.sleep(0.3)                       # 讓控制器先把「到位」旗標清掉
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._read_holding(self.REG_DONE, 1)[0] == 1:
                cur = self.read_pose()
                err = max(abs(a - b) for a, b in zip(cur[:3], pose[:3]))
                if err < 1.0:
                    return cur
            time.sleep(0.05)
        raise RuntimeError(f"移動逾時 {timeout}s（目標 {pose}），請看教導器是否有警報")

    def close(self):
        self.client.close()


if __name__ == "__main__":
    r = DeltaRobot()
    p = r.read_pose()
    print("目前位姿 X,Y,Z(mm) RX,RY,RZ(deg):")
    print("  " + ", ".join(f"{v:.3f}" for v in p))
    print("請和教導器畫面上的數字比對：一致才代表暫存器位址/單位/字序設定正確。")
    r.close()
