"""
用控制器上的 RL 程式 (rl/pbl_path.lua) 走連續路徑。

Modbus 的 GO 指令一次只能送一個點，每段都會停；這裡改成：
  1. 把整條路徑的點寫進 Device Buffer (0x1020 起)
  2. 用 Modbus 啟動 RL 程式 (0x0220 = RL ID，0x0228 = 6 開啟並執行)
  3. 寫 0x1000 = 1，RL 程式讀出所有點，用 MovL + PASS() 連續走完
  4. 走完把 RL 停掉 (0x0228 = 3)，之後 Python 的 GO 指令才不會跟 RL 搶手臂

單獨執行 (不需要相機)：
  python rl_path.py test     只做自我測試，手臂不會動
  python rl_path.py square   在目前位置的同一高度畫 30 mm 方形 (空中，會動！)
"""
import time

RL_ID = 1                   # DRAStudio Pro 裡這個專案的 RL ID (0x0220)
RL_VER = 1                  # 要跟 pbl_path.lua 的 VER 一樣

REG_RL_ID, REG_RL_MODE, REG_RL_STATUS = 0x0220, 0x0228, 0x0213
REG_MOTION = 0x00E0         # 0 = 停止，1 = 運動中
REG_GO_CMD = 0x0300         # 寫 1000 = Motion Stop

BUF = 0x1000
MSG = 0x1F00                # RL 回報的文字 (0x1EFF = 長度)，點資料不能寫到這裡
PTS = 0x1020
REC = 10
MAX_PTS = (MSG - 1 - PTS) // REC
CHUNK = 10                  # 一次寫幾個 word (控制器一次讀 100 個會回 IllegalAddress，先保守用 10)

ST_RUN, ST_DONE, ST_TEST_DONE, ST_ERR = 2, 3, 6, 9
KIND_MOVL, KIND_MOVP = 0, 1


def _dw(val_mm):
    v = int(round(val_mm * 1000)) & 0xFFFFFFFF
    return [v & 0xFFFF, v >> 16]


def _from_dw(lo, hi):
    v = lo | (hi << 16)
    return (v - (1 << 32) if v & 0x80000000 else v) / 1000.0


class RLPathError(RuntimeError):
    pass


class RLPath:
    def __init__(self, arm, rl_id=RL_ID):
        self.arm = arm
        self.rl_id = rl_id

    # ---------- RL 程式啟動 / 停止 ----------
    def rl_status(self):
        return self.arm._read(REG_RL_STATUS, 1)[0]

    def start(self, timeout=8.0):
        """RL 已經在跑 (例如在 DRAStudio 手動按 Start) 就直接用，之後也不會把它停掉"""
        self.started_by_us = False
        if self.rl_status() == 1 and self.arm._read(BUF + 2, 1)[0] == RL_VER:
            return
        self.started_by_us = True
        self.arm._write(BUF + 2, 0)
        self.arm._write(BUF, 0)
        self.arm._write(REG_RL_ID, self.rl_id)
        self.arm._write(REG_RL_MODE, 6)              # Open & Run
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.arm._read(BUF + 2, 1)[0] == RL_VER:
                return
            time.sleep(0.1)
        raise RLPathError(f"RL 程式沒有啟動 (RL ID {self.rl_id}，RL 狀態 {self.rl_status()})。"
                          "請確認 DRAStudio Pro 已上傳 pbl_path.lua、RL ID 正確、控制器在 Auto 模式")

    def stop(self):
        if not getattr(self, "started_by_us", True):
            return
        try:
            self.arm._write(REG_RL_MODE, 3)
        except Exception as e:
            print(f"⚠️ 停止 RL 失敗：{e}")

    def abort(self):
        """緊急：先停運動再停 RL"""
        try:
            self.arm._write(REG_GO_CMD, 1000)
        finally:
            self.started_by_us = True             # 緊急狀況一律停掉 RL
            self.stop()

    # ---------- 交握 ----------
    def read_msg(self):
        """RL 寫在 0x1100 的文字訊息 (錯誤原因 / 除錯資訊)"""
        n = min(self.arm._read(MSG - 1, 1)[0], 250)
        if n == 0:
            return ""
        chars = []
        for i in range(0, n, 10):                 # 控制器一次讀太多會回 IllegalAddress
            try:
                chars += self.arm._read(MSG + i, min(10, n - i))
            except RuntimeError:
                for j in range(i, min(i + 10, n)):
                    chars += self.arm._read(MSG + j, 1)
        return bytes(c & 0xFF for c in chars).decode("utf-8", errors="replace")

    def _cmd(self, cmd, want, timeout):
        want = want if isinstance(want, tuple) else (want,)
        self.arm._write(MSG - 1, 0)
        self.arm._write(BUF, cmd)
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self.arm._read(BUF, 1)[0]
            if st in want:
                return
            if st == ST_ERR:
                raise RLPathError(f"RL 程式回報錯誤：{self.read_msg() or '(沒有訊息)'}")
            time.sleep(0.02)
        raise RLPathError(f"等 RL 回應逾時 (指令 {cmd}，0x1000 = {self.arm._read(BUF, 1)[0]})")

    def write_points(self, points):
        """points: [(x, y, z, speed%, kind, overlap_mm), ...]"""
        if not points:
            raise RLPathError("路徑是空的")
        if len(points) > MAX_PTS:
            raise RLPathError(f"點數 {len(points)} 超過上限 {MAX_PTS}")
        regs = []
        for x, y, z, spd, kind, ovl in points:
            regs += _dw(x) + _dw(y) + _dw(z)
            regs += [int(spd), int(kind), int(round(max(ovl, 0) * 10)), 0]
        for i in range(0, len(regs), CHUNK):
            self.arm._write(PTS + i, regs[i:i + CHUNK])
        self.arm._write(BUF + 1, len(points))

    def self_test(self, points):
        """確認：RL 讀到的第 1 點 = 寫進去的；RL 的目前位置 = Python 讀到的 (座標系一致)"""
        self.arm._write(BUF, 0)
        time.sleep(0.1)
        self._cmd(5, ST_TEST_DONE, 3.0)
        r = self.arm._read(0x1010, 12)
        echo = [_from_dw(r[i], r[i + 1]) for i in range(0, 6, 2)]
        rl_pose = [_from_dw(r[i], r[i + 1]) for i in range(6, 12, 2)]
        if max(abs(a - b) for a, b in zip(echo, points[0][:3])) > 0.01:
            raise RLPathError(f"RL 讀到的點 {echo} ≠ 寫入的 {points[0][:3]} (數值格式不一致)")
        py_pose = self.arm.pose()[:3]
        if max(abs(a - b) for a, b in zip(rl_pose, py_pose)) > 1.0:
            raise RLPathError(f"RL 的目前位置 {[round(v, 2) for v in rl_pose]} ≠ Modbus 讀到的 "
                              f"{[round(v, 2) for v in py_pose]}：座標系 (UF/TF) 不一致，不能執行")
        return echo, rl_pose

    # ---------- 執行 ----------
    def run(self, points, rpy, timeout=120.0, keep_running=False):
        """走完整條路徑才回傳；rpy = (RX, RY, RZ) 度"""
        self.arm._write(BUF + 3, 0)                  # 執行進度歸零 (呼叫端用它判斷手臂動了沒)
        self.start()
        try:
            self.write_points(points)
            self.arm._write(BUF + 4, _dw(rpy[0]) + _dw(rpy[1]) + _dw(rpy[2]))
            self.self_test(points)
            self.arm._write(BUF + 3, 0)
            self._cmd(1, (ST_RUN, ST_DONE), 3.0)
            t0 = time.time()
            last = -1
            while True:
                st = self.arm._read(BUF, 1)[0]
                if st == ST_DONE:
                    break
                if st == ST_ERR:
                    raise RLPathError(f"RL 執行路徑時出錯：{self.read_msg() or '(沒有訊息)'}")
                if self.rl_status() != 1:
                    raise RLPathError(f"RL 程式停了 (狀態 {self.rl_status()})，請看示教器 / DRAStudio 警報")
                if time.time() - t0 > timeout:
                    raise RLPathError("路徑執行逾時")
                i = self.arm._read(BUF + 3, 1)[0]
                if i != last:
                    last = i
                    print(f"\r  RL 執行中：第 {i}/{len(points)} 點", end="", flush=True)
                time.sleep(0.05)
            # 最後一段可能還在減速：等手臂真的停下來
            while self.arm._read(REG_MOTION, 1)[0] != 0 and time.time() - t0 < timeout:
                time.sleep(0.02)
            print(f"\r  ✅ RL 路徑走完：{len(points)} 點，{time.time() - t0:.1f} s          ")
        except BaseException:
            print()
            self.abort()
            raise
        if not keep_running:
            self.stop()
            time.sleep(0.3)


def _main():
    import sys
    import glue_edge as g
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    arm = g.Arm()
    rl = RLPath(arm)
    p = arm.pose()
    x, y, z = p[:3]
    print(f"目前位置 X {x:.2f}  Y {y:.2f}  Z {z:.2f}  RX {p[3]:.2f}  RY {p[4]:.2f}  RZ {p[5]:.2f}")
    if mode == "test":
        rl.start()
        try:
            pts = [(x, y, z, 5, KIND_MOVL, 0)]
            rl.write_points(pts)
            arm._write(BUF + 4, _dw(p[3]) + _dw(p[4]) + _dw(p[5]))
            echo, rl_pose = rl.self_test(pts)
            print(f"RL 訊息：{rl.read_msg()}")
            print(f"✅ 自我測試通過：RL 讀到的點 {echo}，RL 的目前位置 {rl_pose}")
        finally:
            rl.stop()
    elif mode in ("names", "probe"):
        import os
        cmd, done, fname, sep = ((7, 8, "rl_globals.txt", ",") if mode == "names"
                                 else (11, 12, "rl_probe.txt", " | "))
        rl.start()
        try:
            parts = []
            for page in range(200):
                arm._write(BUF + 1, page)
                rl._cmd(cmd, done, 30.0)
                s = rl.read_msg()
                if not s:
                    break
                parts.append(s)
                print(f"\r  讀取第 {page + 1} 頁…", end="", flush=True)
            text = "".join(parts)
            path = os.path.join(g.DATA_DIR, fname)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text.replace(sep, "\n"))
            print("\n" + text.replace(sep, "\n"))
            print(f"\n已存到 {path}")
        finally:
            rl.stop()
    elif mode == "nudge":
        if z < g.Z_MIN + 20:
            sys.exit(f"目前 Z {z:.1f} 太低，先把手臂移到空中")
        input("手臂會用 RL MovL 往上移 10 mm (速度 5%)，按 Enter 開始，Ctrl+C 取消")
        rl.run([(x, y, z + 10, 5, KIND_MOVL, 0)], p[3:6])
        q = arm.pose()
        print(f"移動後 X {q[0]:.2f}  Y {q[1]:.2f}  Z {q[2]:.2f}  RX {q[3]:.2f}  RY {q[4]:.2f}  RZ {q[5]:.2f}"
              f"  (預期 Z {z + 10:.2f}，其他不變)")
    elif mode == "square":
        s, ovl, spd = 30.0, 3.0, 5
        if z < g.Z_MIN + 20:
            sys.exit(f"目前 Z {z:.1f} 太低，先把手臂移到空中 (至少 Z_MIN + 20 = {g.Z_MIN + 20:.0f})")
        input(f"手臂會在目前高度畫 {s:.0f} mm 方形 (速度 {spd}%，轉角混合 {ovl} mm)，按 Enter 開始，Ctrl+C 取消")
        corners = [(x + s, y), (x + s, y + s), (x, y + s), (x, y)]
        pts = [(cx, cy, z, spd, KIND_MOVL, ovl) for cx, cy in corners]
        rl.run(pts, p[3:6])
    else:
        sys.exit("用法：python rl_path.py [test|names|probe|nudge|square]")


if __name__ == "__main__":
    _main()
