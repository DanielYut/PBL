"""
只讓手臂回 HOME (拍照位置)，不開相機、不偵測任何東西。

HOME 座標、安全高度、速度都沿用 glue_edge.py 的設定 (X_HOME ... RZ_HOME、Z_TRAVEL、SPEED_FAST)，
改那邊的設定這裡就跟著變。

動作：
  1. 印出控制器狀態 (操作模式、Servo、速度倍率、RL、錯誤代碼)
  2. RL 程式在跑 / 暫停中 → 先停掉 (RL 沒關時控制器不吃 Modbus 移動指令)
  3. 目前位置比 Z_TRAVEL 低 → 先垂直抬到 Z_TRAVEL (避免筆拖過工件)
  4. 移到 HOME。送出後 3 秒手臂都沒動 → 停下來印出可能原因 (不會乾等 60 秒)

執行 (在 main 資料夾)：
  python go_home.py         會先問一次，按 Enter 才動
  python go_home.py -y      不問，直接回 HOME
"""
import math
import sys
import time

import glue_edge as g
import rl_path

REG_SERVO = (0x0006, 0x0007, 0x0000)   # J1J2、J3J4、J5J6：0x0101 = 兩軸都 ON
REG_MOTION = 0x00E0                    # 0 停止、1 運動中
REG_OVERRIDE = 0x0246                  # 速度倍率 (0.1 %)
START_TIMEOUT = 3.0                    # 送出指令後幾秒內要開始動


def status(arm):
    """印出狀態，回傳 (servo 全開?, 速度倍率 %)"""
    servo = [arm._read(a, 1)[0] for a in REG_SERVO]
    servo_on = all(s == 0x0101 for s in servo)
    ovr = arm._read(REG_OVERRIDE, 1)[0] / 10.0
    print(f"  {rl_path.RLPath(arm).diag()}")
    print(f"  Servo J1J2/J3J4/J5J6 = {' / '.join(f'0x{s:04X}' for s in servo)} "
          f"({'全部 ON' if servo_on else '⚠️ 有軸沒開'})，速度倍率 {ovr:.1f}%"
          + ("  ⚠️ 倍率太低，幾乎不會動" if ovr < 1 else ""))
    return servo_on, ovr


def stop_rl(arm):
    st = arm._read(rl_path.REG_RL_STATUS, 1)[0]
    if st == 0:
        return True
    print(f"⚠️ RL 程式沒關 (狀態 {st}：1 執行中 / 2 中斷點 / 3 暫停 / 4 Pre-run) → 先停掉")
    arm._write(rl_path.REG_RL_MODE, 3)
    t0 = time.time()
    while arm._read(rl_path.REG_RL_STATUS, 1)[0] != 0 and time.time() - t0 < 3.0:
        time.sleep(0.05)
    return arm._read(rl_path.REG_RL_STATUS, 1)[0] == 0


def move_checked(arm, x, y, z, label):
    """送出移動；START_TIMEOUT 秒內沒開始動就丟錯，有動就等到位"""
    p0 = arm.pose()
    if math.dist(p0[:3], (x, y, z)) < 1.0:
        print(f"  {label}：已經在目標位置")
        return
    print(f"  {label} → ({x:.1f}, {y:.1f}, {z:.1f})")
    arm.send_move(x, y, z, speed=g.SPEED_FAST)
    t0 = time.time()
    while time.time() - t0 < START_TIMEOUT:
        if arm._read(REG_MOTION, 1)[0] == 1 or math.dist(arm.pose()[:3], p0[:3]) > 0.5:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError(f"{label}：指令送出 {START_TIMEOUT:.0f} 秒手臂都沒動")
    arm.wait_idle(60.0)


def main():
    arm = g.Arm()
    try:
        x, y, z, rx, ry, rz = arm.pose()
        print(f"目前位置 X {x:.2f}  Y {y:.2f}  Z {z:.2f}  RX {rx:.2f}  RY {ry:.2f}  RZ {rz:.2f}")
        print(f"HOME     X {g.X_HOME:.2f}  Y {g.Y_HOME:.2f}  Z {g.Z_HOME:.2f}  "
              f"RX {g.RX_HOME:.2f}  RY {g.RY_HOME:.2f}  RZ {g.RZ_HOME:.2f}")
        if math.dist((x, y, z), (g.X_HOME, g.Y_HOME, g.Z_HOME)) < 1.0 and \
                max(abs(rx - g.RX_HOME), abs(ry - g.RY_HOME), abs(rz - g.RZ_HOME)) < 0.5:
            print("✅ 已經在 HOME 了，不用動")
            return
        print("控制器狀態：")
        servo_on, _ = status(arm)
        if not servo_on:
            sys.exit("❌ Servo 沒有全部開啟：請在示教器 Servo ON 後再試")
        if z < g.Z_TRAVEL:
            print(f"  (會先垂直抬到 Z_TRAVEL = {g.Z_TRAVEL:.0f}，再移到 HOME)")

        if "-y" not in sys.argv[1:]:
            input("手臂會移動，請確認周圍淨空。按 Enter 開始，Ctrl+C 取消")

        if not stop_rl(arm):
            sys.exit("❌ RL 停不下來，請在 DRAStudio / 示教器按 Stop 後再試")

        try:
            if z < g.Z_TRAVEL:
                move_checked(arm, x, y, g.Z_TRAVEL, "垂直抬起")
            move_checked(arm, g.X_HOME, g.Y_HOME, g.Z_HOME, "回 HOME")
        except RuntimeError as e:
            print(f"❌ {e}")
            print("目前控制器狀態：")
            status(arm)
            print("可能原因：\n"
                  "  - 示教器在 T1/T2 手動模式：要按住致能開關，或切到 Auto 模式\n"
                  "  - 有警報 (上面的錯誤代碼不是 0x0000)：在示教器清除警報\n"
                  "  - 速度倍率是 0 / 很低：在示教器調高\n"
                  "  - 示教器或 DRAStudio 正在控制手臂 (例如還在 JOG 畫面)：先離開\n"
                  "  - HOME 超出可達範圍：檢查 glue_edge.py 的 X_HOME ... RZ_HOME")
            sys.exit(1)
        x, y, z = arm.pose()[:3]
        print(f"✅ 已回 HOME (X {x:.2f}  Y {y:.2f}  Z {z:.2f})")
    except KeyboardInterrupt:
        print("\n已取消")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
