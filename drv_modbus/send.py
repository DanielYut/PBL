import time

from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.constants import Endian
from drv_modbus import request

MovP = 0
MovL = 1

POS_TOLERANCE = 1.0   # 跟目標差多少 mm 以內算到位


def _write(c, addr, value):
    """寫暫存器；控制器會把閒置太久的連線關掉，失敗時自動重連再寫 (最多 3 次)"""
    last = None
    for _ in range(3):
        try:
            request.ensure_connected(c)
            if isinstance(value, list):
                rr = c.write_registers(addr, value, 2)
            else:
                rr = c.write_register(addr, value, 2)
            if not rr.isError():
                return
            last = rr
        except Exception as e:
            last = e
        print(f"⚠️ 寫入 0x{addr:04X} 失敗，重新連線再試")
        c.close()
        time.sleep(0.3)
    raise RuntimeError(f"寫入暫存器 0x{addr:04X} 失敗：{last}")


def Go_Position(c, x, y, z, rx, ry, rz, speed = 100, mov = 0, block = True, timeout = 60.0):
    builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.LITTLE)
    builder.add_32bit_int(int(x  * 1000))
    builder.add_32bit_int(int(y  * 1000))
    builder.add_32bit_int(int(z  * 1000))
    builder.add_32bit_int(int(rx * 1000))
    builder.add_32bit_int(int(ry * 1000))
    builder.add_32bit_int(int(rz * 1000))
    payload = builder.to_registers()
    _write(c, 0x0324, speed)
    _write(c, 0x0330, payload)

    if mov == MovP:
        print("MovP")
        _write(c, 0x0300, 301)
    if mov == MovL:
        print("MovL")
        _write(c, 0x0300, 302)

    print("Start moving...")
    if block == False:
        return

    # 先等一下，讓控制器把上一個動作留下的「到位」旗標清掉，
    # 否則會誤判已到位，下一個指令在手臂還在動時送出而被忽略
    time.sleep(0.3)

    # 到位條件：旗標 = 1 而且 實際位置跟目標差 < POS_TOLERANCE
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if request.Get_Pose_Flag(c) == 1:
                pose = request.Get_TCP_Pose(c)
                if pose and max(abs(a - b) for a, b in zip(pose[:3], (x, y, z))) < POS_TOLERANCE:
                    print("Move done!")
                    return
        except Exception as e:      # 讀取時斷線 → 下一輪會自動重連
            print(f"⚠️ 讀取手臂狀態失敗：{e}")
            c.close()
            time.sleep(0.3)
        time.sleep(0.05)
    raise RuntimeError(f"移動逾時 {timeout}s，目標 ({x:.1f}, {y:.1f}, {z:.1f})。"
                       f"請看教導器是否有警報、點位是否超出範圍")

def Suction_ON(c):
    _write(c, 0x02FE, 1)

def Suction_OFF(c):
    _write(c, 0x02FE, 0)

def Jog_Position(c, x, y, z, rx, ry, rz):
    if x > 0:
        c.write_registers(0x0300, 601, 2)
    if x < 0:
        c.write_registers(0x0300, 602, 2)

    if y > 0:
        c.write_registers(0x0300, 603, 2)
    if y < 0:
        c.write_registers(0x0300, 604, 2)

    if z > 0:
        c.write_registers(0x0300, 605, 2)
    if z < 0:
        c.write_registers(0x0300, 606, 2)

    if rx > 0:
        c.write_registers(0x0300, 607, 2)
    if rx < 0:
        c.write_registers(0x0300, 608, 2)

    if ry > 0:
        c.write_registers(0x0300, 609, 2)
    if ry < 0:
        c.write_registers(0x0300, 610, 2)

    if rz > 0:
        c.write_registers(0x0300, 611, 2)
    if rz < 0:
        c.write_registers(0x0300, 612, 2)

def Jog_Stop(c):
    c.write_registers(0x0300, 0, 2)

def gripper_ON(c):
    _write(c, 0x02FE, 64)

def gripper_OFF(c):
    _write(c, 0x02FE, 128)
