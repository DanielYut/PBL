"""
只控制夾爪張合 (放筆 / 換筆用)，不開相機、手臂不會移動。

夾爪的 DO 設定沿用 glue_edge.py (REG_IO、GRIP_OPEN、GRIP_CLOSE)，改那邊這裡就跟著變。

執行 (在 main 資料夾)：
  python gripper.py            互動模式：o 打開 / c 夾緊 / q 離開 (直接按，不用 Enter)
  python gripper.py open       打開後結束
  python gripper.py close      夾緊後結束
"""
import sys
import time

import glue_edge as g


def main():
    arm = g.Arm()
    try:
        cmd = sys.argv[1].lower() if len(sys.argv) > 1 else ""
        if cmd in ("open", "o"):
            arm.grip(close=False)
            print("✋ 夾爪已打開")
            return
        if cmd in ("close", "c"):
            arm.grip(close=True)
            print("✊ 夾爪已夾緊")
            return
        if cmd:
            sys.exit("用法：python gripper.py [open|close]")

        print("o: 打開   c: 夾緊   q: 離開  (直接按鍵，不用 Enter)")
        print("放筆：按 o 打開 → 把筆放進夾爪 (筆尖朝下) → 扶著筆按 c 夾緊")
        g.flush_keys()
        while True:
            key = g.poll_key(1)
            time.sleep(0.05)                 # 沒開影像視窗時 waitKey 會馬上回來 → 自己等，不要空轉
            if key == ord('o'):
                arm.grip(close=False)
                print("✋ 打開")
            elif key == ord('c'):
                arm.grip(close=True)
                print("✊ 夾緊")
            elif key in (ord('q'), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        arm.close()


if __name__ == "__main__":
    main()
