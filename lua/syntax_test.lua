--[[
語法測試 (第一次上傳 path_runner.lua 之前先跑這支)

用途：確認控制器吃得懂手冊 (MS2 RL 手冊) 的語法，並確認 Python 和 Lua 的 DW 字序一致
執行前：用示教器把手臂移到工作區中間、下方空曠的地方 (會往 +X、+Y 各走 30 mm)
執行後要確認：
  1. 沒有語法錯誤
  2. 轉角 (第 2 個點) 有圓滑過去，沒有停一下
  3. 「結果」頁籤印出 123 和 123456
]]

SpdL(10)
AccL(30)
DecL(30)
PassMode("DISTANT", 100)
SetOverlapDistance(10)

local base = ReadRobotCPos()
print("start x=" .. base.x .. " y=" .. base.y .. " z=" .. base.z)

local x0, y0 = base.x, base.y
MovL(base ..X(x0 + 30) ..Y(y0), PASS())        -- 往 +X 30 mm (連續)
MovL(base ..X(x0 + 30) ..Y(y0 + 30), PASS())   -- 再往 +Y 30 mm (連續)
MovL(base ..X(x0) ..Y(y0))                      -- 回起點 (停)

WriteModbus(0x1000, "W", 123)
print(ReadModbus(0x1000, "W"))                  -- 應該印 123
WriteModbus(0x1002, "DW", 123456)
print(ReadModbus(0x1002, "DW"))                 -- 應該印 123456
print("done")
