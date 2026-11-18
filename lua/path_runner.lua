--[[
path_runner.lua —— 控制器上的「路徑執行器」
用 DRAStudio 上傳一次，記下它的 RL ID，填到 glue_edge.py 的 RUNNER_RL_ID。
之後由 Python 透過 Modbus 寫 0x0220 (RL ID) + 0x0228 = 6 (Open & Run) 啟動，每次啟動走一條路徑就結束。

Python 先下好筆 (筆尖已在路徑第 1 點)，再把「剩下的點」寫進暫存器：
  0x1000  W   狀態  Python 先清 0；Lua：1 = 執行中，2 = 完成，9 = 參數錯誤
  0x1002  DW  點數 N
  0x1004  DW  直線速度 (%)
  0x1006  DW  轉角平滑距離 (×1000，單位 mm；手冊範圍 1~100 mm)
  0x1008  DW  Lua 讀到的 N (回寫給 Python 檢查 DW 字序)
  0x1010~ DW  每點 x, y, z (×1000，單位 mm)，一點 6 個暫存器

姿態 (RX/RY/RZ、肘/肩/腕) 沿用啟動當下的手臂姿態 (Python 下筆時用的 Home 姿態)
]]

local ADDR_STATUS  = 0x1000
local ADDR_N       = 0x1002
local ADDR_SPEED   = 0x1004
local ADDR_OVERLAP = 0x1006
local ADDR_ECHO    = 0x1008
local ADDR_PTS     = 0x1010
local MAX_PTS      = 680        -- (0x2000 - 0x1010) / 6
local CHUNK        = 30         -- 一次讀幾個點

WriteModbus(ADDR_STATUS, "W", 1)

local n   = ReadModbus(ADDR_N, "DW")
local spd = ReadModbus(ADDR_SPEED, "DW")
local ovl = ReadModbus(ADDR_OVERLAP, "DW") / 1000
WriteModbus(ADDR_ECHO, "DW", n)

if n == nil or n < 1 or n > MAX_PTS or spd == nil or spd < 1 or spd > 100 then
    print("path_runner: bad params n=" .. tostring(n) .. " spd=" .. tostring(spd))
    WriteModbus(ADDR_STATUS, "W", 9)
else
    -- 1. 先把所有點讀進來 (移動中不再讀 Modbus，避免卡頓)
    local pts = {}
    local i = 0
    while i < n do
        local k = n - i
        if k > CHUNK then k = CHUNK end
        local raw = MultiReadModbus(ADDR_PTS + i * 6, k * 3, "DW")
        for j = 0, k - 1 do
            pts[i + j + 1] = { raw[j * 3 + 1] / 1000, raw[j * 3 + 2] / 1000, raw[j * 3 + 3] / 1000 }
        end
        i = i + k
    end

    -- 2. 連續走：除了最後一點，全部用 PASS (不停、轉角平滑)
    if ovl < 1 then ovl = 1 end
    SpdL(spd)
    AccL(50)
    DecL(50)
    PassMode("DISTANT", 100)
    SetOverlapDistance(ovl)

    local base = ReadRobotCPos()
    for idx = 1, n do
        local p = pts[idx]
        local tgt = base ..X(p[1]) ..Y(p[2]) ..Z(p[3])
        if idx < n then
            MovL(tgt, PASS())
        else
            MovL(tgt)               -- 最後一點確實停到位
        end
    end

    WriteModbus(ADDR_STATUS, "W", 2)
end
