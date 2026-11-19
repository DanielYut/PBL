-- ============================================================
-- PBL 連續路徑執行器 (台達 DRV，MS2 機器人語言)
-- 平常不用下載這個檔：控制器上裝的是 rl/loader.lua (只裝一次)，
-- main/rl_path.py 每次啟動 RL 時會把這個檔透過 Modbus 送進去執行。改完直接跑 Python 就生效。
-- (也可以像以前一樣直接放進 DRAStudio 專案的 main.lua，一樣能用)
--
-- 為什麼需要它：Modbus 的 GO 指令 (0x0300 = 301/302) 一次只能送一個點，每段都會停；
-- RL 程式裡的 MovL + PASS() 可以讓控制器在轉角處提早轉彎，整條路徑連續不停。
--
-- 交握 (Device Buffer 0x1000~0x1FFF，全部用 16-bit word 讀寫，32-bit 數值低位字在前)：
--   0x1000  指令/狀態：Python 寫 1 = 執行路徑、5 = 自我測試
--                      RL 回 2 = 執行中、3 = 完成、6 = 測試完成、9 = 錯誤
--   0x1001  點數 N
--   0x1002  版本號 (RL 啟動後寫入，Python 用來確認 RL 已經在跑)
--   0x1003  目前執行到第幾點
--   0x1004  RX、RY、RZ (各 2 word，0.001 度)
--   0x100A  PASS 模式：0 = DISTANT (轉角磨圓，每段仍加減速)，1 = DEC (速度疊合，連續不減速)
--   0x100B  DEC 模式的疊合 % (1~100，100 = 前段減速和下段加速完全重疊)
--   0x1010  自我測試回傳區：第 1 點 X、Y、Z，RL 讀到的目前位置 X、Y、Z (各 2 word，0.001 mm)
--   0x1020  起每點 10 word：X、Y、Z (各 2 word，0.001 mm)、速度 %、種類 (0 = MovL，1 = MovP)、
--           轉角混合距離 (0.1 mm，0 = 到位才走下一段)、保留 (最多到 0x1EFE，約 380 點)
--   0x1EFF  訊息長度；0x1F00 起 訊息文字 (每 word 一個字元，錯誤原因 / 除錯資訊)
-- ============================================================

local BASE = 0x1000
local PTS = 0x1020
local REC = 10
local VER = 3   -- 2：加 PASS DEC 模式 (0x100A/0x100B)；3：改由 loader.lua 載入

local function u16(v)
    if v < 0 then v = v + 65536 end
    return v
end

local function s32(lo, hi)
    local v = u16(lo) + u16(hi) * 65536
    if v >= 2147483648 then v = v - 4294967296 end
    return v
end

local function w16(v)
    if v > 32767 then v = v - 65536 end
    return v
end

local function writeDW(addr, val)
    local v = math.floor(val * 1000 + 0.5)
    if v < 0 then v = v + 4294967296 end
    WriteModbus(addr, "W", w16(v % 65536))
    WriteModbus(addr + 1, "W", w16(math.floor(v / 65536)))
end

-- 把文字寫進 0x1F00 起 (每個 word 一個字元，0x1EFF = 長度)，Python 讀出來顯示
local MSG = 0x1F00
local function report(msg)
    msg = string.sub(tostring(msg), 1, 250)
    WriteModbus(MSG - 1, "W", #msg)
    for i = 1, #msg do
        WriteModbus(MSG + i - 1, "W", string.byte(msg, i))
    end
end

-- 讀目前位置的函式：手冊範例寫 ReadRobotCPos，語法欄寫 ReadRobotCPOS，實機兩個哪個存在不一定
local CPOS_NAME, CPOS = nil, nil
for _, name in ipairs({"ReadRobotCPos", "ReadRobotCPOS", "ReadRobotWorldPos"}) do
    local fn = _G and _G[name]
    if CPOS == nil and fn ~= nil then
        CPOS_NAME, CPOS = name, fn
    end
end

local function readRec(i)
    local r = MultiReadModbus(PTS + (i - 1) * REC, REC, "W")
    return s32(r[1], r[2]) / 1000, s32(r[3], r[4]) / 1000, s32(r[5], r[6]) / 1000,
           u16(r[7]), u16(r[8]), u16(r[9]) / 10
end

-- 這台韌體沒有 CreatePoint / GP / ReadRobotCPos (2026-11-18 用 probe 查過)。
-- 點位是一個表 (跟 P[1] 同格式)：x/y/z 單位 0.001 mm、a/b/c 單位 0.001 度 (= RX/RY/RZ)，
-- 另有 Elbow/Shoulder/Flip/UF/TF/Coord/PS/Active/JntRotCnt。
-- 做法：複製 P[1] (含 metatable) 當模板，填入座標和目前姿態。
local function copyTable(src)
    local t = {}
    for k, v in pairs(src) do
        if type(v) == "table" then
            local c = {}
            for i, w in pairs(v) do c[i] = w end
            t[k] = c
        else
            t[k] = v
        end
    end
    local mt = getmetatable(src)
    if mt ~= nil then setmetatable(t, mt) end
    return t
end

local function makePoint(x, y, z, a, b, c)
    local p = copyTable(P[1])
    p.x = math.floor(x * 1000 + 0.5)
    p.y = math.floor(y * 1000 + 0.5)
    p.z = math.floor(z * 1000 + 0.5)
    p.a = math.floor(a * 1000 + 0.5)
    p.b = math.floor(b * 1000 + 0.5)
    p.c = math.floor(c * 1000 + 0.5)
    p.Elbow = RobotElbow()
    p.Shoulder = RobotShoulder()
    p.Flip = RobotFlip()
    local j = RobotJRC()
    local jrc = {}
    for i = 1, 8 do jrc[i] = j[i] or 0 end
    p.JntRotCnt = jrc
    return p
end

local function readOri()
    local o = MultiReadModbus(BASE + 4, 6, "W")
    return s32(o[1], o[2]) / 1000, s32(o[3], o[4]) / 1000, s32(o[5], o[6]) / 1000
end

local STAGE = ""   -- 目前執行到哪一步，出錯時一起回報

-- MovL/MovP 不吃自己組的點位表 (MovP_C 會 attempt to index a nil value)。
-- 改用台達官方範例的做法：點位表編號。CopyPoint(新編號, 1) 拿 P1 的 UF/TF/姿態當模板，
-- WritePoint 改座標 (mm / 度)，再 MovL(編號)。P1 本身不會被改。
-- 每個點用自己的編號 (SLOT0 + i)，PASS 預先規劃下一段時不會被覆蓋。
local SLOT0 = 500                  -- 編號 500 起當暫存點，點位表這段不要放自己教的點
local function setSlot(id, x, y, z, a, b, c)
    CopyPoint(id, 1)
    WritePoint(id, "X", x)
    WritePoint(id, "Y", y)
    WritePoint(id, "Z", z)
    WritePoint(id, "RX", a)
    WritePoint(id, "RY", b)
    WritePoint(id, "RZ", c)
end

local function runPath()
    STAGE = "read N"
    local n = u16(ReadModbus(BASE + 1, "W"))
    STAGE = "readOri"
    local rx, ry, rz = readOri()
    -- 先把所有點寫進點位表 (手臂還沒動)。動起來之後迴圈裡只剩 MovL，
    -- 下一段指令才來得及在 PASS 轉彎區之前送出，否則每點都會減速
    local recs = {}
    for i = 1, n do
        STAGE = "readRec " .. i
        local x, y, z, spd, kind, ovl = readRec(i)
        if spd < 1 then spd = 5 end
        STAGE = "setSlot " .. i
        local p = SLOT0 + i
        setSlot(p, x, y, z, rx, ry, rz)
        local bx, by, bz = ReadPoint(p, "X"), ReadPoint(p, "Y"), ReadPoint(p, "Z")
        if math.abs(bx - x) > 0.05 or math.abs(by - y) > 0.05 or math.abs(bz - z) > 0.05 then
            error("point " .. i .. " write mismatch: want " .. x .. "," .. y .. "," .. z
                  .. " got " .. tostring(bx) .. "," .. tostring(by) .. "," .. tostring(bz))
        end
        recs[i] = {p, spd, kind, ovl}
    end
    -- 0x100A = PASS 模式 (0 = DISTANT 距離疊合，1 = DEC 速度疊合)，0x100B = DEC 的疊合 %
    -- 手冊 1.7：DISTANT 每段還是會加減速 (只是轉角磨圓)；DEC 100% = 前段減速和下段加速完全重疊，速度接得起來
    STAGE = "PassMode"
    local dec = u16(ReadModbus(BASE + 10, "W")) == 1
    if dec then
        local t = u16(ReadModbus(BASE + 11, "W"))
        if t < 1 or t > 100 then t = 100 end
        PassMode("DEC", 100)
        SetOverlapTime(t)
    else
        PassMode("DISTANT", 100)
    end
    for i = 1, n do
        local p, spd, kind, ovl = recs[i][1], recs[i][2], recs[i][3], recs[i][4]
        WriteModbus(BASE + 3, "W", i)
        if kind == 1 then
            STAGE = "MovP " .. i
            MovP(p, SPD(spd))
        elseif ovl > 0 and i < n then
            if not dec then
                if ovl < 1 then ovl = 1 end
                STAGE = "SetOverlapDistance " .. i
                SetOverlapDistance(ovl)
            end
            STAGE = "MovL+PASS " .. i
            MovL(p, SPD(spd) + PASS())
        else
            STAGE = "MovL " .. i
            MovL(p, SPD(spd))
        end
    end
    STAGE = "done"
end

local function field(c, k, i)
    local v = c[k]
    if v == nil then v = c[string.upper(k)] end
    if v == nil then v = c[i] end
    return v
end

local function selfTest()
    report("selfTest: readRec")
    local x, y, z = readRec(1)
    report("selfTest: point1 = " .. tostring(x) .. ", " .. tostring(y) .. ", " .. tostring(z))
    writeDW(0x1010, x)
    writeDW(0x1012, y)
    writeDW(0x1014, z)
    if CPOS == nil then                   -- 這台韌體沒有 ReadRobotCPos，改用 RobotX/Y/Z
        writeDW(0x1016, RobotX())
        writeDW(0x1018, RobotY())
        writeDW(0x101A, RobotZ())
        -- 用第 1 點 + 0x1004 的角度組一個點，回報內容給 Python 檢查 (不移動)
        local rx, ry, rz = readOri()
        local p = makePoint(x, y, z, rx, ry, rz)
        local s = "mt=" .. tostring(getmetatable(p) ~= nil) .. " pt:"
        for _, k in ipairs({"x", "y", "z", "a", "b", "c", "Elbow", "Shoulder", "Flip", "UF", "TF", "Coord", "PS", "Active"}) do
            s = s .. " " .. k .. "=" .. tostring(p[k])
        end
        report(s)
        return
    end
    local c = CPOS()
    local info = tostring(CPOS_NAME) .. " -> " .. type(c) .. ":"
    if type(c) == "table" then
        for k, v in pairs(c) do info = info .. " " .. tostring(k) .. "=" .. tostring(v) end
    end
    print(info)
    report(info)
    local cx, cy, cz = field(c, "x", 1), field(c, "y", 2), field(c, "z", 3)
    if cx == nil and RobotX ~= nil then
        cx, cy, cz = RobotX(), RobotY(), RobotZ()
    end
    writeDW(0x1016, cx)
    writeDW(0x1018, cy)
    writeDW(0x101A, cz)
end

-- 列出控制器所有全域名稱 (分頁，每頁 250 字)，Python 寫頁碼到 0x1001
local ALLNAMES = nil
local function listNames()
    if ALLNAMES == nil then
        local names = {}
        for k, v in pairs(_G) do
            names[#names + 1] = tostring(k) .. ":" .. string.sub(type(v), 1, 1)
        end
        table.sort(names)
        ALLNAMES = table.concat(names, ",")
    end
    local page = u16(ReadModbus(BASE + 1, "W"))
    report(string.sub(ALLNAMES, page * 250 + 1, page * 250 + 250))
end

-- 探查點位格式 (Point()、P 表、姿態函式)，分頁回傳
local function dump(v, depth)
    if type(v) ~= "table" then return tostring(v) end
    if depth <= 0 then return "{...}" end
    local parts, n = {}, 0
    for k, x in pairs(v) do
        n = n + 1
        if n > 25 then parts[#parts + 1] = "..." break end
        parts[#parts + 1] = tostring(k) .. "=" .. dump(x, depth - 1)
    end
    return "{" .. table.concat(parts, " ") .. "}"
end

local function keysOf(t, maxn)
    if type(t) ~= "table" then return type(t) end
    local ks, n = {}, 0
    for k, _ in pairs(t) do
        n = n + 1
        if n > maxn then ks[#ks + 1] = "..." break end
        ks[#ks + 1] = tostring(k)
    end
    return "[" .. table.concat(ks, ",") .. "]"
end

local function pdump(label, fn, ...)
    local r = {pcall(fn, ...)}
    local s = label .. ": ok=" .. tostring(r[1])
    for i = 2, #r do s = s .. " r" .. (i - 1) .. "=" .. dump(r[i], 3) end
    return s
end

local PROBE = nil
local function probe()
    if PROBE == nil then
        local L = {}
        L[#L + 1] = "P type=" .. type(P) .. " keys=" .. keysOf(P, 10)
        if type(P) == "table" then
            for k, v in pairs(P) do
                L[#L + 1] = "P[" .. tostring(k) .. "]=" .. dump(v, 3)
                break
            end
        end
        L[#L + 1] = pdump("Point()", Point)
        L[#L + 1] = pdump("Point(13 args)", Point, "t1", 400, 0, 500, 180, 0, 0, 0, 0, 0, 0, 0, {0,0,0,0,0,0,0,0})
        L[#L + 1] = pdump("Point(1)", Point, 1)
        L[#L + 1] = pdump("ReadPoint(P[1],X)", function() return ReadPoint(P[1], "X") end)
        L[#L + 1] = "pose X/Y/Z/A/B/C=" .. tostring(RobotX()) .. "," .. tostring(RobotY()) .. "," .. tostring(RobotZ())
            .. "," .. tostring(RobotA()) .. "," .. tostring(RobotB()) .. "," .. tostring(RobotC())
        L[#L + 1] = "RX/RY/RZ=" .. tostring(RobotRX()) .. "," .. tostring(RobotRY()) .. "," .. tostring(RobotRZ())
        L[#L + 1] = "Elbow/Shoulder/Flip=" .. tostring(RobotElbow()) .. "," .. tostring(RobotShoulder())
            .. "," .. tostring(RobotFlip()) .. " JRC=" .. dump(RobotJRC(), 2)
        L[#L + 1] = "UF=" .. tostring(UF) .. " TF=" .. tostring(TF)
        L[#L + 1] = "xRL keys=" .. keysOf(xRL, 40)
        L[#L + 1] = "motion keys=" .. keysOf(motion, 40)
        L[#L + 1] = "fun keys=" .. keysOf(fun, 40)
        PROBE = table.concat(L, " | ")
    end
    local page = u16(ReadModbus(BASE + 1, "W"))
    report(string.sub(PROBE, page * 250 + 1, page * 250 + 250))
end

-- 原地測試各種 MovL/MovP 寫法 (目標 = 目前位置，手臂不會移動)，回報哪些可以用
local function moveTest()
    local x, y, z = RobotX(), RobotY(), RobotZ()
    local a, b, c = RobotRX(), RobotRY(), RobotRZ()
    local id = SLOT0
    local s = "pos " .. string.format("%.2f,%.2f,%.2f", x, y, z)
    local ok, err = pcall(setSlot, id, x, y, z, a, b, c)
    s = s .. " | setSlot:" .. (ok and "ok" or tostring(err))
    if not ok then report(s) return end
    -- 讀回來確認單位是 mm/度 而且真的寫進去了，否則不准動 (避免點位跑到奇怪的地方)
    local back = {ReadPoint(id, "X"), ReadPoint(id, "Y"), ReadPoint(id, "Z")}
    s = s .. " back=" .. tostring(back[1]) .. "," .. tostring(back[2]) .. "," .. tostring(back[3])
    for k, v in ipairs({x, y, z}) do
        if type(back[k]) ~= "number" or math.abs(back[k] - v) > 0.05 then
            report(s .. " | MISMATCH, no move")
            return
        end
    end
    local tests = {
        {"MovL(id)",               function() MovL(id) end},
        {"MovL(id,SPD)",           function() MovL(id, SPD(5)) end},
        {"MovL(id,SPD+PASS)",      function() MovL(id, SPD(5) + PASS()) end},
        {"MovL(id,SPD,PASS)",      function() MovL(id, SPD(5), PASS()) end},
        {"MovL(id,'PASS')",        function() MovL(id, "PASS") end},
        {"MovP(id,SPD)",           function() MovP(id, SPD(5)) end},
        {"MovL(id) end",           function() MovL(id) end},
    }
    for k, t in ipairs(tests) do
        local ok2, e2 = pcall(t[2])
        s = s .. " | " .. k .. ":" .. (ok2 and "ok" or string.sub(tostring(e2), -22))
    end
    report(s)
end

local function traceback(e)
    if debug ~= nil and debug.traceback ~= nil then
        return debug.traceback(tostring(e), 2)
    end
    return tostring(e)
end

local function try(fn, name)
    STAGE = ""
    local ok, err = xpcall(fn, traceback)
    if not ok then
        local msg = name .. " error @" .. STAGE .. ": " .. tostring(err)
        print(msg)
        pcall(report, msg)
    end
    return ok
end

WriteModbus(BASE, "W", 0)
WriteModbus(BASE + 3, "W", 0)
WriteModbus(BASE + 2, "W", VER)

while true do
    local cmd = ReadModbus(BASE, "W")
    if cmd == 1 then
        WriteModbus(BASE, "W", 2)
        if try(runPath, "runPath") then
            WriteModbus(BASE, "W", 3)
        else
            WriteModbus(BASE, "W", 9)
        end
    elseif cmd == 7 then
        if try(listNames, "listNames") then
            WriteModbus(BASE, "W", 8)
        else
            WriteModbus(BASE, "W", 9)
        end
    elseif cmd == 11 then
        if try(probe, "probe") then
            WriteModbus(BASE, "W", 12)
        else
            WriteModbus(BASE, "W", 9)
        end
    elseif cmd == 13 then
        if try(moveTest, "moveTest") then
            WriteModbus(BASE, "W", 14)
        else
            WriteModbus(BASE, "W", 9)
        end
    elseif cmd == 5 then
        if try(selfTest, "selfTest") then
            WriteModbus(BASE, "W", 6)
        else
            WriteModbus(BASE, "W", 9)
        end
    end
    DELAY(0.02)
end
