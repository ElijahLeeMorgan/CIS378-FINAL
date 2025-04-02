SickleManager = {}
SickleManager.__index = SickleManager

local DEBUG_SPEED = 80

local DIRECTIONS = {
    DOWN = { 0, 1 },
    LEFT = { -1, 0 },
    UP = { 0, -1 },
    RIGHT = { 1, 0 }
}

local right = true

local function make_sickle(_x, _y, _dir, _speed)
    local new_sickle = Sickle:new(_x, _y, _dir, _speed)
    return new_sickle
end

local VERTICALS = {50, 62, 74, 86, 98, 110, 122, 134, 146, 158, 170, 182, 194}
local HORIZONTALES = {114, 102, 91, 80, 69, 56} --first 2 are good

local WAVES = {
    debug = {
        { VERTICALS[4], -10, DIRECTIONS.DOWN },
    },
    -- TOP WAVES
    top_left = {
        { VERTICALS[2],  -10, DIRECTIONS.DOWN },
        { VERTICALS[3],  -10, DIRECTIONS.DOWN },
        { VERTICALS[4],  -10, DIRECTIONS.DOWN },
        { VERTICALS[5],  -10, DIRECTIONS.DOWN },
        { VERTICALS[6],  -10, DIRECTIONS.DOWN },
        { VERTICALS[7], -10, DIRECTIONS.DOWN },
    },
    top_right = {
        { VERTICALS[9], -21,  DIRECTIONS.DOWN },
        { VERTICALS[10], -21,  DIRECTIONS.DOWN },
        { VERTICALS[11], -80,  DIRECTIONS.DOWN },
        { VERTICALS[12], -80,  DIRECTIONS.DOWN },
        { VERTICALS[13], -136, DIRECTIONS.DOWN },
    },
    top_double_line = {
        { VERTICALS[4],  -10, DIRECTIONS.DOWN },
        { VERTICALS[5],  -10, DIRECTIONS.DOWN },
        { VERTICALS[8], -10, DIRECTIONS.DOWN },
        { VERTICALS[9], -10, DIRECTIONS.DOWN },
    },
    top_double_outside = {
        { VERTICALS[2],  -5, DIRECTIONS.DOWN },
        { VERTICALS[3],  -5, DIRECTIONS.DOWN },
        { VERTICALS[10],  -5, DIRECTIONS.DOWN },
        { VERTICALS[11],  -5, DIRECTIONS.DOWN },
    },
    top_double_left = {
        { VERTICALS[2],  -10, DIRECTIONS.DOWN },
        { VERTICALS[3],  -10, DIRECTIONS.DOWN },
        { VERTICALS[4], -10, DIRECTIONS.DOWN },
    },
    top_double_right = {
        { VERTICALS[6],  -10, DIRECTIONS.DOWN },
        { VERTICALS[7],  -10, DIRECTIONS.DOWN },
        { VERTICALS[8], -10, DIRECTIONS.DOWN },
    },
    top_full_a = {
        { VERTICALS[2],  -17,  DIRECTIONS.DOWN },
        { VERTICALS[1],  -133, DIRECTIONS.DOWN },
        { VERTICALS[6], -132, DIRECTIONS.DOWN },
        { VERTICALS[5], -17,  DIRECTIONS.DOWN },
        { VERTICALS[1],  -58,  DIRECTIONS.DOWN },
        { VERTICALS[2],  -185, DIRECTIONS.DOWN },
        { VERTICALS[8], -184, DIRECTIONS.DOWN },
        { VERTICALS[10], -58,  DIRECTIONS.DOWN },
    },
    top_full_b = {
        { VERTICALS[11],  -99,  DIRECTIONS.DOWN },
        { VERTICALS[9], -47,  DIRECTIONS.DOWN },
        { VERTICALS[8], -155, DIRECTIONS.DOWN },
        { VERTICALS[8], -97,  DIRECTIONS.DOWN },
        { VERTICALS[7],  -13,  DIRECTIONS.DOWN },
        { VERTICALS[10], -14,  DIRECTIONS.DOWN },
    },
    -- LEFT WAVES
    left_low = {
        { -10,  HORIZONTALES[1], DIRECTIONS.RIGHT },
        { -150, HORIZONTALES[2], DIRECTIONS.RIGHT },
        { -190, HORIZONTALES[1], DIRECTIONS.RIGHT },
    },
    
    left_high_single = {
        { -10, HORIZONTALES[2], DIRECTIONS.RIGHT },
    },
    left_full = { --TODO: Fix 
        { -20,  HORIZONTALES[3], DIRECTIONS.RIGHT },
        { -62,  HORIZONTALES[3],  DIRECTIONS.RIGHT },
        { -110, HORIZONTALES[4], DIRECTIONS.RIGHT },
        { -140, HORIZONTALES[3], DIRECTIONS.RIGHT },
        { -180, HORIZONTALES[1], DIRECTIONS.RIGHT },
    },
    left_high = {
        { -20, HORIZONTALES[5],  DIRECTIONS.RIGHT },
        { -20, HORIZONTALES[4], DIRECTIONS.RIGHT },
        { -90,  HORIZONTALES[1],  DIRECTIONS.RIGHT },
        { -20,  HORIZONTALES[3], DIRECTIONS.RIGHT },
    },

    -- RIGHT WAVES
    right_high = { --TODOL Fix this
        { 347, HORIZONTALES[5], DIRECTIONS.LEFT },
        { 538, HORIZONTALES[5],  DIRECTIONS.LEFT },
        { 254, HORIZONTALES[5],  DIRECTIONS.LEFT },
        { 345, HORIZONTALES[4],  DIRECTIONS.LEFT },
        { 436, HORIZONTALES[3],  DIRECTIONS.LEFT },
    },
    right_low = {
        { 271, HORIZONTALES[1], DIRECTIONS.LEFT },
        { 288, HORIZONTALES[2],  DIRECTIONS.LEFT },
        { 346, HORIZONTALES[2],  DIRECTIONS.LEFT },
        { 360, HORIZONTALES[3], DIRECTIONS.LEFT },
        { 458, HORIZONTALES[1], DIRECTIONS.LEFT },
        { 490, HORIZONTALES[1],  DIRECTIONS.LEFT },
    },
    right_low_single = {
        { 243, HORIZONTALES[1], DIRECTIONS.LEFT },
    },
    
    final_wave = {
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
    },
    xxx_a = {
        { 75,  -226, DIRECTIONS.DOWN },
        { 121, -155, DIRECTIONS.DOWN },
        { 160, -95,  DIRECTIONS.DOWN },
        { 191, -17,  DIRECTIONS.DOWN },
        { 270, 99,   DIRECTIONS.LEFT },
        { 363, 99,   DIRECTIONS.LEFT },
        { 452, 99,   DIRECTIONS.LEFT },
        { 535, 99,   DIRECTIONS.LEFT },
    },
    
    alternating_left_right = {
        { 250, HORIZONTALES[5],  DIRECTIONS.LEFT },
        { -10, HORIZONTALES[3],  DIRECTIONS.RIGHT },
        { 250, HORIZONTALES[1], DIRECTIONS.LEFT },
    },
}

function SickleManager:on_two_sec()
    if right then
        self:spawn_sickles(WAVES.right_low_single, 150)
        logger.debug("spawning right")
        right = false
    else
        self:spawn_sickles(WAVES.left_high_single, 150)
        logger.debug("spawning left")
        right = true
    end
end

function SickleManager:spawn_debug_wave()
    self:spawn_sickles(WAVES.debug, DEBUG_SPEED)
end

function SickleManager:new()
    local _sickle_manager = setmetatable({}, SickleManager)
    _sickle_manager.active_sickles = {}
    _sickle_manager.timers = {}
    _sickle_manager.tmr_every_2s = Timer:new(60 * 2,function ()
        _sickle_manager:on_two_sec() 
    end , true)
    _sickle_manager.tmr_every_4s = Timer:new(60 * 4,
        function() _sickle_manager:spawn_sickles(WAVES.left_high_single, 120) end, true)

    table.insert(_sickle_manager.timers, _sickle_manager.tmr_every_2s)
    table.insert(_sickle_manager.timers, _sickle_manager.tmr_every_4s)



    _sickle_manager.update_actions = {
        [30] = nil,
        [29] = function()
            _sickle_manager:spawn_sickles(WAVES.alternating_left_right, 150)
            --_sickle_manager.tmr_every_4s:start()
        end,
        [28] = nil,
        [27] = function() _sickle_manager:spawn_sickles(WAVES.top_double_line, 190) end,
        [26] = function() 
            _sickle_manager.tmr_every_2s:start()
            _sickle_manager:spawn_sickles(WAVES.top_double_outside, 270)

        end,
        [25] = function() _sickle_manager:spawn_sickles(WAVES.left_high, 150) end,
        [24] = function() _sickle_manager:spawn_sickles(WAVES.top_double_left, 220) end,
        [23] = function() _sickle_manager:spawn_sickles(WAVES.top_double_right, 250) end,
        [22] = nil,
        [21] = function() _sickle_manager:spawn_sickles(WAVES.right_low, 160) end,
        [20] = nil,
        [19] = nil,
        [18] = function() _sickle_manager:spawn_sickles(WAVES.left_full, 150) end,
        [17] = nil,
        [16] = nil,
        [15] = function() _sickle_manager:spawn_sickles(WAVES.top_right, 220) end,
        [14] = nil,
        [13] = function() _sickle_manager:spawn_sickles(WAVES.top_full_a, 220) end,
        [12] = nil,
        [11] = function() _sickle_manager:spawn_sickles(WAVES.top_left, 220) end,
        [10] = function() _sickle_manager:spawn_sickles(WAVES.left_low, 170) end,
        [9] = nil,
        [8] = nil,
        [7] = function() _sickle_manager:spawn_sickles(WAVES.left_low, 120) end,
        [6] = nil,
        [5] = function() _sickle_manager:spawn_sickles(WAVES.top_full_a, 200) end,
        [4] = nil,
        [3] = function() _sickle_manager:spawn_sickles(WAVES.right_high, 130) end,
        [2] = function() _sickle_manager:spawn_sickles(WAVES.left_full, 150) end,
        [1] = on_player_win,
    }
    _sickle_manager.debug_key_functions = {
        ["1"] = function() _sickle_manager:spawn_debug_wave() end,
        ["2"] = function() _sickle_manager:spawn_sickles(WAVES.top_full_b, DEBUG_SPEED) end,
        ["3"] = function() _sickle_manager:spawn_sickles(WAVES.top_left, DEBUG_SPEED) end,
        ["4"] = function() _sickle_manager:spawn_sickles(WAVES.top_right, DEBUG_SPEED) end,
        ["5"] = function() _sickle_manager:spawn_sickles(WAVES.left_full, DEBUG_SPEED) end,
        ["6"] = function() _sickle_manager:spawn_sickles(WAVES.left_high, DEBUG_SPEED) end,
        ["7"] = function() _sickle_manager:spawn_sickles(WAVES.left_low, DEBUG_SPEED) end,
        ["8"] = function() _sickle_manager:spawn_sickles(WAVES.right_high, DEBUG_SPEED) end,
        ["9"] = function() _sickle_manager:spawn_sickles(WAVES.right_low, DEBUG_SPEED) end,
    }

    return _sickle_manager
end



function SickleManager:reset()
    self.tmr_every_2s:stop()
    self.tmr_every_4s:stop()

    for k in pairs(self.active_sickles) do
        if not self.active_sickles[k].body:isDestroyed() then
            self.active_sickles[k].body:destroy()
        end
        self.active_sickles[k] = nil
    end
end

function SickleManager:update(dt)
    for t in all(self.timers) do
        t:update()
    end

    for s in all(self.active_sickles) do
        s:update(dt)
        if s.life_timer <= 0 then
            if s.alive then
                s.body:destroy()
            end
            del(self.active_sickles, s)
        else

        end
    end
end

function SickleManager:draw()
    for s in all(self.active_sickles) do
        s:draw()
    end

end

function SickleManager:on_every_second(_seconds_in)
    local action = self.update_actions[_seconds_in]
    if action then
        action()
    else
        return
    end
end

function SickleManager:spawn_sickles(_pattern, _speed)
    for p in all(_pattern) do
        local n_s = make_sickle(p[1], p[2], { p[3][1], p[3][2] }, _speed)
        table.insert(self.active_sickles, n_s)
    end
end

local screen_width = 240
local screen_hieght = 120

function draw_sickle_lanes()
    love.graphics.push("all")
    love.graphics.setColor(love.math.colorFromBytes(255, 255, 255, 30))

    -- Horizontals 
    love.graphics.line(0, 64, screen_width, 64)
    love.graphics.line(0, 75, screen_width, 75)
    love.graphics.line(0, 86, screen_width, 86)
    love.graphics.line(0, 97, screen_width, 97)
    love.graphics.line(0, 108, screen_width, 108)

    -- Verticals 
    love.graphics.line(45, 0, 45, screen_hieght)
    love.graphics.line(56, 0, 56, screen_hieght)
    love.graphics.line(68, 0, 68, screen_hieght)
    love.graphics.line(80, 0, 80, screen_hieght)
    love.graphics.line(92, 0, 92, screen_hieght)
    love.graphics.line(104, 0, 104, screen_hieght)
    love.graphics.line(116, 0, 116, screen_hieght)
    love.graphics.line(128, 0, 128, screen_hieght)
    love.graphics.line(140, 0, 140, screen_hieght)
    love.graphics.line(152, 0, 152, screen_hieght)
    love.graphics.line(164, 0, 164, screen_hieght)
    love.graphics.line(176, 0, 176, screen_hieght)
    love.graphics.line(188, 0, 188, screen_hieght)
    love.graphics.line(200, 0, 200, screen_hieght)

    love.graphics.pop()
end
