JumpSfx = {}
JumpSfx.__index = JumpSfx


local jump_effect_sheet = love.graphics.newImage("asset/image/jump_effect.png")
local jump_grid = anim8.newGrid(8, 8, jump_effect_sheet:getWidth(), jump_effect_sheet:getHeight())

function JumpSfx:new()
    local _jump_sfx = setmetatable({}, JumpSfx)
    _jump_sfx.curr_animation = anim8.newAnimation(jump_grid(('1-6'), 1), 0.07, 'pauseAtEnd')
    _jump_sfx.x = 0
    _jump_sfx.y = 0
    _jump_sfx.curr_animation:pause()
    return _jump_sfx
end

function JumpSfx:update(dt)
    self.curr_animation:update(dt)
end

function JumpSfx:do_animation(x, y)
    self.x = x
    self.y = y + 4
    self.curr_animation:resume()
    self.curr_animation:gotoFrame(1)
end

function JumpSfx:draw()
    self.curr_animation:draw(jump_effect_sheet, self.x, self.y, 0, 1, 1, 4, 4)
end
