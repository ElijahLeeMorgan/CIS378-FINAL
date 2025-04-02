Platform = {}
Platform.__index = Platform


function Platform:new()
    local _platform = setmetatable({}, Platform)
    _platform.image = love.graphics.newImage("asset/image/platform.png")
    _platform.x = 122
    _platform.y = 124
    _platform.w = _platform.image:getWidth()
    _platform.h = _platform.image:getHeight()
    _platform.ox = _platform.w / 2
    _platform.oy = _platform.h / 2
    _platform.body = love.physics.newBody(world, _platform.x, _platform.y, "static")
    _platform.shape = love.physics.newRectangleShape(_platform.w, _platform.h)
    _platform.fixture = love.physics.newFixture(_platform.body, _platform.shape)
    _platform.body:setAwake(true)
    _platform.fixture:setUserData({ obj_type = "Ground", owner = _platform })
    _platform.fixture:setCategory(3)
    _platform.fixture:setMask(3)
    return _platform
end

function Platform:draw()
    love.graphics.draw(self.image, self.x, self.y, 0, 1, 1, self.ox, self.oy)
end
