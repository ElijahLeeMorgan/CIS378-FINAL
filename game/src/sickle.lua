Sickle = {}
Sickle.__index = Sickle


local ice_sickle_sheet = love.graphics.newImage("asset/image/ice_sickle_sheet.png")
local sickle_grid = anim8.newGrid(16, 16, ice_sickle_sheet:getWidth(), ice_sickle_sheet:getHeight())
local break_sfx = love.audio.newSource("asset/audio/ice_break.wav", "static")

break_sfx:setVolume(0.1)


function Sickle:new(_x, _y, _moving_dir, _speed)
    local _sickle = setmetatable({}, Sickle)

    _sickle.animations = {
        default = anim8.newAnimation(sickle_grid(('1-2'), 1), 0.1),
        shatter = anim8.newAnimation(sickle_grid(('3-8'), 1), 0.02, 'pauseAtEnd')
    }
    _sickle.curr_animation = _sickle.animations["default"]
    _sickle.x = 0
    _sickle.y = 0
    _sickle.moving_dir = _moving_dir
    _sickle.alive = true
    _sickle.rotation = 0
    _sickle.life_timer = 300
    _sickle.speed = _speed
    _sickle.w, _sickle.h = _sickle.curr_animation:getDimensions()
    _sickle.body = love.physics.newBody(world, _sickle.x, _sickle.y, "dynamic")
    _sickle.shape = love.physics.newRectangleShape(_sickle.w - 12, _sickle.h - 3)
    _sickle.fixture = love.physics.newFixture(_sickle.body, _sickle.shape)
    _sickle.fixture:setUserData({ obj_type = "Sickle", owner = _sickle })
    _sickle.fixture:setCategory(1)
    _sickle.fixture:setMask(1)
    _sickle.body:setGravityScale(0)
    _sickle.body:setFixedRotation(true)
    _sickle.body:setPosition(_x, _y)
    _sickle:set_rotation()
    _sickle.body:setLinearVelocity(_sickle.speed * _sickle.moving_dir[1], _sickle.speed * _sickle.moving_dir[2])
    return _sickle
end

function Sickle:update(dt)
    self.curr_animation:update(dt)
    if self.alive then
        self.x = self.body:getX()
        self.y = self.body:getY()
    end

    self.life_timer = self.life_timer - 1

    if self.curr_animation.status == "paused" then
        self.life_timer = 0
    end
end

function Sickle:on_ground_contact()
    self:shatter()
end

function Sickle:shatter()
    self.alive = false
    self.body:destroy()
    self.curr_animation = self.animations["shatter"]
    break_sfx:stop()
    break_sfx:play()
end

function Sickle:draw()
    --TODO: Only draw if on screen
    self.curr_animation:draw(ice_sickle_sheet, self.x, self.y, math.rad(self.rotation), 1, 1, self.w / 2, self.h / 2)
end

function Sickle:set_rotation()
    if do_tables_match(self.moving_dir, { 1, 0 }) then
        self.body:setAngle(math.rad(90))
        self.rotation = 0
    elseif do_tables_match(self.moving_dir, { -1, 0 }) then
        self.body:setAngle(math.rad(90))
        self.rotation = 180
    elseif do_tables_match(self.moving_dir, { 0, 1 }) then
        self.rotation = 90
    end
end
