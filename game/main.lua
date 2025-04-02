require("src.const")


is_debug_on = false

if is_debug_on then
    love.profiler = require('lib.profile')
end

love = require("love")
lume = require("lib.lume")
anim8 = require("lib.anim8")
logger = require("lib.log")

world = love.physics.newWorld(0, 950, true)

love.graphics.setDefaultFilter("nearest", "nearest")
-- Load Assets
-- TODO: Add get_asset function to all
local title_music = love.audio.newSource("asset/audio/snowy_c.ogg", "stream")
local bg_music = love.audio.newSource("asset/audio/8_bit_iced_village.ogg", "stream")
local snow_flake = get_asset("image", SNOWFLAKE_FILE)
local death_marker = love.graphics.newImage('asset/image/death_marker.png')
local background = get_asset("image", BACKGROUND_FILE)
local title_img = love.graphics.newImage("asset/image/title.png")
local pause_img = love.graphics.newImage("asset/image/pause.png")



require("lib.color")
require("src.player")
require("src.platform")
require("src.sickle_manager")
require("src.sickle")
require("lib.kgo.debug")
require("lib.kgo.timer")
require("src.start_area")
local socket = require("socket")

local font = nil

gamestate = nil
local death_markers = {}
local seconds_left = 60
local tick = 0
local player_attempt = 0
local player_total_attempts = 0
local has_won

-- Pause screen
local is_paused = false
local p_index = 1


local snow_system = love.graphics.newParticleSystem(snow_flake, 1000)
local player = Player:new()
local platfrom = Platform:new()
local sickle_manager = SickleManager:new()
local start_area = StartArea:new()

-- Game Modifications, sends player and sickle data via a UDP socket.
-- Start AI Assisted Code
local udp

function openSocket()
    udp = socket.udp()
    udp:settimeout(0)
    udp:setpeername("127.0.0.1", 12345)
end

-- Sends relevant data to localhost:12345 for processing by another program.
function sendData()
    if not udp then
        logger.error("Socket is not open. Call openSocket() first.")
        return
    end

    local data = {
        player = {
            x = player.x,
            y = player.y,
            is_alive = player.is_alive
        },
        sickles = {},
        timer = seconds_left -- Add the timer time to the data
    }

    for _, sickle in ipairs(sickle_manager.active_sickles) do
        table.insert(data.sickles, { x = sickle.x, y = sickle.y })
    end

    local serializedData = lume.serialize(data)
    udp:send(serializedData)
end

function closeSocket()
    if udp then
        udp:close()
        udp = nil
    end
end
-- End AI Assisted Code

function love.load()
    if is_debug_on then
        logger.level = logger.Level.DEBUG
        logger.debug("Entering debug mode")
        love.profiler.start()
    else
        logger.level = logger.Level.INFO
        logger.info("logger in INFO mode")
    end
    local joysticks = love.joystick.getJoysticks()
    contected_controller = joysticks[1]
    init_snow()
    load_game()
    window = { translateX = 0, translateY = 0, scale = 4, width = GAME_W, height = GAME_H }
    width, height = love.graphics.getDimensions()
    love.window.setMode(width, height, { resizable = true, borderless = false })
    resize(width, height)
    bg_music:setLooping(true)
    title_music:setLooping(true)

    title_music:play()
    title_music:setVolume(0.3)
    bg_music:setVolume(0.3)
    world:setCallbacks(beginContact, endContact, preSolve, postSolve)
    font = love.graphics.newFont("asset/font/mago2.ttf", 16)
    love.graphics.setFont(font)
    --gamestate = gamestates.title
    start_game()
    font:setFilter("nearest")

    if is_debug_on then
        if bg_music:isPlaying() then
            bg_music:pause()
        end
        if title_music:isPlaying() then
            title_music:pause()
        end
    end
end

function reset_game()
    player_attempt = player_attempt + 1
    seconds_left = 30
    sickle_manager:reset()
    player:reset()
end

function init_snow()
    snow_system:setParticleLifetime(5, 10)
    snow_system:setEmissionRate(100)
    snow_system:setEmissionArea("normal", 240 / 4, 0)
    snow_system:setSpeed(1, 3)
    snow_system:setPosition(240 / 2, -6)
    snow_system:setSizes(0.7, 0.6, 0.5)
    snow_system:setSizeVariation(1)
    snow_system:setSpinVariation(1)
    snow_system:setLinearAcceleration(-2, 3, 2, 10)
    snow_system:setColors(1, 1, 1, 1, 1, 1, 1, 0)
end

function start_game()
    --start_area = nil
    openSocket()
    gamestate = gamestates.game
    title_music:stop()
    bg_music:stop()
    bg_music:play()
    reset_game()
end

function on_player_win()
    gamestate = gamestates.win
end

function love.joystickpressed(joystick, button)
    -- 1 = X
    -- 7 = pause
    if button == 7 then
        toggle_pause()
        return
    end

    if is_paused then
        if button == 12 then
            move_pause_arrow("up")
        elseif button == 13 then
            move_pause_arrow("down")
        end
        if button == 1 then
            handle_pause_action()
            return
        end
    end
    logger.debug(button)
    if button == 1 then
        if gamestate == gamestates.retry then
            reset_game()
            gamestate = gamestates.game
            return
        else
            player:jump()
        end
    end
end

function love.keypressed(key)
    --logger.debug(key)
    if not is_paused then
        if key == "escape" then
            if is_debug_on then
                love.event.quit()
            end
            toggle_pause()
            return
        end
        if key == "m" then
            if bg_music:isPlaying() then
                bg_music:pause()
            else
                bg_music:play()
            end
        end

        if key == "space" or key == "w" or key == "up" then
            if gamestate == gamestates.retry then
                reset_game()
                gamestate = gamestates.game
                return
            else
                player:jump()
            end
        end

        if is_debug_on then
            for i = 1, 9 do
                if key == (tostring(i)) then
                    sickle_manager.debug_key_functions[key]()
                end
            end
        end
    else
        if key == "up" or key == "w" then
            --logger.debug("Pressed up on pause menu")
            move_pause_arrow("up")
        elseif key == "down" or key == "s" then
            --logger.debug("Pressed down on pause menu")
            move_pause_arrow("down")
        end
        if key == "space" or key == "return" then
            handle_pause_action()
        end
    end
end

function move_pause_arrow(dir)
    if dir == "up" then
        p_index = clamp(1, p_index - 1, 3)
    elseif dir == "down" then
        p_index = clamp(1, p_index + 1, 3)
    end
end

function handle_pause_action()
    logger.debug("pressed space - index " .. p_index)
    if p_index == 1 then
        toggle_pause()
    elseif p_index == 2 then
        return
    elseif p_index == 3 then
        love.event.quit()
    end
end

function love.update(dt)
    if not is_paused then
        snow_system:update(dt)
        if gamestate == gamestates.title then
            if check_collision(player.hitbox, start_area) then
                start_area:increase()
            else
                start_area:decrease()
            end
            update_title(dt)
        elseif gamestate == gamestates.game then
            update_game(dt)
        else
            update_gameover(dt)
        end
    end
end

function update_title(dt)
    start_area:update(dt)
    player:update(dt)
    world:update(dt)
    sickle_manager:update(dt) --TODO: Remove after testing
end

function update_game(dt)
    tick = tick + 1
    if seconds_left >= 1 then
        if tick == 60 then
            seconds_left = seconds_left - 1
            tick = 0
            sickle_manager:on_every_second(seconds_left)
        end
    end
    if seconds_left == 0 and player.is_alive then
        has_won = true
        gamestate = gamestates.win
    end
    world:update(dt)
    sickle_manager:update(dt)
    player:update(dt)

    sendData()
end

function update_gameover(dt)
    return
end

function spawn_death_marker(_x, _y)
    table.insert(death_markers, { _x, _y })
end

function love.joystickremoved(joystick)
    local js = joystick
    contected_controller = nil
    print(js:getName())
end

function love.joystickadded(joystick)
    contected_controller = joystick
    print()
end

function love.draw()
    love.graphics.translate(window.translateX, window.translateY)
    love.graphics.scale(window.scale)
    love.graphics.draw(background, 0, 0)
    draw_snow()
    if is_debug_on then
        draw_sickle_lanes()
    end
    
    if gamestate == gamestates.title then
        draw_title()
        start_area:draw()
        platfrom:draw()
        player:draw()
        sickle_manager:draw() --TODO: Remove after testing
    end
    if gamestate == gamestates.game then
        draw_game()
    end
    if gamestate == gamestates.retry then
        draw_gameover()
    end
    if gamestate == gamestates.win then
        draw_win()
    end

    if is_paused then
        draw_pause()
    end

    if gamestate ~= gamestates.title then
        draw_death_markers()
        draw_time_left()
    end

    if is_debug_on then
        love.graphics.print("gs: " .. tostring(get_gs_str(gamestate)), 0, 0, 0, 0.8, 0.8)
        love.graphics.print("fps: " .. tostring(love.timer.getFPS()), 0, 8, 0, 0.8, 0.8)
    end
end

function draw_title()
    love.graphics.draw(title_img, 60, 20, 0, 0.19, 0.19)
    love.graphics.print("by Dave", 120, 37)
end

function draw_pause()
    love.graphics.draw(pause_img, 0, 0)
    love.graphics.print("_", 49, 40 + (p_index * 8))
    love.graphics.print("Resume", 57, 50, 0, 0.8, 0.8)
    love.graphics.print("stats", 57, 58, 0, 0.8, 0.8)
    love.graphics.print("quit", 57, 66, 0, 0.8, 0.8)
end

function toggle_pause()
    is_paused = not is_paused
end

function draw_game()
    love.graphics.push("all")
    draw_hud()
    love.graphics.pop()

    --if is_debug_on then
        --draw_world()
    --end

    platfrom:draw()
    player:draw()
    sickle_manager:draw()
end

function draw_time_left()
    love.graphics.setColor(love.math.colorFromBytes(255, 255, 255, 100))
    love.graphics.print(seconds_left, 110, 15, 0, 3, 3)
    love.graphics.setColor(255, 255, 255)
end

function draw_world()
    for _, body in pairs(world:getBodies()) do
        for _, fixture in pairs(body:getFixtures()) do
            local shape = fixture:getShape()

            if shape:typeOf("CircleShape") then
                local cx, cy = body:getWorldPoints(shape:getPoint())
                love.graphics.circle("fill", cx, cy, shape:getRadius())
            elseif shape:typeOf("PolygonShape") then
                love.graphics.polygon("fill", body:getWorldPoints(shape:getPoints()))
            else
                love.graphics.line(body:getWorldPoints(shape:getPoints()))
            end
        end
    end
end

function draw_death_markers()
    for dm in all(death_markers) do
        love.graphics.draw(death_marker, dm[1], dm[2], 0, 0.2, 0.2, death_marker:getWidth() / 2,
            death_marker:getHeight() / 2)
    end
end

function draw_snow()
    love.graphics.draw(snow_system, 0, -6)
end

function draw_hud()
    love.graphics.print("Attempt: " .. tostring(player_attempt), 180, 0, 0, 1, 1)
end

function draw_gameover()
    love.graphics.print(seconds_left, 110, 15, 0, 3, 3)
    if math.floor(love.timer.getTime()) % 2 == 0 then
        love.graphics.print("jump to try again", 65, 70, 0, 1, 1)
    end
end

function draw_win()
    if math.floor(love.timer.getTime()) % 2 == 0 then
        love.graphics.print("you win", 60, 70, 0, 1, 1)
        love.graphics.print("thanks for playing", 60, 80, 0, 1, 1)
    end
    platfrom:draw()
    player:draw()
end

function play_sound(_sound)
    love.audio.stop(_sound)
    love.audio.play(_sound)
end

function go_to_gameover()
    gamestate = gamestates.retry
end

function clamp(_min, _val, _max)
    return math.max(_min, math.min(_val, _max));
end

function all(_list)
    local i = 0
    return function()
        i = i + 1; return _list[i]
    end
end

function del(_table, _item)
    for i, v in ipairs(_table) do
        if v == _item then
            _table[i] = _table[#_table]
            _table[#_table] = nil
            return
        end
    end
end

function check_collision(a, b)
    return a.x < b.x + b.w and
        b.x < a.x + a.w and
        a.y < b.y + b.h and
        b.y < a.y + a.h
end

function do_tables_match(_table_1, _table_2)
    return table.concat(_table_1) == table.concat(_table_2)
end

function save_game()
    data = {}
    data.has_won = has_won
    data.player_total_attempts = player_total_attempts + player_attempt
    serialized = lume.serialize(data)
    love.filesystem.write("sickle.sav", serialized)
end

function load_game()
    if love.filesystem.getInfo("sickle.sav") then
        file = love.filesystem.read("sickle.sav")
        data = lume.deserialize(file)
        player.has_won = data.has_won or false
        player_total_attempts = data.player_total_attempts or 0
        player_attempt = data.player_attempt or 1
        if is_debug_on then
            player.has_won = false
        end
    end
end

function beginContact(a, b, coll)
    --x, y = coll:getNormal()
    obj_a = a:getUserData()
    obj_b = b:getUserData()

    logger.debug(obj_a.obj_type .. " hit " .. obj_b.obj_type)

    -- Checking contact vector to prevent player from clinging to side of platform
    if obj_a.obj_type == "Player" and obj_b.obj_type == "Ground" and coll:getNormal() ~= { 1, 0 } then
        player:on_ground_contact()
    end
    if obj_a.obj_type == "Player" and obj_b.obj_type == "Sickle" then
        player:on_sickle_contact(obj_b.owner)
    end
    if obj_a.obj_type == "Ground" and obj_b.obj_type == "Sickle" then
        obj_b.owner:on_ground_contact()
    end
end

function endContact(a, b, coll) end

function preSolve(a, b, coll) end

function postSolve(a, b, coll, normalimpulse, tangentimpulse) end

function resize(w, h)                          
    local w1, h1 = window.width, window.height
    local scale = math.min(w / w1, h / h1)
    window.translateX, window.translateY, window.scale = (w - w1 * scale) / 2, (h - h1 * scale) / 2, scale
end

function love.resize(w, h)
    resize(w, h)
end

function love.quit()
    save_game()
    logger.info("The application is closing.")
    -- Perform your cleanup tasks here.
    closeSocket()
    if is_debug_on then
        love.profiler.stop()
        print(love.profiler.report(30))
    end
end
