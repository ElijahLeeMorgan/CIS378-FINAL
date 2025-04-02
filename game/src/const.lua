GAME_W = 240
GAME_H = 136

ASSET_DIR = "asset/"
IMG_DIR = "image/"
FONT_DIR = "font/"
SFX_DIR = "audio/"

-- TODO: Add the rest... maybe, not sure if better than full paths 
SNOWFLAKE_FILE = "snow.png"
BACKGROUND_FILE = "background.png"
DEATH_MARKER = "death_marker.png"
PLAYER_SHEET = "player_sheet.png"
PLAYER = "player.png"
CROWN = "crown.png"

gamestates = {
    title = 0,
    credit = 0.1,
    info = 0.2,
    game = 1,
    retry = 1.1,
    win = 2,
    pause=3,
}

function get_gs_str(num)
	for state_name, val in pairs(gamestates) do
        if val == num then
            return state_name
        end
    end
end

function get_asset(type, file)
    local _middle = ""
    if type == "image" then
        _middle = IMG_DIR
        local _img = love.graphics.newImage(ASSET_DIR..IMG_DIR..file)
        return _img
    elseif type == "sound" then
        _middle = SFX_DIR
    elseif type == "font" then
        _middle = FONT_DIR
    end
end