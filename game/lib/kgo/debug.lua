

function draw_hitbox(hitbox, color)
    love.graphics.push("all")
    changeFontColor(color)
    love.graphics.rectangle("line", hitbox.x, hitbox.y, hitbox.w, hitbox.h)
    love.graphics.pop()
end