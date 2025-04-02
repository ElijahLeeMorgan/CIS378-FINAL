-- timer.lua

Timer = {}
Timer.__index = Timer

function Timer:new(finished_time, callback, _loop)
    local _timer = setmetatable({}, Timer)
    _timer.time = 0
    _timer.loop = _loop
    _timer.is_paused = true
    _timer.finished_time = finished_time or 1
    _timer.is_finished = false
    _timer.is_running = false
    _timer.on_done_func = callback or function() _timer:print_done() end
    return _timer
end

function Timer:update()
    if not self.is_finished and not self.is_paused then
        self.time = self.time + 1
        if self.time > self.finished_time then
            self.is_finished = true
            self.is_running = false
            self:on_done()
        end
    end
end

function Timer:start()
    self.time = 0
    self.is_finished = false
    self.is_running = true
    self.is_paused = false
end

function Timer:stop()
    self.time = 0
    self.is_running = false
    self.is_paused = true
end

function Timer:pause()
    self.is_paused = not self.is_paused
end

function Timer:print_done()
    print("I'm done")
end

function Timer:on_done()
    self.on_done_func()
    if self.loop == true then
        self:start()
    end
end

return Timer