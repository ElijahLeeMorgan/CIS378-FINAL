import os
import pyautogui
import time

# Start AI Assisted Code
def focus_window(window_title):
    """Focuses on a window with the given title using platform-specific tools."""
    try:
        if os.name == 'posix':  # Unix-like systems
            if os.uname().sysname == 'Darwin':  # macOS
                # Use AppleScript with System Events to focus the window by process name
                # Hardcode love game engine on Mac
                script = f'tell application "System Events" to set frontmost of process "love" to true'
                result = os.system(f"osascript -e '{script}'")
                if result != 0:
                    raise Exception(f"No process found with title: love")
            else:  # Linux
                # Use wmctrl to find and focus the window
                result = os.popen(f"wmctrl -l | grep '{window_title}'").read()
                if not result:
                    raise Exception(f"No window found with title: {window_title}")
                window_id = result.split()[0]
                os.system(f"wmctrl -i -a {window_id}")
                time.sleep(0.5)  # Allow time for the window to focus
        else:
            raise Exception("Unsupported operating system")
    except Exception as e:
        raise Exception(f"Error focusing window: {e}")

def send_keystroke(window_title, keystroke, hold_time=0):
    """Sends a single keystroke to the specified window, with an optional hold time."""
    #focus_window(window_title)
    pyautogui.keyDown(keystroke)
    time.sleep(hold_time)
    pyautogui.keyUp(keystroke)

def send_hotkey(window_title, *keys, hold_time=0):
    """Sends a hotkey combination (simultaneous key presses) to the specified window, with an optional hold time."""
    #focus_window(window_title)
    for key in keys:
        pyautogui.keyDown(key)
    time.sleep(hold_time)
    for key in reversed(keys):  # Release keys in reverse order
        pyautogui.keyUp(key)
# End AI Assisted Code

class GameInteraction:
    def __init__(self, window_title):
        self.window_title = window_title
        focus_window(self.window_title)

    def moveLeft(self, holdTime=0.25):
        send_keystroke(self.window_title, 'left', holdTime)
    
    def moveRight(self, holdTime=0.25):
        send_keystroke(self.window_title, 'right', holdTime)
    
    def jump(self, holdTime=0.10):
        send_keystroke(self.window_title, 'up', holdTime)
    
    def nothing(self, holdTime=0.10):
        time.sleep(holdTime)

    '''
    def doubleJumpUp(self, holdTime1=0.10, holdTime2=0.10):
        send_keystroke(self.window_title, 'up', holdTime1)
        time.sleep(0.1)
        send_keystroke(self.window_title, 'up', holdTime2)
    
    def doubleJumpLeft(self, holdTime1=0.10, holdTime2=0.10):
        send_hotkey(self.window_title, 'left', 'up', holdTime1)
        send_hotkey(self.window_title, 'left', 'up', holdTime2)

    def doubleJumpRight(self, holdTime1=0.10, holdTime2=0.10):
        send_hotkey(self.window_title, 'right', 'up', holdTime1)
        send_hotkey(self.window_title, 'right', 'up', holdTime2)
    '''

# Example usage:
# Replace 'Sickle Dodge' with the title of your target application window.
if __name__ == "__main__":
    time.sleep(2)
    try:
        game = GameInteraction("Sickle Dodge")
        for i in range(0, 45):
            game.moveLeft()
            game.jump()
            game.moveRight()
            game.jump()
    except Exception as e:
        print(e)