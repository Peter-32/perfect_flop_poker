# import pyautogui as g
from time import sleep
# import pyperclip as clip
# g.FAILSAFE = True
# g.PAUSE = 0.0
# g.position()

# def c():
#     g.click(button='left')
#
# def c2():
#     g.click(clicks=2)
#
# def rc():
#     g.click(button='right')
#
# def paste():
#     g.keyDown('ctrl')
#     g.keyDown('v')
#     g.keyUp('v')
#     g.keyUp('ctrl')
#
# def copy_text():
#     g.keyDown('ctrl')
#     g.keyDown('c')
#     g.keyUp('c')
#     g.keyUp('ctrl')


# Click on screen
# g.moveTo(791, 398)
# c()



    # sleep_length = 0.06
    # is_success = True
    # for _ in range(100):
    #     # Clear all
    #     g.moveTo(959, 396)
    #     c()
    #
    #     # Flop
    #     clip.copy(final_flop_string)
    #     sleep(sleep_length*5)
    #     g.moveTo(784, 320)
    #     c()
    #     paste()
    #     sleep(sleep_length*5)
    #
    #     # Hand range 1
    #     clip.copy(my_hands_string)
    #     sleep(sleep_length)
    #     g.moveTo(972, 156)
    #     c()
    #     paste()
    #     sleep(sleep_length*5)
    #
    #     # Hand range 2 (Hopefully it doesn't copy the first hand twice)
    #     clip.copy("ABC")
    #     clip.copy(opponents_hands_string)
    #     sleep(sleep_length)
    #     g.moveTo(999, 179)
    #     c()
    #     paste()
    #     sleep(sleep_length)
    #
    #     # Evaluate button
    #     g.moveTo(1377, 394)
    #     c()
    #
    #     # # Stop monteo carlo (Decided to use Enumerate all)
    #     # sleep(sleep_length)
    #     # g.moveTo(1206, 398)
    #     # c()
    #
    #     # Copy equity
    #     sleep(sleep_length*3.3)
    #     g.moveTo(1396, 158)
    #     c2()
    #     copy_text()
    #     sleep(sleep_length/3)
    #
    #     # Save raw equity
    #     raw_equity_string = clip.paste()
    #
        # Possibly repeat loop
    # try:
    #     raw_equity = float(raw_equity_string.replace("%",""))/100
    #     is_success = True
    # except:
    #     is_success = False
    #     sleep_length = sleep_length*1.1
    #
    # if is_success:
    #     break
