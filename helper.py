import pyautogui
import time
import numpy as np
import tensorflow as tf
print("Loading model....")
model = tf.keras.models.load_model("model_.h5")
print("Model loaded")


dislike_path = "buttons/dislike.JPG"
like_path = "buttons/like.JPG"
t_path = "buttons/t.JPG"


def is_tinder_open():
    if pyautogui.locateOnScreen(t_path, grayscale=True, confidence=.5):
        return True
    else:
        False


def press_like():
    if pyautogui.locateOnScreen(like_path, grayscale=True, confidence=.75):

        time.sleep(1)
        print("Like")
        print(pyautogui.locateOnScreen(
            like_path, grayscale=True, confidence=.75))
        pyautogui.click(pyautogui.locateOnScreen(
            like_path, grayscale=True, confidence=.75))


def press_dislike():
    if pyautogui.locateOnScreen(dislike_path, grayscale=True, confidence=.75):

        time.sleep(1)
        print("Dislike")
        print(pyautogui.locateOnScreen(
            dislike_path, grayscale=True, confidence=.75))
        pyautogui.click(pyautogui.locateOnScreen(
            dislike_path, grayscale=True, confidence=.75))


def pred(img):
    return model.predict(img.reshape((1, 224, 224, 3)).astype(np.float32)/255)/10
