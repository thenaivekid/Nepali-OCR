import cv2
import pytesseract
import numpy as np

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    bg_removed = cv2.bitwise_and(image, image, mask=opening)
    # contrast_enhanced = cv2.equalizeHist(bg_removed)
    
    sharp_edge = cv2.bilateralFilter(bg_removed, 9, 75, 75)
    return sharp_edge


def get_text_from_image(image):
    custom_config = r'-l eng+nep --oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text


def pipeline(image):
    image = preprocess(image)
    text = get_text_from_image(image)
    print(text)
    return text

