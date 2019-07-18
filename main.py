'''
    Author: Gabriel Assunção Domene
    Date: 01/07/2019
    Objective: Use BGS detection for finding cars in street
'''
import cv2
import numpy as np
from random import randint
# Colors value
BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
YELLOW_COLOR = (0, 255, 255)

def draw_contours(value, frame, left_pass, right_pass, up_pass, down_pass):
    '''Detection and coloring contours'''
    contours, _ = cv2.findContours(value, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    points_list = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    number_car = 0
    left_pass = left_pass
    right_pass = right_pass
    up_pass = up_pass
    down_pass = down_pass
    for cnt in contours:
        initial_x, initial_y, width, height = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if (width >= 10 and height >=10 and area >= 70):
            c_x = int(initial_x+width/2)
            c_y = int(initial_y+height/2) 
            cv2.rectangle(frame, (initial_x, initial_y), (initial_x+width, initial_y+height), GREEN_COLOR, 2)
            cv2.circle(frame, (c_x, c_y), 3, RED_COLOR, -1)
            number_car += 1
            points_list.append((c_x, c_y))
            if ((c_x <= 236 and c_x >= 225) and (c_y <= 155 and c_y >= 105)):
                left_pass += 1
            elif((c_x >= 418 and c_x <= 425) and (c_y >= 170 and c_y <= 226)):
                right_pass += 1
            elif((c_x >= 335 and c_x <= 376) and (c_y >= 48 and c_y <= 55)):
                up_pass += 1
            elif((c_x >= 264 and c_x <= 320) and (c_y >= 250 and c_y <= 265)):
                down_pass += 1
            
    text = 'Car(s) ' + str(number_car)
    cv2.putText(frame, text, (10, 50), font, 1, RED_COLOR, 2, cv2.LINE_AA)

    return left_pass, right_pass, up_pass, down_pass, points_list

def filter_mask(fg_mask):
    '''Morph operations for filtering'''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Fill any small holes
    closing = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations = 3)

    return dilation

def main():
    '''Main function for calling functions of script'''
    # cap = cv2.VideoCapture('drone.mp4')
    cap = cv2.VideoCapture('alternate.mp4')
    # fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(500, 6, 0.9, .1)

    # subtractor = cv2.bgsegm.createBackgroundSubtractorMOG(history=15)
    subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=40)

    # cv2.putText(resized_frame, str(0), (200, 100), font, 1, WHITE_COLOR, 2, cv2.LINE_AA)
    width = int(cap.get(3)/2)
    height = int(cap.get(4)/2)

    img = np.zeros((height, width, 3), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    left_pass, right_pass, up_pass, down_pass, total = 0, 0, 0, 0, 0
    while True:
        kernel = np.ones((5, 5), np.uint8)
        _, frame = cap.read()
        resized_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        
        # Linhas de contagem
        # Embaixo
        cv2.line(resized_frame, (264, 256), (320, 256), BLUE_COLOR, 2)
        # Direita
        cv2.line(resized_frame, (418, 170), (418, 226), BLUE_COLOR, 2)
        # Cima
        cv2.line(resized_frame, (335, 52), (376, 52), BLUE_COLOR, 2)
        # Esquerda
        cv2.line(resized_frame, (236, 105), (236, 155), BLUE_COLOR, 2)

        # cv2.putText(resized_frame, str(0), (200, 100), font, 1, WHITE_COLOR, 2, cv2.LINE_AA)
        # Morph
        mask = subtractor.apply(resized_frame)
        fg_mask = filter_mask(mask)

        # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (7, 7))
        # dilate = cv2.dilate(fg_mask, kernel)
        # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, (3,3))
        # blur = cv2.GaussianBlur(opening, (11, 11), 0)   
        pre_contours = fg_mask.copy()

        left_pass, right_pass, up_pass, down_pass, points = draw_contours(fg_mask, resized_frame, left_pass, right_pass, up_pass, down_pass)
        cv2.putText(resized_frame, str(left_pass), (200, 100), font, 1, WHITE_COLOR, 2, cv2.LINE_AA)
        cv2.putText(resized_frame, str(right_pass), (430, 270), font, 1, WHITE_COLOR, 2, cv2.LINE_AA)
        cv2.putText(resized_frame, str(up_pass), (400, 52), font, 1, WHITE_COLOR, 2, cv2.LINE_AA)
        cv2.putText(resized_frame, str(down_pass), (220, 270), font, 1, WHITE_COLOR, 2, cv2.LINE_AA)
        total = left_pass + up_pass + down_pass + right_pass
        for idx, point in enumerate(points):
            
            color = (randint(0, 255), randint(0, 255), randint(0, 255),)
            cv2.circle(img, point, 1, color, -1)
        cv2.putText(resized_frame, 'Total: ' + str(total), (10, 280), font, 1, YELLOW_COLOR, 2, cv2.LINE_AA)


        # Display
        cv2.imshow('fgmask', fg_mask)
        # cv2.imshow('Mask', mask)
        # cv2.imshow('Opening', opening)
        cv2.imshow('resized_frame', resized_frame)
        cv2.imshow('black', img)
        # cv2.moveWindow('Mask', 0, 0)
        # cv2.moveWindow('Opening', 700, 0)
        cv2.moveWindow('fgmask', 700, 0)
        cv2.moveWindow('resized_frame', 0, 0)
        cv2.moveWindow('black', 700, 600)
        k = cv2.waitKey(75) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()
