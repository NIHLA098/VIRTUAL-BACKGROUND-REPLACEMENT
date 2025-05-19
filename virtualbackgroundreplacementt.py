import cv2
import numpy as np

input_image = cv2.imread("static image/inputstaticimage.jpg")
if input_image is None:
    exit()

virtual_bg = cv2.imread("backgroundimage.jpg/background.jpg")
if virtual_bg is None:
    exit()

virtual_bg = cv2.resize(virtual_bg, (input_image.shape[1], input_image.shape[0]))

hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

mask = cv2.inRange(hsv, lower_green, upper_green)
mask_inv = cv2.bitwise_not(mask)

foreground = cv2.bitwise_and(input_image, input_image, mask=mask_inv)
background = cv2.bitwise_and(virtual_bg, virtual_bg, mask=mask)

final_output = cv2.add(foreground, background)

cv2.imshow("Virtual Background", final_output)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('output_image.jpg', final_output)