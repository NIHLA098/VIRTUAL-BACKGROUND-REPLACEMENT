import cv2
import cv2.aruco as aruco
import numpy as np

cap = cv2.VideoCapture(0)
overlay_image = cv2.imread("C:/User/ASUS/Downloads/overlay image.jpg")  # Image to overlay on the marker

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()  # Correct method
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        for corner in corners:
            int_corners = corner[0].astype(int)
            pts_dst = int_corners

            h, w = overlay_image.shape[:2]
            pts_src = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
            pts_src = np.array(pts_src, dtype=float)
            pts_dst = np.array(pts_dst, dtype=float)

            matrix, _ = cv2.findHomography(pts_src, pts_dst)
            warped = cv2.warpPerspective(overlay_image, matrix, (frame.shape[1], frame.shape[0]))

            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(pts_dst), 255)
            frame = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
            frame = cv2.add(frame, warped)

    cv2.imshow('AR Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()