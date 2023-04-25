import cv2

img = cv2.imread('opencv/road.jpg', 0)
img = cv2.resize(img, (200,200), fx=0.5, fy=0.5)
img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imshow('Image', img)
cv2.waitKey(10000)
cv2.destroyAllWindows()