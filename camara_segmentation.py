#Codigo basado en el código de Alejandro Armenta Arellano 
import cv2
import numpy as np
import time

IMG_ROW_RES = 480
IMG_COL_RES = 640

def init_camera():
    video_capture = cv2.VideoCapture(0)
    ret = video_capture.set(3, IMG_COL_RES)
    ret = video_capture.set(4, IMG_ROW_RES)
    return video_capture

def acquire_image(video_capture):
    ret, frame = video_capture.read()
    scaled_rgb_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    scaled_rgb_frame = scaled_rgb_frame[:, :, ::-1]
    return frame, scaled_rgb_frame

def show_frame(name, frame):
    cv2.imshow(name, frame)

lastPublication = 0.0
PUBLISH_TIME = 10

video_capture = init_camera()

while True:
    bgr_video, scaled_rgb_frame = acquire_image(video_capture)

    if np.abs(time.time() - lastPublication) > PUBLISH_TIME:
        try:
            print("No remote action needed ...")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)

        lastPublication = time.time()

    # umbralización global
    _, thresh = cv2.threshold(cv2.cvtColor(bgr_video, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ret, tozero = cv2.threshold(bgr_video, 100, 255, cv2.THRESH_TOZERO)

    gray_video = cv2.cvtColor(bgr_video, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(gray_video, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    _, th_bin_img = cv2.threshold(gray_video, 124, 255, cv2.THRESH_BINARY_INV)
    SE = np.ones((15, 15), np.uint8)
    ero_img = cv2.erode(th_bin_img, SE, iterations=1)
    ret, markers = cv2.connectedComponents(ero_img)
    watershed = cv2.watershed(bgr_video, markers)

    show_frame('RGB image', bgr_video)
    show_frame('Contornos', adaptive)
    show_frame('Global Segmentation', thresh)
    show_frame('Watershed', cv2.applyColorMap(watershed.astype(np.uint8), cv2.COLORMAP_CIVIDIS))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
