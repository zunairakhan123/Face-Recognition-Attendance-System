import cv2
import os

name = input("Enter teacher name: ")

path = f"dataset/{name}"
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)   # This is "plugs in" to webcam. The 0 usually means our primary laptop camera , 1 means external USB webcam , 2 means Another connected camera .

count = 0            # This acts as a counter so the computer knows how many photos we've taken (it stops at 23). 

while count < 25:
    ret, frame = cap.read()   # Grabs a single frame from the camera.
    cv2.imshow("Capture", frame)  # Shows that frame in a window so you can see yourself.

    key = cv2.waitKey(1)     # Waits for you to press a key on your keyboard.

    if key == ord('c'):      # It saves the current frame as a .jpg file inside the teacher's folder.
        cv2.imwrite(f"{path}/{name}_{count}.jpg", frame)
        count += 1
        print("Image saved")

    if key == ord('q'):   # It breaks the loop and closes the program immediately
        break

cap.release()          # Turns off the webcam
cv2.destroyAllWindows()    # Closes the video window on your screen.