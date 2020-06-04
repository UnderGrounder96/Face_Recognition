import sys
import cv2 as cv

# Refactored https://realpython.com/face-detection-in-python-using-a-webcam/

def cascade_detect(frame, cascade):
  """Convert frame to gray then detect faces"""
  gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  return cascade.detectMultiScale(
    gray_image,
    scaleFactor = 1.15,
    minNeighbors = 5,
    minSize = (30, 30),
    flags = cv.CASCADE_SCALE_IMAGE
  )

def detections_draw(frame, faces):
  """Draw a rectangle around the faces"""
  for (x, y, w, h) in faces:
    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

def main():
  """Core function program"""
  if len(sys.argv) != 1:
    print("Arguments provided! Exiting...")
    return 1

  cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

  video_capture = cv.VideoCapture(0)

  while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    faces = cascade_detect(frame, cascade)
    detections_draw(frame, faces)

    # Display the resulting frame
    cv.imshow('Live Feed', frame)

    # Quit program on 'q' key press, otherwise show next frame
    if cv.waitKey(1) & 0xFF == ord('q'):
      break

    # Once completed release capture
  video_capture.release()
  cv.destroyAllWindows()

  return 0

if __name__ == "__main__":
  sys.exit(main())
