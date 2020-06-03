import sys
import cv2 as cv

# Refactored https://realpython.com/face-recognition-with-python/

def cascade_detect(cascade, image):
  gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  return cascade.detectMultiScale(
    gray_image,
    scaleFactor = 1.15,
    minNeighbors = 5,
    minSize = (30, 30),
    flags = cv.CASCADE_SCALE_IMAGE
  )

def detections_draw(image, detections):
  for (x, y, w, h) in detections:
    print("({0}, {1}, {2}, {3})".format(x, y, w, h))
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

def main():
  args_len = len(sys.argv)

  if not args_len > 1:
    print("No arguments provided! Exiting...")
    return 1

  cascade_path = "haarcascade_frontalface_default.xml"
  cascade = cv.CascadeClassifier(cascade_path)

  for i in range(1,args_len):
    image = cv.imread(sys.argv[i])

    if image is None:
      print("Error loading image #{0}! Exiting...".format(i))
      return 2

    detections = cascade_detect(cascade, image)
    detections_draw(image, detections)

    print("Image #{0} - has {1} face(s)!".format(i, len(detections)))
    print()
    cv.imshow("Objects found", image)
    cv.waitKey(3000)

if __name__ == "__main__":
  sys.exit(main())
