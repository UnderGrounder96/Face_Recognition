import sys
import cv2 as cv

# Refactored https://realpython.com/face-recognition-with-python/

def cascade_detect(image, cascade):
  """Convert image to gray then detect faces"""
  gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  return cascade.detectMultiScale(
    gray_image,
    scaleFactor = 1.15,
    minNeighbors = 5,
    minSize = (30, 30),
    flags = cv.CASCADE_SCALE_IMAGE
  )

def detections_draw(image, faces):
  """Draw a rectangle around the faces"""
  for (x, y, w, h) in faces:
    print("({0}, {1}, {2}, {3})".format(x, y, w, h))
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

def main():
  """Core function program"""
  args_len = len(sys.argv)

  if not args_len > 1:
    print("No arguments provided! Exiting...")
    return 1

  cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

  for i in range(1,args_len):
    image = cv.imread(sys.argv[i])

    if image is None:
      print("Error loading image #{0}! Exiting...".format(i))
      return 2

    faces = cascade_detect(image, cascade)
    detections_draw(image, faces)

    print()
    print("Image #{0} - has {1} face(s)!".format(i, len(faces)))
    print()

    # Display the resulting image
    cv.imshow("Image Window", image)

    # Quit program on 'q' key press, otherwise show next image
    if cv.waitKey(5000) & 0xFF == ord('q'):
      break

  # Once completed release window
  cv.destroyAllWindows()

  return 0

if __name__ == "__main__":
  sys.exit(main())
