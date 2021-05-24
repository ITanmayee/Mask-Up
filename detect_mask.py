# import the necessary packages
import imutils
from imutils.video import VideoStream
import cv2

# start the video streaming
vs = VideoStream(src=0).start()

while True:
    # grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

    # show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
         break
		
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
