# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import face_recognition
from face_recognition.api import load_image_file
import glob
import csv



def get_name(frame):
	faces_encodings = []
	faces_names = []
	_path = os.path.join('/home/administrator/Desktop/Mask-Up/dataset/Emp_Images/' + 'Emp_Images/')

	list_of_files = [i for i in glob.glob(_path+'*.jpg')]

	number_files = len(list_of_files)
	names = list_of_files.copy()                


	for i in range(number_files):
		globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
		globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
		faces_encodings.append(globals()['image_encoding_{}'.format(i)])

		names[i] = names[i].replace(_path, "")  
		faces_names.append(names[i]) 

	face_locations = []
	face_encodings = []
	face_names = []
	process_this_frame = True

	
	rgb_small_frame = frame[:, :, ::-1]
	if process_this_frame:
		face_locations = face_recognition.face_locations( rgb_small_frame)
		face_encodings = face_recognition.face_encodings( rgb_small_frame, face_locations)
		face_names = []
		for face_encoding in face_encodings:
			matches = face_recognition.compare_faces (faces_encodings, face_encoding)
			name = "Unknown"
			face_distances = face_recognition.face_distance( faces_encodings, face_encoding)
			best_match_index = np.argmin(face_distances)
			if matches[best_match_index]:
				name = faces_names[best_match_index]
				print(name)
			face_names.append(name.split('.')[0])

	return face_names 


def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	faces = []
	locs = []
	preds = []
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY))
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	return (locs, preds)


prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


maskNet = load_model("mask_detector.model")


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
count = 0
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
	count = count + 1
	
	for (box, pred) in zip(locs, preds):
		
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		result = label
		
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if count == 100 or key == ord("q"):
		if result == "No Mask":
			l=get_name(frame)
			print(l)
	
			with open('/Users/alekh/Face-Mask-Detection/dataset/emp.csv') as csv_file:
					csv_reader= csv.reader(csv_file, delimiter=',')
					for row in csv_reader:
						if row[0] == l[0]:
							mobile = row[8]
							client = clx.xms.Client(service_plan_id='a16f94c6233349158903ac417e09ebcc',token='c737e9a1b7e649c7b01e540ed29c88b1')
							create = clx.xms.api.MtBatchTextSmsCreate()
							create.sender = '+447537454915'
							create.recipients = {'+91'+ mobile}
							print('Message sent to '+ row[9])
							create.body = 'Please wear your mask'
							client.create_batch(create)
		break
	

cv2.destroyAllWindows()
vs.stop()


