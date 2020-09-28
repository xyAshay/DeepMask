from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.video import VideoStream
import imutils
import numpy as np
import cv2


prototxt = r"face_detector/deploy.prototxt"
weights = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxt, weights)
DeepFace = load_model("DeepFaceMask.model")

def detect_masks(frame, faceNet, DeepFace):
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (100, 100),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    preds = []
    locs = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(width - 1, endX), min(height - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (100, 100))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
        faces = np.array(faces, dtype='float32')
        preds = DeepFace.predict(faces, batch_size=32)
    return(locs, preds)
    
def main():
	stream = VideoStream(src=1).start()
	while True:
		frame = stream.read()
		frame = imutils.resize(frame, width=600)
		(locs, preds) = detect_masks(frame, faceNet, DeepFace)
		for(box, pred) in zip(locs, preds):
		    (startX, startY, endX, endY) = box
		    (mask, no_mask) = pred
		    label = 'Mask' if mask > no_mask else 'No Mask'
		    color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
		    label = "{}: {:.2f}%".format(label, max(mask, no_mask) * 100)
		    cv2.putText(frame, label, (startX, startY - 10),
		                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2
		               )
		    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.imshow("DeepFace", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
		    break
	cv2.destroyAllWindows()
	stream.stop()
	
if (__name__ == "__main__"):
	main()
