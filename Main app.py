from tkinter import *
from tkinter import messagebox

import mysql.connector

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
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)


def fetchEmpInfo(empid):
    con=mysql.connector.connect(host='localhost',database='success200',user='root',password='')
    cur=con.cursor()
    try:
        fetch_qry="SELECT * from emp_entry where empid=(%s)"
        cur.execute(fetch_qry,(empid,))
        rec=cur.fetchall()
        row=list(rec[0])
        return(row)
    except:
        print("notfount")
        return ["notfound","notfound"]

window = Tk()

window.title("200 SUCCESS")

window.geometry('350x200')

lbl = Label(window, text="Employee ID: ")

lbl.grid(column=0, row=0)

txt = Entry(window,width=25)

txt.grid(column=1, row=0)


lbl2 = Label(window, text="Temperature in F: ")

lbl2.grid(column=0, row=1)

txt2 = Entry(window,width=6)

txt2.grid(column=1, row=1)



def clicked():
    
    res = ""
    lasttemp = 0.0
    empId = txt.get()
    bodyTemp = float(txt2.get())
    empRec=fetchEmpInfo(empId)
    empName=empRec[1]
    if(empName=="notfound"):
        messagebox.showinfo('Access Denied', 'Your ID doesn\'t match with our Organization.')
    else:
        empEmail=empRec[7]
        empMob=str(empRec[6])
        lastTemp=str(empRec[5])
        empLoc=empRec[3]
        empLCT=str(empRec[4])
        if((bodyTemp >= 97)&(bodyTemp <= 99)):
                lab=""
                i=0
                # load our serialized face detector model from disk
                prototxtPath = r"face_detector\deploy.prototxt"
                weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
                faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
                # load the face mask detector model from disk
                maskNet = load_model("mask_detector.model")
                # initialize the video stream
                print("[INFO] starting video stream...")
                vs = VideoStream(src=0).start()

                while True:
                    i=i+1
                    # grab the frame from the threaded video stream and resize it
                    # to have a maximum width of 400 pixels
                    frame = vs.read()
                    frame = imutils.resize(frame, width=400)
                            # detect faces in the frame and determine if they are wearing a
                            # face mask or not
                    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

                            # loop over the detected face locations and their corresponding
                            # locations
                    for (box, pred) in zip(locs, preds):
                                    # unpack the bounding box and predictions
                        (startX, startY, endX, endY) = box
                        (mask, withoutMask) = pred

                        # determine the class label and color we'll use to draw
                        # the bounding box and text
                        label = "Mask" if mask > withoutMask else "No Mask"
                        lab=label
                        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                        # include the probability in the label
                        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                                    # display the label and bounding box rectangle on the output
                                    # frame
                        cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


                    # show the output frame
                    cv2.imshow("Frame", frame)
                    key = cv2.waitKey(1) & 0xFF

                    if(i>=6):
                        break
                            
                    # if the `q` key was pressed, break from the loop
                    if key == ord("q"):
                        break
                if(lab=="Mask"):
                    res="mask"
                    print("\n\nDoor Opened, You May go in!\n\n")
                elif(lab=="No Mask"):
                    res="nomask"
                    print("\n\nKindly Wear Mask, Else door cannot be opened\n\n")
                    # do a bit of cleanup
                cv2.destroyAllWindows()
                vs.stop()
                time.sleep(2)

        else:
            res="hightemp"
            print("\n\nHigh Body Temperature.\nDoor Cannot be opened.\nInformation sent to higher authority.\n\n")
            from_addr='rameshofvijay@gmail.com'
            to_addr=['ponprabhakarpg@gmail.com','vijayramesh26301@gmail.com','praganesh.28@gmail.com']
            msg=MIMEMultipart()
            msg['From']=from_addr
            msg['To']=" ,".join(to_addr)
            msg['subject']='URGENT - Abnormal Temperature for an Employee'

            body = 'Employee ID : '+empId+'\n'+'Employee Name : '+empName+'\n'+'Employee  Area : '+empLoc+'\n'+'Date of Last Covid Test  : '+empLCT+'\n'+'Temperature : '+str(bodyTemp)+'\n'+'Temperature Yesterday : '+lastTemp

            msg.attach(MIMEText(body,'plain'))

            email=from_addr
            password='vijay@321'

            mail=smtplib.SMTP('smtp.gmail.com',587)
            mail.ehlo()
            mail.starttls()
            mail.login(email,password)
            text=msg.as_string()
            mail.sendmail(from_addr,to_addr,text)
            mail.quit()
        if(res=="mask"):
            messagebox.showinfo('Authentication Success', 'Dear '+empName+',\n      Door is opened, You may go in')
        if(res=="nomask"):
            messagebox.showinfo('Authentication Failed', 'Dear '+empName+',\n      Door cannot be opened, Wear your mask.')
        if(res=="hightemp"):
            messagebox.showinfo('Danger', 'Dear '+empName+',\n      Since you have high body temperature, you can get back to your home. Message has been sent to Higher authority')
    txt.configure(text= "")
    txt2.configure(text="")

btn = Button(window, text="Submit", command=clicked)

btn.grid(column=1, row=3)

window.mainloop()
