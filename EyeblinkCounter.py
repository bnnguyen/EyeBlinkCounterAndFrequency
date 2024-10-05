# https://youtu.be/-TVUwH1PgBs

# import opencv library
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import time

start_time = time.time()

# define a video capture object
cap = cv2.VideoCapture(0)

detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 45])

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []
blinkLine = 30.8

counter = 0
color = (255, 0, 255)
blinkCounter = 0

while True:

    # capture video frame by frame
    ret, frame = cap.read()

    # frame = cv2.resize(frame, (1920, 1080)) #this sets the size of the frame to the size of the laptop screen.
    # frame = cv2.resize(frame, (640, 360))

    # detecting face
    frame, faces = detector.findFaceMesh(frame, draw=False)

    # if something in 'faces' list, get face (only having 1 face)
    if faces:

        face = faces[0]
        for id in idList:
            # draw circle on img, center at face[id], radius, color
            cv2.circle(frame, face[id], 5, color, cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]

        lengthVer,_ = detector.findDistance(leftUp, leftDown) #if ",_" is not added, it will not be a number but a list.
        lengthHor,_ = detector.findDistance(leftLeft, leftRight)

        cv2.line(frame, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(frame, leftLeft, leftRight, (0, 200, 0), 3)

        ratio = (lengthVer/lengthHor)*100
        ratioList.append(ratio)
        if len(ratioList) > 2:
            ratioList.pop(0)
        ratioAverage = sum(ratioList) / len(ratioList)
        if ratioAverage <= blinkLine and counter == 0:
            blinkCounter += 1
            color = (0, 200, 0)
            counter = 1
        elif ratioAverage > blinkLine:
            counter = 0
            color = (255, 0, 255)

        current_time = time.time()
        elapsed_time = current_time - start_time

        cvzone.putTextRect(frame, f"Elapsed time: {elapsed_time}", (50, 50), colorR=color)
        cvzone.putTextRect(frame, f"Blink count: {blinkCounter}", (50, 100), colorR=color)
        cvzone.putTextRect(frame, f"Blink frequency: {blinkCounter / elapsed_time}", (50, 150), colorR=color)

        imgPlot = plotY.update(ratioAverage, color=color)
        # cv2.imshow('imagePlot', imgPlot)
        # frame = cv2.resize(frame, (640, 360))
        imgStack = cvzone.stackImages([frame, imgPlot], 2, 1)
    else:
        # frame = cv2.resize(frame, (640, 360))
        imgStack = cvzone.stackImages([frame, frame], 2, 1)

    # cv2.imshow('frame', frame)
    cv2.imshow('frame', imgStack)

    # the 'q' button is set as the quitting button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# # After the loop release the cap object
cap.release()
# # Destroy all the windows
cv2.destroyAllWindows()