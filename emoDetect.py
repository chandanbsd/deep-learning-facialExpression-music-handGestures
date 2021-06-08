import cv2, numpy as np, argparse, time, glob, os, sys, subprocess, pandas, random, math, ctypes, win32con
#Define variables and load classifier
import HandTrackingModule as htm
import pyautogui as p
camnumber = 0
detected = 0
count = 0
count1 = 0
Player = 1
emotion_rec = 10

folderPath = "Images"
myList = ['0.png', '1.png', '2.png', '3.png', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.png', '10.png']
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    image = cv2.resize(image, (450, 337))
    overlayList.append(image)


#hand detetction
detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

#face detection
wCam, hCam = 640, 480

video_capture = cv2.VideoCapture(0)

cv2.moveWindow("image", 0,0)

facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
fishface = cv2.face.FisherFaceRecognizer_create()

try:
    fishface.read("trained_emoclassifier.xml")
except:
    print("Mode not trained.")


facedict = {}
actions = {}
emotions = ["angry", "happy", "sad", "neutral"]


df = pandas.read_excel("EmotionLinks.xlsx") #open Excel file
actions["angry"] = [x for x in df.angry.dropna()] #We need de dropna() when columns are uneven in length, which creates NaN values at missing places. The OS won't know what to do with these if we try to open them.
actions["happy"] = [x for x in df.happy.dropna()]
actions["sad"] = [x for x in df.sad.dropna()]
actions["neutral"] = [x for x in df.neutral.dropna()]


def open_stuff(filename): #Open the file, credit to user4815162342, on the stackoverflow link in the text above
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener ="open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])

def crop_face(clahe_image, face):
    facedict.clear()
    for (x, y, w, h) in face:
        faceslice = clahe_image[y:y+h, x:x+w]
        faceslice = cv2.resize(faceslice, (350, 350))

    facedict["face%s" %(len(facedict)+1)] = faceslice
    return faceslice

def recognize_emotion():
    global Player
    global emotion_rec
    for x in facedict.keys():
        pred, conf = fishface.predict(facedict[x])
        cv2.imwrite("images\\%s.jpg" %x, facedict[x])
    recognized_emotion = emotions[pred]

    #face image value
    if recognized_emotion == "happy":
        emotion_rec = 0

    elif recognized_emotion == "sad":
        emotion_rec = 1

    elif recognized_emotion == "angry":
        emotion_rec = 2

    elif recognized_emotion == "neutral":
        emotion_rec = 3

    else:
        pass

    print("I think you're %s" %recognized_emotion)
    print(pred)
    print(conf)

    actionlist = [x for x in actions[recognized_emotion]] #get list of actions/files for detected emotion
    #random.shuffle(actionlist) #Randomly shuffle the list
        
    open_stuff(actionlist[0]) #Open the first entry in the list
    Player = 1

def grab_webcamframe():
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    return clahe_image

def detect_face():
    global detected
    clahe_image = grab_webcamframe()
    face = facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face) == 1:
        faceslice = crop_face(clahe_image, face)
        detected = 1
        return faceslice
    else:
        print("no/multiple faces detected, passing over frame")

def run_detection():
    detect_face()
    global detected
    if detected == 1:
        recognize_emotion()
        detected = 0

hand_detected = 9 #default hand image number


#positioning window
cv2. namedWindow("image")
cv2.moveWindow("image", 0,50)

cv2. namedWindow("Emotion detected")
cv2.moveWindow("Emotion detected", 450,50)

cv2. namedWindow("Hand detected")
cv2.moveWindow("Hand detected", 900,50)


while True:
    detected = 0
    if Player == 0:
        run_detection()
    success, img = video_capture.read() 
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = facecascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #print(lmList)
    img = cv2.resize(img, (450, 337))
    if len(lmList) != 0:
        fingers = []
        
        #finger position
        thumb = not(((lmList[5][1] > lmList[4][1]) & (lmList[17][1] < lmList[4][1]))|((lmList[5][1] < lmList[4][1]) & (lmList[17][1] > lmList[4][1])))
        indexF = lmList[8][2] < lmList[6][2]
        middleF = lmList[12][2] < lmList[10][2]
        ringF = lmList[16][2] < lmList[14][2]
        pinkyF = lmList[20][2] < lmList[18][2]

        if ((lmList[5][2] < lmList[0][2]) & (lmList[17][2] < lmList[0][2]) & (((lmList[5][1] > lmList[0][1]) & (lmList[17][1] < lmList[0][1])) | ((lmList[5][1] < lmList[0][1]) & (lmList[17][1] > lmList[0][1])))):
            pass

            if indexF and pinkyF and not(middleF) and not(ringF):
                Player = 0
                cv2.putText(img, "detecting emotion",(10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
                hand_detected =5

            elif not(thumb) and indexF and not(pinkyF) and not(middleF) and not(ringF):
                if count == 0:
                    p.press("playpause")
                    cv2.putText(img, "play/pause",(10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
                    count1 = 0
                    hand_detected = 6
                
                count=count+1
                if count > 20:
                    count = 0

            elif not(thumb) and indexF and not(pinkyF) and middleF and not(ringF):
                p.press("volumeup")
                cv2.putText(img, "Increasing volume",(10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
                hand_detected = 8
                count = 0
                count1 = 0

                
            elif not(thumb) and indexF and not(pinkyF) and middleF and ringF:
                p.press("volumedown")
                cv2.putText(img, "Decreasing volume",(10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
                hand_detected = 7
                count = 0
                count1 = 0
                
            elif not(thumb) and indexF and pinkyF and middleF and ringF:
                count = 0
                if count1 == 0:
                    p.press("nexttrack")
                    cv2.putText(img, "Next Track",(10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
                    hand_detected = 4    
                
                count1=count1+1
                if count1 > 20:
                    count1 = 0

            elif thumb and indexF and not(pinkyF) and not(middleF) and not(ringF):
                actionlist = [x for x in actions["happy"]]
                open_stuff(actionlist[0])
                emotion_rec = 0

            elif thumb and indexF and not(pinkyF) and middleF and not(ringF):
                actionlist = [x for x in actions["sad"]]
                open_stuff(actionlist[0])
                emotion_rec = 1

            elif thumb and indexF and not(pinkyF) and middleF and ringF:
                actionlist = [x for x in actions["angry"]]
                open_stuff(actionlist[0])
                emotion_rec = 2

            elif thumb and indexF and pinkyF and middleF and ringF:
                actionlist = [x for x in actions["neutral"]]
                open_stuff(actionlist[0])
                emotion_rec = 3

            else:
                pass
    
    cv2.imshow("Hand detected", overlayList[hand_detected])
    cv2.imshow("Emotion detected", overlayList[emotion_rec])
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break