import cv2, numpy as np, argparse, time, glob, os, sys, subprocess, pandas, random, math, ctypes, win32con
#Define variables and load classifier
import HandTrackingModule as htm
import pyautogui as p
camnumber = 0
detected = 0
count = 0
count1 = 0
Player = 0

#hand detetction
detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

#face detection
wCam, hCam = 640, 480

video_capture = cv2.VideoCapture(0)

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
    for x in facedict.keys():
        pred, conf = fishface.predict(facedict[x])
        cv2.imwrite("images\\%s.jpg" %x, facedict[x])
    recognized_emotion = emotions[pred]
    print("I think you're %s" %recognized_emotion)
    print(pred)
    print(conf)

    actionlist = [x for x in actions[recognized_emotion]] #get list of actions/files for detected emotion
    #random.shuffle(actionlist) #Randomly shuffle the list
    open_stuff(actionlist[0]) #Open the first entry in the list
    Player = 1

def grab_webcamframe():
    ret, frame = video_capture.read()
    cv2.imshow("image", frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    cv2.imshow("game", clahe_image)
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

while True:
    detected = 0
    if Player == 0:
        run_detection()
    success, img = video_capture.read() 
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)

    if len(lmList) != 0:
        fingers = []
        #thumb
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        #print(fingers)
        totalFingers = fingers.count(1)

        if totalFingers == 0:
            pass

        elif totalFingers == 1:
            if count == 0:
                p.press("playpause")
                count1 = 0
            
            count=count+1
            if count > 20:
                count = 0

        elif totalFingers == 2:
            p.press("volumeup")
            count = 0
            count1 = 0
            
        elif totalFingers == 3:
            p.press("volumedown")
            count = 0
            count1 = 0
            
        elif totalFingers == 4:
            #p.press("nexttrack")
            count = 0

            if count1 == 0:
                p.press("playpause")
            
            count1=count1+1
            if count1 > 20:
                count1 = 0

        else:
            pass

    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break