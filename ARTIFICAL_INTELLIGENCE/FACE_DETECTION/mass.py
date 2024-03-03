import cv2

# Use raw string literal to avoid issues with backslashes in file paths
harcascade = r"C:\Users\123\Desktop\C0DESOFT\CODESOFT-ARTIFICAL INTELLIGENCE\ARTIFICIAL INTELIGENCE\mass-face-dec\haarcascade_frontalface_default.xml"
facecascade = cv2.CascadeClassifier(harcascade)

cap=cv2.VideoCapture(0)

cap.set(3,640)  #width
cap.set(4,480)  #height

while True:
    sucess,img = cap.read()

    # No need to redefine facecascade inside the loop
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    face=facecascade.detectMultiScale(img_gray, 1.1, 4)

    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),2)
    
    cv2.imshow("Face",img)    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release VideoCapture object outside of the loop
cap.release()
cv2.destroyAllWindows()
