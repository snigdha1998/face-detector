import cv2


# Read the input image
img = cv2.imread('OriginalImage.jpg')
cv2.imshow('Original Image',img)
print("height,width,channel=",img.shape)

#split into channels
b,g,r=cv2.split(img)
cv2.imwrite('Blue Channel.jpg',b)
cv2.imwrite('Green Channel.jpg',g)
cv2.imwrite('Red Channel.jpg',r)

#merged image
merged=cv2.merge((b,g,r))
cv2.imwrite('Merged Image.jpg',merged)


# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Greyed Image',gray)

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
print("No. of faces found :",len(faces))

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)    
    
    #detect eye in each face
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(roi_gray,1.1,2)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


# Display the output
cv2.imshow('FaceDetected', img)
cv2.imwrite('FaceDetected.jpg', img)
