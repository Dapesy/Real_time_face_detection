import cv2 as cv

face_cascades = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv.VideoCapture(0)

fourcc = cv.VideoWriter_fourcc(*'X264')  # Define the codec (X264 for compactibility)
output_filename = 'output1.mp4'  # Name of the output video file
frame_width = int(cap.get(3))  # Width of the frames
frame_height = int(cap.get(4))  # Height of the frames
frame_rate = cap.get(cv.CAP_PROP_FPS)  # Frames per second (adjust as needed)

out = cv.VideoWriter(output_filename, fourcc, frame_rate, (frame_width, frame_height))

while True:
    ret, frame = cap.read()

    if ret == True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face = face_cascades.detectMultiScale(gray, 1.1, 4)

        # drawing rectangle around the face 
        for (x,y,w,h) in face:
            cv.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)

        out.write(frame)

        cv.imshow('video',frame)
        key = cv.waitKey(30) & 0xff
        if key ==27:
            break
    else:
        break
cap.release()
out.release()
cv.destroyAllWindows()
