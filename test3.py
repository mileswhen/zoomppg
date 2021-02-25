import dlib
import cv2
import numpy as np


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("zoomppg/models/shape_predictor_81_face_landmarks.dat")
cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        landmarks = predictor(gray, box=face)
        lpoints = []
        lcheek = []
        rcheek = []
        fhead = []
        for n in range(0, 81):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            lpoints.append((x,y))

            if n == 29:
                rcheek.append((x,y))
                lcheek.append((x,y))
            elif n in [2, 41]:
                rcheek.append((x,y))
            elif n in [14, 46]:
                lcheek.append((x,y))
            elif n in [19, 24, 71]:
                fhead.append((x,y))


        mask = np.zeros(frame.shape[:2], dtype='uint8')
        convex1 = cv2.convexHull(np.array(fhead))
        convex2 = cv2.convexHull(np.array(rcheek))
        convex3 = cv2.convexHull(np.array(lcheek))
        cv2.fillConvexPoly(mask, convex1, 255)
        cv2.fillConvexPoly(mask, convex2, 255)
        cv2.fillConvexPoly(mask, convex3, 255)
        intensity = cv2.mean(frame, mask)
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        print(intensity)
        print(np.average(masked[masked != 0]))

        # delaunay
        rect = cv2.boundingRect(np.array(lpoints))
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(lpoints)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            cv2.line(frame, pt1, pt2, (255, 255, 255), 1)
            cv2.line(frame, pt2, pt3, (255, 255, 255), 1)
            cv2.line(frame, pt1, pt3, (255, 255, 255), 1)

    cv2.imshow("frame", frame)
    
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()