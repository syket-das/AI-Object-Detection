import cv2


def reScaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)




video = cv2.VideoCapture("object.mp4")


classifier_file= "cars.xml"

car_tracker = cv2.CascadeClassifier(classifier_file)

human_tracker = cv2.CascadeClassifier("fullbody.xml")



while True:
    read_successful, frame = video.read()

    if read_successful:
        frame_resized = reScaleFrame(frame, scale=.2)
        grayscaled_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
    else:
        break

    

    cars = car_tracker.detectMultiScale(grayscaled_frame)
    humans = human_tracker.detectMultiScale(grayscaled_frame)
    
    for (x,y,w,h) in cars:
        cv2.rectangle(frame_resized , (x,y), (x+w , y+h), (255,0,255), 2)
        cv2.putText(frame_resized, "Vehicle",
                    (x-5, y-5), 0, 0.4, (0, 0, 255), 1)

    for (x,y,w,h) in humans:
        cv2.rectangle(frame_resized , (x,y), (x+w , y+h), (0,255,255), 2)
        cv2.putText(frame_resized, "padestrain", (x-5, y-10), 0, 0.4, (0,255,0), 1)



    # print(cars)




    cv2.imshow("detector", frame_resized)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break




video.release()
