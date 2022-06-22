# Camera
import cv2
import numpy as np
from imutils.video import VideoStream
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.utils import load_img, img_to_array

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# model = load_model('Final.h5') #model converse
model = load_model('Final.h5')
# Create a VideoCapture object and read from input file
# cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture('test_new.mp4')
try:
    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')
# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')


# frame
currentframe = 0
# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")
# Read until video is completed
while(cap.isOpened()):
# Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
	# Display the resulting frame
        # if video is still left continue creating images
        name = './data/frame' + str(currentframe) + '.jpg'
        print ('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)
        filename = name
        predict = ['adidas','nike','no_shoe','converse']
        predict = np.array(predict)
        img = load_img(filename,target_size=(150,150))
        img = load_img(filename,target_size=(150,150))
        img = img_to_array(img)
        img = img.reshape(1,150,150,3)
        img = img.astype('float32')
        img = img/255
        result = np.argmax(model.predict(img),axis=-1)
        predict[result]
        if(result==0):
            a = "Brand: ADIDAS"
        if(result==2):
            a = "Brand: NIKE"
        if(result==3):
            a = "Khong mang giay"
        if(result==1):
            a = "converse"

        # org
        org = (30, 170)
        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Line thickness of 2 px
        thickness = 2
        cv2.putText(frame,a,org,font,fontScale,color,2)
        cv2.imshow('Frame', frame)
        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1

	    # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Break the loop
    else:
	    break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
