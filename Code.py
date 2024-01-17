import cv2
 # Importing OpenCV Library for basic image processing functions
import numpy as np # Numpy for array related functions and mathamatical functions
import dlib
from imutils import face_utils
from playsound import playsound
import winsound


cam_port = 0
vidcap = cv2.VideoCapture(cam_port)

active = 0
drowsy = 0
sleepy = 0
status = ""
color = (0,0,0)
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # We use the model

#already trained, to predict the landmarks which detects the face in 68 different points
def eye_dist(point_a, point_b):
	distance = np.linalg.norm(point_a - point_b)
	return distance

def blinked(p1, p2, p3, p6, p5, p4): # a eye has six landmark points
	height = eye_dist(p2, p6) + eye_dist(p3, p5) # short distance of the eye has 2 points on top and 2 points below
	width = eye_dist(p1, p4) # long distance of eye has 2 points on either sides of the eye
	ratio = height / (2.0 * width)
  
  #Checking if blinking of eye is occuring
	if ratio > 0.25:
		return 2 # denotes eye open
	elif ratio > 0.21 and ratio <= 0.25:
		return 1 # denotes eye half closed
	else:
		return 0 # denotes eye closed


while(True):
  ret, image = vidcap.read()
  
  #print("Original Image")
  #cv2.imshow('Original Image',image)
  scaled_width = 900
  scaled_height = 700
  scaled_points = (scaled_width, scaled_height)
  #print("Resized Image")
  resize_img = image
  #cv2.imshow('Cropped Image',resize_img)

  imgGray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
  #print("GrayScale")
  #cv2.imshow('Grey Scaled Image',imgGray)

  imgContours = resize_img.copy()
  # apply binary thresholding
  ret, thresh = cv2.threshold(imgGray, 150, 255, cv2.THRESH_BINARY)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE , cv2.CHAIN_APPROX_NONE)

  #cv2.imshow('Segmentation',(cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)))



  denoised31=cv2.fastNlMeansDenoising(image,None,6,6)
  #cv2.imshow('Laplacian Filter',denoised31)

  blurred23=cv2.GaussianBlur(denoised31,(3,3),0)
  #cv2.imshow('Guassian Filter',blurred23)

  median_blur= cv2.medianBlur(image, 3)
  #cv2.imshow('Median Filter',blurred23)

  gauss = cv2.GaussianBlur(image, (7,7), 0)
  unsharp_image = cv2.addWeighted(denoised31, 2, gauss, -1, 0)
  #cv2.imshow('Unsharp Masking and Boost Filtering',unsharp_image)


  new_image = cv2.Laplacian(denoised31,cv2.CV_8UC1)
  lapimg = cv2.add(denoised31, new_image)
  #cv2.imshow('Laplacian Filter',lapimg)





  #Hori1 = np.concatenate((imgGray, denoised31), axis=1)
  Hori2 = np.concatenate((denoised31, blurred23), axis=1)
  Hori3 = np.concatenate((image, median_blur), axis=1)
  #Hori4 = np.concatenate((image, lapimg), axis=1)

  #cv2.imshow('In built function from openCV', Hori1)
  #cv2.imshow('Guassian Filter', Hori2)
  #cv2.imshow('Meadian Filter', Hori3)
  #cv2.imshow('Laplacian Filter', Hori4)

  
  #cv2.imshow('Filtered image', denoised31)
  #cv2.imshow('sharped image', unsharp_image)

  imgGray2 = unsharp_image.copy()
  
  bounding_box1 = face_detector(imgGray2)
  for d in bounding_box1:
    bounded_box = cv2.rectangle(imgGray2, (d.left(), d.top()), (d.right(), d.bottom()), 0, 1)
    cv2.imshow('Face Detection',bounded_box)

 
  
  for d in bounding_box1:
    p1 = d.top()
    p2 = d.bottom() 
    p3 = d.left()
    p4 = d.right()    
    cropped_image = imgGray[p3:p4, p1:p2]
    #cv2.imshow('test6',cropped_image)

    #Since we have detected face now we have to detect eyes
    eye = predictor(imgGray, d)
    eye = face_utils.shape_to_np(eye)
    # The numbers are the landmarks which represents the eyes(index is decreased by 1 since its a list)
    # Passing the points of each eye to the compute_blinking_ratio function we calculate the ratio for both the eye
    left_eye_blink = blinked(eye[36], eye[37], eye[38], eye[41], eye[40], eye[39])
    right_eye_blink = blinked(eye[42], eye[43], eye[44], eye[47], eye[46], eye[45])
    # Now judge what to do for the eye blinks
    print(left_eye_blink)
    print(right_eye_blink)
    if(left_eye_blink == 0 or right_eye_blink == 0):
      sleepy = sleepy + 1
      drowsy = 0
      active = 0
      #print("1")
      if(sleepy > 6):
        status = "SLEEPY !!"
        color = (255, 0, 0)
        # Playing the alert beep
        frequency = 2000 # Set frequency to 2000 Hertz
        duration = 1000 # Set duration to 1000 ms == 1 second
        winsound.Beep(frequency, duration)
        ##playsound('sound.wav')
        
      elif(left_eye_blink == 1 or right_eye_blink == 1):
        drowsy = drowsy + 1
        sleepy = 0
        active = 0
        #print("2")
        if(drowsy > 6):
          status = "DROWSY !"
          color = (0, 0, 255)
          # Playing the alert beep
          frequency = 2000 # Set frequency to 2000 Hertz
          duration = 1000 # Set duration to 1000 ms == 1 second
          winsound.Beep(frequency, duration)
          ##playsound('sound.wav')

      else:
        #print("3")
        active = active + 1
        drowsy = 0
        sleep = 0
        if(active > 6):
          status = "ACTIVE"
          color = (0, 255, 0)
      ##cv2.putText(d, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3) # frame displayed along with specified colour    

      cv2.putText(image, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3) # frame displayed along with specified colour
      # to display the landmark points in the frame
      for n in range(0, 68):
        (x, y) = eye[n]
        cv2.circle(image, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", d)
    cv2.imshow("Result of detector", image)
       
  


  

  if cv2.waitKey(1) & 0xFF == ord('q'):
        break




vidcap.release()
cv2.destroyAllWindows()

