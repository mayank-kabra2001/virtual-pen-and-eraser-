import cv2 
import numpy as np 
import time 

def nothing(x) :
	pass

#initialize webcam feed
cap = cv2.VideoCapture(0) 

cap.set(3 , 1280)
cap.set(4 , 720)

#create a window for trackers 
cv2.namedWindow("Trackbars")

# Making window size adjustable
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# Now create 6 trackbars that will control the lower and upper range of
# H,S and V channels. The Arguments are like this: Name of trackbar,
# window name, range,callback function. For Hue the range is 0-179 and
# for S,V its 0-255.
cv2.createTrackbar("L - H" , "Trackbars" , 0 , 179 , nothing)
cv2.createTrackbar("L - S" , "Trackbars" , 0 , 255 , nothing)
cv2.createTrackbar("L - V" , "Trackbars" , 0 , 255 , nothing)
cv2.createTrackbar("U - H" , "Trackbars" , 279 , 179 , nothing)
cv2.createTrackbar("U - S" , "Trackbars" , 255 , 255 , nothing)
cv2.createTrackbar("U - V" , "Trackbars" , 255 , 255 , nothing)

# Load these 2 images and resize them to the same size.
pen_img = cv2.resize(cv2.imread('pen.png',1), (50, 50))
eraser_img = cv2.resize(cv2.imread('eraser.jpg',1), (50, 50))



while True :

##################################################################################
# Step 1: Find Color range of target Pen and save it
####################################################################################

	#start reading webcam feed frame by frame 
	ret , frame = cap.read() 
	if not ret :
		break 

	#flip the frame horizontally 
	frame = cv2.flip(frame , 1)

	# convert BGR to HSV
	hsv = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV)

	#get the new values of trackbar in real time as user changes them 
	l_h = cv2.getTrackbarPos("L - H" , "Trackbars")
	l_s = cv2.getTrackbarPos("L - S" , "Trackbars")
	l_v = cv2.getTrackbarPos("L - V" , "Trackbars")
	u_h = cv2.getTrackbarPos("U - H" , "Trackbars")
	u_s = cv2.getTrackbarPos("U - S" , "Trackbars")
	u_v = cv2.getTrackbarPos("U - V" , "Trackbars")

	#set lower and upper hsv range according to the value selected  by the tracker 
	lower_range = np.array([l_h , l_s , l_v])
	upper_range = np.array([u_h , u_s , u_v])


	#filter the image and get the bnary mask , where white represent  you target color 
	mask = cv2.inRange(hsv , lower_range , upper_range)

	#you can visualize the real part of target color 
	res = cv2.bitwise_and(frame , frame , mask = mask)

	#if esc is pressed exit 
	key = cv2.waitKey(1) 
	if key == 27: 
		break 

	thearray = [[l_h , l_s , l_v] , [u_h , u_s , u_v]]

	#if user pres s print this array 
	if key == ord('s'):
		print(thearray) 

	#also save this array as penval.npy 
	np.save('penval' , thearray)


##############################################################################################
#Step 2: Maximizing the Detection Mask and Getting rid of the noise
##############################################################################################


	# this variable determines if we want to load color range from memory or use the defined ines 
	load_from_disk = True 

	if load_from_disk : 
		penval = np.load('penval.npy')


	#creating A 5X5  kernel for morphological ops 
	kernel = np.ones((5,5) , np.uint8) 


	#if your reading from mem then load the upper and lower ranges from here 
	if load_from_disk :
		lower_range = penval[0] 
		upper_range = penval[1] 

	#otherwise define your custom values 
	else : 
		lower_range = np.array([26 , 80 , 147])
		upper_range = np.array([81 , 255, 255]) 


	#perform morphological ops to get rid of noise
	mask = cv2.erode(mask , kernel , iterations = 1) 
	mask = cv2.dilate(mask ,kernel , iterations = 2)

	res = cv2.bitwise_and(frame , frame , mask = mask)

	#converting the binary mask to 3 channel image 
	mask_3 = cv2.cvtColor(mask , cv2.COLOR_GRAY2BGR)

	#stack the mask , original frame and the filtered result 
	stacked = np.hstack((mask_3 , frame , res))

	#show this stacked frame at 40% of the size
	cv2.imshow('Trackbars ' , cv2.resize(stacked , None , fx = 0.4 , fy = 0.4))


##################################################################################################
#Step 3: Tracking the Target Pen
##################################################################################################

	# Initializing the canvas on which we will draw upon
	canvas = None
	
	# Initilize x1,y1 points
	x1,y1=0,0

	#threshold for filter noise 
	noiseth = 800 

	# Threshold for wiper, the size of the contour must be bigger than for us to clear the canvas 
	wiper_thresh = 100000

	# A variable which tells when to clear canvas, if its True then we clear the canvas
	clear = False

	#create a background subtractor object 
	backgroundObject = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

	#threshold for background to check amount of disruption 
	background_threshold = 600 

	#variables which tells you if you're using a pen or eraser .
	switch = 'Pen' 

	#wth this variable we will monitor the time between previous switch . 
	last_switch = time.time() 

	#take the top left of the frame and apply the background subtractor there
	top_left = frame[0 : 50 , 0 : 50] 
	fgmask = backgroundObject.apply(top_left) 


	#note the number of pixels that are white , this is the level of disruption .
	switch_thresh = np.sum(fgmask == 255) 

	# If the disruption is greater than background threshold and there has been some time after the previous switch then you. can change the object type.
	if switch_thresh > background_threshold and (time.time() - last_switch) > 1: 

		# save the time of the switch 
		last_switch = time.time() 

		if switch == 'Pen' : 
			switch = 'Eraser' 
		else: 
			switch = 'Pen' 


	# find contours in the frame
	contours , hierarchy = cv2.findContours(mask , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE) 


	if contours and cv2.contourArea(max(contours , key = cv2.contourArea)) > noiseth : 

		#grab the biggest contour w.r.t area 
		c = max(contours , key = cv2.contourArea) 

		#get bounding box coordinates around that contour 
		x , y, w, h = cv2.boundingRect(c) 

		# Get the area of the contour
		area = cv2.contourArea(c)

		#draw the bounding box 
		cv2.rectangle (frame , (x,y) , (x+w , y+w) , (0 , 25 , 255) , 2)

###############################################################################################
#Step 4: Drawing with the Pen
################################################################################################

		# Initialize the canvas as a black image of the same size as the frame.
		if canvas is None:
			canvas = np.zeros_like(frame)

	    # If there were no previous points then save the detected x2,y2 coordinates as x1,y1.
	    # This is true when we writing for the first time or when writing again when the pen had disappeared from view.
		if x1 == 0 and y1 == 0:
	  		x1 , y1 = x , y 

#################################################################################################
#Step 6: Adding the Eraser Functionality
##################################################################################################
		
		else : 
			if switch == 'Pen': 
			  	#draw the line on canvas 
			  	canvas = cv2.line(canvas , (x1 ,y1) , (x, y) , [255 ,0 , 0] , 5) 
			
			else : 
		    	# draw circle for eraser 
				cv2.circle(canvas , (x ,y) , 20 , [0 ,0 ,0] , -1) 

	    #after the line drawn the new points become the previous points 
		x1 , y1 = x , y


#################################################################################################
#Step 5: Adding An Image Wiper
#################################################################################################
	    # Now if the area is greater than the wiper threshold then set the clear variable to True and warn User.	
		# if area > wiper_thresh: 
		# 	cv2.putText(canvas , 'clearing canvas' , (100, 200) , cv2.FONT_HERSHEY_SIMPLEX , 2 , (0, 0 , 255) , 5 , cv2.LINE_AA) 
		# 	clear = True 

	else: 

		# if there were no contours detected the make x1 , y1 =0 
		x1 , y1 = 0 , 0 

	# When c is pressed clear the canvas
	if key == ord('c'):
		canvas = None


	#merge the canvas and the frame 
	frame = cv2.add(frame , canvas) 

	_ , mask = cv2.threshold(cv2.cvtColor(canvas , cv2.COLOR_BGR2GRAY) , 20 , 255 , cv2.THRESH_BINARY) 

	foreground = cv2.bitwise_and(canvas , canvas , mask = mask)
	background = cv2.bitwise_and(frame , frame , mask = cv2.bitwise_not(mask)) 
	frame = cv2.add(foreground , background) 


    # Clear the canvas after 1 second if the clear variable is true
	if clear == True:

	    time.sleep(1)

	    canvas = None
	   
	    clear = False
		

#################################################################################################
# END 
##################################################################################################

	#converting the binary mask to 3 channel image 
	mask_3 = cv2.cvtColor(mask , cv2.COLOR_GRAY2BGR)

	#stack the mask , original frame and the filtered result 
	stacked = np.hstack((mask_3 , frame , res))


	cv2.imshow('image' , frame) 


cv2.destroyAllWindows() 
cap.release()



