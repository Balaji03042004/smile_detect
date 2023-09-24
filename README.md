# Smart-Auto-Selfie-capture-application-by-detecting-smile

We will create a project that will take pictures every time you smile because everyone loves to see them.
This is a beginner-friendly machine learning project that will make use of the openCV library.


## Regarding the Project
Describe OpenCV.
An open-source computer vision library with a focus on real-time applications is called OpenCV.
The primary areas of concentration include video collection and processing, as well as image processing and analysis (such as face and object detection).
We don't need to bother about training and testing algorithms because it has a lot of built-in features and pre-trained models.


## Project requirements
We must be aware of the following in order to carry out this project:
1. Basic Python ideas; 2. Fundamentals of openCV.
You may use pip installer from the command line to install the library: install opencv-python using pip




### How to Develop the Project in Steps

Haarcascade_frontalface_default.xml and Haarcascade_smile.xml files are required for this project.
Please download these files and the project code using the link provided in the previous step.
Make sure to download these files into a separate folder inside the project folder because it is excellent programming practise to create distinct directories for each files.

Haar Cascade: This ML object detection approach uses a 2D grid (considered as a matrix in this case) to identify items in an image or video.
This approach uses a cascade function that has been trained using a large number of positive and negative photos to find items in other images.
It is trainable to recognise practically any thing. This project will involve making use of these trained files. 
There are four steps in the algorithm:
Feature Selection by Haar
Producing Integral Pictures
Adaboost Instruction 
Cascading Classifiers


### Steps Involved to implement Smile Detection and Selfie Capture Project

1. The openCV library is initially imported.
2. Next, launch the webcam using the cv2 function named VideoCapture in the second line.
3. Next, add the files for the haarcascade to the python file.
4. Since a video is just a collection of photos, we'll execute an endless while loop for it.
5. Next, we use read() to read images from the movie.
6. Using the fundamental openCV functions cvtColor() and BGR2GRAY, we will transform the image to a grayscale version because grayscale images are better for feature detection.
7. We will now read faces using the detectMultiscale() function, passing the grey image, ScaleFactor, and minNeighbors, along with the haarcascade file that has previously been included.
   

minNeighbors: A parameter defining the minimum number of neighbours required to keep a rectangle.
   If a face is found, the rectangle() method of cv2 will be used to draw the face's outer boundary. This method takes five arguments: an image, an initial point (x, y), an endpoint of the principal diagonal (x + width, y + height), the colour of the rectangular periphery, and the thickness of the drawn rectangular periphery.
   If a face is found, we will also look for a grin, and if that smile is found as well, we will print the image and save it where we want to save it.Imwrite(), which requires the two parameters image and location, will be used to store the photos.



