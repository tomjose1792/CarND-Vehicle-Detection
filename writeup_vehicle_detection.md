#Vehicle-Detection Project

## Writeup



**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/Cars_hog_features.png
[image2]: ./images/Non_cars_hog_features.png
[image3]: ./images/SW_1.png
[image4]: ./images/SW_2.png
[image5]: ./images/SW_3.png
[image6]: ./images/HeatMap_0.png
[image7]: ./images/HeatMap_1.png
[image8]: ./images/HeatMap_2.png
[image9]: ./images/HeatMap_3.png
[image10]: ./images/HeatMap_4.png
[image11]: ./images/HeatMap_5.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

Firstly , I used 3 types of features to be extracted from images for image classification by the classifier (SVM). They were HOG features, spatial features and color histogram features. 

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The class for HOG features is in `get_hog_features()`. It is contained in the code cell 2 of the IPython notebook.

Code Cell 7 shows the visualisation of the HOG features. I started by reading in a `vehicle` and `non-vehicle` image.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Vehicle
 
![alt text][image1]

Non-vehicle

![alt text][image2]

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them (like above pics) to get a feel for what the `skimage.hog()` output looks like.

I finally settled for the `LUV` color space and then converted to grayscale and applied HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` on the images.


#### 2. Explain how you settled on your final choice of HOG parameters.

It was basically trial and error method for choosing the parameters. Started with the parameter values used in the Udacity tutorials. I have pretty much retained the same values of the parameters for the project.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The classes `img_features()` and `extract_features()` extracted HOG features, color histogram features. Optimised parameters were passed for optimised extraction of features. Data augmentation was alsocarried out by flipping images.
The feature extraction of the car and non car images were carried out for classifer training. Normalisation is done to ensure that the classifier is not dominated by just a few subsets. `StandardScaler()` method from `sklearn` is used for normalisation.
The dataset was divided into 80% fro training set and 20% for test set.  

Refer code cell 5.The classifier is a linear SVM and was trained. The test accuracy was 98.9%.  

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the the sliding window algorithm, `slide_window()`,code cell 8, explained in the Udacity tutorials. This class generates a list of boxes over the images. `draw_boxes()` used to draw the boxes. `search_windows()` class used to extract features at each window positions for classifications. These classes from the udactiy tutorials werre used. The overlapping parameters were set depending on the output from classifier. Then, settled for 85% overlap for the output I got. 



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

These were the output image after the classification by the SVC.

![alt text][image3]

![alt text][image4]

![alt text][image5]

For optimisation I varied the `slide_window()` parameters like the `y_start_stop=[]`, `xy_window=[]`, `xy_overlap=[]` for best results. Finally, the classifier identified the cars,but there are false positives seen in the images. This is avoided by various filtering methods explained in the following Rubric points.


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/tomjose1792/CarND-Vehicle-Detection/blob/master/project_solved.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For filtering the false positives, I used the heat map approach. I pretty much used the functions provided by the Udacity tutorials. The positions of positive detections in each image is identified.  From the positive detections, created a heatmap(`add_heat()`),code cell 13, and then thresholded that map to identify vehicle positions. Then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. The each blob is assumed to correspond to a vehicle. Then bounding boxes are constructed to cover the area of each blob detected. 

Here's an example result showing the heatmap from the test images `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the images, `draw_labeled_box()`(code cell 15) :

### Here are the test images and their corresponding heatmaps and labels:

![alt text][image6]


![alt text][image7]

![alt text][image8]

![alt text][image9]

![alt text][image10]

![alt text][image11]


After the filtering the result has not been perfect. therefore I searched online forums for methods to make the filtering process better. So, for the video implementation, consequently, I made modifications by using function the `draw_labeled_bboxes()`, similar to the Udacity tutorials, code cell 14 and `process_frame()` for mutiscale windows over the frames for better classification. Also, to reduce jitters , a `filt()` function was used which applies a simple low-pass filter on the new and the previous cars boxes coordinates and sizes makes the car boundaries on the video to be smooth.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* Well, firstly the assigning of parameter values for ideal outputs were a bit tricky. It was more of a trial and error method I considered.

* Secondly, the algorithm can fail in different light conditions, since different color spaces capture different levels of features. So, may be an optimsed color space might have to be selected before feature extractions.  
* In the video actually when the cars ovelap a single box is created, maybe having separate boxes for the cars might be ideal in those situations.

