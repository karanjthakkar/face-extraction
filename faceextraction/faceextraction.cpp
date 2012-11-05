/**************************************************************************************************************************
@file faceextraction.cpp
@author Karan Thakkar
@brief Uses the Haar Cascade Classifier to detect face in a video feed (webcam used here) and extracts 100 training samples
**************************************************************************************************************************/

#include "stdafx.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"

#include <iostream>

using namespace std;
using namespace cv;

/** Function Headers **/
void detectAndCrop(IplImage* frame );

/** Global variables **/
int i=0;
CvHaarClassifierCascade* face_cascade = 0;		//Haar Classifier for face
CvMemStorage* pStorageface = 0;					// memory for detector to use
char buf[50];
char buf1[50];
int minWidth = MAXINT32, minHeight = MAXINT32;
    
RNG rng(12345);

/**
 * @function main
 */
int main( int argc, const char** argv )
{
	CvCapture* capture;
	IplImage* frame = 0;
	//-- 1. Load the cascade (WARNING: Use your OpenCV installation path)
	face_cascade = (CvHaarClassifierCascade *)cvLoad("C:\\OpenCV-2.4.2\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt_tree.xml", 0 , 0, 0);
	
	//Use this loop to capture some training data for your face recognition algorithm
	for (i=0; i<=100; i++)
	{
		capture = cvCaptureFromCAM(1);
		frame = cvQueryFrame( capture );
		//Store the raw images in a folder named Training in C: drive
		sprintf(buf, "C:\\Training\\raw%d.jpg", i);
		cvSaveImage(buf, frame, 0);
		//Apply the classifier to the frame
	    detectAndCrop(frame);
	}
  
	//Crop the extracted faces to the same size and store them in the same folder
	for (i=0; i<=100; i++)
	{

		sprintf(buf, "C:\\Training\\extract%d.jpg", i);
		frame = cvLoadImage(buf, CV_LOAD_IMAGE_UNCHANGED);
		IplImage* tmpsize = cvCreateImage(Size(minWidth, minHeight), frame->depth, frame->nChannels);	
		cvResize(frame, tmpsize, INTER_CUBIC);
		sprintf(buf1, "C:\\Training\\extract%d.jpg", i);
		cvSaveImage(buf1, tmpsize, 0);

	}
	
	// clean up and release resources
    cvReleaseImage(&frame);
	if(face_cascade) cvReleaseHaarClassifierCascade(&face_cascade);
	if(pStorageface) cvReleaseMemStorage(&pStorageface);

	return 0;

}

/**
 * @function detectAndCrop
 */
void detectAndCrop(IplImage* frame )
{
	CvSeq * pFaceRectSeq;               // memory-access interface
	pStorageface = cvCreateMemStorage(0);
	cvClearMemStorage(pStorageface);

    // detect faces in image
	pFaceRectSeq = cvHaarDetectObjects
		(frame, face_cascade, pStorageface,
		1.1,                        // increase search scale by 10% each pass
		3,                          // merge groups of three detections
		0,						    // 0(entire region) or CV_HAAR_DO_CANNY_PRUNING(skip regions unlikely to contain a face)
		cvSize(40, 40)  			// smallest size face to detect = 40x40
		);            

	CvRect* r = (CvRect*)cvGetSeqElem(pFaceRectSeq, 0);
	CvPoint pt1 = { r->x, r->y };
	CvPoint pt2 = { r->x + r->width, r->y + r->height };
	
	//Use this to find out the minWidth and minHeight among all the faces.
	if (r->width < minWidth || r->height < minHeight)
	{

		minWidth = r->width;
		minHeight = r->height;

	}

	cvSetImageROI(frame, cvRect(r->x, r->y, r->width, r->height));
	IplImage *cropImage = cvCreateImage(cvGetSize(frame), frame->depth, frame->nChannels);
	sprintf(buf1, "C:\\Training\\extract%d.jpg", i);
	cvSaveImage(buf1, frame, 0);

}

         