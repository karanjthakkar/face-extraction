This code uses the Haar Cascade Classifier to detect face in a video feed (webcam used here) and extracts 100 training samples

The training samples and raw images are stored in a folder named Training in C: drive

The code has been tested on the following configuration:

1. Windows 7 Professional (64-bit)
2. OpenCV 2.4.2
3. Visual Studio 2012 Ultimate

The build configuration for the project in Visual Studio was x64(Release).


For users using OpenCV for the first time in a Visual Studio project, a custom property sheet has been provided "OpenCV.props"
WARNING: This property sheet can be used only if you have the OpenCV installation path as: C:\OpenCV-2.4.2\opencv


For any feedback, comment, advice, bugs, please contact me at: karanjthakkar@gmail.com