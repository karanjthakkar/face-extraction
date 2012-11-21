# Face Extraction

This code uses the Haar Cascade Classifier to detect face in a video feed (webcam used here) and extracts 100 training samples

The training samples and raw images are stored in a folder named Training in C: drive

The code has been tested on the following configuration:

1. Windows 7 Professional (64-bit)
2. OpenCV 2.4.2
3. Visual Studio 2012 Ultimate

The build configuration for the project in Visual Studio was x64(Release).

For users using OpenCV for the first time in a Visual Studio project, a custom property sheet has been provided **OpenCV.props**
**WARNING**: This property sheet can be used only if:

1. You have the OpenCV installation path as: `C:\OpenCV-2.4.2\opencv`
2. You are using a 64-bit Windows 7 installation

# Downloading the source code

For users unfamiliar with `github`: You can download the source code as a zip file by clicking [here](https://github.com/karanjthakkar/face-extraction/archive/master.zip)

For users who have used github before: You can simply fork the repository(if you intend to make changes and pull them to master) and/or clone it on your local machine

#Tutorial

If you are looking at a tutorial to help you use OpenCV with Visual Studio 2012 on Windows 7 64-bit, then [this blog post is just for you](http://karanjthakkar.wordpress.com/2012/11/21/usin-opencv-2-4-2-with-visual-studio-2012-on-windows-7-64-bit/)

# LICENSE

[Simplified BSD](http://en.wikipedia.org/wiki/BSD_licenses#2-clause_license_.28.22Simplified_BSD_License.22_or_.22FreeBSD_License.22.29)(See the [LICENSE](https://github.com/karanjthakkar/face-extraction/blob/master/LICENSE.txt) file)

For any feedback, comment, advice, bugs, please contact me at:
**karanjthakkar [at] gmail [dot] com**