#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
int size = 5;
std::vector<Rect>capture;
Mat Imageblurred;

static bool doMosaic(Mat pic, int size)
{
	for (int i = 0; i < pic.cols - size; i += size)
		for (int j = 0; j < pic.rows - size; j += size)
		{
			Rect m = Rect(i, j, size, size);
			Mat saic = pic(m);
			saic.setTo(mean(saic));
		}
	return true;
}

static bool doBlur()
{
	for (size_t i = 0; i< capture.size(); i++)
	{
		Mat io = Imageblurred(capture[i]);
		doMosaic(io, size);
	}
	imshow("sample",Imageblurred);
	return true;
}

int main(int argc, const char** argv)
{

	CascadeClassifier cascade_face;
	cascade_face.load("haarcascade_frontalface_alt.xml");
	VideoCapture Devicecapture;
	Devicecapture.open(0);
Mat Framecapture;

	Mat Framegrayscale;
	namedWindow("outputCapture", 1);

	
			Devicecapture >> Framecapture;
			imshow("output capture", Framecapture);
		cvtColor(Framecapture, Framegrayscale, CV_BGR2GRAY);
		equalizeHist(Framegrayscale, Framegrayscale);

		cascade_face.detectMultiScale(Framegrayscale, capture, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		for (size_t i = 0; i < capture.size(); i++)
		{
			Point pt1(capture[i].x +capture[i].width, capture[i].y + capture[i].height);
			Point pt2(capture[i].x, capture[i].y);
			rectangle(Framecapture, pt1, pt2, cvScalar(0, 255, 0, 0), 1, 8, 0);

			
			imshow("image capture",Framecapture);
			Framecapture.copyTo(Imageblurred);
		
			if (capture[i].width > 0 && capture[i].height > 0)
			{
				Mat Imageblurredcopy;
				Imageblurred.copyTo(Imageblurredcopy);
				Mat io = Imageblurredcopy(capture[i]);
				bitwise_not(io, io);
				imshow("sample1", Imageblurredcopy);
				
			}
			
			doBlur();
		}

		while (true)
		{
			int key = waitKey(0);

			if (key == 27)
				break;

			if (key == 's') 
			{
				imwrite("result",Imageblurred);
			}

			if (key == 'i') 
			{
				size += 5;
				Framecapture.copyTo(Imageblurred);
				doBlur();
			}

			if (key == 'd') 
			{
				size = size == 5 ? 5 : size - 5;
				Framecapture.copyTo(Imageblurred);
				doBlur();
			}

			if (key == 27) 
			{
				
				Framecapture.copyTo(Imageblurred);
				doBlur();
			}
		}
		imshow("ouput", Imageblurred);
		
	
	return 0;
}
