
#pragma once
#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdio.h>
#include <math.h>
#include <string>

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "cuda_runtime_api.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"

#include <npp.h>


#define BATCH_SIZE 1
#define INPUT_W 640
#define INPUT_H 640
#define INPUT_SIZE 640
#define IsPadding 1
#define NUM_CLASS 20
#define NMS_THRESH 0.45
#define CONF_THRESH 0.25
#define PROB_THRESH 0.80

using namespace std;
using namespace cv;


// 中点坐标宽高
struct Bbox {
	float x;
	float y;
	float w;
	float h;
	float score;
	int classes;
};



class yolov5
{
public:
	yolov5();
public:
	//void preprocess(cv::Mat &img, float dstdata_arr[]);  //速度太慢了，不知道为啥。。。。
	//void cudaResize(cv::Mat &image, cv::Mat &rsz_img);
	void preprocess(cv::Mat& img, float data[]);
	void pred2box(float *preds, std::vector<Bbox> &out);
	float IOUCalculate(const Bbox &det_a, const Bbox &det_b);
	void NmsDetect(std::vector<Bbox> &detections);
	std::vector<Bbox> rescale_box(std::vector<Bbox> &out, int width, int height);
	cv::Mat renderBoundingBox(cv::Mat image, const std::vector<Bbox> &bboxes);

	//nvinfer1::IExecutionContext* engine_context_yolo = nullptr;
public:
	//std::vector<std::string> class_names = { "normal","normal", "normal", "normal",
	//			"normal", "xirou", "cancer", "normal", "normal",
	//			"normal", "normal","normal", "normal","normal",
	//			"normal", "normal", "normal", "normal","normal" };


	std::vector<std::string> class_names = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
		"traffic light","fire hydrant", "stop sign", "parking meter", "bench","bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack","umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush" };


};

