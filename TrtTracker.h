
#ifndef TRTTRACKER_H
#define TRTTRACKER_H
#pragma once
#include <vector>
#include "./deepsort_include/deepsort.h"
#include "./deepsort_include/logging.h"
#include <ctime>
#include "yolov5.h"

static Logger gLogger;

class TrtTracker
{
public:
	TrtTracker(std::string ReidModelPath);
	~TrtTracker();

	std::vector<DetectBox> getDetections(std::vector<Bbox> detectBox,int frmeId);
	void run(vector<DetectBox> sortDetectBox,cv::Mat &frame);
	void showDetection(cv::Mat& img, std::vector<DetectBox>& boxes);


private:
	DeepSort* DS;


};
#endif
