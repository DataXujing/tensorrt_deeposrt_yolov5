#include "TrtTracker.h"



TrtTracker::TrtTracker(std::string ReidModelPath) {
	//ÊµÀý»¯DeepSort
	// DeepSort(std::string modelPath, int batchSize, int featureDim, int gpuID, ILogger* gLogger);
	DS = new DeepSort(ReidModelPath, 128, 256, 0, &gLogger);
}

TrtTracker::~TrtTracker() {

}


void TrtTracker::showDetection(cv::Mat& img, std::vector<DetectBox>& boxes) {
	//cv::Mat temp = img.clone();
	for (auto box : boxes) {
		cv::Point lt(box.x1, box.y1);
		cv::Point br(box.x2, box.y2);
		cv::rectangle(img,lt,br, cv::Scalar(255, 33, 115), 2, cv::LINE_8, 0);

		int baseLine;
		//std::string lbl = cv::format("ID:%d_Label:%d_Conf:%.2f", (int)box.trackID, (int)box.classID, box.confidence);
		std::string lbl = cv::format("%d-%d", (int)box.classID, (int)box.trackID);
		cv::Size labelSize = getTextSize(lbl, cv::FONT_HERSHEY_TRIPLEX, 0.5, 1, &baseLine);

		rectangle(img, cv::Point(box.x1, box.y1 - round(1.5*labelSize.height)),
			cv::Point(box.x1 + round(1.0*labelSize.width), box.y1+ baseLine), cv::Scalar(255, 33, 115), cv::FILLED);
		cv::putText(img, lbl, lt, cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255,255, 255));
	}

}


std::vector<DetectBox> TrtTracker::getDetections(std::vector<Bbox> detectBox, int frmeId) {
	std::vector<DetectBox>  sortBox;
	for (int i = 0; i < detectBox.size(); i++) {
		//x1,y1,x2,y2,conf,label
		DetectBox bb(detectBox[i].x - detectBox[i].w / 2, detectBox[i].y - detectBox[i].h / 2,
			detectBox[i].x + detectBox[i].w / 2, detectBox[i].y + detectBox[i].h / 2,
			detectBox[i].score, detectBox[i].classes);
		sortBox.push_back(bb);
	}

	return sortBox;
}

void TrtTracker::run(vector<DetectBox> sortDetectBox,cv::Mat &frame) {

	cv::Mat frame_1;
	cv::cvtColor(frame, frame_1, cv::COLOR_BGR2RGB);
	DS->sort(frame_1, sortDetectBox);
	showDetection(frame, sortDetectBox);

}