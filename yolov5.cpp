#include "yolov5.h"


yolov5::yolov5(){

}

//void yolov5::cudaResize(cv::Mat &image, cv::Mat &rsz_img)
//{
//	int outsize = rsz_img.cols * rsz_img.rows * sizeof(uchar3);
//
//	int inwidth = image.cols;
//	int inheight = image.rows;
//	int memSize = inwidth * inheight * sizeof(uchar3);
//
//	NppiSize srcsize = { inwidth, inheight };
//	NppiRect srcroi = { 0, 0, inwidth, inheight };
//	NppiSize dstsize = { rsz_img.cols, rsz_img.rows };
//	NppiRect dstroi = { 0, 0, rsz_img.cols, rsz_img.rows };
//
//	uchar3* d_src = NULL;
//	uchar3* d_dst = NULL;
//	cudaMalloc((void**)&d_src, memSize);
//	cudaMalloc((void**)&d_dst, outsize);
//	cudaMemcpy(d_src, image.data, memSize, cudaMemcpyHostToDevice);
//
//	// nvidia npp 图像处理
//	nppiResize_8u_C3R((Npp8u*)d_src, inwidth * 3, srcsize, srcroi,
//		(Npp8u*)d_dst, rsz_img.cols * 3, dstsize, dstroi,
//		NPPI_INTER_LINEAR);
//
//
//	cudaMemcpy(rsz_img.data, d_dst, outsize, cudaMemcpyDeviceToHost);
//
//	cudaFree(d_src);
//	cudaFree(d_dst);
//}

void yolov5::preprocess(cv::Mat& img, float data[]) {
	int w, h, x, y;
	float r_w = INPUT_W / (img.cols*1.0);
	float r_h = INPUT_H / (img.rows*1.0);
	if (r_h > r_w) {
		w = INPUT_W;
		h = r_w * img.rows;
		x = 0;
		y = (INPUT_H - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = INPUT_H;
		x = (INPUT_W - w) / 2;
		y = 0;
	}
	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
	//cudaResize(img, re);
	cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

	int i = 0;
	for (int row = 0; row < INPUT_H; ++row) {
		uchar* uc_pixel = out.data + row * out.step;
		for (int col = 0; col < INPUT_W; ++col) {
			data[i] = (float)uc_pixel[2] / 255.0;
			data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
			data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
			uc_pixel += 3;
			++i;
		}
	}

}


//void yolov5::preprocess(cv::Mat &img, float dstdata_arr[]) {
//	int w, h, x, y;
//	float r_w = INPUT_W / (img.cols*1.0);
//	float r_h = INPUT_H / (img.rows*1.0);
//	if (r_h > r_w) {
//		w = INPUT_W;
//		h = r_w * img.rows;
//		x = 0;
//		y = (INPUT_H - h) / 2;
//	}
//	else {
//		w = r_h * img.cols;
//		h = INPUT_H;
//		x = (INPUT_W - w) / 2;
//		y = 0;
//	}
//	cv::Mat re(h, w, CV_8UC3);
//	cv::Mat img_rgb;
//	cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
//	cv::resize(img_rgb, re, re.size(), 0, 0, cv::INTER_LINEAR);
//	cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
//	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
//
//	cv::Mat img_rgb_float;
//	out.convertTo(img_rgb_float, CV_32FC3, 1 / 255.0); // 转float 归一化
//
//	std::vector<cv::Mat> rgbChannels(3);
//	std::vector<float> dstdata;
//	cv::split(img_rgb_float, rgbChannels);
//
//
//	for (auto i = 0; i < rgbChannels.size(); i++) {
//		std::vector<float> data = std::vector<float>(rgbChannels[i].reshape(1, 1));
//
//		for (int j = 0; j < data.size(); j++) {
//			if (i == 0) {
//				dstdata.push_back(data[j]);
//			}
//			else if (i == 1) {
//				dstdata.push_back(data[j]);
//			}
//			else {
//				dstdata.push_back(data[j]);
//			}
//		}
//	}
//
//	std::copy(dstdata.begin(), dstdata.end(), dstdata_arr);
//}

void yolov5::pred2box(float *preds, std::vector<Bbox> &out) {

	Bbox box;
	for (int i = 0; i < 25200; i++) {
		box.x = preds[i * 85];  //25 85
		box.y = preds[i * 85 + 1];
		box.w = preds[i * 85 + 2];
		box.h = preds[i * 85 + 3];

		float max_conf = 0.0;
		int max_id = 0;
		for (int j = 0; j < 80; j++) {
			if (preds[i * 85 + 5 + j] >= max_conf) {
				max_conf = preds[i * 85 + 5 + j] * preds[i * 85 + 4];
				max_id = j;
			}
		}
		box.score = max_conf;
		box.classes = max_id;

		//结肠息肉和早癌是第5和6
		//if (box.classes != 5 && box.classes != 6) {
		//	continue;
		//}

		//coco 0是person
		if (box.classes != 0) {
			continue;
		}


		if (box.score >= CONF_THRESH) {
			out.push_back(box);
		}

	}


}

float yolov5::IOUCalculate(const Bbox &det_a, const Bbox &det_b) {
	cv::Point2f center_a(det_a.x, det_a.y);
	cv::Point2f center_b(det_b.x, det_b.y);
	cv::Point2f left_up(std::min(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2),
		std::min(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2));
	cv::Point2f right_down(std::max(det_a.x + det_a.w / 2, det_b.x + det_b.w / 2),
		std::max(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2));
	float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
	float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
	float inter_l = det_a.x - det_a.w / 2 > det_b.x - det_b.w / 2 ? det_a.x - det_a.w / 2 : det_b.x - det_b.w / 2;
	float inter_t = det_a.y - det_a.h / 2 > det_b.y - det_b.h / 2 ? det_a.y - det_a.h / 2 : det_b.y - det_b.h / 2;
	float inter_r = det_a.x + det_a.w / 2 < det_b.x + det_b.w / 2 ? det_a.x + det_a.w / 2 : det_b.x + det_b.w / 2;
	float inter_b = det_a.y + det_a.h / 2 < det_b.y + det_b.h / 2 ? det_a.y + det_a.h / 2 : det_b.y + det_b.h / 2;
	if (inter_b < inter_t || inter_r < inter_l)
		return 0;
	float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
	float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
	if (union_area == 0)
		return 0;
	else
		return inter_area / union_area - distance_d / distance_c;

}

void yolov5::NmsDetect(std::vector<Bbox> &detections) {
	sort(detections.begin(), detections.end(), [=](const Bbox &left, const Bbox &right) {
		return left.score > right.score;
		});

	for (int i = 0; i < (int)detections.size(); i++)
		for (int j = i + 1; j < (int)detections.size(); j++)
		{
			float iou = IOUCalculate(detections[i], detections[j]);
			if (iou > NMS_THRESH)
				detections[j].score = 0;
		}

	detections.erase(std::remove_if(detections.begin(), detections.end(), [](const Bbox &det)
		{ return det.score == 0; }), detections.end());

}

std::vector<Bbox> yolov5::rescale_box(std::vector<Bbox> &out, int width, int height) {
	float gain = 640.0 / std::max(width, height);
	float pad_x = (640.0 - width * gain) / 2;
	float pad_y = (640.0 - height * gain) / 2;

	std::vector<Bbox> boxs;
	Bbox box;
	for (int i = 0; i < (int)out.size(); i++) {
		box.x = (out[i].x - pad_x) / gain;
		box.y = (out[i].y - pad_y) / gain;
		box.w = out[i].w / gain;
		box.h = out[i].h / gain;
		box.score = out[i].score;
		box.classes = out[i].classes;

		boxs.push_back(box);
	}

	return boxs;

}


cv::Mat yolov5::renderBoundingBox(cv::Mat image, const std::vector<Bbox> &bboxes) {
	for (const auto &rect : bboxes)
	{

		cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
		cv::rectangle(image, rst, cv::Scalar(255, 204, 0), 2, cv::LINE_8, 0);
		//cv::rectangle(image, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2), cv::Point(rect.x + rect.w / 2, rect.y + rect.h / 2), cv::Scalar(255, 204,0), 3);

		int baseLine;
		std::string label = class_names[rect.classes] + ": " + std::to_string(rect.score * 100).substr(0, 4) + "%";
		cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_TRIPLEX, 0.5, 1, &baseLine);
		//int newY = std::max(rect.y, labelSize.height);
		rectangle(image, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - round(1.5*labelSize.height)),
			cv::Point(rect.x - rect.w / 2 + round(1.0*labelSize.width), rect.y - rect.h / 2 + baseLine), cv::Scalar(255, 204, 0), cv::FILLED);
		cv::putText(image, label, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 204, 255));


	}
	return image;
}