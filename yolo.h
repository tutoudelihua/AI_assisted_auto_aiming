#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
//#include<string>
//using namespace std;
//using namespace cv;
//using namespace dnn;

struct Output {
	int id;//结果类别id
	float confidence;//结果置信度
	cv::Rect box;//矩形框
};

class Yolo {
public:
	Yolo() {
	}
	~Yolo() {}
	static Yolo& getInstance();
	bool readModel(cv::dnn::Net& net, std::string netPath, bool isCuda);
	bool Detect(cv::Mat& SrcImg, cv::dnn::Net& net, std::vector<Output>& output);
	void drawPred(cv::Mat& img, std::vector<Output> result, std::vector<cv::Scalar> color);

private:
	const float netAnchors[3][6] = { { 10.0, 13.0, 16.0, 30.0, 33.0, 23.0 },{ 30.0, 61.0, 62.0, 45.0, 59.0, 119.0 },{ 116.0, 90.0, 156.0, 198.0, 373.0, 326.0 } };
	const float netStride[3] = { 8, 16.0,32 };
	const int netWidth = 640;
	const int netHeight = 640;
	float nmsThreshold = 0.25;
	float boxThreshold = 0.1f;
	float classThreshold = 0.25;
	std::vector<std::string> className = { "enermybody", "enermyhead"};

};

