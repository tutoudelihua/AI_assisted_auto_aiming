#include"yolo.h";
using namespace std;
using namespace cv;
using namespace dnn;
Yolo& Yolo::getInstance()
{
	static Yolo instance;
	return instance;
}

bool Yolo::readModel(Net& net, string netPath, bool isCuda = false) {
	try {
        net = readNet(netPath);
        //net = cv::dnn::readNetFromONNX(netPath);
	}
	catch (const std::exception&) {
		return false;
	}
	//cuda
	if (isCuda) {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}
	//cpu
	else {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	return true;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

bool Yolo::Detect(Mat& SrcImg, Net& net, vector<Output>& output) {
	bool isYoloV5 = true;
    cv::Size inputSize(netWidth, netHeight);
    cv::Mat netInputImg;
    cv::resize(SrcImg, netInputImg, inputSize);

    cv::Mat blob;
    blobFromImage(netInputImg, blob, 1 / 255.0, inputSize, cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> netOutputImg;
    try {
        net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
    } catch (const std::exception& e) {
        std::cerr << "Error during forward pass: " << e.what() << std::endl;
        return false;
    }


    int rows = netOutputImg[0].size[1]; // n+4
    int dimensions = netOutputImg[0].size[2]; // 8400

    if (dimensions > rows) // YOLO V8以后的输出是[1,N,8400]
    {
		isYoloV5 = false;
        rows = netOutputImg[0].size[2];
        dimensions = netOutputImg[0].size[1];

        netOutputImg[0] = netOutputImg[0].reshape(1, dimensions);
        cv::transpose(netOutputImg[0], netOutputImg[0]);
    }

    float* data = (float*)netOutputImg[0].data;

    float x_factor = (float)SrcImg.cols / netWidth;
    float y_factor = (float)SrcImg.rows / netHeight;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i)
    {
        if (!isYoloV5)
        {
            float* classes_scores = data + 4;

            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;

            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

            if (maxClassScore > classThreshold)
            {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int(x * x_factor);
                int top = int(y * y_factor);

                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        else // yolov5
        {
            float confidence = data[4];

            if (confidence >= classThreshold)
            {
                float* classes_scores = data + 5;

                cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;

                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

                if (max_class_score > classThreshold)
                {
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left = int(x * x_factor);
                    int top = int(y * y_factor);

                    int width = int(w * x_factor);
                    int height = int(h * y_factor);

                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }

        data += dimensions;
    }

    // NMS
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, classThreshold, nmsThreshold, nms_result);

    for (size_t i = 0; i < nms_result.size(); ++i) {
        int idx = nms_result[i];
        Output result;
        result.id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }

    return !output.empty();
}

//bool Yolo::Detect(Mat& SrcImg, Net& net, vector<Output>& output) {
//    //cout << "Starting detection..." << endl;
//
//    Mat blob;
//    int col = SrcImg.cols;
//    int row = SrcImg.rows;
//    int maxLen = MAX(col, row);
//    Mat netInputImg = SrcImg.clone();
//    if (maxLen > 1.2 * col || maxLen > 1.2 * row) {
//        Mat resizeImg = Mat::zeros(maxLen, maxLen, CV_8UC3);
//        SrcImg.copyTo(resizeImg(Rect(0, 0, col, row)));
//        netInputImg = resizeImg;
//    }
//
//    //cout << "Creating blob from image..." << endl;
//    blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(0, 0, 0), true, false);
//
//    //cout << "Setting input blob..." << endl;
//    net.setInput(blob);
//
//    std::vector<cv::Mat> netOutputImg;
//    //cout << "Running forward pass..." << endl;
//    try {
//        //net.enableWinograd(false); //南化国产CPU不支持卷积计算
//        net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
//    }
//    catch (const std::exception& e) {
//        cerr << "Error during forward pass: " << e.what() << endl;
//        return false;
//    }
//
//    std::vector<int> classIds; // 结果id数组
//    std::vector<float> confidences; // 结果每个id对应置信度数组
//    std::vector<cv::Rect> boxes; // 每个id矩形框
//    float ratio_h = (float)netInputImg.rows / netHeight;
//    float ratio_w = (float)netInputImg.cols / netWidth;
//    int net_width = className.size() + 5; // 输出的网络宽度是类别数+5
//    float* pdata = (float*)netOutputImg[0].data;
//
//    //cout << "Processing output..." << endl;
//    for (int stride = 0; stride < 3; stride++) { // stride
//        int grid_x = (int)(netWidth / netStride[stride]);
//        int grid_y = (int)(netHeight / netStride[stride]);
//        for (int anchor = 0; anchor < 3; anchor++) { // anchors
//            const float anchor_w = netAnchors[stride][anchor * 2];
//            const float anchor_h = netAnchors[stride][anchor * 2 + 1];
//            for (int i = 0; i < grid_y; i++) {
//                for (int j = 0; j < grid_x; j++) {
//                    float box_score = pdata[4]; // 获取每一行的box框中含有某个物体的概率
//                    if (box_score > boxThreshold) {
//                        cv::Mat scores(1, className.size(), CV_32FC1, pdata + 5);
//                        Point classIdPoint;
//                        double max_class_socre;
//                        minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
//                        max_class_socre = (float)max_class_socre;
//                        if (max_class_socre > classThreshold) {
//                            // rect [x,y,w,h]
//                            float x = pdata[0];
//                            float y = pdata[1];
//                            float w = pdata[2];
//                            float h = pdata[3];
//
//                            Output result;
//                            result.id = classIdPoint.x;
//                            result.confidence = max_class_socre * box_score;
//                            result.box = Rect(x* ratio_w, y* ratio_h, w* ratio_w, h* ratio_h);
//                            output.push_back(result);
//                        }
//                    }
//                    pdata += net_width; // 下一行
//                }
//            }
//        }
//    }
//
//   // cout << "Detection completed. Number of outputs: " << output.size() << endl;
//
//    if (output.size())
//        return true;
//    else
//        return false;
//}

void Yolo::drawPred(Mat& img, vector<Output> result, vector<Scalar> color) {
	for (int i = 0; i < result.size(); i++) {
      
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		rectangle(img, result[i].box, color[result[i].id], 2, 8);

		string label = className[result[i].id] + ":" + to_string(result[i].confidence);

		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}
	imshow("test", img);
	//imwrite("out.bmp", img);
	waitKey();
	//destroyAllWindows();

}

