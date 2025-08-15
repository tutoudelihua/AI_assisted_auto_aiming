#include "Screenshot.h"
#include"yolo.h";
#include <opencv2/opencv.hpp>
#include<windows.h>

#include <graphics.h>
#include <iostream>
#include <conio.h>

const long long INF = 0x3f3f3f3f3f3f3f3fLL;
bool flag = 1;//0 -> body 1 -> head

std::atomic<bool> running(true);
void keyListener() {
    while (running) {
        if (_kbhit()) {
            int ch = _getch();
            if (ch == 27) {
                running = false;
                break;
            }
            //std::cout << "检测到按键: " << static_cast<char>(ch) << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

signed main(int argc, char* argv[]) {

   HINSTANCE hUser32 = LoadLibrary(L"User32.dll");
    if (hUser32)
    {
        typedef BOOL(WINAPI* LPSetProcessDPIAware)(void);
        LPSetProcessDPIAware pSetProcessDPIAware = (LPSetProcessDPIAware)GetProcAddress(hUser32, "SetProcessDPIAware");
        if (pSetProcessDPIAware)
        {
            pSetProcessDPIAware();
        }
        FreeLibrary(hUser32);
    }

    Yolo& yolo = Yolo::getInstance();

    cv::dnn::Net net;
    const std::string netPath = "model/yolo11n_valorant_head_body.onnx";

    try {
        if (!yolo.readModel(net, netPath, false)) {
            throw std::runtime_error("readModel failed");
        }
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    POINT p;
    MOUSEMSG m;
    std::thread listener(keyListener);

    std::cout << "按ESC键退出" << std::endl;
    while (running) {
        Screenshot screenshot;
        cv::Mat img = screenshot.getScreenshot();
        imwrite("screenshot.jpg", img);
		img = cv::imread("screenshot.jpg");
        //cv::Mat img = cv::imread("test.jpg");

        std::vector<Output> outputs;
        std::vector<cv::Scalar> color;

        try {
            if (!yolo.Detect(img, net, outputs) && net.empty()) {
                throw std::runtime_error("Detect failed");
            }
            else {
                /*if (MouseHit()) {
                    m = GetMouseMsg();
                    if (m.mkMButton == WM_MBUTTONDBLCLK) {
                        flag = 1 - flag;
                    }
                }*/

                GetCursorPos(&p);
                int x = p.x, y = p.y;
                long long tmp = INF;
                for (int i = 0; i < outputs.size(); ++i) {
                    if (outputs[i].id == flag && abs(x - outputs[i].box.x) * abs(x - outputs[i].box.x) + abs(y - outputs[i].box.y) * abs(y - outputs[i].box.y) < tmp) {
                        x = outputs[i].box.x;
                        y = outputs[i].box.y;
                        tmp = abs(x - outputs[i].box.x) * abs(x - outputs[i].box.x) + abs(y - outputs[i].box.y) * abs(y - outputs[i].box.y);
                    }
                }
                if (x != p.x || y != p.y)
                    SetCursorPos(x, y);
                //flag = !flag;
                //color = { cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255) };
                //yolo.drawPred(img, outputs, color);
            }
        }
        catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            return -1;
        }
        //std::this_thread::sleep_for(std::chrono::seconds(1));

    }
    listener.join();
    return 0;
}

