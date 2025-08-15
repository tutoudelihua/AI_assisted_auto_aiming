#pragma once
#include <Windows.h>
#include <opencv2/opencv.hpp>

class Screenshot
{
public:
    Screenshot();
    static double getZoom();
    cv::Mat getScreenshot();
    cv::Mat getScreenshot(int x, int y, int width, int height);

private:
    int m_width;
    int m_height;
    HDC m_screenDC;
    HDC m_compatibleDC;
    HBITMAP m_hBitmap;
    LPVOID m_screenshotData = nullptr;
};
