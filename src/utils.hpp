#pragma once

#include "net.h"
#include <sys/time.h>
using namespace std;
struct Bbox {
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    ncnn::Mat keypoints;
    float ppoint[10];
    float regreCoord[4];
};

class Rect
{
public:
    float x;
    float y;
    float width;
    float height;
public:
	Rect(){};

	Rect(int x, int y, int width, int height)
    {
        this->x = x;
        this->y = y;
        this->width = width;
        this->height = height;
    }
	int area() const
    {
        int area = height * width;
        return area;
    }
};

class Point
{
public:
    float x;
    float y;
public:
	Point(){};

	Point(int x, int y)
    {
        this->x = x;
        this->y = y;
    }
};