#ifndef MTCNN_HDFACE
#define MTCNN_HDFACE

#include "net.h"
#include <vector>
#include <map>
#include "cmath"
#include <iostream>
#include <algorithm>
#include "utils.hpp"

#include "anchor_generator.h"
#include "config.h"
#include "tools.h"

#include "PNet.id.h"
#include "RNet.id.h"
#include "ONet.id.h"
#include "LNet.id.h"

#include "PNet.mem.h"
#include "RNet.mem.h"
#include "ONet.mem.h"
#include "LNet.mem.h"

#include "retinaface.id.h"
#include "retinaface.mem.h"
// double get_current_time()
// {
//     struct timeval tv;
//     gettimeofday(&tv, NULL);
//     return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
// }

class HDface
{
public:
	HDface();
	void init(int minsize);
	~HDface();
    void detectMaxFace(ncnn::Mat &img_, std::vector<Bbox> &finalBbox);

private:
	void generateBbox(ncnn::Mat score, ncnn::Mat location,std::vector<Bbox> &boundingBox_,float scale);
	void nms(std::vector<Bbox> &boundingBox_, const float overlap_threshold, std::string modelname = "Union");
	void nmsTwoBoxs(std::vector<Bbox> &boundingBox_, std::vector<Bbox> &previousBox_, const float overlap_threshold, std::string modelname = "Union");
    void extractMaxFace(std::vector<Bbox> &boundingBox_);
    void refine(std::vector<Bbox> &vecBbox, const int &height, const int &width, bool square);

    void inference_pnet(float scale);
    void inference_rnet();
    void inference_onet();
	void inference_lnet(std::vector<Bbox> &boundingBox_);

	ncnn::Net Pnet, Rnet, Onet, Lnet;
	ncnn::Mat img;

	std::vector<Bbox> firstBbox_, secondBbox_, thirdBbox_;
    std::vector<Bbox> firstPreviousBbox_, secondPreviousBbox_, thirdPrevioussBbox_;
    int img_w, img_h;

	const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
	const float norm_vals[3] = { (float)1 / 128 ,(float)1 / 128 ,(float)1 / 128 };

	const float threshold[3] = { 0.5f, 0.5f, 0.7f };
	const float nms_threshold[4] = { 0.5f, 0.7f, 0.7f, 0.5f };
	int m_minsize = 100;
	const int MIN_DET_SIZE = 12;
	const float m_pre_facetor = 0.5f;
    int m_num_threads = 1;
};

#endif