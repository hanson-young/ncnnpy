#ifndef HDETECTOR_H
#define HDETECTOR_H

#include "HDface.h"
#include "net.h"



class HDetector {
public:
    HDetector(int minsize);
    ~HDetector();
	void detect_maxface(ncnn::Mat &ncnn_img, std::vector<float>& maxface_info);
	void retina_detector(ncnn::Mat &ncnn_img, std::vector<float>& maxface_info);
private:
    void onetRefine(Rect &facebox, const int &height, const int &width, bool square);
	HDface face_detector;
    ncnn::Net retinaface;
	std::vector<Bbox> detected_faces;
	std::vector<Bbox> sorted_faces;

	float m_keyPoints[10];
	float m_maxfacelds[10];
    Rect refined_max_face;
	const float zoom_factor = 0.8f;
	const float max_angle = 45.f;
    const float mean_vals[3] = {127.5, 127.5, 127.5};
    const float norm_vals[3] = {0.0078125, 0.0078125, 0.0078125};
};

#endif