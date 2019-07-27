#include "hdetector.h"

//Construction
HDetector::HDetector(int minsize)
{
    face_detector.init(minsize);
	retinaface.load_param(retinaface_param_bin);
	retinaface.load_model(retinaface_bin);
}


//Destruction
HDetector::~HDetector() {
    retinaface.clear();
}

void HDetector::retina_detector(ncnn::Mat &ncnn_img, std::vector<float>& faces_info)
{
    extern float pixel_mean[3];
    extern float pixel_std[3];

    ncnn_img.substract_mean_normalize(pixel_mean, pixel_std);
	ncnn::Extractor _extractor = retinaface.create_extractor();
	_extractor.input(retinaface_param_id::BLOB_data, ncnn_img);


    std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
        int stride = _feat_stride_fpn[i];
        ac[i].Init(stride, anchor_cfg[stride], false);
    }

    std::vector<Anchor> proposals;
    proposals.clear();
    int blob_index[] = {
        retinaface_param_id::BLOB_face_rpn_cls_prob_reshape_stride32,
        retinaface_param_id::BLOB_face_rpn_bbox_pred_stride32,
        retinaface_param_id::BLOB_face_rpn_landmark_pred_stride32,

        retinaface_param_id::BLOB_face_rpn_cls_prob_reshape_stride16,
        retinaface_param_id::BLOB_face_rpn_bbox_pred_stride16,
        retinaface_param_id::BLOB_face_rpn_landmark_pred_stride16,   

        retinaface_param_id::BLOB_face_rpn_cls_prob_reshape_stride8,
        retinaface_param_id::BLOB_face_rpn_bbox_pred_stride8,
        retinaface_param_id::BLOB_face_rpn_landmark_pred_stride8, 
    };
    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
        ncnn::Mat cls;
    	ncnn::Mat reg;
    	ncnn::Mat pts;
        _extractor.extract(blob_index[i * 3 + 0], cls);
        _extractor.extract(blob_index[i * 3 + 1], reg);
        _extractor.extract(blob_index[i * 3 + 2], pts);

        ac[i].FilterAnchor(cls, reg, pts, proposals);
        for (int r = 0; r < proposals.size(); ++r) {
            proposals[r].print();
        }
    }
  
    // nms
    std::vector<Anchor> result;
    nms_cpu(proposals, nms_threshold, result);

    for(int i = 0; i < result.size(); i ++)
    {
        float xmin = result[i].finalbox.x;
        float ymin = result[i].finalbox.y;
        float xmax = result[i].finalbox.width;
        float ymax = result[i].finalbox.height;
        faces_info.push_back(result[i].score);
        faces_info.push_back(xmin);
        faces_info.push_back(ymin);
        faces_info.push_back(xmax);
        faces_info.push_back(ymax);
        for (int j = 0; j < result[i].pts.size(); ++j) {
            float x = result[i].pts[j].x;
            float y = result[i].pts[j].y;
            faces_info.push_back(x);
            faces_info.push_back(y);
        }
    }
}

void HDetector::detect_maxface(ncnn::Mat &ncnn_img, std::vector<float>& maxface_info)
{
    ncnn_img.substract_mean_normalize(mean_vals, norm_vals);
    face_detector.detectMaxFace(ncnn_img, sorted_faces);
    float score = 0.f;
    if (sorted_faces.size()) {
        Bbox max_face = sorted_faces[0];
        score = max_face.score;
        //TODO:Need to check if filter condition is right.

        float k = (float)(max_face.ppoint[6] - max_face.ppoint[5]) / (float)(max_face.ppoint[1] - max_face.ppoint[0] + 1e-6);
        float rad = std::atan(k);
        float face_angle = (float)std::abs((rad * 180 / 3.14159));
        std::cout<<"face_angle"<<face_angle<<std::endl;
        //限制角度在+-30度
        //Crooking necks is filtered out.
        if (face_angle < max_angle)
        {
            float _keypoints[10];
            memcpy(m_maxfacelds, max_face.ppoint, 10 * sizeof(float));
            memcpy(_keypoints, m_maxfacelds, 10 * sizeof(float));

            std::sort(_keypoints, _keypoints + 5,
                      std::greater<float>());    //Sort coordinate-x
            std::sort(_keypoints + 5, _keypoints + 10,
                      std::greater<float>());//Sort coordinate-y

            int top = (int) _keypoints[9];
            int bottom = (int) _keypoints[5];

            int center_x = int((_keypoints[0] + _keypoints[1] + _keypoints[2] + _keypoints[3] +
                                _keypoints[4]) / 5.0f);
            int center_y = int((_keypoints[5] + _keypoints[6] + _keypoints[7] + _keypoints[8] +
                                _keypoints[9]) / 5.0f);
            int temptopbut = (int) ((bottom - top) * std::cos(rad) / zoom_factor);
            top = center_y - temptopbut;
            bottom = center_y + temptopbut;

            int left = (int) _keypoints[4];
            int right = (int) _keypoints[0];
            int templeftright = (int) ((right - left) * std::cos(rad) / zoom_factor);
            left = center_x - templeftright;
            right = center_x + templeftright;
            std::cout<<"box-> left:"<<left<<" top:"<<top<<" right:"<<right<<" bottom:"<<bottom<<std::endl;
            refined_max_face = Rect(left, top, right - left, bottom - top);
            onetRefine(refined_max_face, ncnn_img.h, ncnn_img.w, true);

        } else {
            memset(m_maxfacelds, 0, 10 * sizeof(float));
            refined_max_face = Rect(0, 0, 0, 0);
        }
    } else {
        memset(m_maxfacelds, 0, 10 * sizeof(float));
        refined_max_face = Rect(0, 0, 0, 0);
    }
    
    maxface_info[0] = score;
    maxface_info[1] = refined_max_face.x;
    maxface_info[2] = refined_max_face.y;
    maxface_info[3] = refined_max_face.x + refined_max_face.width;
    maxface_info[4] = refined_max_face.y + refined_max_face.height;

    for (int i = 0; i < 5; i++) 
    {
        maxface_info[2 * i + 5] = m_maxfacelds[i];
        maxface_info[2 * i + 5 + 1] = m_maxfacelds[i + 5];
    }
}

inline void HDetector::onetRefine(Rect &facebox, const int &height, const int &width, bool square)
{
    if (!facebox.area())
    {
        return;
    }
    float maxSide = 0;
    float h = 0, w = 0;
    float x1 = 0, y1 = 0;

    x1 = facebox.x;
    y1 = facebox.y;

    if (square)
    {
        w = facebox.width + 1;
        h = facebox.height + 1;
        maxSide = (h > w) ? h : w;
        x1 = static_cast<float>(x1 + w * 0.5 - maxSide * 0.5);
        y1 = static_cast<float>(y1 + h * 0.5 - maxSide * 0.5);
        --maxSide;// = maxSide - 1;
        facebox.width = static_cast<int>(round(maxSide));
        facebox.height = static_cast<int>(round(maxSide));
        facebox.x = static_cast<int>(round(x1));
        facebox.y = static_cast<int>(round(y1));
    }

    //boundary check
    if (facebox.x < 0)facebox.x = 0;
    if (facebox.y < 0)facebox.y = 0;
    if (facebox.width + facebox.x > width)facebox.width = width - facebox.x - 1;
    if (facebox.height + facebox.y > height)facebox.height = height - facebox.y - 1;
}
