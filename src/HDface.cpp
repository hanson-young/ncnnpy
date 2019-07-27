#include "HDface.h"

bool cmpareScore(Bbox lsh, Bbox rsh) {
	if (lsh.score < rsh.score)
		return true;
	else
		return false;
}

bool cmpareArea(Bbox lsh, Bbox rsh) {
	if (lsh.area < rsh.area)
		return false;
	else
		return true;
}

HDface::HDface()
{

}

void HDface::init(int minsize)
{
    this->m_minsize = minsize;
    Pnet.load_param(PNet_param_bin);
    Pnet.load_model(PNet_bin);
    Rnet.load_param(RNet_param_bin);
    Rnet.load_model(RNet_bin);
    Onet.load_param(ONet_param_bin);
    Onet.load_model(ONet_bin);
    Lnet.load_param(LNet_param_bin);
    Lnet.load_model(LNet_bin);
}

HDface::~HDface()
{
    Pnet.clear();
    Rnet.clear();
    Onet.clear();
    Lnet.clear();
}


void HDface::generateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<Bbox> &boundingBox_, float scale)
{
	const int stride = 2;
	const int cellsize = 12;

	float *p = score.channel(0);//score.data + score.cstep;//

	Bbox bbox;
	float inv_scale = 1.0f / scale;

	for (int row = 0; row < score.h; row++) {
		for (int col = 0; col < score.w; ++col) {
			if (*p > 0.6f)
			{
				//OK(need to check the col & row)
				bbox.score = *p;
				bbox.x1 = round((stride * col + 1) * inv_scale);
				bbox.y1 = round((stride * row + 1) * inv_scale);
				bbox.x2 = round((stride * col + 1 + cellsize) * inv_scale);
				bbox.y2 = round((stride * row + 1 + cellsize) * inv_scale);


				bbox.area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1);

				const int index = row * score.w + col;
				for (int channel = 0; channel < 4; channel++) {
					bbox.regreCoord[channel] = location.channel(channel)[index];
				}
				boundingBox_.push_back(bbox);
			}
			p++;
		}
	}
}

void HDface::nms(std::vector<Bbox> &boundingBox_, const float overlap_threshold, std::string modelname) {
	if (boundingBox_.empty()) {
		return;
	}
	std::sort(boundingBox_.begin(), boundingBox_.end(), cmpareScore);
	float IOU = 0;
	float maxX = 0;
	float maxY = 0;
	float minX = 0;
	float minY = 0;
	std::vector<int> vPick;
	int nPick = 0;
	std::multimap<float, int> vScores;
	const int num_boxes = boundingBox_.size();
	vPick.resize(num_boxes);
	for (int i = 0; i < num_boxes; ++i) {
		vScores.insert(std::pair<float, int>(boundingBox_[i].score, i));
	}
	while (vScores.size() > 0) {
		int last = vScores.rbegin()->second;
		vPick[nPick] = last;
		nPick += 1;
		for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();) {
			int it_idx = it->second;
			maxX = std::max(boundingBox_.at(it_idx).x1, boundingBox_.at(last).x1);
			maxY = std::max(boundingBox_.at(it_idx).y1, boundingBox_.at(last).y1);
			minX = std::min(boundingBox_.at(it_idx).x2, boundingBox_.at(last).x2);
			minY = std::min(boundingBox_.at(it_idx).y2, boundingBox_.at(last).y2);
			//maxX1 and maxY1 reuse
			maxX = ((minX - maxX + 1)>0) ? (minX - maxX + 1) : 0;
			maxY = ((minY - maxY + 1)>0) ? (minY - maxY + 1) : 0;
			//IOU reuse for the area of two bbox
			IOU = maxX * maxY;
			if (!modelname.compare("Union"))
				IOU = IOU / (boundingBox_.at(it_idx).area + boundingBox_.at(last).area - IOU);
			else if (!modelname.compare("Min")) {
				IOU = IOU / ((boundingBox_.at(it_idx).area < boundingBox_.at(last).area) ? boundingBox_.at(it_idx).area : boundingBox_.at(last).area);
			}
			if (IOU > overlap_threshold) {
				it = vScores.erase(it);
			}
			else {
				it++;
			}
		}
	}

	vPick.resize(nPick);
	std::vector<Bbox> tmp_;
	tmp_.resize(nPick);
	for (int i = 0; i < nPick; i++) {
		tmp_[i] = boundingBox_[vPick[i]];
	}
	boundingBox_ = tmp_;
}

void HDface::inference_pnet(float scale) {
    //first stage
    int hs = (int)(img_h * scale);
    int ws = (int)(img_w * scale);
    ncnn::Mat in;
    resize_bilinear(img, in, ws, hs);
    ncnn::Extractor ex = Pnet.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(this->m_num_threads);
    ex.input(PNet_param_id::BLOB_data, in);
    ncnn::Mat score_, location_, PReLU_1;

    ex.extract(PNet_param_id::BLOB_PReLU_1, PReLU_1);
    ex.extract(PNet_param_id::BLOB_Sigmoid_1, score_);
    ex.extract(PNet_param_id::BLOB_ConvNd_5, location_);

    std::vector<Bbox> boundingBox_;

    generateBbox(score_, location_, boundingBox_, scale);
    nms(boundingBox_, nms_threshold[0]);

    firstBbox_.insert(firstBbox_.end(), boundingBox_.begin(), boundingBox_.end());
    boundingBox_.clear();
}

void HDface::inference_rnet() {
    secondBbox_.clear();
    int count = 0;
    for (std::vector<Bbox>::iterator it = firstBbox_.begin(); it != firstBbox_.end(); it++) {
        ncnn::Mat tempIm;
        copy_cut_border(img, tempIm, (*it).y1, img_h - (*it).y2, (*it).x1, img_w - (*it).x2);
        ncnn::Mat in;
        resize_bilinear(tempIm, in, 24, 24);
        ncnn::Extractor ex = Rnet.create_extractor();
        ex.set_num_threads(this->m_num_threads);
        ex.set_light_mode(true);
        ncnn::Mat score, bbox;
        ex.input(RNet_param_id::BLOB_data, in);
        ncnn::Mat MaxPool2d_1, MaxPool2d_1_ret, MaxPool2d_2, MaxPool2d_2_ret;
        ex.extract(RNet_param_id::BLOB_MaxPool2d_1, MaxPool2d_1);
        copy_cut_border(MaxPool2d_1, MaxPool2d_1_ret, 0, 1, 0, 1);
        ex.input(RNet_param_id::BLOB_MaxPool2d_1, MaxPool2d_1_ret);
        ex.extract(RNet_param_id::BLOB_MaxPool2d_2, MaxPool2d_2);
        copy_cut_border(MaxPool2d_2, MaxPool2d_2_ret, 0, 1, 0, 1);
        ex.input(RNet_param_id::BLOB_MaxPool2d_2, MaxPool2d_2_ret);
        ex.extract(RNet_param_id::BLOB_Sigmoid_1, score);

        float *p = score.channel(0);

        if (*p > threshold[1]) {
            ex.extract(RNet_param_id::BLOB_Addmm_3, bbox);
            for (int channel = 0; channel < 4; channel++) {
                it->regreCoord[channel] = (float)bbox.channel(0)[channel];//*(bbox.data+channel*bbox.cstep);
            }
            it->area = (it->x2 - it->x1) * (it->y2 - it->y1);
            it->score = *p;//*(score.data+score.cstep);
            secondBbox_.push_back(*it);
        }
    }
}

void HDface::inference_lnet(std::vector<Bbox> &boundingBox_)
{
    ncnn::Mat croppedimg, in;
    copy_cut_border(img, croppedimg, boundingBox_[0].y1, img_h - boundingBox_[0].y2, boundingBox_[0].x1, img_w - boundingBox_[0].x2);

    resize_bilinear(croppedimg, in, 48, 48);
    ncnn::Mat score, bbox, keyPoint, MaxPool2d_3, MaxPool2d_3_ret;
    ncnn::Extractor lex = Lnet.create_extractor();
    lex.input(ONet_param_id::BLOB_data, in);
    lex.extract(ONet_param_id::BLOB_MaxPool2d_3, MaxPool2d_3);
    copy_cut_border(MaxPool2d_3, MaxPool2d_3_ret, 0, 1, 0, 1);
    lex.input(ONet_param_id::BLOB_MaxPool2d_3, MaxPool2d_3_ret);
    lex.extract(ONet_param_id::BLOB_Addmm_4, keyPoint);

    for (int num = 0; num < 5; num++) {
        (boundingBox_[0].ppoint)[num] = boundingBox_[0].x1 + (boundingBox_[0].x2 - boundingBox_[0].x1) * keyPoint.channel(0)[2 * num];
        (boundingBox_[0].ppoint)[num + 5] = boundingBox_[0].y1 + (boundingBox_[0].y2 - boundingBox_[0].y1) * keyPoint.channel(0)[2 * num + 1];
    }

}

void HDface::inference_onet() {
    thirdBbox_.clear();
    for (std::vector<Bbox>::iterator it = secondBbox_.begin(); it != secondBbox_.end(); it++) {
        ncnn::Mat tempIm;
        copy_cut_border(img, tempIm, (*it).y1, img_h - (*it).y2, (*it).x1, img_w - (*it).x2);
        ncnn::Mat in;
        resize_bilinear(tempIm, in, 48, 48);
        ncnn::Extractor ex = Onet.create_extractor();
        ex.set_num_threads(this->m_num_threads);
        ex.set_light_mode(true);

        ex.input(ONet_param_id::BLOB_data, in);
        ncnn::Mat score, bbox, keyPoint, MaxPool2d_3, MaxPool2d_3_ret;
        ex.extract(ONet_param_id::BLOB_MaxPool2d_3, MaxPool2d_3);
        copy_cut_border(MaxPool2d_3, MaxPool2d_3_ret, 0, 1, 0, 1);
        ex.input(ONet_param_id::BLOB_MaxPool2d_3, MaxPool2d_3_ret);

        ex.extract(ONet_param_id::BLOB_Sigmoid_1, score);

        float *p = score.channel(0);
        if (*p > threshold[2]) {

            ex.extract(ONet_param_id::BLOB_Addmm_3, bbox);

            for (int channel = 0; channel < 4; channel++) {
                it->regreCoord[channel] = (float)bbox.channel(0)[channel];
            }
            it->area = (it->x2 - it->x1) * (it->y2 - it->y1);
            it->score = *p;

            thirdBbox_.push_back(*it);
        }
    }
}


void HDface::refine(std::vector<Bbox> &vecBbox, const int &height, const int &width, bool square) {
    if (vecBbox.empty()) {
        std::cout << "Bbox is empty!!" << std::endl;
        return;
    }
    float bbw = 0, bbh = 0, maxSide = 0;
    float h = 0, w = 0;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    for (std::vector<Bbox>::iterator it = vecBbox.begin(); it != vecBbox.end(); it++) {
        bbw = (*it).x2 - (*it).x1 + 1;
        bbh = (*it).y2 - (*it).y1 + 1;
        x1 = (*it).x1 + (*it).regreCoord[0] * bbw;
        y1 = (*it).y1 + (*it).regreCoord[1] * bbh;
        x2 = (*it).x2 + (*it).regreCoord[2] * bbw;
        y2 = (*it).y2 + (*it).regreCoord[3] * bbh;


        if (square) {
            w = x2 - x1 + 1;
            h = y2 - y1 + 1;
            maxSide = (h > w) ? h : w;
            x1 = x1 + w * 0.5 - maxSide * 0.5;
            y1 = y1 + h * 0.5 - maxSide * 0.5;
            (*it).x2 = round(x1 + maxSide - 1);
            (*it).y2 = round(y1 + maxSide - 1);
            (*it).x1 = round(x1);
            (*it).y1 = round(y1);
        }

        //boundary check
        if ((*it).x1 < 0)(*it).x1 = 0;
        if ((*it).y1 < 0)(*it).y1 = 0;
        if ((*it).x2 > width)(*it).x2 = width - 1;
        if ((*it).y2 > height)(*it).y2 = height - 1;

        it->area = (it->x2 - it->x1) * (it->y2 - it->y1);
    }
}

void HDface::nmsTwoBoxs(std::vector<Bbox> &boundingBox_, std::vector<Bbox> &previousBox_,
                            const float overlap_threshold, std::string modelname) {
    if (boundingBox_.empty()) {
        return;
    }
    sort(boundingBox_.begin(), boundingBox_.end(), cmpareScore);
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    //std::cout << boundingBox_.size() << " ";
    for (std::vector<Bbox>::iterator ity = previousBox_.begin(); ity != previousBox_.end(); ity++) {
        for (std::vector<Bbox>::iterator itx = boundingBox_.begin(); itx != boundingBox_.end();) {
            int i = itx - boundingBox_.begin();
            int j = ity - previousBox_.begin();
            maxX = std::max(boundingBox_.at(i).x1, previousBox_.at(j).x1);
            maxY = std::max(boundingBox_.at(i).y1, previousBox_.at(j).y1);
            minX = std::min(boundingBox_.at(i).x2, previousBox_.at(j).x2);
            minY = std::min(boundingBox_.at(i).y2, previousBox_.at(j).y2);
            //maxX1 and maxY1 reuse
            maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
            maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
            //IOU reuse for the area of two bbox
            IOU = maxX * maxY;
            if (!modelname.compare("Union"))
                IOU = IOU / (boundingBox_.at(i).area + previousBox_.at(j).area - IOU);
            else if (!modelname.compare("Min")) {
                IOU = IOU /
                      ((boundingBox_.at(i).area < previousBox_.at(j).area) ? boundingBox_.at(i).area
                                                                           : previousBox_.at(
                                      j).area);
            }
            if (IOU > overlap_threshold && boundingBox_.at(i).score > previousBox_.at(j).score) {
                //if (IOU > overlap_threshold) {
                itx = boundingBox_.erase(itx);
            }
            else {
                itx++;
            }
        }
    }
}

void HDface::extractMaxFace(std::vector<Bbox> &boundingBox_) {
    if (boundingBox_.empty()) {
        return;
    }
    sort(boundingBox_.begin(), boundingBox_.end(), cmpareArea);
    for (std::vector<Bbox>::iterator itx = boundingBox_.begin() + 1; itx != boundingBox_.end();) {
        itx = boundingBox_.erase(itx);
    }
}

void HDface::detectMaxFace(ncnn::Mat &img_, std::vector<Bbox> &finalBbox) {
    firstPreviousBbox_.clear();
    secondPreviousBbox_.clear();
    thirdPrevioussBbox_.clear();
    firstBbox_.clear();
    secondBbox_.clear();
    thirdBbox_.clear();
	finalBbox.clear();
    //norm
    img = img_;
    img_w = img.w;
    img_h = img.h;


    //pyramid size
    float min_edge = img_w < img_h ? img_w : img_h;
    float current_scale = (float)MIN_DET_SIZE / m_minsize;
    min_edge *= current_scale;
    float scale_factor = this->m_pre_facetor;
    std::vector<float> scales_;

    while (min_edge > MIN_DET_SIZE)
    {
        scales_.push_back(current_scale);
        min_edge *= scale_factor;
        current_scale *= scale_factor;
    }
    sort(scales_.begin(), scales_.end());

    //Change the sampling process.
    for (size_t i = 0; i < scales_.size(); i++) {
        //first stage
        inference_pnet(scales_[i]);
        nms(firstBbox_, nms_threshold[0]);
        nmsTwoBoxs(firstBbox_, firstPreviousBbox_, nms_threshold[0]);
        if (firstBbox_.size() < 1) {
            firstBbox_.clear();
            continue;
        }
        firstPreviousBbox_.insert(firstPreviousBbox_.end(), firstBbox_.begin(),
                                  firstBbox_.end());

        refine(firstBbox_, img_h, img_w, true);

        //second stage
        inference_rnet();
        nms(secondBbox_, nms_threshold[1]);

        nmsTwoBoxs(secondBbox_, secondPreviousBbox_, nms_threshold[0]);
        secondPreviousBbox_.insert(secondPreviousBbox_.end(), secondBbox_.begin(),
                                   secondBbox_.end());
        if (secondBbox_.size() < 1) {
            firstBbox_.clear();
            secondBbox_.clear();
            continue;
        }
        refine(secondBbox_, img_h, img_w, true);

        //third stage
        inference_onet();
        if (thirdBbox_.size() < 1) {
            firstBbox_.clear();
            secondBbox_.clear();
            thirdBbox_.clear();
            continue;
        }

        refine(thirdBbox_, img_h, img_w, false);
        nms(thirdBbox_, nms_threshold[2], "Min");

        if (thirdBbox_.size() > 0) {
            extractMaxFace(thirdBbox_);

            inference_lnet(thirdBbox_);
            finalBbox = thirdBbox_;//if largest face size is similar,
            break;
        }
    }
    img.release();
}
