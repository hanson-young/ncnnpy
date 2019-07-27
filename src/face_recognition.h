#ifndef RECOGNITION_H
#define RECOGNITION_H
#include"net.h"
#include "cpu.h"
#include <sys/time.h>
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
#include "np2mat.hpp"
#include "face_recog.mem.h"
#include "face_recog.id.h"
using namespace boost::python;

namespace ncnnpy {

    class FaceRecognition
    {
    private:
        int m_num_threads;
        int m_power;
        int m_debug;
        std::vector<float> m_features;
        ncnn::Net m_net;

    public:
        void set_features(ncnn::Mat output);
        object get_features();
        void inference(PyObject *np_array);
        void init(int num_threads, int power);
    };

} //end namespace ncnnpy


#endif