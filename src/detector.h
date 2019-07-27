#ifndef DETECTOR_H
#define DETECTOR_H
#include"net.h"
#include "cpu.h"
#include <sys/time.h>
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
#include "np2mat.hpp"
#include "hdetector.h"
using namespace boost::python;

namespace ncnnpy {
    class FaceDetector
    {
    private:

        HDetector *m_hdetector;
        int m_num_threads;
        int m_power;
        std::vector<float> m_faces_info;
        std::vector<float> m_maxface_info;
        ncnn::Net m_net;

    public:
        object get_maxface(PyObject *np_array);
        object get_allface(PyObject *np_array) ;

        void init(int num_threads, int power, int minface );
    };

} //end namespace ncnnpy


#endif