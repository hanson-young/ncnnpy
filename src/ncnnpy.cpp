#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include"net.h"
#include "cpu.h"
#include <sys/time.h>
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
#include "np2mat.hpp"
#include "face_recognition.h"
#include"detector.h"

namespace ncnnpy {

void FaceRecognition::set_features(ncnn::Mat output)
{
    this->m_features.clear();
    for (int j=0; j<output.w; j++)
        this->m_features.push_back(output[j]);
}

object FaceRecognition::get_features()
{
    npy_intp size = this->m_features.size();

    /* const_cast is rather horrible but we need a writable pointer
        in C++11, vec.data() will do the trick
        but you will still need to const_cast
    */

    float * data = size ? const_cast<float *>(&this->m_features[0]) 
        : static_cast<float *>(NULL); 
    if(this->m_debug)
        std::cout<<"b !"<<std::endl;
    // create a PyObject * from pointer and data 
    
    // PyObject * pyObj = PyArray_SimpleNewFromData( 1, &size, NPY_FLOAT, data );
    PyObject * pyObj  = PyArray_SimpleNewFromData(1, &size, NPY_FLOAT, data);
    if(this->m_debug)
        std::cout<<"a !"<<std::endl;
    handle<> array( pyObj );

    /* The problem of returning arr is twofold: firstly the user can modify
    the data which will betray the const-correctness 
    Secondly the lifetime of the data is managed by the C++ API and not the 
    lifetime of the numpy array whatsoever. But we have a simple solution..
    */
    return object(array); // copy the object. numpy owns the copy now.
}

void FaceRecognition::init(int num_threads = 1, int power = 0)
{
    this->m_features.clear();
    this->m_num_threads = num_threads;
    this->m_power = power;

    ncnn::set_cpu_powersave(this->m_power);
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(this->m_num_threads);
    this->m_net.load_param(recog_param_bin);
    this->m_net.load_model(recog_bin);

}

void FaceRecognition::inference(PyObject *np_array) 
{
    ncnn::Mat face_mat = ncnnpy::fromNDArrayToMat(np_array);
    if (face_mat.w != 112 || face_mat.h != 112) {
        PyErr_SetString(PyExc_TypeError,
                        "Incompatible sizes for matrix multiplication.");
        throw_error_already_set();
    }
    ncnn::Mat output;

    ncnn::Extractor _extractor = this->m_net.create_extractor();
    _extractor.input(recog_param_id::BLOB_data, face_mat);
    _extractor.extract(recog_param_id::BLOB_fc1, output);
    this->set_features(output);
}


void FaceDetector::init(int num_threads = 1, int power = 0, int minface = 80)
{
    this->m_maxface_info.resize(15);
    this->m_faces_info.clear();
    this->m_num_threads = num_threads;
    this->m_power = power;
    
    ncnn::set_cpu_powersave(this->m_power);
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(this->m_num_threads);

    this->m_hdetector = new HDetector(minface);
}

object FaceDetector::get_maxface(PyObject *np_array) 
{
    ncnn::Mat img_mat = ncnnpy::fromNDArrayToMat(np_array);
    this->m_hdetector->detect_maxface(img_mat, this->m_maxface_info);
    npy_intp size[2] = {int(this->m_maxface_info.size()/15),15};

    float * data = size ? const_cast<float *>(this->m_maxface_info.data()): static_cast<float *>(NULL); 

    PyObject * pyObj  = PyArray_SimpleNewFromData(2, size, NPY_FLOAT, data);

    handle<> array( pyObj );
    return object(array);
}

object FaceDetector::get_allface(PyObject *np_array) 
{
    ncnn::Mat img_mat = fromNDArrayToMat(np_array);
    this->m_faces_info.clear();
    this->m_hdetector->retina_detector(img_mat, this->m_faces_info);
    npy_intp size[2] = {int(this->m_faces_info.size()/15),15};

    float * data = size ? const_cast<float *>(this->m_faces_info.data()): static_cast<float *>(NULL);
    PyObject * pyObj  = PyArray_SimpleNewFromData(2, size, NPY_FLOAT, data);
    handle<> array( pyObj );
    return object(array);
}

BOOST_PYTHON_MODULE (ncnnpy) {
    import_array();
    class_<FaceDetector>("FaceDetector")
        .def("init", &FaceDetector::init)
        .def("get_maxface", &FaceDetector::get_maxface)
        .def("get_allface", &FaceDetector::get_allface)
        ;
    class_<FaceRecognition>("FaceRecognition")
        .def("init", &FaceRecognition::init)
        .def("get_features", &FaceRecognition::get_features)
        .def("inference", &FaceRecognition::inference)
        ;
}


} //end namespace ncnnpy
