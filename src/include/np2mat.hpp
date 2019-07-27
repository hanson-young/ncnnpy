/*
 * CVBoostConverter.hpp
 *
 *  Created on: Mar 20, 2014
 *      Author: Gregory Kramida
 *   Copyright: (c) 2014 Gregory Kramida
 *     License: MIT
 */

#ifndef NP2MAT_HPP_
#define NP2MAT_HPP_

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <boost/python.hpp>
#include <cstdio>
#include "net.h"

namespace pbcvt{

static int failmsg(const char *fmt, ...);

//===================   STANDALONE CONVERTER FUNCTIONS     =========================================

PyObject* fromMatToNDArray(const ncnn::Mat& m);
ncnn::Mat fromNDArrayToMat(PyObject* o);
} // end namespace pbcvt
#endif /* NP2MAT_HPP_ */
