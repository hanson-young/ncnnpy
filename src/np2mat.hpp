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
#include <iostream>
#include "net.h"

namespace ncnnpy{

static int failmsg(const char *fmt, ...);

ncnn::Mat fromNDArrayToMat(PyObject* o);
} // end namespace ncnnpy
#endif /* NP2MAT_HPP_ */
