/*
 * CV2BoostConverter.cpp
 *
 *  Created on: May 21, 2015
 *      Author: Gregory Kramida
 *   Copyright: 2015 Gregory Kramida
 */
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#include "np2mat.hpp"
namespace ncnnpy{

//===================   ERROR HANDLING     =========================================================
static int failmsg(const char *fmt, ...)
		{
	char str[1000];

	va_list ap;
	va_start(ap, fmt);
	vsnprintf(str, sizeof(str), fmt, ap);
	va_end(ap);

	PyErr_SetString(PyExc_TypeError, str);
	return 0;
}

ncnn::Mat fromNDArrayToMat(PyObject* o) 
{
	ncnn::Mat m;
    // !PyArray_Check(o)
	if (0)
    {
		failmsg("argument is not a numpy array");
        return m;
	} 
    else 
    {
		PyArrayObject* oarr = (PyArrayObject*) o;
        
		bool needcopy = false, needcast = false;
		int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
		int type = typenum == NPY_FLOAT;

		if (type < 0) 
        {
                failmsg("Argument data type is not supported");
				return m;
		} 
        int ndims = PyArray_NDIM(oarr);
		if (ndims != 3) {
			failmsg("Dimensionality of argument must be 3");
			return m;
		}
        npy_intp* sizes = PyArray_DIMS(oarr);
		m = ncnn::Mat((int)sizes[2], (int)sizes[1], (int)sizes[0], PyArray_DATA(oarr));
	}
	return m;
}
} //end namespace ncnnpy