#include <Python.h>
#include "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h"

static PyObject*
calculate_momentum_squared (PyObject *dummy, PyObject *args)
{

    PyObject *diffractionZone = NULL;
    PyObject *diffractionBins = NULL;
    PyObject *brillouinZone = NULL;
    PyObject *k_grid = NULL;
    PyObject *energy_bands = NULL;
    PyObject *wave_real = NULL;
    PyObject *wave_imag = NULL;
    PyObject *energy_bins = NULL;
   
    double fermi_energy; // Shuld be regular float
    double temperature;

    if (!PyArg_ParseTuple(args, "OOOOOOOOdd", &diffractionZone, &diffractionBins, &brillouinZone, &k_grid, &energy_bands, &wave_real, &wave_imag, &energy_bins, &fermi_energy, &temperature))
        return NULL;

    printf("Process started..\n");
    if(PyErr_CheckSignals()) goto aborted;    

    npy_intp *shape;

    int nBands = PyArray_SHAPE(energy_bands)[1];
    int nWaves = PyArray_SHAPE(wave_real)[2];
    int k_size = PyList_Size(k_grid);

    int EELS_E_dim = 0;
    double dE;
    double energy_offset;
    

    shape=PyArray_SHAPE(energy_bins);
    EELS_E_dim = shape[0];
    energy_offset = *(double *) PyArray_GETPTR1(energy_bins,0);
    dE= *(double *) PyArray_GETPTR1(energy_bins,1)-energy_offset;

    npy_intp dims[4];
    dims[0] = EELS_E_dim;
    
    dims[1] = *(int*) PyArray_GETPTR1(diffractionBins, 0);
    dims[2] = *(int*) PyArray_GETPTR1(diffractionBins, 1);
    dims[3] = *(int*) PyArray_GETPTR1(diffractionBins, 2);

    int energyIndex = 0;
    int qIndex[3] = {0,0,0};

    double dQ[3] = {*(double*)PyArray_GETPTR1(diffractionZone,0)/(dims[1]), *(double*)PyArray_GETPTR1(diffractionZone,1)/(dims[2]), *(double*)PyArray_GETPTR1(diffractionZone,2)/(dims[3])};

    double ****EELS = (double****)malloc( dims[0]*dims[1]*dims[2]*dims[3] * sizeof(double***));
    for (int e = 0; e < dims[0]; e++){  
        EELS[e] = (double ***)malloc( dims[1]*dims[2]*dims[3] * sizeof(double**));
        for (int i = 0; i < dims[1]; i++){  
            EELS[e][i] = (double **)malloc( dims[2]*dims[3] * sizeof(double*));
            for (int j = 0; j < dims[2]; j++){  
                EELS[e][i][j] = (double *)malloc( dims[3] * sizeof(double*));
                for (int k = 0; k < dims[3]; k++){  
                    EELS[e][i][j][k] = (i*dQ[0])*(i*dQ[0])+(j*dQ[1])*(j*dQ[1])+(k*dQ[2])*(k*dQ[2]);
                } 
            } 
        }
    }

    PyArrayObject *ArgsArray = PyArray_SimpleNew(4, dims, NPY_DOUBLE);
    double *p = (double *) PyArray_DATA(ArgsArray);

    for (int e = 0; e < dims[0]; e++){ 
        for (int i = 0; i < dims[1]; i++){  
            for (int j = 0; j < dims[2]; j++){  
                for (int k = 0; k < dims[3]; k++){  
                    memcpy(p, &EELS[e][i][j][k], sizeof(double));
                    p += 1;
                }
            }
        }
    }

    printf("Process calculated momentum\n", iterations);
    return Py_BuildValue("O", ArgsArray);

    aborted:
        printf("Process aborted. \n");
        return Py_None;
}


static PyMethodDef methods[] = {
    //PyName	CppName	TakingArguments	Description
    {"calculate_momentum_squared", calculate_momentum_squared, METH_VARARGS, "Calculates momentum scaling"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef myModule = {
	PyModuleDef_HEAD_INIT,
	"_momentum_squared",
	"A C-extended program for calculating momentum squared weights",
	-1, //global state = -1
	methods
};

PyMODINIT_FUNC PyInit_cpyeels (void)
{
    import_array();
    return PyModule_Create(&myModule);
}