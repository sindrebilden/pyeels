#include <Python.h>
#include "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h"


static PyObject*
calculate_spectrum (PyObject *dummy, PyObject *args)
{
    PyObject *arg1 = NULL; // Should be a 3x1 iterable 
    PyObject *arg2 = NULL; // Should be a 3x1 iterable 
    double fermiEnergy; // Shuld be regular float

    if (!PyArg_ParseTuple(args, "OOd", &arg1, &arg2, &fermiEnergy))
        return NULL;


    PyObject *mesh = NULL; 
    PyObject *k_grid = NULL; 
    PyObject *k_list = NULL; 
    PyObject *k_vec = NULL; 
    PyObject *energy_bands=NULL;
    PyObject *wave_vectors=NULL;


    int nDim = 0;
    npy_intp *shape;

    int nBands = 0;
    int nWaves = 0;
    int k_size, sub_k_size;
    if (PyTuple_Check(arg1)){

        //#########   MESH

        if(PyTuple_Size(arg1) > 3){ 
            mesh = PyTuple_GetItem(arg1,0);

            if (PyArray_Check(mesh)){   // Convert ndarray to list
                mesh = PyArray_ToList(mesh);
            }
            if (PyList_Check(mesh)){    // Convert list to tuple
                mesh = PyList_AsTuple(mesh);
            }
            if (!PyTuple_Check(mesh)){    // Check for tuple
                PyErr_SetString(PyExc_TypeError, "Mesh must be list, array or tuple");
                return Py_None;
            }
            /*
            if (PyTuple_Size(mesh) < 3) return Py_None;

            for(int i = 0; i < 3; i++){
                if (!PyTuple_GetItem(arg1,i)){
                    return Py_None;
                } else {
                    mesh[i] = PyLong_AsLong(PyTuple_GetItem(arg1,i));
                }
            */

            //#########   K LIST

            k_grid = PyTuple_GetItem(arg1,1);
            if (PyList_Check(k_grid)){
                k_size = PyList_Size(k_grid);
                //k_list = (double ***) malloc(k_size);

                printf("k_size: %i\n", k_size);

                for(int i = 0; i<k_size; i++){

                    k_list = PyList_GetItem(k_grid,i);
                    sub_k_size = PyList_Size(k_list);

                    //printf("sub_k_size: %i\n", sub_k_size);

                    k_vec = PyList_GetItem(k_list,0);

                    if (PyList_Check(k_vec)){
                        /*
                        //k_list[i] = (double **) malloc(sub_k_size);
                        for(int j = 0; j<sub_k_size; j++){
                            k_vec = PyList_GetItem(k_list,j);

                            //k_list[i][j] = (double *) malloc(3);

                            if (PyList_Size(k_vec) < 3){
                                return Py_None;
                            } else {
                                for(int k = 0; k<3; k++){
                                    //k_list[i][j][k] = PyFloat_AsDouble(PyList_GetItem(k_vec_py,k));
                                    //printf("%.2f\t",k_list[i][j][k]);
                                    printf("%.2f\t", PyFloat_AsDouble(PyList_GetItem(k_vec,k)));

                                }
                                printf("\n");
                            }
                        }
                        */
                        
                    } else {
                        return Py_None;
                    }
                }
            } else {
                PyErr_SetString(PyExc_TypeError, "List of k-points must be a nested list");
                return Py_None;
            }

            //#########   ENERGY BANDS       
            printf("\n");
            energy_bands = PyTuple_GetItem(arg1,2);
            double *energy;
            if(PyArray_Check(energy_bands)){
                int nDim = PyArray_NDIM(energy_bands);   //number of dimensions
                npy_intp *shape;
                shape = PyArray_SHAPE(energy_bands);

                if (nDim == 2){
                    if (!(shape[0] == k_size)){
                        if(shape[1] == k_size){
                            energy_bands = PyArray_Transpose(energy_bands, (1,0));
                        } else {
                            PyErr_SetString(PyExc_TypeError, "No dimension matches the length of gridpoints.");
                            return Py_None;    
                        }

                    } else {
                        nDim = PyArray_NDIM(energy_bands);   //number of dimensions
                        shape = PyArray_SHAPE(energy_bands);
                        nBands = shape[1];
                        /*
                        for (int k = 0; k < shape[0]; k++){
                            for (int band = 0; band < shape[1]; band++){
                                energy = (double *) PyArray_GETPTR2(energy_bands, k, band);
                                printf("%.2f \t",*energy);
                            }printf("\n");
                        }
                        */
                    }
                } else {
                    PyErr_SetString(PyExc_TypeError, "Wrong number of dimensions in element 3, must be ndim=2.");
                }

            } else {
                PyErr_SetString(PyExc_TypeError, "Band structure must be numpy array.");
                return Py_None;
            }

            //#########   WAVE VECTRORS

            wave_vectors = PyTuple_GetItem(arg1,3);
            double *wave = NULL;
            if(PyArray_Check(wave_vectors)){
                nDim = PyArray_NDIM(wave_vectors);   //number of dimensions
                shape = PyArray_SHAPE(wave_vectors);

                if (nDim == 3){
                    if (!(shape[0] == k_size)){
                        if(shape[1] == k_size){
                            if(shape[0] == nBands){
                                energy_bands = PyArray_Transpose(energy_bands, (1,0,2));
                            } else if(shape[2] == nBands){
                                energy_bands = PyArray_Transpose(energy_bands, (1,2,0));
                            } else {
                                PyErr_SetString(PyExc_TypeError, "No dimension matches the number of bands.");
                                return Py_None;   
                            }
                        } else if(shape[2] == k_size){
                            if(shape[1] == nBands){
                                energy_bands = PyArray_Transpose(energy_bands, (2,1,0));
                            } else if(shape[0] == nBands){
                                energy_bands = PyArray_Transpose(energy_bands, (2,0,1));
                            } else {
                                PyErr_SetString(PyExc_TypeError, "No dimension matches the number of bands.");
                                return Py_None;   
                            }
                        } else {
                            PyErr_SetString(PyExc_TypeError, "No dimension matches the length of gridpoints.");
                            return Py_None;    
                        }

                    } else {
                        if(!(shape[1] == nBands)){
                            if(shape[2] == nBands){
                                energy_bands = PyArray_Transpose(energy_bands, (0,2,1));
                            } else {
                                PyErr_SetString(PyExc_TypeError, "No dimension matches the number of bands.");
                                return Py_None;   
                            }
                        } else {

                            shape = PyArray_SHAPE(wave_vectors); // Redefine shape after transposing
                            nWaves = shape[2];
                            /*
                            for (int k = 0; k < shape[0]; k++){
                                for (int band = 0; band < shape[1]; band++){
                                    for (int v = 0; v < shape[2]; v++){
                                        wave = (double *) PyArray_GETPTR3(wave_vectors, k, band, v); //Complex?
                                        printf("%.2f  ",*wave);
                                    }printf("\t");
                                }printf("\n");
                            }
                            */
                        }
                    }
                } else {
                    PyErr_SetString(PyExc_TypeError, "Wrong number of dimensions in element 4, must be ndim=3.");
                }

            } else {
                PyErr_SetString(PyExc_TypeError, "Wave vectors must be numpy array.");
                return Py_None;
            }

        } else {
            PyErr_SetString(PyExc_TypeError, "First argument must be tuple of size 4.");
            return Py_None;
        }
    } else {
            PyErr_SetString(PyExc_TypeError, "First argument must be tuple.");
            return Py_None;
    }


    int EELS_E_dim = 0;
    double *dE_p;
    double *energy_offset_p;
    double dE;
    double energy_offset;
    

    //######### ENERGY BINNING
    
    if (PyArray_Check(arg2)){
        if(PyArray_NDIM(arg2)==1){
            shape=PyArray_SHAPE(arg2);
            EELS_E_dim = shape[0];
            energy_offset_p = (double *) PyArray_GETPTR1(arg2,0);
            dE_p = (double *) PyArray_GETPTR1(arg2,1);
            energy_offset = *energy_offset_p;
            dE = *dE_p-energy_offset;
            printf("ENERGYOFFSET:%.2f\tENERGYSTEP:%.2f\n",energy_offset,dE);
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Second argument must be ndarray.");
        return Py_None;   
    }


    // Create EELS-histogram structure
    npy_intp dims[4];
    dims[0] = EELS_E_dim;
    dims[1] = PyLong_AsLong(PyTuple_GetItem(mesh,0));
    dims[2] = PyLong_AsLong(PyTuple_GetItem(mesh,1));
    dims[3] = PyLong_AsLong(PyTuple_GetItem(mesh,2));


    int energyIndex = 0;
    int qIndex[3] = {0,0,0};

    double dQ[3] = {1.0/(dims[1]), 1.0/(dims[2]), 1.0/(dims[3])};

    printf("MOMENT_STEP: ");
    for(int i = 0; i < 3; i++){
        printf("%.3f,  ",dQ[i]);
    }printf("\nDone!");



    double ****EELS = (double****)malloc( dims[0]*dims[1]*dims[2]*dims[3] * sizeof(double***));

    for (int e = 0; e < dims[0]; e++){  
        printf("%i,",e);
        EELS[e] = (double ***)malloc( dims[1]*dims[2]*dims[3] * sizeof(double**));
        for (int i = 0; i < dims[1]; i++){  
            EELS[e][i] = (double **)malloc( dims[2]*dims[3] * sizeof(double*));
            for (int j = 0; j < dims[2]; j++){  
                EELS[e][i][j] = (double *)malloc( dims[3] * sizeof(double*));
                for (int k = 0; k < dims[3]; k++){  
                    EELS[e][i][j][k] = 0.0;
                } 
            } 
        }
    }printf("\n");

    /*
    // Initialize zero values
    //double EELS[dims[0]][dims[1]][dims[2]][dims[3]];

    
    //double ****EELS = malloc(dims[0]*sizeof((double*)))
    printf("EELS=");
    for (int e = 0; e < dims[0]; e++){  
        printf("%i",e);
        for (int i = 0; i < dims[1]; i++){  
            for (int j = 0; j < dims[2]; j++){  
                for (int k = 0; k < dims[3]; k++){  
                    EELS[e][i][j][k] = 0.0;
                } 
            } 
        }
    }printf("\n");
    */

    // Loop over k-grid

    double *initial_energy;
    double *final_energy;
    double energyTransfer;
    double probability;
    double *v_i;
    double *v_f;
    double momTrans[3] = {0.0,0.0,0.0};
    double q_squared;

    double xi,yi,zi;

    printf("\n\nEf=%.3f \n\n",fermiEnergy);

    int iterations = 0;
    for (int initial_k = 0; initial_k < k_size; initial_k++){
       for (int final_k = 0; final_k < k_size; final_k++){
            //printf("\nk(%i)=>k(%i):",initial_k, final_k);
            for (int initial_band = 0; initial_band < nBands-1; initial_band++){
                initial_energy = (double *) PyArray_GETPTR2(energy_bands, initial_k, initial_band);
                if (*initial_energy > fermiEnergy) continue; //Speeds up calculation
 
            
                for (int final_band = initial_band; final_band < nBands; final_band++){

                    final_energy =  (double *) PyArray_GETPTR2(energy_bands, final_k, final_band);
                    if (*final_energy < fermiEnergy) continue; //Speeds up calculation

                    energyTransfer = *final_energy-*initial_energy;
                    energyIndex = (energyTransfer-energy_offset)/dE;
                    
                    probability = 0;
       
                    
                    for (int v = 0; v < nWaves; v++){                      
                        v_i = (double *) PyArray_GETPTR3(wave_vectors, initial_k, initial_band,v);
                        v_f = (double *) PyArray_GETPTR3(wave_vectors, final_k, final_band,v);

                        probability += *v_i*(*v_f);
                        /*
                        probability += PyComplex_RealAsDouble(v_i)*PyComplex_RealAsDouble(v_f)+PyComplex_ImagAsDouble(v_i)*PyComplex_ImagAsDouble(v_i);
                        */
                    }
                    probability = probability*probability;
                
                    //printf("\t %.2f \t %i \t|\t",energyTransfer, energyIndex);



                    PyObject *initial_k_list;
                    PyObject *final_k_list;
                    PyObject *initial_k_vec;
                    PyObject *final_k_vec;
                    
                    initial_k_list = PyList_GetItem(k_grid,initial_k);
                    final_k_list = PyList_GetItem(k_grid,final_k);

                    for (int initial_k_index = 0; initial_k_index < PyList_Size(initial_k_list); initial_k_index++){
                        initial_k_vec = PyList_GetItem(initial_k_list, initial_k_index);
                        for (int  final_k_index = 0; final_k_index < PyList_Size(final_k_list); final_k_index++){
                            final_k_vec = PyList_GetItem(final_k_list, final_k_index);
                            

                            for(int q = 0; q < 3; q++){
                                momTrans[q] =  (PyFloat_AsDouble(PyList_GetItem(final_k_vec,q))-PyFloat_AsDouble(PyList_GetItem(initial_k_vec,q)));
                                if(momTrans[q]>0.5){
                                    momTrans[q] = momTrans[q]-1.0;
                                } 
                                if(momTrans[q]<-0.5){
                                    momTrans[q] = momTrans[q]+1.0;

                                }
                                
                                qIndex[q] = (momTrans[q]+0.5)/dQ[q]; //0.5 +- 1e-5
                            }
                            /*
                            if(initial_k == 0 && final_k == 1){

                                xi = PyFloat_AsDouble(PyList_GetItem(final_k_vec,0));
                                yi = PyFloat_AsDouble(PyList_GetItem(final_k_vec,1));
                                zi = PyFloat_AsDouble(PyList_GetItem(final_k_vec,2));
                                //printf("(%.2f, %.2f, %.2f)=>",xi,yi,zi);

                                xi = PyFloat_AsDouble(PyList_GetItem(initial_k_vec,0));
                                yi = PyFloat_AsDouble(PyList_GetItem(initial_k_vec,1));
                                zi = PyFloat_AsDouble(PyList_GetItem(initial_k_vec,2));
                                
                                printf("(%.2f, %.2f, %.2f)\t",xi,yi,zi);

                                printf("[%i]->[%i]: E=%.3f=>[%i]<%i : \t",initial_k_index,final_k_index,energyTransfer, energyIndex, EELS_E_dim);
                                printf("(%.2f, %.2f, %.2f)=>",momTrans[0],momTrans[1],momTrans[2]);
                                printf("( %i,  %i,  %i)\n",qIndex[0],qIndex[1],qIndex[2]);
                                
                            }
                            */
                            q_squared = (momTrans[0]*momTrans[0]+momTrans[1]*momTrans[1]+momTrans[2]*momTrans[2]);
                            //printf("[%.2f,%.2f,%.2f]=%.3f\n",momTrans[0],momTrans[1],momTrans[2],q_squared);
                            //printf("[%i,%i,%i,%i]=%.3f\n",energyIndex,qIndex[0],qIndex[1],qIndex[2],q_squared);
                            if (q_squared > 0) {
                                if(energyIndex < EELS_E_dim){
                                    if(qIndex[0] < dims[1] && qIndex[0] >= 0){
                                        if(qIndex[1] < dims[2] && qIndex[1] >= 0){
                                            if(qIndex[2] < dims[3] && qIndex[2] >= 0){
                                                //printf("(%i, %i, %i), %.2f \n",qIndex[0],qIndex[1],qIndex[2],sqrt(q_squared));
                                                EELS[energyIndex][qIndex[0]][qIndex[1]][qIndex[2]] += (probability/(q_squared*q_squared));
                                            }
                                        }
                                    } else {

                                    }
                                    /* else {
                                        printf("( %i,  %i,  %i)\n",qIndex[0],qIndex[1],qIndex[2]);
                                    }*/
                                }
                            }
                        }                
                    }
                    iterations ++;




                }        
            }
        }
    }
    
    //PyArrayObject *EELS_Py = (PyArrayObject *) PyArray_SimpleNewFromData(4,dims,NPY_FLOAT64,EELS);

    // PyArray_SimpleNew allocates the memory needed for the array.
    PyArrayObject *ArgsArray = PyArray_SimpleNew(4, dims, NPY_DOUBLE);

    // The pointer to the array data is accessed using PyArray_DATA()
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





    printf("\n\n Iteration: %i \n\n", iterations);
    return Py_BuildValue("O", ArgsArray);
    //return Py_BuildValue("O&", EELS);
}


static PyMethodDef methods[] = {
    //PyName	CppName	TakingArguments	Description
    {"calculate_spectrum", calculate_spectrum, METH_VARARGS, "Takes a band structure and simulates all transitions, returns a four dimentional histogram"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef myModule = {
	PyModuleDef_HEAD_INIT,
	"_spectrum",
	"A C-extended program for calculating EEL-spectra",
	-1, //global state = -1
	methods
};

PyMODINIT_FUNC PyInit__spectrum (void)
{
    import_array();
    return PyModule_Create(&myModule);
}