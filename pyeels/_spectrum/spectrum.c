#include <Python.h>
#include "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h"

double fermiDirac(double energy, double fermiLevel, double temperature){
    return 1.0/(exp((energy-fermiLevel)/temperature)+1.0);
}

static PyObject*
calculate_spectrum (PyObject *dummy, PyObject *args)
{

    PyObject *diffractionZone = NULL;
    PyObject *diffractionBins = NULL;
    PyObject *brillouinZone = NULL;
    PyObject *k_grid = NULL;
    PyObject *energy_bands = NULL;
    PyObject *wave_vectors = NULL;
    PyObject *energy_bins = NULL;
   
    double fermi_energy; // Shuld be regular float
    double temperature;

    if (!PyArg_ParseTuple(args, "OOOOOOOdd", &diffractionZone, &diffractionBins, &brillouinZone, &k_grid, &energy_bands, &wave_vectors, &energy_bins, &fermi_energy, &temperature))
        return NULL;

    npy_intp *shape;

    int nBands = PyArray_SHAPE(energy_bands)[1];
    int nWaves = PyArray_SHAPE(wave_vectors)[2];
    int k_size = PyList_Size(k_grid);

    /*
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
    */



    int EELS_E_dim = 0;
    double dE;
    double energy_offset;
    

    //######### ENERGY BINNING
    

    shape=PyArray_SHAPE(energy_bins);
    EELS_E_dim = shape[0];
    energy_offset = *(double *) PyArray_GETPTR1(energy_bins,0);
    dE= *(double *) PyArray_GETPTR1(energy_bins,1)-energy_offset;

    printf("ENERGYOFFSET:%.2f\tENERGYSTEP:%.2f\n",energy_offset,dE);





    // Create EELS-histogram structure
    npy_intp dims[4];
    dims[0] = EELS_E_dim;
    
    //dims[1] = PyLong_AsLong(PyTuple_GetItem(mesh,0));
    //dims[2] = PyLong_AsLong(PyTuple_GetItem(mesh,1));
    //dims[3] = PyLong_AsLong(PyTuple_GetItem(mesh,2));
    
    dims[1] = *(int*) PyArray_GETPTR1(diffractionBins, 0);
    dims[2] = *(int*) PyArray_GETPTR1(diffractionBins, 1);
    dims[3] = *(int*) PyArray_GETPTR1(diffractionBins, 2);



    int energyIndex = 0;
    int qIndex[3] = {0,0,0};

    double dQ[3] = {*(double*)PyArray_GETPTR1(diffractionZone,0)/(dims[1]), *(double*)PyArray_GETPTR1(diffractionZone,1)/(dims[2]), *(double*)PyArray_GETPTR1(diffractionZone,2)/(dims[3])};



    printf("MOMENT_STEP: ");
    for(int i = 0; i < 3; i++){
        printf("%.3f,  ",dQ[i]);
    }printf("\n \n");


    printf("Brillouin Zone:\n");
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            printf("%f\t", *(double *) PyArray_GETPTR2(brillouinZone, i, j));
        } 
        printf("\n");
    }printf("\n");



    double ****EELS = (double****)malloc( dims[0]*dims[1]*dims[2]*dims[3] * sizeof(double***));

    for (int e = 0; e < dims[0]; e++){  
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
    }



    // Loop over k-grid



    double *initial_energy;
    double *final_energy;
    double energyTransfer;
    double probability;
    double v_i;
    double v_f;
    double momTrans[3] = {0.0,0.0,0.0};
    double q_squared;
    double fermiValueI, fermiValue;

    printf("\n\nEf=%.3f \t",fermi_energy);
    printf("T=%.3f K ",temperature);
    temperature = temperature * 8.93103448276e-5;
    printf("(kT=%.3f eV)\n\n",temperature);
    

    int iterations = 0;
    for (int initial_k = 0; initial_k < k_size; initial_k++){
        for (int final_k = 0; final_k < k_size; final_k++){

            //printf("\nk(%i)=>k(%i):",initial_k, final_k);
            for (int initial_band = 0; initial_band < nBands-1; initial_band++){
                initial_energy = (double *) PyArray_GETPTR2(energy_bands, initial_k, initial_band);
                fermiValueI = fermiDirac(*initial_energy,fermi_energy,temperature);
                if (fermiValueI < 1e-5) continue; //Speeds up calculation
 
            
                for (int final_band = initial_band; final_band < nBands; final_band++){

                    final_energy =  (double *) PyArray_GETPTR2(energy_bands, final_k, final_band);

                    fermiValue = fermiValueI * (1.0-fermiDirac(*final_energy, fermi_energy, temperature));

                    if (fermiValue < 1e-5) continue; //Speeds up calculation


                    energyTransfer = *final_energy-*initial_energy;
                    energyIndex = (energyTransfer-energy_offset)/dE;
                    //printf("%.3f - %.3f = %.3f => [%i]\t", energyTransfer, energy_offset, (energyTransfer-energy_offset), energyIndex );
                    probability = 0;
                    
                    for (int v = 0; v < nWaves; v++){                      
                        v_i = *(double *) PyArray_GETPTR3(wave_vectors, initial_k, initial_band,v);
                        v_f = *(double *) PyArray_GETPTR3(wave_vectors, final_k, final_band,v);
                        //printf("(%f * %f)\n", v_i, v_f);,
                        probability += v_i * v_f;
                    }
                    probability = probability*probability;
                
                    //printf("\t %.2f \t %i \t|\n",energyTransfer, energyIndex);



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
                            

                            //for(int qAxis = 0; qAxis < 3; qAxis++) {
                                //momProj[qAxis] = 0;
                            for(int q = 0; q < 3; q++){
                                momTrans[q] =  (PyFloat_AsDouble(PyList_GetItem(final_k_vec,q))-PyFloat_AsDouble(PyList_GetItem(initial_k_vec,q)));
                                if(momTrans[q]>*(double*)PyArray_GETPTR1(diffractionZone,q)*0.5){
                                    //printf("%.2f =>", momTrans[q]);
                                    momTrans[q] = momTrans[q]-*(double*)PyArray_GETPTR1(diffractionZone,q);
                                    //SKJEKK HER!
                                    //printf("%.2f\n", momTrans[q]);
                                } 
                                if(momTrans[q]<*(double*)PyArray_GETPTR1(diffractionZone,q)*(-0.5)){
                                    //printf("(%.2f>%.2f) %.2f",*(double*)PyArray_GETPTR1(diffractionZone,q)*(-0.5), momTrans[q], momTrans[q]);
                                    momTrans[q] = momTrans[q]+*(double*)PyArray_GETPTR1(diffractionZone,q);
                                    //printf("+%.2f=%.2f\n",*(double*)PyArray_GETPTR1(diffractionZone,q), momTrans[q]);
                                }
                                qIndex[q] = (momTrans[q]+*(double*)PyArray_GETPTR1(diffractionZone,q)*0.5)/dQ[q];
    //                                    momProj[qAxis] += momTrans[q]* *(double *)PyArray_GETPTR2(brillouinZone, qAxis,q);
                            }
                                //qIndex[qAxis] += (momProj[q]+0.5)/dQ[q]; //0.5 +- 1e-5
                            //}

                            q_squared = (momTrans[0]*momTrans[0]+momTrans[1]*momTrans[1]+momTrans[2]*momTrans[2]);
                            //printf("[%.2f,%.2f,%.2f]=>",momTrans[0],momTrans[1],momTrans[2],q_squared);
                            //printf("[%i,%i,%i,%i]\t",energyIndex,qIndex[0],qIndex[1],qIndex[2]);
                            if (q_squared > 0) {
                                if(energyIndex < EELS_E_dim){
                                    if(qIndex[0] < dims[1] && qIndex[0] >= 0){
                                        if(qIndex[1] < dims[2] && qIndex[1] >= 0){
                                            if(qIndex[2] < dims[3] && qIndex[2] >= 0){
                                                //printf("Y(%i, %i, %i, %i), %.2f \n",energyIndex, qIndex[0], qIndex[1], qIndex[2],probability*sqrt(q_squared));
                                                EELS[energyIndex][qIndex[0]][qIndex[1]][qIndex[2]] += (probability*fermiValue/(q_squared*q_squared));
                                                iterations ++;
                                            }// else {
                                                //printf("N(%i, %i, %i, %i), %.2f \n",energyIndex, qIndex[0], qIndex[1], qIndex[2],probability*sqrt(q_squared));
                                            //    printf("Z");
                                            //}
                                        }// else {
                                                //printf("N(%i, %i, %i, %i), %.2f \n",energyIndex, qIndex[0], qIndex[1], qIndex[2],probability*sqrt(q_squared));
                                        //    printf("Y");
                                        //}
                                    }// else {
                                                //printf("N(%i, %i, %i, %i), %.2f \n",energyIndex, qIndex[0], qIndex[1], qIndex[2],probability*sqrt(q_squared));
                                    //    printf("X");
                                    //}
                                }
                            }
//                            printf("\n");
                        }                
                    }

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





    printf("\n\n Iterations: %i \n\n", iterations);
    return Py_BuildValue("O", ArgsArray);
    //return Py_BuildValue("d", temperature);
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

PyMODINIT_FUNC PyInit_cpyeels (void)
{
    import_array();
    return PyModule_Create(&myModule);
}