#include <Python.h>
#include <numpy/arrayobject.h>

double fermiDirac(double energy, double fermiLevel, double temperature){
    return 1.0/(exp((energy-fermiLevel)/temperature)+1.0);
}

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

    if (!PyArg_ParseTuple(args, "OOO", &diffractionBins, &brillouinZone, &energy_bins))
        return NULL;

    printf("Process started..\n");
    if(PyErr_CheckSignals()) goto aborted;    

    npy_intp *shape;
    shape=PyArray_SHAPE(energy_bins);

    npy_intp dims[4];
    dims[0] = shape[0];
    dims[1] = *(int*) PyArray_GETPTR1(diffractionBins, 0);
    dims[2] = *(int*) PyArray_GETPTR1(diffractionBins, 1);
    dims[3] = *(int*) PyArray_GETPTR1(diffractionBins, 2);

    int energyIndex = 0;
    int qIndex[3] = {0,0,0};


    double dQ[3] = {1.0/(dims[1]-1), 1.0/(dims[2]-1), 1.0/(dims[3]-1)};

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

    double temp, sub_temp;
    for (int i = 0; i < dims[1]; i++){  
        for (int j = 0; j < dims[2]; j++){  
            for (int k = 0; k < dims[3]; k++){
                temp = 0.0;
                //printf("%i, %i, %i \n", (i-(dims[1]-1)/2), (j-(dims[2]-1)/2), (k-(dims[3]-1)/2));
                
                // Loop through x,y,z and 
                for (int q = 0; q < 3; q++) {
                    sub_temp  = (i-(dims[1]-1)/2)*dQ[0]* *(double*) PyArray_GETPTR2(brillouinZone, 0, q); //a
                    sub_temp += (j-(dims[2]-1)/2)*dQ[1]* *(double*) PyArray_GETPTR2(brillouinZone, 1, q); //b
                    sub_temp += (k-(dims[3]-1)/2)*dQ[2]* *(double*) PyArray_GETPTR2(brillouinZone, 2, q); //c 
                    
//                    printf("%f, ",sub_temp);

                    temp += sub_temp*sub_temp; //x^2
                }
                printf("\n");
                for (int e = 0; e < dims[0]; e++){  
                    EELS[e][i][j][k] = temp;

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

    // Free memory from c-object
    for (int e = 0; e < dims[0]; e++){  
        for (int i = 0; i < dims[1]; i++){  
            for (int j = 0; j < dims[2]; j++){  
                free(EELS[e][i][j]);
            } 
            free(EELS[e][i]);
        }
        free(EELS[e]);
    }
    free(EELS);

    printf("Process calculated momentum\n");
    return Py_BuildValue("O", ArgsArray);

    aborted:
        printf("Process aborted. \n");
        return Py_None;
}

static PyObject*
calculate_spectrum (PyObject *dummy, PyObject *args)
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

    if (!PyArg_ParseTuple(args, "OOOOOOOdd", &diffractionBins, &brillouinZone, &k_grid, &energy_bands, &wave_real, &wave_imag, &energy_bins, &fermi_energy, &temperature))
        return NULL;

    printf("Process started..\n");
    if(PyErr_CheckSignals()) goto aborted;    

    npy_intp *shape;

    int nBands = PyArray_SHAPE(energy_bands)[1];
    int nWaves = PyArray_SHAPE(wave_real)[2];
    int k_size = PyArray_SHAPE(k_grid)[0];

    int EELS_E_dim = 0;
    double dE;
    double energy_offset;
    

    //######### ENERGY BINNING
    shape=PyArray_SHAPE(energy_bins);
    EELS_E_dim = shape[0];
    energy_offset = *(double *) PyArray_GETPTR1(energy_bins,0);
    dE= *(double *) PyArray_GETPTR1(energy_bins,1)-energy_offset;

    // Create EELS-histogram structure
    npy_intp dims[4];
    dims[0] = EELS_E_dim;
    
    dims[1] = *(int*) PyArray_GETPTR1(diffractionBins, 0);
    dims[2] = *(int*) PyArray_GETPTR1(diffractionBins, 1);
    dims[3] = *(int*) PyArray_GETPTR1(diffractionBins, 2);

    int energyIndex = 0;

    int qIndex[3] = {0,0,0};
    double dQ[3] = {1.0/(dims[1]), 1.0/(dims[2]), 1.0/(dims[3])};

    // Initialize EELS
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

    double *initial_energy;
    double *final_energy;
    double energyTransfer;
    double probability;
    double p_real;
    double p_imag;
    double v_real_i;
    double v_real_f;
    double v_imag_i;
    double v_imag_f;
    double momTrans[3] = {0.0,0.0,0.0};
    double q_squared;
    double fermiValueI, fermiValue;
    double k_squared;
    double k_temp;

    //Convert temperature to thermal energy
    temperature = temperature * 8.93103448276e-5;


    long unsigned int iterations = 0;
    for (int initial_k = 0; initial_k < k_size; initial_k++){
        for (int final_k = 0; final_k < k_size; final_k++){
            initial_energy = (double *) PyArray_GETPTR2(energy_bands, initial_k, 0);
            fermiValueI = fermiDirac(*initial_energy,fermi_energy,temperature);
            if (fermiValueI < 1e-10) continue; //Speeds up calculation

            final_energy =  (double *) PyArray_GETPTR2(energy_bands, final_k, 1);
            fermiValue = fermiValueI * (1.0-fermiDirac(*final_energy, fermi_energy, temperature));

            if (fermiValue < 1e-10) continue; //Speeds up calculation

            energyTransfer = *final_energy-*initial_energy;
            energyIndex = (energyTransfer-energy_offset)/dE;

            p_real = 0;
            p_imag = 0;
            
            for (int v = 0; v < nWaves; v++){      
                v_real_i = *(double *) PyArray_GETPTR3(wave_real, initial_k, 0,v);
                v_real_f = *(double *) PyArray_GETPTR3(wave_real, final_k, 1,v);
                
                v_imag_i = *(double *) PyArray_GETPTR3(wave_imag, initial_k, 0,v);
                v_imag_f = *(double *) PyArray_GETPTR3(wave_imag, final_k, 1,v);

                p_real += (v_real_f*v_real_i+v_imag_f*v_imag_i);
                p_imag += (v_real_f*v_imag_i-v_imag_f*v_real_i);
            }
            probability = (p_real*p_real+p_imag*p_imag);
                            
            if(PyErr_CheckSignals()) goto aborted;

            for(int q = 0; q < 3; q++){
                momTrans[q] =  (*(double *) PyArray_GETPTR2(k_grid, final_k, q)-*(double *) PyArray_GETPTR2(k_grid, initial_k, q));
                if(momTrans[q]>0.5){
                    momTrans[q] = momTrans[q]-1.0;
                } 
                if(momTrans[q]<=-0.5){
                    momTrans[q] = momTrans[q]+1.0;
                }
                qIndex[q] = (momTrans[q]+0.5)/dQ[q];
            }

            if(energyIndex < EELS_E_dim){
                if(qIndex[0] < dims[1] && qIndex[0] >= 0){
                    if(qIndex[1] < dims[2] && qIndex[1] >= 0){
                        if(qIndex[2] < dims[3] && qIndex[2] >= 0){
                            /*
                            Write:
                            for (int e = 0; e<dims[0]; e++){
                                Some smearing (e-energyIndex) or gaussian
                                EELS[e][qIndex[0]][qIndex[1]][qIndex[2]] += (fermiValue)/(e-energyIndex);
                                See Gajdos or Adler
                            }
                            */


                            /* JUST A TEST
                            if (initial_k == final_k){

                                k_squared = 0;
                                for(int q = 0; q < 3; q++){
                                    k_temp = (*(double *) PyArray_GETPTR2(k_grid, initial_k, q));
                                    k_squared += k_temp*k_temp;
                                }

                                EELS[energyIndex][qIndex[0]][qIndex[1]][qIndex[2]] += (probability*fermiValue*k_squared);
                            } else {
                                EELS[energyIndex][qIndex[0]][qIndex[1]][qIndex[2]] += (probability*fermiValue);
                            }
                            */
                            EELS[energyIndex][qIndex[0]][qIndex[1]][qIndex[2]] += (probability*fermiValue);

                            iterations ++;
                            //printf("%i\n", iterations);
                        }
                    }
                }
            }        
        }
    }

    // Transfer C data to a Python object
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

    // Free memory from c-object
    for (int e = 0; e < dims[0]; e++){  
        for (int i = 0; i < dims[1]; i++){  
            for (int j = 0; j < dims[2]; j++){  
                free(EELS[e][i][j]);
            } 
            free(EELS[e][i]);
        }
        free(EELS[e]);
    }
    free(EELS);

    printf("Process ended with %lu sucessful transitions..\n", iterations);

    PyErr_Clear();
    return Py_BuildValue("O", ArgsArray);

    aborted:
        // Free memory from c-object
        for (int e = 0; e < dims[0]; e++){  
            for (int i = 0; i < dims[1]; i++){  
                for (int j = 0; j < dims[2]; j++){  
                    free(EELS[e][i][j]);
                } 
                free(EELS[e][i]);
            }
            free(EELS[e]);
        }
        free(EELS);

        printf("Process aborted. \n");
        return Py_None;
}


static PyObject*
scan_spectrum (PyObject *dummy, PyObject *args)
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

    if (!PyArg_ParseTuple(args, "OOOOOOOdd", &diffractionBins, &brillouinZone, &k_grid, &energy_bands, &wave_real, &wave_imag, &energy_bins, &fermi_energy, &temperature))
        return NULL;

    printf("Process started..\n");
    if(PyErr_CheckSignals()) goto aborted;    

    npy_intp *shape;

    int nBands = PyArray_SHAPE(energy_bands)[1];
    int nWaves = PyArray_SHAPE(wave_real)[2];
    int k_size = PyArray_SHAPE(k_grid)[0];

    int EELS_E_dim = 0;
    double dE;
    double energy_offset;
    

    //######### ENERGY BINNING
    shape=PyArray_SHAPE(energy_bins);
    EELS_E_dim = shape[0];
    energy_offset = *(double *) PyArray_GETPTR1(energy_bins,0);
    dE= *(double *) PyArray_GETPTR1(energy_bins,1)-energy_offset;

    // Create EELS-histogram structure
    npy_intp dims[4];
    dims[0] = EELS_E_dim;
    
    dims[1] = *(int*) PyArray_GETPTR1(diffractionBins, 0);
    dims[2] = *(int*) PyArray_GETPTR1(diffractionBins, 1);
    dims[3] = *(int*) PyArray_GETPTR1(diffractionBins, 2);

    int energyIndex = 0;

    int qIndex[3] = {0,0,0};
    double dQ[3] = {1.0/(dims[1]), 1.0/(dims[2]), 1.0/(dims[3])};

    // Initialize EELS
    double *EELS = (double*)malloc( dims[0] * sizeof(double));
    for (int e = 0; e < dims[0]; e++){  
        EELS[e] = 0.0;
    }

    double *initial_energy;
    double *final_energy;
    double energyTransfer;
    double probability;
    double p_real;
    double p_imag;
    double v_real_i;
    double v_real_f;
    double v_imag_i;
    double v_imag_f;
    double momTrans[3] = {0.0,0.0,0.0};
    double q_squared;
    double fermiValueI, fermiValue;

    temperature = temperature * 8.93103448276e-5;


    long unsigned int iterations = 0;
    int final_k = 0;
    for (int initial_k = 0; initial_k < k_size; initial_k++){
        final_k = initial_k;
        initial_energy = (double *) PyArray_GETPTR2(energy_bands, initial_k, 0);
        fermiValueI = fermiDirac(*initial_energy,fermi_energy,temperature);
        if (fermiValueI < 1e-10) continue; //Speeds up calculation

        final_energy =  (double *) PyArray_GETPTR2(energy_bands, final_k, 1);
        fermiValue = fermiValueI * (1.0-fermiDirac(*final_energy, fermi_energy, temperature));

        if (fermiValue < 1e-10) continue; //Speeds up calculation

        energyTransfer = *final_energy-*initial_energy;
        energyIndex = (energyTransfer-energy_offset)/dE;

        if(PyErr_CheckSignals()) goto aborted;

        for(int q = 0; q < 3; q++){
            momTrans[q] =  (*(double *) PyArray_GETPTR2(k_grid, final_k, q)-*(double *) PyArray_GETPTR2(k_grid, initial_k, q));
            if(momTrans[q]>0.5){
                momTrans[q] = momTrans[q]-1.0;
            } 
            if(momTrans[q]<=-0.5){
                momTrans[q] = momTrans[q]+1.0;
            }
            qIndex[q] = (momTrans[q]+0.5)/dQ[q];
        }

        if(energyIndex < EELS_E_dim){
            /*
            Write:
            for (int e = 0; e<EELS_E_dim; e++){
                Some smearing (e-energyIndex) or gaussian
                EELS[e] += (fermiValue)/(e-energyIndex); //how to treat imaginary part?
                See Gajdos or Adler
            }
            */
            EELS[energyIndex] += (fermiValue);
            iterations ++;
            //printf("%i\n", iterations);
        }        
    }

    // Transfer C data to a Python object
    PyArrayObject *ArgsArray = PyArray_SimpleNew(1, dims[0], NPY_DOUBLE);
    double *p = (double *) PyArray_DATA(ArgsArray);

    for (int e = 0; e < dims[0]; e++){ 
        memcpy(p, &EELS[e], sizeof(double));
        p += 1;  
    }

    printf("Process ended with %lu sucessful transitions..\n", iterations);
    PyErr_Clear();
    return Py_BuildValue("O", ArgsArray);

    aborted:
        printf("Process aborted. \n");
        return Py_None;
}

static PyMethodDef methods[] = {
    //PyName	CppName	TakingArguments	Description
    {"calculate_spectrum", calculate_spectrum, METH_VARARGS, "Takes a band structure and simulates all transitions, returns a four dimentional histogram."},
    {"scan_spectrum", scan_spectrum, METH_VARARGS, "Takes a band structure and simulates the direct transitions of a band structure witout wavefunctions, returns a intensity as function of energy."},
    {"calculate_momentum_squared", calculate_momentum_squared, METH_VARARGS, "Calculates momentum scaling."},
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