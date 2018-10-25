#include "retrieve.h"
#include <unistd.h>

Retrieve::Retrieve()
{
    //Nothing for now
}

Retrieve::~Retrieve()
{
    //Nothing for now
}

void Retrieve::run()
{
    printf("In Retrieve Run\n");
}

int Retrieve::call_py(int argc, const char** argv)
{
    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;
    int i;

    if (argc < 3) {
        fprintf(stderr,"Usage: call pythonfile funcname [args]\n");
        return 1;
    }

    char cwd[256];
    if (getcwd(cwd, sizeof(cwd)) == NULL)
      perror("getcwd() error");
    else
      printf("current working directory is: %s\n", cwd);

    Py_Initialize();

    pName = PyUnicode_DecodeFSDefault(argv[2]);
    /* Error checking of pName left out */

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, argv[3]);
        /* pFunc is a new reference */

        if (pFunc==NULL){
            printf("pFunc is NULL. func name is %s\n",argv[1]);
        }
        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New(argc - 4);
            for (i = 0; i < argc - 4; ++i) {
                pValue = PyLong_FromLong(atoi(argv[i + 4]));
                if (!pValue) {
                    Py_DECREF(pArgs);
                    Py_DECREF(pModule);
                    fprintf(stderr, "Cannot convert argument\n");
                    return 1;
                }
                /* pValue reference stolen here: */
                PyTuple_SetItem(pArgs, i, pValue);
            }
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);
            if (pValue != NULL) {
                printf("Result of call: %ld\n", PyLong_AsLong(pValue));
                Py_DECREF(pValue);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return 1;
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", argv[2]);
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", argv[1]);
        return 1;
    }
    if (Py_FinalizeEx() < 0) {
        return 120;
    }
    return 0;

}
