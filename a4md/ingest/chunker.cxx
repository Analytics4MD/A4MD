#include "chunker.h"
#include "pycall.h"

Chunker::Chunker()
{
	initialize();
}

Chunker::~Chunker()
{
	finalize();
}

PdbChunker::PdbChunker(std::string file_path, std::string log_path, std::string py_path, std::string py_script, std::string py_def) 
{
    m_file_path = file_path;
    m_log_path = log_path;
    m_py_path = py_path;
    m_py_script = py_script;
    m_py_def = py_def;

    /*setenv("PYTHONPATH", py_path.c_str(), 1);
    if (!Py_IsInitialized())
        Py_Initialize();

    m_py_func = load_py_function(py_script, py_def);*/
}

PdbChunker::~PdbChunker() 
{
	/*Py_Finalize();*/
}

void PdbChunker::initialize() 
{
	setenv("PYTHONPATH", m_py_path.c_str(), 1);
    if (!Py_IsInitialized())
        Py_Initialize();

    m_py_func = load_py_function(m_py_script, m_py_def);
}

void PdbChunker::finalize()
{
	Py_Finalize();
}


int extract_frame(PyObject *py_func, std::string file_path, std::string log_path, double **data)
{
	PyObject *py_retValue;
    PyObject *py_args;
    py_args = PyTuple_Pack(2, PyUnicode_FromString(file_path.c_str()), PyUnicode_FromString(log_path.c_str()));
    py_retValue = PyObject_CallObject(py_func, py_args);
    Py_DECREF(py_args);
    int nCA = 0;

    // Get partial CA coordinates 
    PyObject *py_num;
    py_num = PyList_GetItem(py_retValue, 0);
    if (PyLong_AsSsize_t(py_num) < 0) {
        Py_DECREF(py_retValue);
        return -1;
    } else {
        PyObject *py_CA;
        py_CA = PyList_GetItem(py_retValue, 1);
        int nCA;
        nCA = PyList_Size(py_CA);           
        *data = (double *) malloc (nCA * 3 * sizeof (double));
        
        int i, j;
        PyObject *item;
        for (i = 0; i < nCA; i++) {
            item = PyList_GetItem(py_CA, i);
            for (j = 0; j < 3; j++) { 
                (*data)[i * 3 + j] = PyFloat_AsDouble(PyTuple_GetItem(item, j));
            }
            Py_DECREF(item);
        }
        // Clean up
        Py_DECREF(py_CA);
        Py_DECREF(py_retValue);
        return nCA;
    }
}

std::vector<Chunk> PdbChunker::chunks_from_file(int num_chunks)
{
    std::vector<Chunk> chunks;
    // throw NotImplementedException();
    //read from persistent storage
    //logic to iterate from last read chunk to num_chunks
    //ch1.data = python_get_frame();

    double *data = NULL;
    for (int i = 0; i < num_chunks; i++) {
    	int size = extract_frame(m_py_func, m_file_path, m_log_path, &data);
    	Chunk chunk;
    	chunk.data = data;
    	chunk.size = size;
    	chunks.push_back(chunk);
    }

    return chunks;
}
