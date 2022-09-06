#include "../include/cv_runner.h"
#include "../include/cv_chunk.h"

CVRunner::CVRunner(char* module_name, char* function_name, char* py_path)
: PyRunner(module_name, function_name, py_path) 
{
	printf("---===== Initialized CVRunner\n");
}

CVRunner::~CVRunner()
{ 
	printf("---===== Finalized CVRunner\n");
}

template <typename T>
std::vector<T> listToVector(PyObject* incoming) 
{
    std::vector<T> data;
    if (std::is_same<T, int>::value)
    {
        if (PyList_Check(incoming)) 
        {
            for(Py_ssize_t i = 0; i < PyList_Size(incoming); i++) 
            {
                data.push_back(PyLong_AsLong(PyList_GetItem(incoming, i)));
            }
        } 
        else 
        {
            fprintf(stderr, "Passed PyObject pointer was not a list!");
        }
    }
    else if (std::is_same<T, double>::value)
    {
        if (PyList_Check(incoming)) 
        {
            for(Py_ssize_t i = 0; i < PyList_Size(incoming); i++) 
            {
                data.push_back(PyFloat_AsDouble(PyList_GetItem(incoming, i)));
            }
        } 
        else 
        {
            fprintf(stderr, "Passed PyObject pointer was not a list");
        }
    }
    else 
        fprintf(stderr, "T should be integer or double\n");
    
    return data;
}

template <typename T>
PyObject* vectorTolist(std::vector<T> vecs) 
{
	PyObject* result;
	if (vecs.empty())
	{
		return result;
	}
	int num = vecs.size();
	result = PyList_New(num);
	int i = 0;
	for(auto vec: vecs)
	{
		PyObject *val;
		if (std::is_same<T, int>::value)
		{
			val = PyLong_FromLong(vec);
		}
		else if (std::is_same<T, double>::value)
		{
			val = PyFloat_FromDouble(vec);
		}
		else
		{
			fprintf(stderr, "T should be integer or double\n");
			return result;
		}
		PyList_SetItem(result, i, val);
		i++;
	}
	return result;
}

void CVRunner::input_chunk(Chunk* chunk) 
{

}

Chunk* CVRunner::output_chunk(unsigned long int chunk_id) 
{
	Chunk* result;
	return result;
}

Chunk* CVRunner::direct_chunk(Chunk* chunk) 
{
	Chunk* result;
	if (!m_py_module)
    {
        PyErr_Print();
        fprintf(stderr,"---===== ERROR: CVRunner::direct_chunk Import %s failed. See for an error message above\n", m_module_name);
        return result;
    }
    if (!m_py_func || !PyCallable_Check(m_py_func))
    {
    	fprintf(stderr,"---===== ERROR: CVRunner::direct_chunk Python function %s is not found in %s\n", m_function_name, m_module_name);
        print_py_error_and_rethrow();
        return result;
    }
    if (chunk == nullptr)
	{
		fprintf(stderr, "---===== ERROR: CVRunner::direct_chunk Input chunk is a null pointer\n");
		return result;
	}

	unsigned long int chunk_id = chunk->get_chunk_id();
	MDChunk *md_chunk = dynamic_cast<MDChunk*>(chunk);
	if (md_chunk == nullptr) 
	{
		fprintf(stderr, "---===== ERROR: CVRunner::direct_chunk Failed to uppercast the chunk to MDChunk");
		return result;
	}
	std::vector<double> x_positions = md_chunk->get_x_positions();
	std::vector<double> y_positions = md_chunk->get_y_positions();
	std::vector<double> z_positions = md_chunk->get_z_positions();
	std::vector<int> input_types = md_chunk->get_types();
	double lx = md_chunk->get_box_lx();
	double ly = md_chunk->get_box_ly();
	double lz = md_chunk->get_box_lz();
	double hx = md_chunk->get_box_hx();
	double hy = md_chunk->get_box_hy(); 
	double hz = md_chunk->get_box_hz(); 
	int step = md_chunk->get_timestep(); 
	if ((x_positions.size() != y_positions.size()) || (y_positions.size() != z_positions.size()) || (z_positions.size() != x_positions.size())) {
		fprintf(stderr, "---===== ERROR: CVRunner::direct_chunk Input chunk does not have consistency in dimension across x, y, z\n");
		return result;
	}
	if ((input_types.size() != x_positions.size()) || (input_types.size() != y_positions.size()) || (input_types.size() != z_positions.size())) {
		fprintf(stderr, "---===== ERROR: CVRunner::direct_chunk Input chunk has the dimension of types array is different from either x, y, z positions\n");
		return result;
	}
	PyObject* py_args = PyTuple_New(6);
    PyObject* py_input_types = vectorTolist<int>(input_types);
    PyTuple_SetItem(py_args, 0, py_input_types);
    PyObject* py_x_positions = vectorTolist<double>(x_positions);
    PyTuple_SetItem(py_args, 1, py_x_positions);
    PyObject* py_y_positions = vectorTolist<double>(y_positions);
    PyTuple_SetItem(py_args, 2, py_y_positions);
    PyObject* py_z_positions = vectorTolist<double>(z_positions);
    PyTuple_SetItem(py_args, 3, py_z_positions);
    PyObject* py_input_box = Py_BuildValue("[dddddd]", lx, ly, lz, hx, hy, hz);
    PyTuple_SetItem(py_args, 4, py_input_box);
    PyObject* py_step = Py_BuildValue("i", step);
    PyTuple_SetItem(py_args, 5, py_step);
    PyObject* py_return = PyObject_CallObject(m_py_func, py_args);
    Py_DECREF(py_args);
    if (py_return == NULL)
	{
	    fprintf(stderr, "---===== ERROR: CVRunner::direct_chunk Python returns NULL\n");
	    print_py_error_and_rethrow();
	    return result;
	}
	std::vector<double> cv_values = listToVector<double>(py_return);

	result = new CVChunk(chunk_id, cv_values);
	Py_DECREF(py_return);
    return result;
}
