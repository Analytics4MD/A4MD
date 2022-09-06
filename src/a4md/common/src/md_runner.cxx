#include "../include/md_runner.h"

#include <iostream>
#include "../include/md_chunk.h"
// #define NO_IMPORT_ARRAY
// #define PY_ARRAY_UNIQUE_SYMBOL PY_ARRAY_API
// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// #include <numpy/arrayobject.h>
// #include <numpy/ndarrayobject.h>
// #include <numpy/npy_common.h>

MDRunner::MDRunner(char* module_name, char* function_name, char* py_path)
: PyRunner(module_name, function_name, py_path) 
{
	printf("---===== Initialized MDRunner\n");
}

MDRunner::MDRunner(char* module_name, char* function_name, char* py_path, char* trajectory_path, int num_atoms, int position)
: PyRunner(module_name, function_name, py_path),
  m_trajectory_path(trajectory_path),
  m_position(position),
  m_num_atoms(num_atoms) 
{
	printf("---===== Initialized MDRunner\n");
	printf("---===== Initialized trajectory path: %s\n", m_trajectory_path.c_str());
	printf("---===== Initialized position: %d\n", m_position);
	printf("---===== Initialized number of atoms: %d\n", m_num_atoms);
}

MDRunner::~MDRunner()
{ 
	printf("---===== Finalized MDRunner\n");
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

void MDRunner::input_chunk(Chunk* chunk)
{
    if (!m_py_module)
    {
        PyErr_Print();
        fprintf(stderr,"---===== ERROR: MDRunner::input_chunk Import %s failed. See for an error message above\n", m_module_name);
        return;
    }
    else
    {
        if (m_py_func && PyCallable_Check(m_py_func))
        {
			if (chunk == nullptr)
			{
				fprintf(stderr, "---===== ERROR: MDRunner::input_chunk Input chunk is a null pointer\n");
				return;
			}
			MDChunk *md_chunk = dynamic_cast<MDChunk*>(chunk);
			if (md_chunk == nullptr) 
			{
				fprintf(stderr, "---===== ERROR: MDRunner::input_chunk Failed to uppercast the chunk");
				return;
			}
			std::vector<double> x_positions = md_chunk->get_x_positions();
			std::vector<double> y_positions = md_chunk->get_y_positions();
			std::vector<double> z_positions = md_chunk->get_z_positions();
			std::vector<int> types = md_chunk->get_types();
			double lx = md_chunk->get_box_lx();
			double ly = md_chunk->get_box_ly();
			double lz = md_chunk->get_box_lz();
			double hx = md_chunk->get_box_hx();
			double hy = md_chunk->get_box_hy(); 
			double hz = md_chunk->get_box_hz(); 
			int step = md_chunk->get_timestep(); 

			if ((x_positions.size() != y_positions.size()) || (y_positions.size() != z_positions.size()) || (z_positions.size() != x_positions.size())) {
				fprintf(stderr, "---===== ERROR: MDRunner::input_chunk Input chunk does not have consistency in dimension across x, y, z\n");
				return;
			}

			if ((types.size() != x_positions.size()) || (types.size() != y_positions.size()) || (types.size() != z_positions.size())) {
				fprintf(stderr, "---===== ERROR: MDRunner::input_chunk Input chunk has the dimension of types array is different from either x, y, z positions\n");
				return;
			}

			PyObject* py_args = PyTuple_New(6);

			int count = x_positions.size();
		    // npy_intp types_dims[] = {count};
		    // PyObject* py_types = PyArray_SimpleNewFromData(1, types_dims, NPY_DOUBLE, static_cast<void*>(types.data()));
		    PyObject* py_types = vectorTolist<int>(types);
		    PyTuple_SetItem(py_args, 0, py_types);

		    // npy_intp positions_dims[] = {count};
		    // PyObject* py_x_positions = PyArray_SimpleNewFromData(1, positions_dims, NPY_DOUBLE, static_cast<void*>(x_positions.data()));
		    PyObject* py_x_positions = vectorTolist<double>(x_positions);
		    PyTuple_SetItem(py_args, 1, py_x_positions);

		    // PyObject* py_y_positions = PyArray_SimpleNewFromData(1, positions_dims, NPY_DOUBLE, static_cast<void*>(y_positions.data()));
		    PyObject* py_y_positions = vectorTolist<double>(y_positions);
		    PyTuple_SetItem(py_args, 2, py_y_positions);

		    // PyObject* py_z_positions = PyArray_SimpleNewFromData(1, positions_dims, NPY_DOUBLE, static_cast<void*>(z_positions.data()));
		    PyObject* py_z_positions = vectorTolist<double>(z_positions);
		    PyTuple_SetItem(py_args, 3, py_z_positions);

		    PyObject* py_box = Py_BuildValue("[dddddd]", lx, ly, lz, hx, hy, hz);
		    PyTuple_SetItem(py_args, 4, py_box);

		    PyObject* py_step = Py_BuildValue("i", step);
		    PyTuple_SetItem(py_args, 5, py_step);

            PyObject* py_return = PyObject_CallObject(m_py_func, py_args);
            Py_DECREF(py_args);
            if (py_return != nullptr)
            {
                printf("---===== MDRunner::input_chunk Succeed to call Python function %s in %s\n", m_function_name, m_module_name);
                Py_DECREF(py_return);
            }
            else
            {
                fprintf(stderr, "---===== ERROR: MDRunner::input_chunk Not succeed to call Python function %s in %s\n", m_function_name, m_module_name);
                print_py_error_and_rethrow();
                return;
            }
        }
        else
        {
            fprintf(stderr,"---===== ERROR: PyRunner::input_chunk Python function %s is not found in %s\n", m_function_name, m_module_name);
            print_py_error_and_rethrow();
            return;
        }
    } 
    return;
}

Chunk* MDRunner::output_chunk(unsigned long int chunk_id)
{
    Chunk* result = nullptr;
    if (!m_py_module)
    {
        PyErr_Print();
        fprintf(stderr,"---===== ERROR: MDRunner::output_chunk Import %s failed. See for an error message above\n", m_module_name);
        return result;
    }
    else
    {
        if (m_py_func && PyCallable_Check(m_py_func))
        {
            PyObject* py_args = PyTuple_New(3);
            PyObject* py_file = Py_BuildValue("s", m_trajectory_path.c_str());
            PyTuple_SetItem(py_args, 0, py_file);
        
            PyObject* py_position = Py_BuildValue("i", m_position);
            PyTuple_SetItem(py_args, 1, py_position);
        
            PyObject* py_natoms = Py_BuildValue("i", m_num_atoms);
            PyTuple_SetItem(py_args, 2, py_natoms);

            PyObject* py_return = PyObject_CallObject(m_py_func, py_args);
            Py_DECREF(py_args);
            if (py_return == nullptr)
			{
			    fprintf(stderr, "---===== ERROR: MDRunner::output_chunk Python returns NULL\n");
			    print_py_error_and_rethrow();
			    return result;
			}
			
			if (PyTuple_Check(py_return))
			{
				int num = PyTuple_Size(py_return);
				printf("---===== MDRunner::output_chunk Size of return Python object: %d\n", num);
			    if (PyTuple_Size(py_return) == 7)
			    {
			        // ToDo: Directly casting type of PyList to vector / array  
			        PyObject *py_types = PyTuple_GetItem(py_return, 0);
			        PyObject *py_x_cords = PyTuple_GetItem(py_return, 1);
			        PyObject *py_y_cords = PyTuple_GetItem(py_return, 2);
			        PyObject *py_z_cords = PyTuple_GetItem(py_return, 3);
			        PyObject* py_box = PyTuple_GetItem(py_return, 4);
			        PyObject* py_step = PyTuple_GetItem(py_return, 5);
			        m_position = PyLong_AsLong(PyTuple_GetItem(py_return, 6));

			        int ret_natoms = PyList_Size(py_types);
			        printf("---===== MDRunner::output_chunk Number of returned atoms : %d\n", ret_natoms);
			        std::vector<int> types = listToVector<int>(py_types);
			        std::vector<double> x_cords = listToVector<double>(py_x_cords);
			        std::vector<double> y_cords = listToVector<double>(py_y_cords);
			        std::vector<double> z_cords = listToVector<double>(py_z_cords);

			        // ToDo: Unhardcode these assignments
			        double box_lx = 0.0, box_ly = 0.0, box_lz = 0.0;
			        double box_hx = 0.0, box_hy = 0.0, box_hz = 0.0;
			        
			        if (PyList_Check(py_box))
			        {
			            if (PyList_Size(py_box) == 6)
			            {                        
			                PyObject* py_box_lx = PyList_GetItem(py_box, 0);
			                if (py_box_lx != Py_None) 
			                {
			                    box_lx = PyFloat_AsDouble(py_box_lx);
			                }
			                PyObject* py_box_ly = PyList_GetItem(py_box, 1);
			                if (py_box_ly != Py_None) 
			                {
			                    box_ly = PyFloat_AsDouble(py_box_ly);
			                }
			                PyObject* py_box_lz = PyList_GetItem(py_box, 2);
			                if (py_box_lz != Py_None) 
			                {
			                    box_lz = PyFloat_AsDouble(py_box_lz);
			                }
			                PyObject* py_box_hx = PyList_GetItem(py_box, 3);
			                if (py_box_hx != Py_None) 
			                {
			                    box_hx = PyFloat_AsDouble(py_box_hx);
			                }
			                PyObject* py_box_hy = PyList_GetItem(py_box, 4);
			                if (py_box_hy != Py_None) 
			                {
			                    box_hy = PyFloat_AsDouble(py_box_hy);
			                }
			                PyObject* py_box_hz = PyList_GetItem(py_box, 5);
			                if (py_box_hz != Py_None) 
			                {
			                    box_hz = PyFloat_AsDouble(py_box_hz);
			                }
			            }
			        }
			        else 
			        {
			            fprintf(stderr, "---===== ERROR: MDRunner::output_chunk Box tupple returned is in wrong format\n");
			            return result;
			        }

			        int timestep = 0;
			        if (py_step != Py_None)
			        {
			            timestep = PyLong_AsLong(py_step);
			        }
			        result = new MDChunk(chunk_id, 
			                            timestep,
			                            types,
			                            x_cords,
			                            y_cords,
			                            z_cords,
			                            box_lx,
			                            box_ly,
			                            box_lz,
			                            box_hx,
			                            box_hy,
			                            box_hz);
			    } 
			    else 
			    {
			        fprintf(stderr, "---===== ERROR: MDRunner::output_chunk Python function returns a list in wrong format\n");
			        return result;
			    }
			}
			else
			{
			    fprintf(stderr, "---===== ERROR: MDRunner::output_chunk Python function returns not a list\n");
			    return result;
			}

			Py_DECREF(py_return);
        }
        else
        {
            fprintf(stderr,"---===== ERROR: MDRunner::output_chunk Python function %s is not found in %s\n", m_function_name, m_module_name);
            print_py_error_and_rethrow();
            return result;
        }  
    }
    return result;
}

Chunk* MDRunner::direct_chunk(Chunk* chunk)
{
	Chunk* result = nullptr;
	if (!m_py_module)
    {
        PyErr_Print();
        fprintf(stderr,"---===== ERROR: MDRunner::direct_chunk Import %s failed. See for an error message above\n", m_module_name);
        return result;
    }
    else
    {
        if (m_py_func && PyCallable_Check(m_py_func))
        {
			if (chunk == nullptr)
			{
				fprintf(stderr, "---===== ERROR: MDRunner::direct_chunk Input chunk is a null pointer\n");
				return result;
			}
			unsigned long int chunk_id = chunk->get_chunk_id();

			MDChunk *md_chunk = dynamic_cast<MDChunk*>(chunk);
			if (md_chunk == nullptr) 
			{
				fprintf(stderr, "---===== ERROR: MDRunner::direct_chunk Failed to uppercast the chunk");
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
				fprintf(stderr, "---===== ERROR: MDRunner::direct_chunk Input chunk does not have consistency in dimension across x, y, z\n");
				return result;
			}

			if ((input_types.size() != x_positions.size()) || (input_types.size() != y_positions.size()) || (input_types.size() != z_positions.size())) {
				fprintf(stderr, "---===== ERROR: MDRunner::direct_chunk Input chunk has the dimension of types array is different from either x, y, z positions\n");
				return result;
			}

			int count = x_positions.size();
			PyObject* py_args = PyTuple_New(6);
		    // npy_intp types_dims[] = {count};
		    // PyObject* py_input_types = PyArray_SimpleNewFromData(1, types_dims, NPY_DOUBLE, static_cast<void*>(input_types.data()));
		    PyObject* py_input_types = vectorTolist<int>(input_types);
		    PyTuple_SetItem(py_args, 0, py_input_types);

		    // npy_intp positions_dims[] = {count};
		    // PyObject* py_x_positions = PyArray_SimpleNewFromData(1, positions_dims, NPY_DOUBLE, static_cast<void*>(x_positions.data()));
		    PyObject* py_x_positions = vectorTolist<double>(x_positions);
		    PyTuple_SetItem(py_args, 1, py_x_positions);

		    // PyObject* py_y_positions = PyArray_SimpleNewFromData(1, positions_dims, NPY_DOUBLE, static_cast<void*>(y_positions.data()));
		    PyObject* py_y_positions = vectorTolist<double>(y_positions);
		    PyTuple_SetItem(py_args, 2, py_y_positions);

		    // PyObject* py_z_positions = PyArray_SimpleNewFromData(1, positions_dims, NPY_DOUBLE, static_cast<void*>(z_positions.data()));
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
			    fprintf(stderr, "---===== ERROR: MDRunner::direct_chunk Python returns NULL\n");
			    print_py_error_and_rethrow();
			    return result;
			}
			
			if (PyTuple_Check(py_return))
			{
				int num = PyTuple_Size(py_return);
				printf("---===== MDRunner::direct_chunk Size of returned Python object: %d\n", num);
			    if (PyTuple_Size(py_return) == 6)
			    {
			        // ToDo: Directly casting type of PyList to vector / array  
			        PyObject *py_output_types = PyTuple_GetItem(py_return, 0);
			        PyObject *py_x_cords = PyTuple_GetItem(py_return, 1);
			        PyObject *py_y_cords = PyTuple_GetItem(py_return, 2);
			        PyObject *py_z_cords = PyTuple_GetItem(py_return, 3);
			        PyObject* py_output_box = PyTuple_GetItem(py_return, 4);
			        PyObject* py_timestep = PyTuple_GetItem(py_return, 5);
			        // m_position = PyLong_AsLong(PyList_GetItem(py_return, 6));

			        int ret_natoms = PyList_Size(py_output_types);
			        printf("---===== MDRunner::direct_chunk Number of returned atoms : %d\n", ret_natoms);
			        std::vector<int> output_types = listToVector<int>(py_output_types);
			        std::vector<double> x_cords = listToVector<double>(py_x_cords);
			        std::vector<double> y_cords = listToVector<double>(py_y_cords);
			        std::vector<double> z_cords = listToVector<double>(py_z_cords);

			        // ToDo: Unhardcode these assignments
			        double box_lx = 0.0, box_ly = 0.0, box_lz = 0.0;
			        double box_hx = 0.0, box_hy = 0.0, box_hz = 0.0;
			        
			        if (PyList_Check(py_output_box))
			        {
			            if (PyList_Size(py_output_box) == 6)
			            {                        
			                PyObject* py_box_lx = PyList_GetItem(py_output_box, 0);
			                if (py_box_lx != Py_None) 
			                {
			                    box_lx = PyFloat_AsDouble(py_box_lx);
			                }
			                PyObject* py_box_ly = PyList_GetItem(py_output_box, 1);
			                if (py_box_ly != Py_None) 
			                {
			                    box_ly = PyFloat_AsDouble(py_box_ly);
			                }
			                PyObject* py_box_lz = PyList_GetItem(py_output_box, 2);
			                if (py_box_lz != Py_None) 
			                {
			                    box_lz = PyFloat_AsDouble(py_box_lz);
			                }
			                PyObject* py_box_hx = PyList_GetItem(py_output_box, 3);
			                if (py_box_hx != Py_None) 
			                {
			                    box_hx = PyFloat_AsDouble(py_box_hx);
			                }
			                PyObject* py_box_hy = PyList_GetItem(py_output_box, 4);
			                if (py_box_hy != Py_None) 
			                {
			                    box_hy = PyFloat_AsDouble(py_box_hy);
			                }
			                PyObject* py_box_hz = PyList_GetItem(py_output_box, 5);
			                if (py_box_hz != Py_None) 
			                {
			                    box_hz = PyFloat_AsDouble(py_box_hz);
			                }
			            }
			        }
			        else 
			        {
			            fprintf(stderr, "---===== ERROR: MDRunner::direct_chunk Box tupple returned is in wrong format\n");
			            return result;
			        }

			        int timestep = 0;
			        if (py_timestep != Py_None)
			        {
			            timestep = PyLong_AsLong(py_timestep);
			        }
			        result = new MDChunk(chunk_id, 
			                            timestep,
			                            output_types,
			                            x_cords,
			                            y_cords,
			                            z_cords,
			                            box_lx,
			                            box_ly,
			                            box_lz,
			                            box_hx,
			                            box_hy,
			                            box_hz);
			    } 
			    else 
			    {
			        fprintf(stderr, "---===== ERROR: MDRunner::direct_chunk Python function returns a list in wrong format\n");
			        return result;
			    }
			}
			else
			{
			    fprintf(stderr, "---===== ERROR: MDRunner::direct_chunk Python function returns not a list\n");
			    return result;
			}

			Py_DECREF(py_return);
		}
		else
        {
            fprintf(stderr,"---===== ERROR: MDRunner::direct_chunk Python function %s is not found in %s\n", m_function_name, m_module_name);
            print_py_error_and_rethrow();
            return result;
        }  
	}
	return result;
}

int MDRunner::get_position()
{
	return m_position;
}

void MDRunner::set_position(int position)
{
	m_position = position;
}
