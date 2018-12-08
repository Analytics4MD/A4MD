#ifndef __RETRIEVE_H__
#define __RETRIEVE_H__
#include <vector>
#include <tuple>

#include <Python.h>


class Retrieve
{
    private:
        int initialize_python();
        PyObject* m_py_module;
        PyObject* m_py_func;
        const char* m_module_name;
        const char* m_function_name;
    public:
        Retrieve(char* module_name,
                 char* function_name);
        ~Retrieve();
        
        int analyze_frame(std::vector<int> types,
                          std::vector<double> x_positions,
                          std::vector<double> y_positions,
                          std::vector<double> z_positions,
                          double x_low,
                          double x_high,
                          double y_low,
                          double y_high,
                          double z_low,
                          double z_high,
                          int step);

};
#endif
