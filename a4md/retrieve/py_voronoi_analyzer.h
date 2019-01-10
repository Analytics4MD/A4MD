#ifndef __PY_VORONOI_ANALYZER_H__
#define __PY_VORONOI_ANALYZER_H__
#include <vector>
#include <Python.h>


class PyVoronoiAnalyzer
{
    private:
        int initialize_python();
        PyObject* m_py_module;
        PyObject* m_py_func;
        const char* m_module_name;
        const char* m_function_name;
    public:
        PyVoronoiAnalyzer(char* module_name,
                          char* function_name);
        ~PyVoronoiAnalyzer();
        
        int analyze_frame(int* types,
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
