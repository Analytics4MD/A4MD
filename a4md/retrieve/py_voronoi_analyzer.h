#ifndef __PY_VORONOI_ANALYZER_H__
#define __PY_VORONOI_ANALYZER_H__
#include <vector>
#include <tuple>

#include <Python.h>


typedef std::vector<std::tuple<double, double, double>> POS_VEC;
class PyVoronoiAnalyzer
{
    private:
        int initialize_python();
        PyObject* m_py_module;
        PyObject* m_py_func;
    public:
        PyVoronoiAnalyzer();
        ~PyVoronoiAnalyzer();
        
        void run();
        int call_py(int, const char**);
        int analyze_frame(char* module_name,
                          char* function_name,
                          int* types,
                          POS_VEC positions,
                          double x_low,
                          double x_high,
                          double y_low,
                          double y_high,
                          double z_low,
                          double z_high,
                          int step);
         int aanalyze_frame(POS_VEC module_name);

};
#endif
