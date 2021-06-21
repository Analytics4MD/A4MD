#include "py_chunks.h"
#include <pybind11/stl.h>

PYBIND11_MODULE(a4md, m) {      
    py::class_<Chunk, PyChunk>(m, "Chunk")
        .def(py::init<>())
        .def(py::init<unsigned long int>())
        .def("getChunkID", &Chunk::get_chunk_id)
        .def("setChunkID", &Chunk::set_chunk_id)
        .def("print", &Chunk::print)
        .def("append", &Chunk::append);

    py::class_<TestChunk, Chunk>(m, "TestChunk")
        .def(py::init<>())
        .def(py::init<unsigned long int, int>());

    m.def("print", &print);

    py::class_<MDChunk, Chunk>(m, "MDChunk")
        .def(py::init<>())
        .def(py::init<unsigned long int, int, std::vector<int> &, std::vector<double> &, std::vector<double> &, std::vector<double> &, double, double, double, double, double, double>())
        .def("getTypes", &MDChunk::get_types)
        .def("getXPositions", &MDChunk::get_x_positions)
        .def("getYPositions", &MDChunk::get_y_positions)
        .def("getZPositions", &MDChunk::get_z_positions)
        .def("getBoxLX", &MDChunk::get_box_lx)
        .def("getBoxLY", &MDChunk::get_box_ly)
        .def("getBoxLZ", &MDChunk::get_box_lz)
        .def("getBoxHX", &MDChunk::get_box_hx)
        .def("getBoxHY", &MDChunk::get_box_hy)
        .def("getBoxHZ", &MDChunk::get_box_hz)
        .def("getTimeStep", &MDChunk::get_timestep);
}