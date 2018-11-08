#ifndef __INGEST_H__
#define __INGEST_H__
#include <stdio.h>
#include "dataspaces.h"
#include "mpi.h"
#include <Python.h>

typedef struct {
	double *data = NULL;
	int size = 0;
} Chunk;

class Ingest
{
	private:
		PyObject* m_py_func;
		Chunk frame;
    public:
        Ingest();
        ~Ingest();
        int extract_frame(char *file_name, char *log_name);
        void run();
};

#endif
