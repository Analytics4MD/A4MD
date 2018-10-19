#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ingester.h"
#include "retriever.h"
#include "extern/argparse/argparse.hpp"

int main (int argc, const char** argv)
{
  ArgumentParser parser;
  parser.addArgument("-p","--python_test", '+');
  parser.parse(argc,argv);

  Ingester ingester;
  ingester.run();

  Retriever retriever;
  retriever.run();
  if (parser.exists("python_test"))
  {
    return retriever.call_py(argc, argv);
  }
  return 0;
}
