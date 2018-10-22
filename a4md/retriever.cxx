#include "retriever.h"

int main (int argc, const char** argv)
{
  Retriever retriever;
  retriever.run();
  return retriever.call_py(argc, argv);
}
