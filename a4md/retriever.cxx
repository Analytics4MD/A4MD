#include "retrieve.h"

int main (int argc, const char** argv)
{
  Retrieve retrieve;
  retrieve.run();
  return retrieve.call_py(argc, argv);
}
