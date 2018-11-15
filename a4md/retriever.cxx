#include "retrieve.h"

int main (int argc, const char** argv)
{
  Retrieve retrieve;
  retrieve.run();
  int types[2];
  types[0] = 0;
  types[1] = 0;
  POS_VEC positions;
  positions.push_back(std::make_tuple(1.0,1.0,1.0));
  double x_low, y_low, z_low,x_high, y_high, z_high;
  x_low=y_low=z_low=0.0;
  x_high=y_high=z_high=1.0;
  printf("x_low %f\n",x_low);
  printf("x_high %f\n",x_high);

  printf("y_low %f\n",y_low);
  printf("y_high %f\n",y_high);

  printf("z_low %f\n",z_low);
  printf("z_high %f\n",z_high);
  char* name = (char*)argv[1];
  char* func = (char*)argv[2];
  retrieve.analyze_frame(name,
                         func,
                         types,
                         positions,
                         x_low,
                         x_high,
                         y_low,
                         y_high,
                         z_low,
                         z_high,
                         0);
  return 0;//retrieve.call_py(argc, argv);
}
