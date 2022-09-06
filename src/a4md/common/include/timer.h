#ifndef __TIMER_H__
#define __TIMER_H__
#include <chrono>

#define timeNow() std::chrono::high_resolution_clock::now()
typedef std::chrono::high_resolution_clock::time_point TimeVar;
typedef std::chrono::duration<double, std::milli> DurationMilli;

#endif
