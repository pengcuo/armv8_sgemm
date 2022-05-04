#ifndef  __DURATION__
#define __DURATION__

#include <chrono>

class Duration {

public :
  Duration() {
    time_ = 0.f;
  }

  void start() {
    start_ = std::chrono::high_resolution_clock::now();
  }

  void end() {
    end_ = std::chrono::high_resolution_clock::now();
  }

  double getDuration() {
    time_ = std::chrono::duration<double, std::milli>(end_ - start_).count();
    return time_;
  }


private :
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_;
    double time_;
};


#endif // __DURATION__
