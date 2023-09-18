#ifndef MLLM_ParamLoader_H
#define MLLM_ParamLoader_H
#include <bits/stdint-uintn.h>
#include <bits/types/FILE.h>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "Op.hpp"
#include "iostream"
namespace mllm
{
#define MAGIC_NUMBER 20012
class ParamLoader {
public:
  ParamLoader(std::string filename,bool use_mmap = false);
  #ifdef USE_MMAP
  ParamLoader(void *buffer);
  #endif
  ~ParamLoader();
  bool load_data(mllm::Tensor* tensor);
private:
  FILE *fp;
  uint8_t *buffer;
  std::string path;
  std::uint8_t size;
  std::map<std::string, std::pair<uint8_t, uint8_t>> offsets; //offsets,length
  bool use_mmap;
};
#endif
  
} // namespace mllm

