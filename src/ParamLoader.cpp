#include "ParamLoader.hpp"
#include "NetParameter.hpp"
#include "Tensor.hpp"
#include <bits/stdint-uintn.h>
#include <cstdio>
#include <string>
#include <utility>
/*
 * ┌───────┬──────────────────────────┬──────────────────────────┬────────┬──────┬─────────┬───────────┬─────────┬─────────┐
 * │       │                          │                          │        │      │         │           │         │         │
 * │       │                          │                          │        │      │         │           │         │         │
 * │       │                          │                          │        │      │         │           │         │         │
 * │       │                          │                          │        │      │         │           │         │         │
 * │       │                          │                          │        │      │         │           │         │         │
 * │       │                          │                          │        │      │         │           │         │         │
 * │ Magic │     Weights Contents 1   │      Weights Contents 2  │ .....  │ Name │ Name    │   Weights │ Offsets │ Weights │
 * │       │                          │                          │        │ Length String  │   Length  │   INT   │  Index  │
 * │       │                          │                          │        │ INT  │         │    INT    │         │  Length │
 * │       │                          │                          │        │      │         │           │         │   INT   │
 * │       │                          │                          │        │      │         │           │         │         │
 * │       │                          │                          │        │      │         │           │         │         │
 * │       │                          │                          │        │      │         │           │         │         │
 * │       │                          │                          │        │      │         │           │         │         │
 * └───────┴──────────────────────────┴──────────────────────────┴────────┴──────┴─────────┴───────────┴─────────┴─────────┘
 *  Weights File Structure
 */

static int read_int(FILE *fp) {
  int tmp;
  fread(&tmp, sizeof(int), 1, fp);
  return tmp;
}
static std::string read_string(FILE *fp) {
  int len = read_int(fp);
  char *tmp = new char[len];
  fread(tmp, sizeof(char), len, fp);
  std::string str(tmp);
  delete[] tmp;
  return str;
}
bool ParamLoader::load_data(mllm::Tensor* tenor) {
  string name = tenor->name_;
#ifndef USE_MMAP
  if (offsets.find(name) == offsets.end()) {
    return false;
  }
  std::pair<uint8_t, uint8_t> offset = offsets[name];
  uint8_t *data = new uint8_t[offset.second];
  fseek(fp, offset.first, SEEK_SET);
  fread(data, sizeof(uint8_t), offset.second, fp);
  //TODO:Data?
  // tenor. = data;
  return true;
#endif
}
ParamLoader::~ParamLoader() {
  if (fp != nullptr) {
    fclose(fp);
  }
}
ParamLoader::ParamLoader(std::string filename, bool use_mmap)
    : path(std::move(filename)), use_mmap(use_mmap) {
  this->fp = fopen(filename.c_str(), "rb");
  if (fp == nullptr) {
    std::cout << "open file failed" << std::endl;
    exit(1);
  }
#ifndef USE_MMAP
  use_mmap = false;
#endif
  fseek(fp, 0, SEEK_SET);
#ifndef USE_MMAP
  int magic = read_int(fp);
  if (magic != MAGIC_NUMBER) {
    std::cout << "magic number error" << std::endl;
    exit(1);
  }
  fseek(fp, 0, SEEK_END);
  this->size = ftell(fp);
  fseek(fp, this->size - sizeof(int), SEEK_CUR);
  int table_len = read_int(fp);
  fseek(fp, this->size - table_len - sizeof(int), SEEK_SET);
  int table_offset = ftell(fp);
  while (table_offset < this->size - sizeof(int)) {

    std::string name = read_string(fp);
    int length = read_int(fp);
    int offset = read_int(fp);
    offsets[name] = std::make_pair(offset, length);
    // table_offset+=name.size()+sizeof(int)+sizeof(int);
    table_offset = ftell(fp);
  }

// int len = sizeof(int);
// while (len<size) {
//     int index = read_int(fp);
//     len+=sizeof(int);
//     std::string name = read_string(fp);
//     int length = read_int(fp);
//     len+=name.size()+sizeof(int)+sizeof(int);
//     offsets[name] = std::make_pair(len,length);
//     len+=length; //Align?
// }
#endif
}
