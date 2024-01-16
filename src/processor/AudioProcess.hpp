//
// Created by Rongjie Yi on 24-1-9.
//

#ifndef AUDIOPROCESS_HPP
#define AUDIOPROCESS_HPP

#include <string>
#include <vector>

std::vector<std::vector<std::vector<std::vector<float>>>> ProcessWAV(std::vector<std::string> waves, int resample_rate = 16000);



#endif //AUDIOPROCESS_HPP
