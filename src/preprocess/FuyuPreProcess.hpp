//
// Created by 咸的鱼 on 2023/12/4.
//

#ifndef FUYUPREPROCESS_HPP
#define FUYUPREPROCESS_HPP
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <imageHelper/stb_image.h>
#include <imageHelper/stb_image_resize2.h>
// #include <imageHelper/stb_image_resize.h>
using std::vector;
namespace mllm {
typedef  vector<vector<vector<vector<float>>>> FourDVector;
struct FuyuBatchEncoding {
    std::vector<int> input_ids;
    std::vector<int> attention_mask;
    std::vector<FourDVector> image_patches;
    std::vector<int> image_patches_indices;
};
struct ImageInfo {
    float * data;
    int width;
    int height;
    int channels;
    int original_width;
    int original_height;
    ImageInfo(float * data, int width, int height, int channels) : data(data), width(width), height(height),original_height(height),original_width(width), channels(channels) {}
    ImageInfo(float * data, int width, int height, int channels,int original_width,int original_height
        ) : data(data), width(width), height(height), channels(channels),original_width(original_width),original_height(original_height) {}

};
enum PaddingType {
    CONSTANT,
};
enum ResampleType {
    BILINEAR,

};
class FuyuPreProcess {
public:
    static std::vector<FourDVector>  PreProcessImages(const std::vector<std::vector<uint8_t>> &images,int height=224,int width=224,bool do_pad=false,bool do_resize=false);
private:
    static std::vector<ImageInfo> PadImages(const std::vector<ImageInfo> &images, int height, int width,float pad_value=1.0,PaddingType padding_type=PaddingType::CONSTANT);
    static std::vector<ImageInfo> ResizeImages(const std::vector<ImageInfo> &images, int height, int width,ResampleType resample_type=ResampleType::BILINEAR);



};

} // mllm

#endif //FUYUPREPROCESS_HPP
