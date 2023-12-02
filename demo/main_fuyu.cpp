#include <vector>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "imageHelper/stb_image.h"

using namespace std;

int main() {
    int width, height, channel;
    unsigned char *data = stbi_load("test.jpg", &width, &height, &channel, 0);
    if (data == nullptr) {
        cout << "load image failed" << endl;
        return -1;
    }
    cout << "width: " << width << " height: " << height << " channel: " << channel << endl;
    vector<float> data_f32(width * height * channel);
    for (int i = 0; i < width * height * channel; i++) {
        data_f32[i] = data[i] / 255.0;
    }
    stbi_image_free(data);
    return 0;
}