#include "CImg/CImg.h"
#include <vector>
#include <iostream>

using namespace cimg_library;
using namespace std;

int main() {
    CImg<unsigned char> image("./lena.jpg");
    vector<float> pixels;

    cimg_forXY(image, x, y) {
        // 将像素值转换为浮点数并存储到vector中
        pixels.push_back(static_cast<float>(image(x, y, 0, 0)));
        pixels.push_back(static_cast<float>(image(x, y, 0, 1)));
        pixels.push_back(static_cast<float>(image(x, y, 0, 2)));
    }

    // 打印vector中的值
    for (const auto &pixel : pixels) {
        cout << pixel << ' ';
    }
    cout << endl;

    return 0;
}