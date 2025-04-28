//
// Created by TifloUser on 12/18/2024.
//
#include "CRNN.h"

bool CRNN::hasGPU = true;
CRNN* CRNN::recognizer = nullptr;

CRNN::CRNN(AAssetManager *mgr, const char *param, const char *bin, bool useGPU) {
    this->Net = new ncnn::Net();
    hasGPU = ncnn::get_gpu_count() > 0;
    this->Net->opt.use_vulkan_compute = useGPU; //hasGPU && useGPU;  // gpu
    this->Net->opt.use_fp16_arithmetic = true;
    this->Net->opt.use_fp16_packed = true;
    this->Net->opt.use_fp16_storage = true;
    this->Net->load_param(mgr,param);
    this->Net->load_model(mgr,bin);
}

CRNN::~CRNN()
{
    delete this->Net;
}

std::string CRNN::recognition(JNIEnv *env, jobject image) {
    ncnn::Mat input;
    input = ncnn::Mat::from_android_bitmap_resize(env, image, ncnn::Mat::PIXEL_RGBA2RGB, 128, 32);
    const float mean_vals[3] = { 127.5, 127.5, 127.5 };
    const float norm_vals[3] = { 1/127.5, 1/127.5, 1/127.5 };

    input.substract_mean_normalize(mean_vals, norm_vals);
    auto ex = this->Net->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    //hasGPU = ncnn::get_gpu_count() > 0;
    //ex.set_vulkan_compute(hasGPU);
    ex.input("in0", input);
    ncnn::Mat output;
    ex.extract("out0", output);

    const std::string charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    std::vector<int> all_ids;
    std::vector<int> unique;

    for (int i = 0; i < output.h; ++i) {
        auto channel = output.row(i);
        auto value = channel[0];
        int max_id = 0;
        for (int j = 0; j < output.w; ++j){
            if (channel[j] > value){
                value = channel[j];
                max_id = j;
            }
        }
        all_ids.push_back(max_id);
    }

    std::vector<int> result = { all_ids[0] };
    for (size_t i = 1; i < all_ids.size(); ++i) {
        if (all_ids[i] != result.back()) {
            result.push_back(all_ids[i]);
        }
    }

    std::vector<int> ids;
    for (int id : result) {
        if (id != 0) {
            ids.push_back(id);
        }
    }

    std::string tokens;
    for (int id : ids) {
        tokens += charset[id - 1];
    }

    return tokens;
}