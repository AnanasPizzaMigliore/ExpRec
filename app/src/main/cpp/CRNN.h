//
// Created by TifloUser on 1/9/2025.
//

#ifndef CRNN_H
#define CRNN_H

#include "net.h"
#include "YoloV5.h"

class CRNN {
public:
    CRNN(AAssetManager *mgr, const char *param, const char *bin, bool useGPU);


    ~CRNN();

    std::string recognition(JNIEnv *env, jobject image);

private:

    ncnn::Net *Net;

public:
    static CRNN *recognizer;
    static bool hasGPU;

};

#endif //CRNN_H
