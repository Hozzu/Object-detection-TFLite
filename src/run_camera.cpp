#include <iostream>
#include <fstream>
#include <cstddef>
#include <cstdio>
#include <unistd.h>
#include <chrono>
#include <string>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/c/c_api_types.h>

#include <opencv2/opencv.hpp>
#include <fastcv/fastcv.h>
#include <qcarcam_client.h>

static tflite::Interpreter * interpreter;

static std::string labels[200]; // label id must be less than 200
static int label_num = 0;

static unsigned long average_inference_time = 0;
static unsigned int num_inference = 0;

static int input_height;
static int input_width;
static int input_channel;

static int output_box_idx;
static int output_class_idx;
static int output_score_idx;
static int output_num_idx;

// Callback function called when the camera frame is refreshed
void qcarcam_event_handler(qcarcam_input_desc_t input_id, unsigned char* buf_ptr, size_t buf_len){
    // Get the camera info
    unsigned int queryNumInputs = 0, queryFilled = 0;
    qcarcam_input_t * pInputs;

    if(qcarcam_query_inputs(NULL, 0, &queryNumInputs) != QCARCAM_RET_OK || queryNumInputs == 0){
        std::cout << "ERROR: The camera is not found.\n";
        exit(-1);
    }

    pInputs = (qcarcam_input_t *)calloc(queryNumInputs, sizeof(*pInputs));       
    if(!pInputs){
        std::cout << "ERROR: Failed to calloc\n";
        exit(-1);
    }

    if(qcarcam_query_inputs(pInputs, queryNumInputs, &queryFilled) != QCARCAM_RET_OK || queryFilled != queryNumInputs){
        std::cout << "ERROR: Failed to get the camera info\n";
        exit(-1);
    }

    int camera_height = pInputs[input_id].res[0].height;
    int camera_width = pInputs[input_id].res[0].width;

    free(pInputs);

    // Change color format from uyuv to rgb
    uint8_t * uv = (uint8_t *)fcvMemAlloc(camera_width * camera_height, 16);
    uint8_t * y = (uint8_t *)fcvMemAlloc(camera_width * camera_height, 16);
    if(uv == NULL || y == NULL){
        std::cout << "ERROR: Failed to fcvMemAlloc\n";
        exit(-1);
    }

    uint8_t * rgb_buf_ptr = new unsigned char[camera_height * camera_width * 3];
    if(rgb_buf_ptr == NULL){
        std::cout << "ERROR: Failed memory allocation\n";
        exit(-1);
    }

    fcvDeinterleaveu8(buf_ptr, camera_width, camera_height, camera_width * 2, (uint8_t *)uv, camera_width, (uint8_t *)y, camera_width);
    fcvColorYCbCr422PseudoPlanarToRGB888u8((uint8_t *)y, (uint8_t *)uv, camera_width, camera_height, camera_width, camera_width, (uint8_t *)rgb_buf_ptr, camera_width * 3);

    // Resize image
    uint8_t * r_buf_ptr = new unsigned char[camera_height * camera_width];
    uint8_t * g_buf_ptr = new unsigned char[camera_height * camera_width];
    uint8_t * b_buf_ptr = new unsigned char[camera_height * camera_width];

    for(int i = 0; i < camera_height * camera_width; i++){
        r_buf_ptr[i] = rgb_buf_ptr[3 * i];
        g_buf_ptr[i] = rgb_buf_ptr[3 * i + 1];
        b_buf_ptr[i] = rgb_buf_ptr[3 * i + 2];
    }

    unsigned char * resize_img_ptr = new unsigned char[input_height * input_width * input_channel];
    unsigned char * r_resize_img_ptr = new unsigned char[input_height * input_width];
    unsigned char * g_resize_img_ptr = new unsigned char[input_height * input_width];
    unsigned char * b_resize_img_ptr = new unsigned char[input_height * input_width];
    fcvScaleu8(r_buf_ptr, camera_width, camera_height, camera_width, r_resize_img_ptr, input_width, input_height, input_width, FASTCV_INTERPOLATION_TYPE_BILINEAR);
    fcvScaleu8(g_buf_ptr, camera_width, camera_height, camera_width, g_resize_img_ptr, input_width, input_height, input_width, FASTCV_INTERPOLATION_TYPE_BILINEAR);
    fcvScaleu8(b_buf_ptr, camera_width, camera_height, camera_width, b_resize_img_ptr, input_width, input_height, input_width, FASTCV_INTERPOLATION_TYPE_BILINEAR);

    for(int i = 0; i < input_height * input_height; i++){
        resize_img_ptr[3 * i] = r_resize_img_ptr[i];
        resize_img_ptr[3 * i + 1] = g_resize_img_ptr[i];
        resize_img_ptr[3 * i + 2] = b_resize_img_ptr[i];
    }

    delete r_buf_ptr, g_buf_ptr, b_buf_ptr, r_resize_img_ptr, g_resize_img_ptr, b_resize_img_ptr;

    // Inference
    if(interpreter->input_tensor(0)->type == kTfLiteFloat32){
        float * f_resize_img_ptr = new float[input_height * input_width * input_channel];

        for(int i = 0; i < input_height * input_width * input_channel; i++){
            double value = ((double)resize_img_ptr[i] - 127.5) / 127.5;
            f_resize_img_ptr[i] = (float) value;
        }

        memcpy(interpreter->typed_input_tensor<float>(0), f_resize_img_ptr, input_height * input_width * input_channel * sizeof(float));
        
        delete f_resize_img_ptr;
    }
    else if(interpreter->input_tensor(0)->type == kTfLiteUInt8){
        memcpy(interpreter->typed_input_tensor<unsigned char>(0), resize_img_ptr, input_height * input_width * input_channel * sizeof(unsigned char));
    }

    auto start = std::chrono::high_resolution_clock::now();
    if(interpreter->Invoke() != kTfLiteOk){
        std::cout << "ERROR: Model execute failed\n";
        exit(-1);
    }
    auto elapsed = std::chrono::high_resolution_clock::now() - start;

    average_inference_time = (average_inference_time * num_inference + std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()) / (num_inference + 1);
    num_inference++;

    // Output of inference
    float * output_locations = interpreter->typed_output_tensor<float>(output_box_idx);
    float * output_classes = interpreter->typed_output_tensor<float>(output_class_idx);
    float * output_scores = interpreter->typed_output_tensor<float>(output_score_idx);
    int output_nums = (int) *(interpreter->typed_output_tensor<float>(output_num_idx));

    // Change color format from rgb to bgr
    for(int i = 0; i < camera_width * camera_height * 3; i += 3){
        unsigned char tmp = rgb_buf_ptr[i];
        rgb_buf_ptr[i] = rgb_buf_ptr[i + 2];
        rgb_buf_ptr[i + 2] = tmp;
    }

    // Draw rectangles
    cv::Mat cvimg(camera_height, camera_width, CV_8UC3, rgb_buf_ptr);
    
    for (int i = 0; i < output_nums; i++){
        //std::cout << i <<  ": , output_classes: " << output_classes[i] << ", output_scores: " << output_scores[i] << ", output_locations: [" << output_locations[i * 4] << "," << output_locations[i * 4 + 1] << "," << output_locations[i * 4 + 2] << ","<< output_locations[i * 4 + 3] << "]\n";

        float score =  output_scores[i];
        if(score < 0.5)
            continue;

        int ymin = output_locations[i * 4] * camera_height;
        int xmin = output_locations[i * 4 + 1] * camera_width;
        int ymax = output_locations[i * 4 + 2] * camera_height;
        int xmax = output_locations[i * 4 + 3] * camera_width;

        int id =  (int)(output_classes[i]) + 1;

        char str[100]; 
        sprintf(str, "class: %s, prob: %.1f", labels[id].c_str(), score);
        cv::putText(cvimg, str, cv::Point(xmin, ymin), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));
        cv::rectangle(cvimg, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), cv::Scalar(0, 0, 255), 2);
    }

    memcpy(rgb_buf_ptr, cvimg.data, camera_width * camera_height * 3 * sizeof(unsigned char));

    // Change color format from bgr to uyuv
    fcvColorRGB888ToYCbCr422PseudoPlanaru8(rgb_buf_ptr, camera_width, camera_height, camera_width * 3, y, uv, camera_width, camera_width);
    fcvInterleaveu8(uv, y, camera_width, camera_height, camera_width, camera_width, buf_ptr, camera_width * 2);

    // Free memory
    delete rgb_buf_ptr;
    delete resize_img_ptr;

    if(uv)
        fcvMemFree(uv);
    if(y)
        fcvMemFree(y);
}

bool run_qcarcam(tflite::Interpreter * interpreter_arg, char * label_path, char * display_path){
    // Parse the label
    std::ifstream labelfile(label_path);
    if(!labelfile.is_open()){
        std::cout << "ERROR: Cannot open the label file.\n";
        return false;
    }

    while(true){
        label_num++;
        std::string line_string;

        if(std::getline(labelfile, line_string)){
            labels[label_num] = line_string;
        }
        else
            break;
    }

    labelfile.close();

    // Get the input tensor info
    interpreter = interpreter_arg;

    TfLiteTensor* input_tensor_0 = interpreter->input_tensor(0);

    TfLiteIntArray* input_dims = input_tensor_0->dims;
    input_height = input_dims->data[1];
    input_width = input_dims->data[2];
    input_channel = input_dims->data[3];

    // Get the output tensor info

    // Default output idx
    output_box_idx = 0;
    output_class_idx = 1;
    output_score_idx = 2;
    output_num_idx = 3;

    // Parse the output idx if possible
    for(int i = 0; i < interpreter->outputs().size(); i++){
        if(strstr(interpreter->GetOutputName(i), "box")){
            output_box_idx = i;
        }
        else if(strstr(interpreter->GetOutputName(i), "class")){
            output_class_idx = i;
        }
        else if(strstr(interpreter->GetOutputName(i), "score")){
            output_score_idx = i;
        }
        else if(strstr(interpreter->GetOutputName(i), "num")){
            output_num_idx = i;
        }
    }

    // Run qcarcam
    if(qcarcam_client_start_preview(display_path, qcarcam_event_handler) != QCARCAM_RET_OK){
        std::cout << "ERROR: Cannot connect to the qcarcam. Please check the display setting file.\n";
        return false;
    }

    // Wait the exit
    std::cout << "\nPress ctrl+c to exit.\n\n";
    int secs = 0;
    while (true){
        sleep(10);
        secs += 10;
        std::cout << std::fixed;
        std::cout.precision(3);
        std::cout << "Average inference speed(0~" << secs << "s): " << 1000000.0 / average_inference_time << "fps\n";
        std::cout << "Average inference speed(0~" << secs << "s): " << average_inference_time << "us\n\n";
    }

    // Stop qcarcam
    if(qcarcam_client_stop_preview() != QCARCAM_RET_OK){
        std::cout << "ERROR: Cannot disconnect the qcarcam.\n";
        return false;
    }

    return true;
}