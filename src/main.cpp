#include <iostream>
#include <cstring>
#include <cstddef>

#include <tensorflow/lite/model_builder.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/interpreter_builder.h>
#include <tensorflow/lite/delegates/hexagon/hexagon_delegate.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/c/c_api_types.h>

bool run_qcarcam(tflite::Interpreter * interpreter_arg, char * label_path, char * display_path);
bool run_image(tflite::Interpreter * interpreter, char * label_path, char * directory_path, char * result_path);

int main(int argc, char ** argv){
    //usage guide
    if(strcmp(argv[1], "-help") == 0 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "--h") == 0){
        std::cout << "Usage: pkshin_detect camera [MODEL] [LABEL] [DISPLAY] [ACCELERATOR]\n";
        std::cout << "camera mode runs the object detection using qcarcam API.\n";
        std::cout << "[MODEL] is path of the model file.\n";
        std::cout << "[LABEL] is path of the label file.\n";
        std::cout << "[DISPLAY] is path of the file defining the display setting.\n\n";
        std::cout << "[ACCELERATOR] specifies the accelerator to run the inference. CPU, GPU, NPU is supported. Default value is CPU.\n\n";
        std::cout << "Usage: pkshin_detect image [MODEL] [LABEL] [IMG_DIR] [RESULT] [ACCELERATOR]\n";
        std::cout << "image mode runs the object detection with jpeg images.\n";
        std::cout << "[MODEL] is path of the model file.\n";
        std::cout << "[LABEL] is path of the label file.\n";
        std::cout << "[IMG_DIR] is path of the directory containing images.\n";
        std::cout << "[RESULT] is path of the result json file.\n\n";
        std::cout << "[ACCELERATOR] specifies the accelerator to run the inference. CPU, GPU, NPU is supported. Default value is CPU.\n\n";
        return true;
    }

    // Argument error checking
    if( (strcmp(argv[1], "camera") != 0 && strcmp(argv[1], "image") != 0) || (strcmp(argv[1], "camera") == 0 && argc < 5) || (strcmp(argv[1], "image") == 0 && argc < 6) ){
        std::cout << "ERROR: The first argument must be camera or image. camera mode requires at least 3 more arguments and image mode requires at least 4 more arguments\n\n";
        std::cout << "Usage: pkshin_detect camera [MODEL] [LABEL] [DISPLAY] [ACCELERATOR]\n";
        std::cout << "camera mode runs the object detection using qcarcam API.\n";
        std::cout << "[MODEL] is path of the model file.\n";
        std::cout << "[LABEL] is path of the label file.\n";
        std::cout << "[DISPLAY] is path of the file defining the display setting.\n\n";
        std::cout << "[ACCELERATOR] specifies the accelerator to run the inference. CPU, GPU, NPU is supported. Default value is CPU.\n\n";
        std::cout << "Usage: pkshin_detect image [MODEL] [LABEL] [IMG_DIR] [RESULT] [ACCELERATOR]\n";
        std::cout << "image mode runs the object detection with jpeg images.\n";
        std::cout << "[MODEL] is path of the model file.\n";
        std::cout << "[LABEL] is path of the label file.\n";
        std::cout << "[IMG_DIR] is path of the directory containing images.\n";
        std::cout << "[RESULT] is path of the result json file.\n\n";
        std::cout << "[ACCELERATOR] specifies the accelerator to run the inference. CPU, GPU, NPU is supported. Default value is CPU.\n\n";
        return false;
    }

    // Load the model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(argv[2]);

    if(model == NULL){
        std::cout << "ERROR: Model load failed. Check the model name.\n";
        return false;
    }

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    if(interpreter == NULL){
        std::cout << "ERROR: Interpreter build failed.\n";
        return false;
    }

    if(interpreter->AllocateTensors() != kTfLiteOk) {
        std::cout << "ERROR: Memory allocation for interpreter failed.\n";
        return false;
    }

    interpreter->SetNumThreads(4);
    std::cout << "INFO: Interperter uses 4 threads.\n";

    // Check input tensor
    for(int i = 0; i < interpreter->inputs().size(); i++){
        std::cout << "INFO: Graph input " << i << ": " << interpreter->GetInputName(i) << std::endl;
    }

    if(interpreter->input_tensor(0)->type != kTfLiteFloat32 && interpreter->input_tensor(0)->type != kTfLiteUInt8){
        std::cout << "ERROR: Graph input type should be kTfLiteFloat32 or kTfLiteUInt8.\n";
        return false;
    }

    // Check output tensors
    bool output_box = false, output_class = false, output_score = false, output_num = false;

    for(int i = 0; i < interpreter->outputs().size(); i++){
        std::cout << "INFO: Graph output " << i << ": " << interpreter->GetOutputName(i) << std::endl;
    }

    for(int i = 0; i < 4; i++){
        if(interpreter->output_tensor(i)->type != kTfLiteFloat32){
            std::cout << "ERROR: Graph output type should be kTfLiteFloat32.\n";
            return false;
        }
    }

    // Set the delegate
    if( (strcmp(argv[1], "camera") == 0 && argc == 5) || (strcmp(argv[1], "image") == 0 && argc == 6) || (strcmp(argv[1], "camera") == 0 && (strcmp(argv[5], "CPU") == 0 || strcmp(argv[5], "cpu") == 0)) || (strcmp(argv[1], "image") == 0 && (strcmp(argv[6], "CPU") == 0 || strcmp(argv[6], "cpu") == 0)) ){
        // No delegate
        std::cout << "INFO: Run with CPU only.\n";
    }
    else if( (strcmp(argv[1], "camera") == 0 && (strcmp(argv[5], "GPU") == 0 || strcmp(argv[5], "gpu") == 0)) || (strcmp(argv[1], "image") == 0 && (strcmp(argv[6], "GPU") == 0 || strcmp(argv[6], "gpu") == 0)) ){
        TfLiteGpuDelegateOptionsV2 gpu_delegate_options = TfLiteGpuDelegateOptionsV2Default();
        auto * gpu_delegate_ptr = TfLiteGpuDelegateV2Create(&gpu_delegate_options);
        if(gpu_delegate_ptr == NULL){
            std::cout << "WARNING: Cannot create gpu delegate. Run without gpu delegate.\n";
            TfLiteGpuDelegateV2Delete(gpu_delegate_ptr);
        }
        else{
            tflite::Interpreter::TfLiteDelegatePtr gpu_delegate(gpu_delegate_ptr, &TfLiteGpuDelegateV2Delete);

            std::cout << "INFO: Run with gpu delegate.\n";
            if (interpreter->ModifyGraphWithDelegate(gpu_delegate.get()) != kTfLiteOk){
                std::cout << "ERROR: Cannot convert model with gpu delegate\n";
                return false;
            }
        }
    }
    else if( (strcmp(argv[1], "camera") == 0 && (strcmp(argv[5], "NPU") == 0 || strcmp(argv[5], "npu") == 0)) || (strcmp(argv[1], "image") == 0 && (strcmp(argv[6], "NPU") == 0 || strcmp(argv[6], "npu") == 0)) ){
        TfLiteHexagonInitWithPath("/usr/lib");

        TfLiteHexagonDelegateOptions npu_delegate_params = {0};
        auto* npu_delegate_ptr = TfLiteHexagonDelegateCreate(&npu_delegate_params);
        if (npu_delegate_ptr == NULL) {
            TfLiteHexagonDelegateDelete(npu_delegate_ptr);
            TfLiteHexagonTearDown();
            std::cout << "WARNING: Cannot create hexagon delegate. Check whether the hexagon library is in /usr/lib/. Run without hexagon delegate.\n";
        }
        else{
            tflite::Interpreter::TfLiteDelegatePtr npu_delegate(npu_delegate_ptr, [](TfLiteDelegate* npu_delegate) {
                TfLiteHexagonDelegateDelete(npu_delegate);
                TfLiteHexagonTearDown();
            });

            std::cout << "INFO: Run with hexagon delegate.\n";
            if(interpreter->ModifyGraphWithDelegate(npu_delegate.get()) != kTfLiteOk){
                std::cout << "ERROR: Cannot convert model with hexagon delegate\n";
                return false;
            }
        }
    }
    else{
        std::cout << "WARNING: [ACCELERATOR] should be CPU, GPU, or NPU. Run with CPU only.\n";
    }

    // Call the appropriate functons for mode
    if(strcmp(argv[1], "camera") == 0){
        std::cout << "INFO: Running the object detection using qcarcam API.\n";
        return run_qcarcam(interpreter.get(), argv[3], argv[4]);
    }
    else if(strcmp(argv[1], "image") == 0){
        std::cout << "INFO: Running the object detection with jpeg images.\n";
        return run_image(interpreter.get(), argv[3], argv[4], argv[5]);
    }
}