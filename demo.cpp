#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace std;
using namespace cv;

struct Object{
    cv::Rect rec;
    int      class_id;
    float    prob;
};



float expit(float x) {
    return 1.f / (1.f + expf(-x));
}


//nms
float iou(Rect& rectA, Rect& rectB)
{
    int x1 = std::max(rectA.x, rectB.x);
    int y1 = std::max(rectA.y, rectB.y);
    int x2 = std::min(rectA.x + rectA.width, rectB.x + rectB.width);
    int y2 = std::min(rectA.y + rectA.height, rectB.y + rectB.height);
    int w = std::max(0, (x2 - x1 + 1));
    int h = std::max(0, (y2 - y1 + 1));
    float inter = w * h;
    float areaA = rectA.width * rectA.height;
    float areaB = rectB.width * rectB.height;
    float o = inter / (areaA + areaB - inter);
    return (o >= 0) ? o : 0;
}

void nms(vector<Object>& boxes,  const double nms_threshold)
{
		vector<int> scores;
    for(int i = 0; i < boxes.size();i++){
			scores.push_back(boxes[i].prob);
		} 
		vector<int> index;
    for(int i = 0; i < scores.size(); ++i){
        index.push_back(i);
    }
		sort(index.begin(), index.end(), [&](int a, int b){
        return scores[a] > scores[b]; }); 
    vector<bool> del(scores.size(), false);
		for(size_t i = 0; i < index.size(); i++){
        if( !del[index[i]]){
            for(size_t j = i+1; j < index.size(); j++){
                if(iou(boxes[index[i]].rec, boxes[index[j]].rec) > nms_threshold){
                    del[index[j]] = true;
                }
            }
        }
    }
		vector<Object> new_obj;
    for(const auto i : index){
				Object obj;
				if(!del[i])
				{
					obj.class_id = boxes[i].class_id;
					obj.rec.x =  boxes[i].rec.x;
					obj.rec.y =  boxes[i].rec.y;
					obj.rec.width =  boxes[i].rec.width;
					obj.rec.height =  boxes[i].rec.height;
					obj.prob =  boxes[i].prob;
				}
				new_obj.push_back(obj);

        
    }
    boxes = new_obj;

    
}

void test() {
 
		// Load model
		std::unique_ptr<tflite::FlatBufferModel> model =
		tflite::FlatBufferModel::BuildFromFile("../detect.tflite");
		// Build the interpreter
		tflite::ops::builtin::BuiltinOpResolver resolver;
		std::unique_ptr<tflite::Interpreter> interpreter;
		tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

		// Resize input tensors, if desired.

		TfLiteTensor* output_locations = nullptr;
		TfLiteTensor* output_classes = nullptr;
		TfLiteTensor* num_detections = nullptr;
		// TfLiteTensor* scores = nullptr;
		//auto cam = cv::VideoCapture(0);
		auto cam = cv::VideoCapture("../car.jpeg");

		std::vector<std::string> labels;

		auto file_name="../labelmap.txt";
		std::ifstream input( file_name );
		for( std::string line; getline( input, line ); )
		{
			labels.push_back( line);

		}

		auto cam_width =cam.get(CAP_PROP_FRAME_WIDTH);
		auto cam_height = cam.get(CAP_PROP_FRAME_HEIGHT);
		while (true) {
			cv::Mat image0;
			auto success = cam.read(image0);
			if (!success) {
				std::cout << "cam fail" << std::endl;
				break;
			}
			cv::Mat image;
			resize(image0, image, Size(300,300));
			interpreter->AllocateTensors();

			uchar* input = interpreter->typed_input_tensor<uchar>(0);

			// feed input
			auto image_height=image.rows;
			auto image_width=image.cols;
			auto image_channels=3;
			int number_of_pixels = image_height * image_width * image_channels;
			int base_index = 0;
			// copy image to input as input tensor
			memcpy(interpreter->typed_input_tensor<uchar>(0), image.data, image.total() * image.elemSize());
			interpreter->SetAllowFp16PrecisionForFp32(true);

			interpreter->SetNumThreads(6);

			interpreter->Invoke();
			output_locations = interpreter->tensor(interpreter->outputs()[0]);
			auto output_data = output_locations->data.f;
			std::vector<float> locations;
			std::vector<float> cls;

			output_classes   = interpreter->tensor(interpreter->outputs()[1]);
			auto out_cls = output_classes->data.f;
			num_detections   = interpreter->tensor(interpreter->outputs()[3]);
			auto nums = num_detections->data.f;
			for (int i = 0; i < 20; i++){
				auto output = output_data[i];
				locations.push_back(output);
				cls.push_back(out_cls[i]);
			}
			int count=0;
			std::vector<Object> objects;

			for(int j = 0; j <locations.size(); j+=4){
					auto ymin=locations[j]*cam_height;
					auto xmin=locations[j+1]*cam_width;
					auto ymax=locations[j+2]*cam_height;
					auto xmax=locations[j+3]*cam_width;
					auto width= xmax - xmin;
					auto height= ymax - ymin;
					
					// auto rec = Rect(xmin, ymin, width, height);
				
					float score = expit(nums[count]); // How has this to be done?
					// std::cout << "score: "<< score << std::endl;
					// if (score < 0.5f) continue;
				
					// auto id=outputClasses;
					Object object;
					object.class_id = cls[count];
					object.rec.x = xmin;
					object.rec.y = ymin;
					object.rec.width = width;
					object.rec.height = height;
					object.prob = score;
					objects.push_back(object);

					count+=1;
				

				}
			nms(objects,0.5);
			RNG rng(12345);
			std::cout << "size: "<<objects.size() << std::endl;
			for(int l = 0; l < objects.size(); l++)
     		 {			
	

					Object object = objects.at(l);
					auto score=object.prob;
					if (score < 0.60f) continue;
					Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
					auto cls = object.class_id;

					cv::rectangle(image0, object.rec,color, 1);
					cv::putText(image0, labels[cls+1], cv::Point(object.rec.x, object.rec.y - 5),
					cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(10, 255, 30));
					std::cout<< cls<< std::endl;

	
			}
			
			cv::imshow("cam", image0);
			auto k = cv::waitKey(3000);
			if (k != 255) {
					break;
			}
		}


}
int main(int argc, char** argv) {
    test();
    return 0;
}
