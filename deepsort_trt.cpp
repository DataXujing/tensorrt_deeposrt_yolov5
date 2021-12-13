//deepsort_trt
//该项目基于deepsort实现目标跟踪
//deepsort中的ReID算法采用TensorRT加速推断
//检测算法我们采用了YOLOV5同样使用TensorRT急速推断
//检测算法与DeepSort是解耦合的，检测算法可以替换为任何算法



#include "TrtTracker.h"
using std::vector;

float h_input[INPUT_SIZE * INPUT_SIZE * 3]; //input_yolov5
float h_output_1[1 * 25200 * 85];  //output_yolov5

int main()
{
    TrtTracker* tracker = new TrtTracker("./model/deepsort.engine");
	yolov5* yolov5x = new yolov5();

	//直接从Engine反序列化
	// 如果基于序列化的engine,直接在engine文件中反序列化
	nvinfer1::IRuntime* engine_runtime = nvinfer1::createInferRuntime(gLogger);
	std::string engine_filepath = "./model/yolov5s.engine";

	std::ifstream file;
	file.open(engine_filepath, std::ios::binary | std::ios::in);
	file.seekg(0, std::ios::end);
	int length = file.tellg();
	file.seekg(0, std::ios::beg);

	std::shared_ptr<char> data(new char[length], std::default_delete<char[]>());
	file.read(data.get(), length);
	file.close();

	//nvinfer1::ICudaEngine* engine_infer = engine_runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
	nvinfer1::ICudaEngine* engine_infer = engine_runtime->deserializeCudaEngine(data.get(), length, nullptr);
	nvinfer1::IExecutionContext* engine_context = engine_infer->createExecutionContext();

	int input_index = engine_infer->getBindingIndex("input_yolov5"); //1x3x640x640
	//std::string input_name = engine_infer->getBindingName(0)
	int output_index_1 = engine_infer->getBindingIndex("output_yolov5");
	;

	std::cout << "输入的index: " << input_index << " 输出的index:output-> " << output_index_1 << std::endl;

	if (engine_context == nullptr)
	{
		std::cerr << "Failed to create TensorRT Execution Context." << std::endl;
	}

	// cached_engine->destroy();
	std::cout << "loaded trt model , do inference" << std::endl;


	cv::VideoCapture capture("test.mp4");//打开视频文件

	//保存视频
	cv::VideoWriter writer;
	int coder = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');//选择编码格式

	double fps = 25.0;
	std::string savefile = "out.mp4";
	writer.open(savefile, coder, fps, cv::Size(1280, 720), true);

	clock_t startTime, endTime;

	int i = 0;
	while (1) {//循环读取图片
		i++;
		cv::Mat frame;//局部变量
		capture >> frame;//读取一帧
		if (frame.empty()) break;

		
		//startTime = clock();//计时开始
		yolov5x->preprocess(frame, h_input);
		void* buffers[2];
		cudaMalloc(&buffers[0], INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));  //<- image
		cudaMalloc(&buffers[1], 1 * 25200 * 85 * sizeof(float)); //<- output
		//endTime = clock();//计时结束
	

	
		cudaMemcpy(buffers[0], h_input, INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float), cudaMemcpyHostToDevice);
		engine_context->executeV2(buffers);

		cudaMemcpy(h_output_1, buffers[1], 1 * 25200 * 85 * sizeof(float), cudaMemcpyDeviceToHost);

		std::vector<Bbox> bboxes;
		yolov5x->pred2box(h_output_1, bboxes);

		yolov5x->NmsDetect(bboxes);
		std::vector<Bbox> pred_boxs;
		pred_boxs = yolov5x->rescale_box(bboxes, frame.cols, frame.rows);
		
		//std::cout << "YOLOv5 推断时间（包括前处理，推断，后处理）：" << (double)(endTime - startTime) / CLOCKS_PER_SEC << " s" << std::endl;
		//startTime = clock();
		std::vector<DetectBox> sortBox = tracker->getDetections(pred_boxs, i);
		tracker->run(sortBox, frame);
		//endTime = clock();

		//std::cout << "DeepSort 推断时间（包括前处理，推断，后处理,可视化）：" << (double)(endTime - startTime) / CLOCKS_PER_SEC << " s" << std::endl;


		//frame = yolov5x->renderBoundingBox(frame, pred_boxs);
		writer.write(frame);

		cv::imshow("Image", frame);//显示图像

		cudaFree(buffers[0]);
		cudaFree(buffers[1]);

		
		cv::waitKey(1);


	}
	capture.release();

	engine_runtime->destroy();
	engine_infer->destroy();





	return 0;



}

