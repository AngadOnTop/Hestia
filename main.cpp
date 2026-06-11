#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

struct Detection {
    cv::Rect box;
    int class_id = -1;
    float confidence = 0.0f;
};

struct LetterboxInfo {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

static std::vector<std::string> loadLabelsFromFile(const std::string& path) {
    std::ifstream file(path);
    std::vector<std::string> labels;
    std::string line;

    while (std::getline(file, line)) {
        if (!line.empty()) labels.push_back(line);
    }

    return labels;
}

static std::vector<std::string> parseLabelsFromMetadata(const std::string& names) {
    std::vector<std::string> labels;
    std::size_t pos = 0;

    while ((pos = names.find('\'', pos)) != std::string::npos) {
        const std::size_t end = names.find('\'', pos + 1);
        if (end == std::string::npos) break;
        labels.push_back(names.substr(pos + 1, end - pos - 1));
        pos = end + 1;
    }

    return labels;
}

static std::vector<std::string> getLabels(const Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    const Ort::ModelMetadata metadata = session.GetModelMetadata();

    if (metadata.LookupCustomMetadataMapAllocated("names", allocator) != nullptr) {
        auto names = metadata.LookupCustomMetadataMapAllocated("names", allocator);
        auto labels = parseLabelsFromMetadata(names.get());
        if (!labels.empty()) return labels;
    }

    return loadLabelsFromFile("classes.txt");
}

static cv::Mat letterbox(const cv::Mat& frame, const cv::Size& target_size, LetterboxInfo& info) {
    const float scale = std::min(
        target_size.width / static_cast<float>(frame.cols),
        target_size.height / static_cast<float>(frame.rows)
    );
    const int resized_w = static_cast<int>(std::round(frame.cols * scale));
    const int resized_h = static_cast<int>(std::round(frame.rows * scale));

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(resized_w, resized_h));

    cv::Mat padded(target_size, CV_8UC3, cv::Scalar(114, 114, 114));
    info.scale = scale;
    info.pad_x = (target_size.width - resized_w) / 2;
    info.pad_y = (target_size.height - resized_h) / 2;
    resized.copyTo(padded(cv::Rect(info.pad_x, info.pad_y, resized_w, resized_h)));

    return padded;
}

static std::vector<float> makeInputTensor(const cv::Mat& frame, const cv::Size& input_size, LetterboxInfo& letterbox_info) {
    cv::Mat input = letterbox(frame, input_size, letterbox_info);
    cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
    input.convertTo(input, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(input, channels);

    std::vector<float> tensor_data(3 * input_size.width * input_size.height);
    const int channel_size = input_size.width * input_size.height;
    for (int c = 0; c < 3; ++c) {
        std::memcpy(tensor_data.data() + c * channel_size, channels[c].data, channel_size * sizeof(float));
    }

    return tensor_data;
}

static std::vector<Detection> runDetections(
    Ort::Session& session,
    const char* input_name,
    const char* output_name,
    const cv::Mat& frame,
    const cv::Size& input_size,
    float confidence_threshold = 0.35f,
    float nms_threshold = 0.45f
) {
    LetterboxInfo letterbox_info;
    std::vector<float> input_tensor = makeInputTensor(frame, input_size, letterbox_info);
    std::array<int64_t, 4> input_shape{1, 3, input_size.height, input_size.width};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_value = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor.data(),
        input_tensor.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = {input_name};
    const char* output_names[] = {output_name};
    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_value,
        1,
        output_names,
        1
    );

    const float* output = outputs.front().GetTensorData<float>();
    const auto output_shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
    const int attributes = static_cast<int>(output_shape[1]);
    const int predictions = static_cast<int>(output_shape[2]);
    const int class_count = attributes - 4;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    for (int i = 0; i < predictions; ++i) {
        float best_score = 0.0f;
        int best_class = -1;

        for (int c = 0; c < class_count; ++c) {
            const float score = output[(4 + c) * predictions + i];
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        if (best_score < confidence_threshold) continue;

        const float cx = output[0 * predictions + i];
        const float cy = output[1 * predictions + i];
        const float w = output[2 * predictions + i];
        const float h = output[3 * predictions + i];

        const float left = (cx - w / 2.0f - letterbox_info.pad_x) / letterbox_info.scale;
        const float top = (cy - h / 2.0f - letterbox_info.pad_y) / letterbox_info.scale;
        const float width = w / letterbox_info.scale;
        const float height = h / letterbox_info.scale;

        cv::Rect box(
            static_cast<int>(std::round(left)),
            static_cast<int>(std::round(top)),
            static_cast<int>(std::round(width)),
            static_cast<int>(std::round(height))
        );
        box &= cv::Rect(0, 0, frame.cols, frame.rows);

        if (box.area() <= 0) continue;
        boxes.push_back(box);
        scores.push_back(best_score);
        class_ids.push_back(best_class);
    }

    std::vector<int> kept_indices;
    cv::dnn::NMSBoxes(boxes, scores, confidence_threshold, nms_threshold, kept_indices);

    std::vector<Detection> detections;
    for (int idx : kept_indices) {
        detections.push_back({boxes[idx], class_ids[idx], scores[idx]});
    }

    return detections;
}

static cv::Scalar colorForClass(int class_id) {
    static const std::array<cv::Scalar, 8> colors{
        cv::Scalar(38, 201, 255),
        cv::Scalar(90, 220, 120),
        cv::Scalar(255, 144, 64),
        cv::Scalar(238, 98, 120),
        cv::Scalar(170, 120, 255),
        cv::Scalar(255, 205, 75),
        cv::Scalar(76, 184, 255),
        cv::Scalar(120, 230, 205)
    };

    return colors[std::max(0, class_id) % colors.size()];
}

static void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections, const std::vector<std::string>& labels) {
    for (const auto& detection : detections) {
        const cv::Scalar color = colorForClass(detection.class_id);
        cv::rectangle(frame, detection.box, color, 2);

        const std::string class_name =
            detection.class_id >= 0 && detection.class_id < static_cast<int>(labels.size())
                ? labels[detection.class_id]
                : "object";
        const std::string text = class_name + " " + cv::format("%.0f%%", detection.confidence * 100.0f);

        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        const int label_y = std::max(detection.box.y, text_size.height + 8);
        const cv::Rect label_bg(
            detection.box.x,
            label_y - text_size.height - 8,
            std::min(text_size.width + 10, frame.cols - detection.box.x),
            text_size.height + baseline + 8
        );

        cv::rectangle(frame, label_bg, color, cv::FILLED);
        cv::putText(
            frame,
            text,
            cv::Point(detection.box.x + 5, label_y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(20, 24, 28),
            1,
            cv::LINE_AA
        );
    }
}

// Returns average pixel brightness (0 = pure black, ~128 = normal image)
static double frameBrightness(const cv::Mat& frame) {
    cv::Scalar mean = cv::mean(frame);
    return (mean[0] + mean[1] + mean[2]) / 3.0;
}

// macOS can "open" a camera but still deliver all-black frames (permissions,
// wrong device, or camera in use). Probe a few frames to see if data is real.
static bool cameraHasPicture(cv::VideoCapture& cap, cv::Mat& frame, int warmup_frames = 15) {
    for (int i = 0; i < warmup_frames; ++i) {
        if (!cap.read(frame) || frame.empty()) return false;
        if (frameBrightness(frame) > 5.0) return true;
    }
    return frameBrightness(frame) > 5.0;
}

static bool openWorkingCamera(cv::VideoCapture& cap, cv::Mat& frame, int preferred_index) {
    const int indices[] = {preferred_index, 0, 1, 2};

    for (int idx : indices) {
        cap.release();
        cap.open(idx, cv::CAP_AVFOUNDATION);
        if (!cap.isOpened()) {
            std::cout << "Camera " << idx << ": not available\n";
            continue;
        }

        std::cout << "Camera " << idx << ": opened "
                  << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
                  << cap.get(cv::CAP_PROP_FRAME_HEIGHT)
                  << ", brightness=" << (cameraHasPicture(cap, frame) ? frameBrightness(frame) : 0.0)
                  << '\n';

        if (!frame.empty() && frameBrightness(frame) > 5.0) {
            std::cout << "Using camera " << idx << '\n';
            return true;
        }

        std::cout << "Camera " << idx << ": black frames (skipped)\n";
    }

    return false;
}

int main(int argc, char* argv[]) {
    int preferred = (argc > 1) ? std::stoi(argv[1]) : 0;
    const std::string model_path = (argc > 2) ? argv[2] : "yolo11n.onnx";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ai_webcam");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    const cv::Size model_input_size(640, 640);
    const std::vector<std::string> labels = getLabels(session);

    cv::VideoCapture cap;
    cv::Mat frame;

    if (!openWorkingCamera(cap, frame, preferred)) {
        std::cerr << "\nNo working camera found.\n";
        std::cerr << "- System Settings -> Privacy & Security -> Camera -> enable Terminal/Cursor\n";
        std::cerr << "- Quit other apps using the camera (FaceTime, Zoom, browser tabs)\n";
        std::cerr << "- Try iPhone camera: ./build/main 1\n";
        return -1;
    }

    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Error: Failed to read frame\n";
            break;
        }

        cv::flip(frame, frame, 1);  // 1 = horizontal mirror (selfie mode)
        const std::vector<Detection> detections = runDetections(
            session,
            input_name.get(),
            output_name.get(),
            frame,
            model_input_size
        );
        drawDetections(frame, detections, labels);

        cv::imshow("Webcam", frame);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
