#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

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
        cv::imshow("Webcam", frame);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}