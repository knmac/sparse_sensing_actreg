#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;

typedef cv::Point3_<double> Point3d;
typedef cv::Point3_<int> Point3i;
typedef cv::Point_<double> Point2d;


/**
 * Camera information structure, containing intrinsic and extrinsic parameters
 * of a frame
 */
struct CameraData {
    int frameID;
    bool valid;
    int width;
    int height;
    int LensModel;
    int ShutterModel;

    cv::Mat K;  // intrinsic
    cv::Mat invK;  // inverse intrinsic
    cv::Mat distortion;  // lens distortion

    double rt[6];  // extrinsic vector
    cv::Mat R;  // rotation matrix
    cv::Mat T;  // translation vector
    cv::Mat P;  // camera matrix
    cv::Mat camCenter;  // camera center
    cv::Mat principleRayDir;  // ray direction
};

/**
 * Data of a whole video
 */
struct VideoData {
    int nframes;
    int start_time;
    int stop_time;
    CameraData *VideoInfo;
};

struct Corpus {
    int nCameras;
    int n3dPoints;
    vector<Point3d> xyz;
    vector<Point3i> rgb;
	vector<int> *threeDIdAllViews;  //2D point in visible view -> 3D index
};
