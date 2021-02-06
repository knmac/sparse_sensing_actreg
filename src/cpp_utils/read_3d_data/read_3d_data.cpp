#include <iostream>
#include "read_3d_data.hpp"

using namespace std;


// ============================================================================
// Helper functions
// ============================================================================
/**
 * Find the maximum number of points in a video sequence to allocate memory
 */
int get_max_n_points(char *fname) {
    int n_points = -1;
    FILE *fp = fopen(fname, "r");

    int pid;
    double x, y, z;
    double u, v;
    int foo;
    double bar;

    while (fscanf(fp, "%d %d %lf %lf %lf %lf %lf %lf",
                       &pid, &foo, &x, &y, &z, &u, &v, &bar) != EOF) {
        if (pid > n_points)
            n_points = pid;
    }
    fclose(fp);

    return n_points;
}


/**
 * Invert the intrinsic matrix
 */
int invert_intrinsic(double *K, cv::Mat &invK) {
    double fx = K[0], skew = K[1], u0 = K[2];
    double fy = K[4], v0 = K[5];
    double fxfy = fx * fy;

    double data[9] = {fy / fxfy, -skew / fxfy, (v0*skew - u0*fy) / fxfy,
                      0.0,       fx / fxfy,    (-v0*fx) / fxfy,
                      0.0,       0.0,          1.0};
    invK = cv::Mat(3, 3, CV_64F, data).clone();

    return 0;
}


/**
 * Get the rotation matrix and translation vector from rt
 */
int get_rotation_translation(double *rt, CameraData &camera) {
	cv::Mat Rmat(3, 3, CV_64F), rvec(3, 1, CV_64F);
	for (int jj = 0; jj < 3; jj++)
		rvec.at<double>(jj) = camera.rt[jj];

	cv::Rodrigues(rvec, Rmat);

    camera.R = Rmat.clone();
    camera.T = cv::Mat(3, 1, CV_64F, rt+3).clone();

    return 0;
}


/**
 * Assemble the camera matrix P = K * [R|T]
 */
int assemble_camera_matrix(CameraData &camera) {
    cv::Mat RT;
    cv::hconcat(camera.R, camera.T, RT);
    camera.P = camera.K * RT;

    return 0;
}


/**
 * Get the camera center location
 */
int get_cam_center(CameraData &camera) {
    cv::Mat R_trans;

    cv::transpose(camera.R, R_trans);
    camera.camCenter = -(R_trans * camera.T);

    return 0;
}


/**
 * Compute the principle ray direction
 */
int get_ray_dir(CameraData &camera, cv::Mat uv1) {
    cv::Mat R_trans;
    cv::Mat rayDir;

    cv::transpose(camera.R, R_trans);
    rayDir = R_trans * camera.invK;
    rayDir = rayDir * uv1;
    rayDir = rayDir / cv::norm(rayDir);
    camera.principleRayDir = rayDir;

    return 0;
}


// ============================================================================
// Main functions to call
// ============================================================================
/**
 * Read inlier data
 *
 * @param fname: (input) path containing the data
 *
 * @param ptid: (output) vector of available point id
 * @param pt3d: (output) array of 3D locations (unavailable points are zeros)
 * @param pt2d: (output) array of 2D locations (unavailable points are zeros)
 */
int read_inliner(char *fname, vector<int> &ptid, Point3d *pt3d, Point2d *pt2d) {
    FILE *fp = fopen(fname, "r");

    int pid;
    double x, y, z;
    double u, v;
    int foo;
    double bar;

    while (fscanf(fp, "%d %d %lf %lf %lf %lf %lf %lf",
                      &pid, &foo, &x, &y, &z, &u, &v, &bar) != EOF) {
        ptid.push_back(pid);
        pt3d[pid] = Point3d(x, y, z);
        pt2d[pid] = Point2d(u, v);
    }
    fclose(fp);

    return 0;
}


/**
 * Read intrinsic and extrinsic parameters of a frame and compute the camera
 * matrix and principle ray direction.
 *
 * The path must contains:
 * path/
 * ├── Intrinsic_[view_id].txt
 * └── CamPose_[view_id].txt
 *
 * @param path: (input) path to the video, containing intrinsic and extrinsic files
 * @param view_id: (input) id of the view (should be 0)
 * @param startF: (input) starting frame index
 * @param stopF: (input) stopping frame index
 *
 * @param vInfo: (output) video information
 */
int read_intrinsic_extrinsic(char *path, int view_id, int startF, int stopF, VideoData &vInfo) {
    const int RADIAL_TANGENTIAL_PRISM = 0;

    // Generate vInfo object
    vInfo.start_time = startF;
    vInfo.stop_time = stopF;
    vInfo.nframes = stopF + 1;
    vInfo.VideoInfo = new CameraData[stopF+1];
    for (int i=0; i<stopF+1; ++i) {
        vInfo.VideoInfo[i].valid = false;
    }

    // ------------------------------------------------------------------------
    // Read intrinsic parameters
	char fname[512];
	sprintf(fname, "%s/Intrinsic_%.4d.txt", path, view_id);

    FILE *fp = fopen(fname, "r");
    int frameID, LensType, ShutterModel;
    int width, height;
    double fx, fy, skew;
    double u0, v0;
    double r0, r1, r2, t0, t1, p0, p1;
    
    while (fscanf(fp, "%d %d %d %d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                      &frameID, &LensType, &ShutterModel, &width, &height,
                      &fx, &fy, &skew, &u0, &v0,
                      &r0, &r1, &r2, &t0, &t1, &p0, &p1) != EOF) {

        if ((startF <= frameID) && (frameID <= stopF)) {
            double _K[9] = {fx,  skew, u0,
                            0.0, fy,   v0,
                            0.0, 0.0,  1.0};
            vInfo.VideoInfo[frameID].K = cv::Mat(3, 3, CV_64F, _K).clone();

			vInfo.VideoInfo[frameID].frameID = frameID;
			vInfo.VideoInfo[frameID].width = width;
            vInfo.VideoInfo[frameID].height = height;
            invert_intrinsic(_K, vInfo.VideoInfo[frameID].invK);
            vInfo.VideoInfo[frameID].LensModel = LensType;
            vInfo.VideoInfo[frameID].ShutterModel = ShutterModel;

            if (LensType == RADIAL_TANGENTIAL_PRISM) {
                double _distort[7] = {r0, r1, r2, t0, t1, p0, p1};
                vInfo.VideoInfo[frameID].distortion = cv::Mat(1, 7, CV_64F, _distort).clone();
            }
        }

        if (frameID > stopF) {
            break;
        }
    }
    fclose(fp);

    // ------------------------------------------------------------------------
    // Read extrinsic parameters
	sprintf(fname, "%s/CamPose_%.4d.txt", path, view_id);

    fp = fopen(fname, "r");
    double rt[6];
    while (fscanf(fp, "%d %lf %lf %lf %lf %lf %lf",
                      &frameID, &rt[0], &rt[1], &rt[2], &rt[3], &rt[4], &rt[5]) != EOF) {
        if ((startF <= frameID) && (frameID <= stopF)) {
            if (abs(rt[3]) + abs(rt[4]) + abs(rt[5]) < 0.001) {
                vInfo.VideoInfo[frameID].valid = false;
                continue;
            }

            vInfo.VideoInfo[frameID].valid = true;
            copy(rt, rt+6, vInfo.VideoInfo[frameID].rt);  // store rt array
            get_rotation_translation(rt, vInfo.VideoInfo[frameID]);  // get R, T
            assemble_camera_matrix(vInfo.VideoInfo[frameID]);  // get P
            get_cam_center(vInfo.VideoInfo[frameID]);

            double _principal[3] = {vInfo.VideoInfo[frameID].width / 2.0,
                                   vInfo.VideoInfo[frameID].height / 2.0,
                                   1.0};
            cv::Mat principal = cv::Mat(3, 1, CV_64F, _principal);
            get_ray_dir(vInfo.VideoInfo[frameID], principal);
        }

        if (frameID > stopF) {
            break;
        }
    }
    fclose(fp);

    return 0;
}


/**
 * Project depth from world plane to camera plane for a new dimensionality.
 * The depth will be rescaled to milimeters using sfm_dist and real_dist.
 * The results are printed out for streaming.
 *
 * @param ptid: (input) vector of available point id
 * @param pt3d: (input) array of 3D locations (unavailable points are zeros)
 * @param pt2d: (input) array of 2D locations (unavailable points are zeros)
 * @param frame_info: (input) video information of that frame
 * @param new_h, new_w: (input) new dimensionality to scale to
 * @param sfm_dist, real_dist: (input) distance to normalize to mm unit
 *
 * @param img: (output) projected depth as a (new_h, new_w) matrix
 * @param projection: (output) projected 2D location on img as an array,
 *      corresponding to point id
 */
int project_depth(vector<int> ptid, Point3d *pt3d, Point2d *pt2d,
                  CameraData frame_info, int new_h, int new_w, double sfm_dist,
                  double real_dist, cv::Mat &img, Point2d *projection) {
    double scale_h = double(frame_info.height) / double(new_h);
    double scale_w = double(frame_info.width) / double(new_w);

    double u, v;
    int k;
    Point3d cam_center(frame_info.camCenter);
    Point3d principle_ray_dir(frame_info.principleRayDir);
    Point3d cam2point;
    double depth;
    for (int i=0; i<ptid.size(); ++i) {
        k = ptid.at(i);
        u = int(round(pt2d[k].x / scale_w));
        v = int(round(pt2d[k].y / scale_h));
        if (u<0 || u>=new_w || v<0 || v>=new_h)
            continue;

        cam2point = pt3d[k] - cam_center;
        depth = cam2point.dot(principle_ray_dir);
        depth = depth / sfm_dist * real_dist;  // convert to mm unit

        img.at<double>(v, u) = depth;
        projection[k] = Point2d(u, v);

        // print to output stream
        cout << setprecision(20)
             << k << "," << v << "," << u << "," << depth << ","
             << pt3d[k].x << "," << pt3d[k].y << "," << pt3d[k].z << " ";
    }

    return 0;
}


// ============================================================================
// Caller
// ============================================================================
int main(int argc, char *argv[]) {
    // parse input as variables for better readability
    char inliers_pth[512], corpus_pth[512];
    int view_id, startF, stopF, new_h, new_w;
    double sfm_dist, real_dist;

    strncpy(inliers_pth, argv[1], 512);
    strncpy(corpus_pth, argv[2], 512);
    view_id = atoi(argv[3]);
    startF = atoi(argv[4]);
    stopF = atoi(argv[5]);
    new_h = atoi(argv[6]);
    new_w = atoi(argv[7]);
    sfm_dist = atof(argv[8]);
    real_dist = atof(argv[9]);

    if (startF != stopF) {
        cerr << "ERROR: startF must be the same as stopF";
        return -1;
    }


    // main routine
    int n_points = get_max_n_points(inliers_pth);

    vector<int> ptid;
    Point3d pt3d[n_points+1];
    Point2d pt2d[n_points+1];
    read_inliner(inliers_pth, ptid, pt3d, pt2d);


    VideoData vInfo;
    read_intrinsic_extrinsic(corpus_pth, view_id, startF, stopF, vInfo);
    CameraData frame_info = vInfo.VideoInfo[startF];

    
    Point2d projection[n_points+1];
    cv::Mat img = cv::Mat(new_h, new_w, CV_64F);
    project_depth(ptid, pt3d, pt2d, frame_info, new_h, new_w, sfm_dist, real_dist,
                  img, projection);

    return 0;
}
