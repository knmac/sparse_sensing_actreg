"""Code to read the camera pose data

https://www.imatest.com/support/docs/pre-5-2/geometric-calibration/projective-camera/
"""
import os
# import time
from struct import unpack

import numpy as np
import cv2
# import skimage.io as sio
from skimage.transform import resize
from scipy.interpolate import Rbf
# import matplotlib.pyplot as plt


class VideoData():
    """Data of a whole video"""
    def __init__(self):
        self.nframes = None
        self.start_time = None
        self.stop_time = None
        self.VideoInfo = None


class CameraData():
    """Camera information structure, containing intrinsic and extrinsic
    parameters of a frame
    """
    def __init__(self):
        self.frameID = None
        self.valid = False
        self.width = None
        self.height = None
        self.LensModel = None
        self.ShutterModel = None
        self.distortion = None

        self.K = None  # intrinsic
        self.invK = None  # inverse intrinsic
        self.rt = None  # extrinsic
        self.R = None  # rotation matrix
        self.T = None  # translation vector
        self.P = None  # camera matrix
        self.camCenter = None  # camera center
        self.principleRayDir = None  # ray direction

    def __str__(self):
        att_lst = ['frameID', 'valid', 'width', 'height', 'LensModel', 'ShutterModel']
        att_lst2 = ['distortion', 'K', 'invK', 'rt', 'R', 'T', 'P', 'camCenter', 'principleRayDir']
        msg = ''
        for item in att_lst:
            msg += '{} = {}\n'.format(item, self.__getattribute__(item))
        for item in att_lst2:
            msg += '{} = \n{}\n'.format(item, self.__getattribute__(item))
        return msg


class Corpus():
    def __init__(self):
        self.nCameras = None
        self.n3dPoints = None
        self.xyz = None
        self.rgb = None
        self.threeDIdAllViews = None  # 2D point in visible view -> 3D index


def invert_intrinsic(K):
    """Invert intrinsic matrix

    K = fx   skew   u0
        0    fy     v0
        0    0      1

    invK = 1/(fx*fy) *
        fy   -skew   v0*skew-u0*fy
        0    fx      -v0*fx
        0    0       fx*fy

    Args:
        K: intrinsic matrix

    Return:
        invK: invert intrinsic matrix
    """
    invK = np.zeros([3, 3], dtype=float)
    fx, skew, u0 = K[0, 0], K[0, 1], K[0, 2]
    fy, v0 = K[1, 1], K[1, 2]
    fxfy = fx * fy

    invK[0, 0] = fy / fxfy
    invK[0, 1] = -skew / fxfy
    invK[0, 2] = (v0*skew - u0*fy) / fxfy
    invK[1, 1] = fx / fxfy
    invK[1, 2] = (-v0*fx) / fxfy
    invK[2, 2] = 1.0
    return invK


def get_ray_dir(iK, R, uv1):
    """Get ray direction

    Args:
        iK: invert intrinsic matrix K
        R: rotation matrix
        uv1: principal vector

    Return:
        rayDir: ray direction
    """
    rayDir = np.matmul(R.transpose(), iK)
    rayDir = np.matmul(rayDir, uv1)
    rayDir = rayDir / np.linalg.norm(rayDir)
    return rayDir


def get_rotation_translation(rt):
    """Get [R|T] matrix from rt vector

    Args:
        rt: the first 3 values are rotation vector
            the last 3 are translation vector

    Return:
        R: 3x3 rotation matrix
        T: translation component of `rt`
    """
    R = np.zeros([3, 3], dtype=float)
    rvec = rt[:3]
    cv2.Rodrigues(rvec, R)

    T = rt[3:]
    return R, T


def get_cam_center(R, T):
    """Get camera center: C = -R't

    Args:
        R: 3x3 rotation matrix
        T: translation vector

    Return:
        C: camera center
    """
    iR = np.transpose(R)
    C = -np.matmul(iR, T)
    return C


def assemble_camera_matrix(K, R, T):
    """Assemble camera matrix P = K [R|T]

    Args:
        K: intrinsic matrix
        R: rotation matrix
        T: translation vector

    Return:
        P: camera matrix
    """
    RT = np.zeros([3, 4], dtype=float)
    RT[:3, :3] = R
    RT[:, 3] = T
    return np.matmul(K, RT)


def lens_distortion_points(img_point, K, distortion):
    """Distort using lens distortion

    Args:
        img_point: undistorted 2d point
        K: intrinsic matrix
        distortion: distortion parameters

    Return:
        Distorted image point
    """
    Kf = K.flatten()
    img_point_x, img_point_y = img_point

    alpha = Kf[0]
    beta = Kf[4]
    gamma = Kf[1]
    u0 = Kf[2]
    v0 = Kf[5]

    ycn = (img_point_y - v0) / beta
    xcn = (img_point_x - u0 - gamma * ycn) / alpha

    r2 = xcn * xcn + ycn * ycn
    r4 = r2 * r2
    r6 = r2 * r4
    X2 = xcn * xcn
    Y2 = ycn * ycn
    XY = xcn * ycn

    a0 = distortion[0]
    a1 = distortion[1]
    a2 = distortion[2]
    p0 = distortion[3]
    p1 = distortion[4]
    s0 = distortion[5]
    s1 = distortion[6]

    radial = 1 + a0 * r2 + a1 * r4 + a2 * r6
    tangential_x = 2.0*p1*XY + p0 * (r2 + 2.0*X2)
    tangential_y = p1 * (r2 + 2.0*Y2) + 2.0*p0*XY
    prism_x = s0 * r2
    prism_y = s1 * r2

    xcn_ = radial * xcn + tangential_x + prism_x
    ycn_ = radial * ycn + tangential_y + prism_y

    img_point_x = alpha * xcn_ + gamma * ycn_ + u0
    img_point_y = beta * ycn_ + v0

    img_point = np.array([img_point_x, img_point_y])
    return img_point


def project_and_distort(WC, P, K, distortion):
    """Project and distort 3D points

    Args:
        WC: 3D point to project and distort
        P: camera matrix
        L: intrinsic matrix
        distortion: distortion parameters

    Return:
        pts: projected and distorted 2D point
    """
    # Homogeneous coordinates
    WC_ = np.expand_dims(np.append(WC, 1.0), axis=1)

    # Projection using camera matrix
    pts = np.matmul(P, WC_)
    pts = np.squeeze(pts / pts[2])[:2]

    # Distortion
    if K is not None:
        pts = lens_distortion_points(pts, K, distortion)
    return pts


def read_corpus(path):
    """Read the corpus of 3d points from a video

    The `path` must contains:
    path/
    └── Corpus/
        ├── Corpus_3D.txt
        ├── Corpus_threeDIdAllViews.dat
        └── CameraToBuildCorpus3.txt

    Args:
        path: path to the directory contaiing `Corpus` of a video

    Return:
        CorpusInfo: corpus of 3D points
        vCorpus_cid_Lcid_Lfid: determine which video frame is used to create
            the corpus
    """
    CorpusInfo = Corpus()

    # Unpack the next integer value (4 bytes) of a binary file from _fin
    def unpack_next(_fin):
        return unpack('i', _fin.read(4))[0]

    # -------------------------------------------------------------------------
    # Read 3D points and color
    fname = os.path.join(path, 'Corpus', 'Corpus_3D.txt')
    assert os.path.isfile(fname)
    with open(fname, 'r') as fp:
        toks = fp.readline().strip().split(' ')
        nCameras, nPoints, useColor = int(toks[0]), int(toks[1]), int(toks[2])
        CorpusInfo.nCameras = nCameras
        CorpusInfo.n3dPoints = nPoints
        CorpusInfo.xyz = np.zeros([nPoints, 3], dtype=float)
        CorpusInfo.rgb = np.zeros([nPoints, 3], dtype=np.uint8)

        if useColor:
            for jj in range(nPoints):
                toks = fp.readline().strip().split(' ')
                _xyz = np.array([float(toks[0]), float(toks[1]), float(toks[2])])
                _rgb = np.array([int(toks[3]), int(toks[4]), int(toks[5])])
                CorpusInfo.xyz[jj] = _xyz
                CorpusInfo.rgb[jj] = _rgb
        else:
            for jj in range(nPoints):
                toks = fp.readline().strip().split(' ')
                _xyz = np.array([float(toks[0]), float(toks[1]), float(toks[2])])
                CorpusInfo.xyz[jj] = _xyz

    # -------------------------------------------------------------------------
    # Read threeDIdAllViews
    CorpusInfo.threeDIdAllViews = [None for _ in range(nCameras)]

    fname = os.path.join(path, 'Corpus', 'Corpus_threeDIdAllViews.dat')
    assert os.path.isfile(fname)
    with open(fname, 'rb') as fin:
        for jj in range(CorpusInfo.nCameras):
            n3D = unpack_next(fin)
            CorpusInfo.threeDIdAllViews[jj] = [None for _ in range(n3D)]
            for ii in range(n3D):
                id3D = unpack_next(fin)
                CorpusInfo.threeDIdAllViews[jj][ii] = id3D
        assert fin.read() == b'', 'Finished before EOF'

    # -------------------------------------------------------------------------
    # Determine which video frame is used to create the corpus
    fname = os.path.join(path, 'Corpus', 'CameraToBuildCorpus3.txt')
    assert os.path.isfile(fname)
    with open(fname, 'r') as fin:
        content = fin.read().splitlines()

    vCorpus_cid_Lcid_Lfid = np.zeros([len(content), 3], dtype=int)
    for jj, line in enumerate(content):
        toks = line.split(' ')
        y, x, z = int(toks[0]), int(toks[1]), int(toks[2])
        vCorpus_cid_Lcid_Lfid[jj] = [x, y, z]

    return CorpusInfo, vCorpus_cid_Lcid_Lfid


def read_intrinsic_extrinsic(path, view_id=0, startF=0, stopF=None):
    """Read intrinsic and extrinsic parameters from all frames of a video

    The `path` must contains:
    path/
    ├── Intrinsic_[view_id].txt
    └── CamPose_[view_id].txt

    Args:
        path: path to the video, containing intrinsic and extrinsic files
        view_id: id of the view
        startF: starting frame index
        stopF: stopping frame index. If None, will parse the last line to get
            this frame index

    Return:
        vInfo: video information containing intrinsic and extrinsic parameters
            of all frames
    """
    RADIAL_TANGENTIAL_PRISM = 0

    # -------------------------------------------------------------------------
    # Read intrinsic parameters
    fname = os.path.join(path, 'Intrinsic_{:04d}.txt'.format(view_id))
    assert os.path.isfile(fname), 'Cannot find {}...'.format(fname)
    with open(fname, 'r') as fin:
        content = fin.read().splitlines()

    # Automatically set the stopF if not given
    if stopF is None:
        stopF = int(content[-1].split(' ')[0])

    # Generate vInfo object
    vInfo = VideoData()
    vInfo.start_time, vInfo.stop_time, vInfo.nframes = startF, stopF, stopF+1
    vInfo.VideoInfo = np.array([CameraData() for _ in range(stopF+1)])

    # Parse each line in the content
    for line in content:
        toks = line.split(' ')

        frameID = int(toks[0])
        LensType, ShutterModel = int(toks[1]), int(toks[2])
        width, height = int(toks[3]), int(toks[4])
        fx, fy = float(toks[5]), float(toks[6])
        skew = float(toks[7])
        u0, v0 = float(toks[8]), float(toks[9])

        if startF <= frameID <= stopF:
            K = np.zeros([3, 3], dtype=float)
            K[0] = [fx, skew, u0]
            K[1] = [0.0, fy, v0]
            K[2] = [0.0, 0.0, 1.0]
            vInfo.VideoInfo[frameID].K = K

            vInfo.VideoInfo[frameID].frameID = frameID
            vInfo.VideoInfo[frameID].width = width
            vInfo.VideoInfo[frameID].height = height
            vInfo.VideoInfo[frameID].invK = invert_intrinsic(K)
            vInfo.VideoInfo[frameID].LensModel = LensType
            vInfo.VideoInfo[frameID].ShutterModel = ShutterModel

            if (LensType == RADIAL_TANGENTIAL_PRISM):
                r0, r1, r2 = float(toks[10]), float(toks[11]), float(toks[12])
                t0, t1 = float(toks[13]), float(toks[14])
                p0, p1 = float(toks[15]), float(toks[16])

                distortion = np.array([r0, r1, r2, t0, t1, p0, p1])
                vInfo.VideoInfo[frameID].distortion = distortion

        if frameID > stopF:
            break

    # -------------------------------------------------------------------------
    # Read extrinsic parameters
    fname = os.path.join(path, 'CamPose_{:04d}.txt'.format(view_id))
    rt = np.zeros(6, dtype=float)

    with open(fname, 'r') as fin:
        content = fin.read().splitlines()
    for line in content:
        toks = line.split(' ')

        frameID = int(toks[0])
        rt[0] = float(toks[1])
        rt[1] = float(toks[2])
        rt[2] = float(toks[3])
        rt[3] = float(toks[4])
        rt[4] = float(toks[5])
        rt[5] = float(toks[6])

        if startF <= frameID <= stopF:
            if (abs(rt[3]) + abs(rt[4]) + abs(rt[5]) < 0.001):
                vInfo.VideoInfo[frameID].valid = False
                continue
            vInfo.VideoInfo[frameID].valid = True
            vInfo.VideoInfo[frameID].rt = rt

            R, T = get_rotation_translation(rt)
            vInfo.VideoInfo[frameID].R = R
            vInfo.VideoInfo[frameID].T = T
            P = assemble_camera_matrix(vInfo.VideoInfo[frameID].K, R, T)
            vInfo.VideoInfo[frameID].P = P
            vInfo.VideoInfo[frameID].camCenter = get_cam_center(R, T)

            principal = np.array([vInfo.VideoInfo[frameID].width / 2,
                                  vInfo.VideoInfo[frameID].height / 2,
                                  1.0])
            principleRayDir = get_ray_dir(vInfo.VideoInfo[frameID].invK,
                                          vInfo.VideoInfo[frameID].R,
                                          principal)
            vInfo.VideoInfo[frameID].principleRayDir = principleRayDir

        if frameID > stopF:
            break

    return vInfo


def project_frame(testFid, vInfo, CorpusInfo, vCorpus_cid_Lcid_Lfid,
                  unique_points=False):
    """Project a frame from 3D to 2D

    For every frame, get the projection of the visble corpus points.
    These points are taken as the points visible in the 2 nearest corpus
    (keyframes) frame

    Args:
        testFid: frameID of the frame to project
        vInfo: camera information of a video, including instrinsic and
            extrinsic parameters
        CorpusInfo: corpus of 3D points
        vCorpus_cid_Lcid_Lfid: determine which video frame is used to create
            the corpus
        unique_points: whether to get the unique point set (with sorted point id)

    Returns:
        projected: (N, 2) projected 2D locations in image plane of visible points
        depths: (N,) depth in camera plane of visible points
        colors: (N, 3) RGB colors of the visible points
        points3d: (N, 3) original 3D locations in world plane of visible points
    """
    if not vInfo.VideoInfo[testFid].valid:
        # print('The selected frame is not localized to the Corpus')
        return None, None, None, None

    # Find the 2 nearest frame and the corpus points in those 2 nn frames
    # because the frame we need may not exist in the corpus
    ii = np.searchsorted(vCorpus_cid_Lcid_Lfid[:, 2], testFid, side='right')
    if ii >= vCorpus_cid_Lcid_Lfid.shape[0]:
        return None, None, None, None
    assert vCorpus_cid_Lcid_Lfid[ii-1, 2] <= testFid <= vCorpus_cid_Lcid_Lfid[ii, 2]
    nn2 = [vCorpus_cid_Lcid_Lfid[ii-1, 1], vCorpus_cid_Lcid_Lfid[ii, 1]]
    try:
        VisbleCorpus3DPointId = CorpusInfo.threeDIdAllViews[nn2[0]] + \
            CorpusInfo.threeDIdAllViews[nn2[1]]
    except IndexError:
        return None, None, None, None

    # Get unique points to reduce size
    if unique_points:
        VisbleCorpus3DPointId = np.unique(VisbleCorpus3DPointId)

    # Project those corpus points to the image
    camI = vInfo.VideoInfo[testFid]
    projected = np.zeros([len(VisbleCorpus3DPointId), 2], dtype=np.float32)
    depths = np.zeros([len(VisbleCorpus3DPointId)], dtype=np.float32)
    colors = np.zeros([len(VisbleCorpus3DPointId), 3], dtype=np.uint8)
    points3d = np.zeros([len(VisbleCorpus3DPointId), 3], dtype=np.float32)
    for ii in range(len(VisbleCorpus3DPointId)):
        id3D = VisbleCorpus3DPointId[ii]
        p3d = CorpusInfo.xyz[id3D]
        rgb = CorpusInfo.rgb[id3D]

        # get the 2d location
        pt2D = project_and_distort(p3d, camI.P, camI.K, camI.distortion)

        # get the depth
        cam2point = p3d - camI.camCenter
        depth = np.dot(cam2point, camI.principleRayDir)

        # Collect results
        projected[ii] = pt2D[0], pt2D[1]
        depths[ii] = depth
        colors[ii] = rgb
        points3d[ii] = p3d

    # with open('{:04d}.txt'.format(testFid), 'w') as fout:
    #     for ii in range(len(VisbleCorpus3DPointId)):
    #         fout.write('{} {:.02f} {:.02f} {:.04f}\n'.format(
    #             VisbleCorpus3DPointId[ii],
    #             projected[ii, 0],
    #             projected[ii, 1],
    #             depths[ii],
    #         ))
    return projected, depths, colors, points3d


def match_point_to_frame(CorpusInfo, vCorpus_cid_Lcid_Lfid):
    """Match all points to all keyframes for quick reference

    Args:
        CorpusInfo: corpus of 3D points. We use threeDIdAllViews in this to get
            the list of all 3D points for all possible keyframes
        vCorpus_cid_Lcid_Lfid: determine which video frame is used to create
            the corpus. We use this to trace back the real frame ID of a keyframe

    Return:
        point_frame_matching: dictionary
            {
                point1: [frame1, frame2, ...],  # all frames contains point1
                point2: [frame1, frame2, ...],  # all franes contains point2
                ...
            }
    """
    assert len(CorpusInfo.threeDIdAllViews) == len(vCorpus_cid_Lcid_Lfid)

    point_frame_matching = {}
    for ii, point_lst in enumerate(CorpusInfo.threeDIdAllViews):
        for point_id in point_lst:
            if point_id not in point_frame_matching:
                point_frame_matching[point_id] = []
            point_frame_matching[point_id].append(vCorpus_cid_Lcid_Lfid[ii, 2])
    return point_frame_matching


def find_points_correspondence(ptid_1, ptid_2, pts_1, pts_2):
    """Find the point correspondence by looking at point id

    Args:
        ptid_1: (list) point id of the 1st frame
        ptid_2: (list) point id of the 2nd frame
        pts_1: (dict) point data of the 1st frame. The key has to match ptid_1
        pts_2: (dict) point data of the 2nd frame. The key has to match ptid_2

    Return:
        matched_1: (ndarray) matched points from the 1st frame
        matched_2: (ndarray) matched points from the 2nd frame
    """
    common_ids = np.intersect1d(ptid_1, ptid_2)
    matched_1 = np.array([pts_1[k] for k in common_ids], dtype=np.float32)
    matched_2 = np.array([pts_2[k] for k in common_ids], dtype=np.float32)
    return matched_1, matched_2


def read_inliner(fname):
    """Read inliner data

    Args:
        fname: (str) path containing the data

    Returns:
        ptid: (list) list of point id
        pt3d: (dict) dictionary of 3D locations
        pt2d: (dict) dictionary of 2D locations
    """
    assert os.path.isfile(fname)

    with open(fname, 'r') as fin:
        content = fin.read().splitlines()
    n_pts = len(content)
    ptid = np.zeros(n_pts, dtype=np.int)
    pt3d = {}
    pt2d = {}

    for i, line in enumerate(content):
        tokens = line.split()
        pid, _, x, y, z, u, v, _ = tokens
        ptid[i] = int(pid)
        pt3d[ptid[i]] = np.array([float(x), float(y), float(z)], dtype=np.float32)
        pt2d[ptid[i]] = np.array([float(u), float(v)], dtype=np.float32)
    return ptid, pt3d, pt2d


def project_depth(ptid, pt3d, pt2d, cam_center, principle_ray_dir,
                  height, width, new_h, new_w):
    """Project depth from world plane to camera plane

    Args:
        ptid: (list) list of point id
        pt3d: (dict) dictionary of 3D locations
        pt2d: (dict) dictionary of 2D locations
        cam_center: camera center 3D location in the frame
        principle_ray_dir: direction of the principle ray
        height: original height of the image (captured by camera)
        width: original width of the image (captured by camera)
        new_h: new height to resize
        new_w: new width to resize

    Return:
        zz: interpolation image
        projection: dictionary of mappings from point_id to (u, v) projection
            location, wrt to (x-axis, y-axis)
    """
    scale_h = height / new_h
    scale_w = width / new_w

    # Collect and rescale 2d points
    img = np.zeros([new_h, new_w])
    projection = {}
    for k in ptid:
        u, v = np.round(pt2d[k] / [scale_w, scale_h]).astype(int)
        if u < 0 or u >= new_w or v < 0 or v >= new_h:
            continue

        cam2point = pt3d[k] - cam_center
        depth = np.dot(cam2point, principle_ray_dir)
        img[v, u] = depth
        projection[k] = (u, v)
    return img, projection


def rbf_interpolate(img, rbf_opts, down_factor=2):
    """Create RBF interpolation image from sparse 2D points

    Args:
        img: 2D image of sparse points

    Return:
        zz: interpolation image
    """
    # Scale down to speed up
    orig_shape = img.shape
    img = img[::down_factor, ::down_factor]

    # Get the unique points
    nz_idx = np.where(img > 0)
    y, x = nz_idx
    z = img[nz_idx]

    # Interpolation
    rbf = Rbf(x, y, z, **rbf_opts)
    yy, xx = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    zz = rbf(xx, yy).transpose([1, 0]).astype(np.float32)

    # Check the results
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(2, 2)
    # axes[0, 0].imshow(zz)
    # axes[1, 0].imshow(rbf(yy, xx).transpose([1, 0]))
    # axes[0, 1].imshow(img)
    # axes[1, 1].scatter(x, y, 50, z)
    # axes[1, 1].set_ylim(axes[1, 1].get_ylim()[::-1])
    # plt.show()

    zz = resize(zz, orig_shape, preserve_range=True)
    return zz
