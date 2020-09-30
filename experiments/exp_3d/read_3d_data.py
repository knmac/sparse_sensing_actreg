"""Code to read the camera pose data

https://www.imatest.com/support/docs/pre-5-2/geometric-calibration/projective-camera/
"""
import os
import time
from struct import unpack

import numpy as np
import cv2
import skimage.io as sio
from skimage.transform import resize
import matplotlib.pyplot as plt


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
        self.valid = None
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
        self.nCameras, self.n3dPoints = None, None
        # self.IDCumView = None
        # self.vTrueFrameId = None
        # self.filenames = None
        # self.camera = CameraData()

        # self.GlobalAnchor3DId = None
        self.xyz = None
        self.rgb = None

        # Need for BA
        # self.viewIdAll3D = None  # 3D -> visible views index
        # self.pointIdAll3D = None  # 3D -> 2D index in those visible views
        # self.uvAll3D = None  # 3D -> uv of that point in those visible views
        # self.scaleAll3D = None  # 3D -> scale of that point in those visible views
        # self.DescAll3D = None  # desc for all 3D

        # Needed for matching and triangulation
        # self.SiftIdAllViews = None  # all views valid sift feature org id
        # self.uvAllViews = None  # all views valid 2D points
        # self.scaleAllViews = None  # all views valid 2D points
        # self.twoDIdAllViews = None  # 2D point in visible view -> 2D index
        self.threeDIdAllViews = None  # 2D point in visible view -> 3D index
        # self.DescAllViews = None  # all views valid desc

        # self.n2DPointsPerView = None
        # self.cn3DPointsPerView = None
        # self.uvAllViews2 = None
        # self.scaleAllViews2 = None
        # self.threeDIdAllViews2 = None
        # self.InlierAllViews2 = None  # 2D point in visible view -> inlier (1) or not (0)

        # Pairwise matching info
        # self.nMatchesCidIPidICidJPidJ = None
        # self.matchedCidICidJ = None
        # self.matchedCidIPidICidJPidJ = None


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
    # fx, fy, skew, u0, v0 = K[0, 0], K[1, 1], K[0, 1], K[2, 0], K[2, 1]

    # ycn = (img_point[1] - v0) / fy
    # xcn = (img_point[0] - u0 - skew * ycn) / fx

    # r2 = xcn * xcn + ycn * ycn
    # r4 = r2 * r2
    # r6 = r2 * r4
    # X2 = xcn * xcn
    # Y2 = ycn * ycn
    # XY = xcn * ycn

    # a0, a1, a2 = distortion[0], distortion[1], distortion[2]
    # p0, p1 = distortion[3], distortion[4]
    # s0, s1 = distortion[5], distortion[6]

    # radial = 1 + a0 * r2 + a1 * r4 + a2 * r6
    # tangential_x = 2.0*p1*XY + p0 * (r2 + 2.0*X2)
    # tangential_y = p1 * (r2 + 2.0*Y2) + 2.0*p0*XY
    # prism_x = s0 * r2
    # prism_y = s1 * r2

    # xcn_ = radial * xcn + tangential_x + prism_x
    # ycn_ = radial * ycn + tangential_y + prism_y

    # img_point[0] = fx * xcn_ + skew * ycn_ + u0
    # img_point[1] = fy * ycn_ + v0

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
    content = open(fname, 'r').read().splitlines()

    vCorpus_cid_Lcid_Lfid = np.zeros([len(content), 3], dtype=int)
    for jj, line in enumerate(content):
        toks = line.split(' ')
        y, x, z = int(toks[0]), int(toks[1]), int(toks[2])
        vCorpus_cid_Lcid_Lfid[jj] = [x, y, z]

    return CorpusInfo, vCorpus_cid_Lcid_Lfid


def read_intrinsic_extrinsic(path, view_id=0, startF=0, stopF=999999):
    """Read intrinsic and extrinsic parameters from all frames of a video

    Args:
        path: path to the video, containing intrinsic and extrinsic files
        view_id: id of the view
        startF: starting frame index
        stopF: stopping frame index

    Return:
        vInfo: video information containing intrinsic and extrinsic parameters
            of all frames
    """
    RADIAL_TANGENTIAL_PRISM = 0

    # -------------------------------------------------------------------------
    # Read intrinsic parameters
    vInfo = VideoData()
    vInfo.VideoInfo = np.array([CameraData() for _ in range(stopF+1)])
    fname = os.path.join(path, 'Intrinsic_{:04d}.txt'.format(view_id))
    assert os.path.isfile(fname), 'Cannot find {}...'.format(fname)

    content = open(fname, 'r').read().splitlines()
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
            if startF <= frameID <= stopF:
                distortion = np.array([r0, r1, r2, t0, t1, p0, p1])
                vInfo.VideoInfo[frameID].distortion = distortion

    # -------------------------------------------------------------------------
    # Read extrinsic parameters
    fname = os.path.join(path, 'CamPose_{:04d}.txt'.format(view_id))
    rt = np.zeros(6, dtype=float)

    content = open(fname, 'r').read().splitlines()
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

    return vInfo


def project_frame(testFid, vInfo, CorpusInfo, vCorpus_cid_Lcid_Lfid):
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
    """
    assert vInfo.VideoInfo[testFid].valid, \
        print('The selected frame is not localized to the Corpus')

    # Find the 2 nearest frame and the corpus points in those 2 nn frames
    # because the frame we need may not exist in the corpus
    ii = np.searchsorted(vCorpus_cid_Lcid_Lfid[:, 2], testFid, side='right')
    if ii >= vCorpus_cid_Lcid_Lfid.shape[0]:
        return None, None
    assert vCorpus_cid_Lcid_Lfid[ii-1, 2] <= testFid <= vCorpus_cid_Lcid_Lfid[ii, 2]
    nn2 = [vCorpus_cid_Lcid_Lfid[ii-1, 1], vCorpus_cid_Lcid_Lfid[ii, 1]]
    VisbleCorpus3DPointId = CorpusInfo.threeDIdAllViews[nn2[0]] + \
        CorpusInfo.threeDIdAllViews[nn2[1]]
    # VisbleCorpus3DPointId = np.unique(VisbleCorpus3DPointId)

    # Project those corpus points to the image
    camI = vInfo.VideoInfo[testFid]
    projected = np.zeros([len(VisbleCorpus3DPointId), 3], dtype=float)
    colors = np.zeros([len(VisbleCorpus3DPointId), 3], dtype=float)
    for ii in range(len(VisbleCorpus3DPointId)):
        id3D = VisbleCorpus3DPointId[ii]
        p3d = CorpusInfo.xyz[id3D]
        rgb = CorpusInfo.rgb[id3D]

        # get the 2d location
        pt2D = project_and_distort(p3d, camI.P, camI.K, camI.distortion)

        # get the depth
        cam2point = p3d - camI.camCenter
        depth = np.dot(cam2point, camI.principleRayDir)

        projected[ii] = pt2D[0], pt2D[1], depth
        colors[ii] = rgb.astype(np.float) / 255

    # with open('{:04d}.txt'.format(testFid), 'w') as fout:
    #     for ii in range(len(VisbleCorpus3DPointId)):
    #         fout.write('{} {:.02f} {:.02f} {:.04f}\n'.format(
    #             VisbleCorpus3DPointId[ii],
    #             projected[ii, 0],
    #             projected[ii, 1],
    #             projected[ii, 2],
    #         ))
    return projected, colors


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


def main():
    path = '/home/knmac/Documents/Dropbox/SparseSensing/3d_projection/P01_08'
    frame_path = '/home/knmac/projects/tmp_extract/frames_full/P01_08/0'

    # Read corpus
    st = time.time()
    CorpusInfo, vCorpus_cid_Lcid_Lfid = read_corpus(path)
    print('Reading corpus: {:.03f}s'.format(time.time() - st))

    # Read camera parameters
    st = time.time()
    vInfo = read_intrinsic_extrinsic(path, stopF=5931)
    print('Reading camera parameters: {:.03f}s'.format(time.time() - st))

    # projected, colors = project_frame(20, vInfo, CorpusInfo, vCorpus_cid_Lcid_Lfid)

    # Project
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    for frame_id in range(30):
        # Project 3D points
        projected, colors = project_frame(frame_id, vInfo, CorpusInfo, vCorpus_cid_Lcid_Lfid)
        if projected is None:
            print(frame_id)
            continue

        # Read frame
        img = sio.imread(os.path.join(frame_path, '{:04}.jpg'.format(frame_id+1)))

        # Mark the projected 2D points
        img2 = np.zeros(img.shape, dtype=np.uint8)
        for i in range(len(projected)):
            xyz = projected[i]
            x, y = np.round(xyz[0]).astype(int), np.round(xyz[1]).astype(int)
            img2[y-5:y+5, x-5:x+5] = (colors[i]*255).astype(np.uint8)
            img[y-5:y+5, x-5:x+5, 0] = 255

        # Resize then show to speed up
        new_shape = [img.shape[0]//4, img.shape[1]//4]
        axes[0].imshow(resize(img, new_shape, anti_aliasing=True, preserve_range=True).astype(np.uint8))
        axes[1].imshow(resize(img2, new_shape, anti_aliasing=True, preserve_range=True).astype(np.uint8))
        fig.suptitle('frame {}'.format(frame_id))

        plt.draw()
        plt.pause(0.01)
    plt.show()

    # fig = plt.figure(figsize=(20, 20))
    # ax = fig.gca(projection='3d')

    # ax.plot([0,500], [0,0], [0,0], color='red')
    # ax.plot([0,0], [0,500], [0,0], color='green')
    # ax.plot([0,0], [0,0], [0,500], color='blue')
    # for testFid in range(5):
    #     projected, colors = project_frame(testFid, vInfo, CorpusInfo, vCorpus_cid_Lcid_Lfid)
    #     ax.scatter(xs=projected[:,0], ys=projected[:,1], zs=1000*projected[:,2], s=3, c=colors)

    #     plt.draw()
    #     plt.pause(0.0001)
    #     # plt.clf()
    # plt.show()


if __name__ == '__main__':
    main()
