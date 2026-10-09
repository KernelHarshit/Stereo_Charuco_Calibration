"""
Stereo calibration with a Charuco board (OpenCV >= 4.7).

Folder layout expected:
    DATA/left/0000.png ... 0041.png
    DATA/right/0000.png ... 0041.png      (same file names = same moment in time)

Steps:
    1. Detect Charuco corners in every image
    2. Calibrate each camera separately  -> K, distortion, RMS error
    3. Stereo calibrate                  -> R, T between the cameras
    4. Stereo rectify + remap            -> rectified images
    5. Draw horizontal epipolar lines on the rectified pairs
"""
import os, glob, json
import cv2
import numpy as np
import os

# ----------------------------- configuration --------------------------------
DATA = r"C:/Users/HP/Downloads/Calibration_pictures"
OUT    = "output"
SQUARES_X, SQUARES_Y = 11, 8      # squares across / down (found by trying both orders)
SQUARE_LEN = 0.032                # 32 mm
MARKER_LEN = 0.024                # 24 mm
DICT       = cv2.aruco.DICT_4X4_50
MIN_CORNERS = 15                  # ignore views with fewer corners than this
SIZE = (2208, 1242)               # (width, height) of the images
# -----------------------------------------------------------------------------

os.makedirs(os.path.join(OUT, "rectified"), exist_ok=True)

# 1. Board definition ---------------------------------------------------------
dictionary = cv2.aruco.getPredefinedDictionary(DICT)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LEN, MARKER_LEN, dictionary)
board.setLegacyPattern(True)      # IMPORTANT: this board was printed with the old (<4.6) layout
board_corners_3d = board.getChessboardCorners()   # (N,3) corner positions in metres, Z=0
detector = cv2.aruco.CharucoDetector(board)


def detect(path):
    """Return (corners Nx2, ids N) or None."""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    corners, ids, _, _ = detector.detectBoard(gray)
    if corners is None or len(corners) == 0:
        return None
    return corners.reshape(-1, 2), ids.ravel()


names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA, "left", "*.png")))
det = {"left": {}, "right": {}}
for side in det:
    for n in names:
        r = detect(os.path.join(DATA, side, n))
        if r is not None:
            det[side][n] = r
    print(f"{side}: board found in {len(det[side])}/{len(names)} images")


# 2. Single-camera calibration -----------------------------------------------
def calibrate_single(side):
    use = [n for n, (c, i) in det[side].items() if len(i) >= MIN_CORNERS]
    while True:
        obj = [board_corners_3d[det[side][n][1]].reshape(-1, 1, 3).astype(np.float32) for n in use]
        img = [det[side][n][0].reshape(-1, 1, 2).astype(np.float32) for n in use]
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, SIZE, None, None)

        # per-view RMS reprojection error
        errs = []
        for o, i, rv, tv in zip(obj, img, rvecs, tvecs):
            proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
            errs.append(float(np.sqrt(np.mean(np.sum((proj - i) ** 2, axis=2)))))

        # drop the worst view if it is a clear outlier, then recalibrate
        w = int(np.argmax(errs))
        if errs[w] > 3 * np.median(errs) and errs[w] > 1.5 and len(use) > 12:
            print(f"  {side}: dropping {use[w]} (error {errs[w]:.2f}px)")
            use.pop(w)
            continue
        return rms, K, dist, use, dict(zip(use, errs))


cam = {}
for side in ("left", "right"):
    rms, K, dist, use, errs = calibrate_single(side)
    cam[side] = dict(rms=rms, K=K, dist=dist, used=use, errs=errs)
    print(f"{side}: {len(use)} images, RMS = {rms:.4f}px")
    print("K =\n", K, "\ndist =", dist.ravel())

# 3. Stereo calibration -------------------------------------------------------
obj_pts, img_l, img_r, pair_names = [], [], [], []
for n in names:
    if n not in det["left"] or n not in det["right"]:
        continue
    (cl, il), (cr, ir) = det["left"][n], det["right"][n]
    common = np.intersect1d(il, ir)                 # corner ids seen by BOTH cameras
    if len(common) < MIN_CORNERS:
        continue
    sel_l = [list(il).index(c) for c in common]
    sel_r = [list(ir).index(c) for c in common]
    obj_pts.append(board_corners_3d[common].reshape(-1, 1, 3).astype(np.float32))
    img_l.append(cl[sel_l].reshape(-1, 1, 2).astype(np.float32))
    img_r.append(cr[sel_r].reshape(-1, 1, 2).astype(np.float32))
    pair_names.append(n)
print("stereo pairs used:", len(pair_names))

KL, dL = cam["left"]["K"], cam["left"]["dist"]
KR, dR = cam["right"]["K"], cam["right"]["dist"]
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 200, 1e-7)

# intrinsics are already good -> keep them fixed, only solve for R, T (and E, F)
stereo_rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    obj_pts, img_l, img_r, KL, dL, KR, dR, SIZE,
    criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
print(f"stereo RMS = {stereo_rms:.4f}px   baseline = {np.linalg.norm(T) * 1000:.1f} mm")
print("R =\n", R, "\nT =", T.ravel())

# 4. Rectification ------------------------------------------------------------
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    KL, dL, KR, dR, SIZE, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
map_l = cv2.initUndistortRectifyMap(KL, dL, R1, P1, SIZE, cv2.CV_32FC1)
map_r = cv2.initUndistortRectifyMap(KR, dR, R2, P2, SIZE, cv2.CV_32FC1)

# quality check: after rectification, matching corners must share the same row (y)
dy = []
for a, b in zip(img_l, img_r):
    ua = cv2.undistortPoints(a, KL, dL, R=R1, P=P1).reshape(-1, 2)
    ub = cv2.undistortPoints(b, KR, dR, R=R2, P=P2).reshape(-1, 2)
    dy += list(np.abs(ua[:, 1] - ub[:, 1]))
dy = np.array(dy)
print(f"rectified row error: mean {dy.mean():.3f}px, median {np.median(dy):.3f}px, "
      f"95% {np.percentile(dy, 95):.3f}px")

# 5. Epipolar-line visualisation ---------------------------------------------
for n in pair_names[:5]:
    L = cv2.remap(cv2.imread(os.path.join(DATA, "left", n)),  *map_l, cv2.INTER_LINEAR)
    Rr = cv2.remap(cv2.imread(os.path.join(DATA, "right", n)), *map_r, cv2.INTER_LINEAR)
    vis = np.hstack([L, Rr])
    for k, y in enumerate(range(40, SIZE[1], 60)):
        cv2.line(vis, (0, y), (vis.shape[1], y), (0, 0, 255) if k % 2 else (0, 255, 255), 2)
    vis = cv2.resize(vis, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(OUT, "rectified", f"rectified_epilines_{n[:-4]}.jpg"), vis)

# Save everything
json.dump(dict(
    left=dict(K=KL.tolist(), dist=dL.ravel().tolist(), rms=cam["left"]["rms"]),
    right=dict(K=KR.tolist(), dist=dR.ravel().tolist(), rms=cam["right"]["rms"]),
    stereo=dict(R=R.tolist(), T=T.ravel().tolist(), rms=stereo_rms,
                row_err_mean=float(dy.mean()), row_err_median=float(np.median(dy))),
    rectify=dict(R1=R1.tolist(), R2=R2.tolist(), P1=P1.tolist(), P2=P2.tolist(), Q=Q.tolist()),
), open(os.path.join(OUT, "calibration_results.json"), "w"), indent=1)
print("done ->", OUT)
