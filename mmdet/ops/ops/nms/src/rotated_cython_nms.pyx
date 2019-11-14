import numpy as np
cimport numpy as np
import cv2
def rotate_cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
    cdef np.ndarray[np.float32_t, ndim=1] x_ctrs = dets[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y_ctrs = dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] widths = dets[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] heights = dets[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] angles = dets[:, 4]
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 5]
    cdef np.ndarray[np.float32_t, ndim=1] areas = heights * widths
    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]
    cdef int ndets = dets.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] suppressed = np.zeros((ndets), dtype=np.int)
    # nominal indices
    cdef int _i, _j
    # sorted indices
    cdef int i, j
    # temp variables for box i's (the box currently under consideration)
    cdef np.float32_t ix_ctr, iy_ctr, ih, iw, ia
    # variables for computing overlap with box j (lower scoring box)
    cdef np.float32_t xx_ctr, yy_ctr, hh, ww, aa
    cdef np.float32_t inter, ovr, ovr1
    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix_ctr = x_ctrs[i]
        iy_ctr = y_ctrs[i]
        iw = widths[i]
        ih = heights[i]
        ia = angles[i]
        r1 = ((ix_ctr, iy_ctr), (iw, ih), ia)
        for _j in range(_i+1,ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx_ctr = x_ctrs[j]
            yy_ctr = y_ctrs[j]
            ww = widths[j]
            hh = heights[j]
            aa = angles[j]
            r2 = ((xx_ctr,yy_ctr),(ww,hh),aa)
            ovr = 0.0
            ovr1 = 0.0
            ovr2 = 0.0
            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints = True)
                inter = cv2.contourArea(order_pts)
                ovr = inter * 1.0 / (areas[i]+areas[j]-inter + 1e-5)
                # ovr1 = inter / iarea
                # ovr2 = inter / areas[j]
            if ovr >= thresh or ovr1 >= 0.95:
                suppressed[j]=1
    return keep
