import numpy as np
import cv2
from filters import inwarp


def radius_of_curvature(bin_img, plt_y_pts, left_x_pix, right_x_pix):
    if left_x_pix.shape[0] >= plt_y_pts.shape[0] and right_x_pix.shape[0] >= plt_y_pts.shape[0]:
        _left_x_pix = left_x_pix[::-1]  # Reverse to match top-to-bottom in y
        _right_x_pix = right_x_pix[::-1]  # Reverse to match top-to-bottom in y

        lane_wid = abs(_right_x_pix[0] - _left_x_pix[0])

        # I consider the height of image in real world is 4.3 meters
        yaxis_m_per_pix = 10 / bin_img.shape[0]
        xaxis_m_per_pix = 3.7 / lane_wid
        car_offset = (
            (bin_img.shape[1]/2) - (((_right_x_pix[0] - _left_x_pix[0])/2) + _left_x_pix[0]))*xaxis_m_per_pix

        real_dim_y = plt_y_pts * yaxis_m_per_pix

        _left_x_pix = _left_x_pix[:len(plt_y_pts)]
        _right_x_pix = _right_x_pix[:len(plt_y_pts)]
        left_fit_coff = np.polyfit(
            real_dim_y, _left_x_pix * xaxis_m_per_pix, 2)
        right_fit_coff = np.polyfit(
            real_dim_y, _right_x_pix * xaxis_m_per_pix, 2)

        """
        - Radius of curvature = (1+(dy/dx)^2)^1.5/|d2y/dx2|
        - equation of 2nd order poly -> aY^2 + bY + c
        - dy/dx = 2a Ymax + b
        - d2y/dx2 = 2a
        """
        real_y_max = np.max(plt_y_pts) * yaxis_m_per_pix
        left_curv = (
            (1+(2 * left_fit_coff[0] * real_y_max + left_fit_coff[1])**2)**1.5) / abs(2 * left_fit_coff[0])
        right_curv = (
            (1+(2 * right_fit_coff[0] * real_y_max + right_fit_coff[1])**2)**1.5) / abs(2 * right_fit_coff[0])

        return (left_curv + right_curv) // 2, car_offset
    else:
        return 0, 0


def draw_poly(org_img, b_img, left_poly, right_poly, rad_cur, car_offset):

    width = b_img.shape[1]
    height = b_img.shape[0]

    y_axis = np.linspace(0, height-1, height)
    left_X = (np.square(y_axis)*left_poly[0]) + \
        ((y_axis)*left_poly[1]) + left_poly[2]
    right_X = (np.square(y_axis)*right_poly[0]) + \
        ((y_axis)*right_poly[1]) + right_poly[2]

    left_X = left_X[(left_X >= 0) & (left_X <= width-1)].astype(np.int32)
    right_X = right_X[(right_X >= 0) & (right_X <= width-1)].astype(np.int32)
    left_Y = np.linspace(height - len(left_X), height - 1, len(left_X))
    right_Y = np.linspace(height - len(right_X), height - 1, len(right_X))

    left_line_XY = np.array([np.transpose(np.vstack([left_X, left_Y]))])
    # print (left_line_XY.shape)
    right_line_XY = np.array(
        [np.flipud(np.transpose(np.vstack([right_X, right_Y])))])
    # print (right_line_XY.shape)
    polygon_XY = np.hstack((left_line_XY, right_line_XY))
    # print(polygon_XY.shape)

    _b_img = np.zeros_like(b_img)
    out_img = np.dstack(
        (_b_img, _b_img, _b_img))
    # plot polygon
    cv2.fillPoly(out_img, polygon_XY.astype(np.int32), (255, 0, 0))
    # Plot the fitted polynomial
    cv2.polylines(out_img, np.int32([left_line_XY]), isClosed=False, color=(
        255, 219, 130), thickness=8)
    cv2.polylines(out_img, np.int32([right_line_XY]), isClosed=False, color=(
        255, 219, 130), thickness=8)

    result_warp = cv2.addWeighted(
        np.dstack(
            (b_img, b_img, b_img))*255, 1, out_img, 0.3, 0)

    window_img_unwrapped = inwarp(out_img)

    result = cv2.addWeighted(
        org_img, 1, window_img_unwrapped, 0.3, 0)

    cv2.putText(result, 'Vehicle is ' + str(-car_offset)
                [0:5] + 'm left of center', (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.putText(result, 'Radius of Curvature = ' + str(int(np.round(rad_cur))
                                                       ) + '(m)', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    return result, result_warp


def hconcat_resize(img_list,
                   interpolation=cv2.INTER_CUBIC):
    # take minimum hights
    h_min = min(img.shape[0]
                for img in img_list)

    # image resizing
    im_list_resize = [cv2.resize(img,
                                 (int(img.shape[1] * h_min / img.shape[0]),
                                  h_min), interpolation=interpolation)
                      for img in img_list]

    # return final image
    return cv2.hconcat(im_list_resize)


def vconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    # take minimum width
    w_min = min(img.shape[1]
                for img in img_list)

    # resizing images
    im_list_resize = [cv2.resize(img,
                                 (w_min, int(img.shape[0]
                                             * w_min / img.shape[1])),
                                 interpolation=interpolation)
                      for img in img_list]
    # return final image
    return cv2.vconcat(im_list_resize)
