import numpy as np


def radius_of_curvature(bin_img, plt_y_pts, left_x_pix, right_x_pix):
    if left_x_pix.shape[0] >= plt_y_pts.shape[0] and right_x_pix.shape[0] >= plt_y_pts.shape[0]:
        left_x_pix = left_x_pix[::-1]  # Reverse to match top-to-bottom in y
        right_x_pix = right_x_pix[::-1]  # Reverse to match top-to-bottom in y

        lane_wid = abs(right_x_pix[0] - left_x_pix[0])

        # I consider the height of image in real world is 4.3 meters
        yaxis_m_per_pix = 4.3 / bin_img.shape[0]
        xaxis_m_per_pix = 3.7 / lane_wid
        car_offset = (
            (bin_img.shape[1]/2) - (((right_x_pix[0] - left_x_pix[0])/2) + left_x_pix[0]))*xaxis_m_per_pix

        real_dim_y = plt_y_pts * yaxis_m_per_pix

        left_x_pix = left_x_pix[:len(plt_y_pts)]
        right_x_pix = right_x_pix[:len(plt_y_pts)]
        left_fit_coff = np.polyfit(real_dim_y, left_x_pix * xaxis_m_per_pix, 2)
        right_fit_coff = np.polyfit(
            real_dim_y, right_x_pix * xaxis_m_per_pix, 2)

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
        raise Exception("Y points number is greater than x point")
