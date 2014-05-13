/* 
 * File:   my_hough.h
 * Author: simone
 *
 * Created on 28 febbraio 2013, 14.32
 */

#ifndef MY_HOUGH_H
#define	MY_HOUGH_H

#include <opencv2/core/core.hpp>

namespace artelab
{

        void houghP( cv::Mat image, cv::OutputArray lines, cv::Mat& accumulator,
                              float rho, float theta, int threshold,
                              double minLineLength, double maxGap );

}

#endif	/* MY_HOUGH_H */