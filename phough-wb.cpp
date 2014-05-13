/* 
 * File:   main.cpp
 * Author: Simone Albertini
 * 
 * E-mail: albertini.simone@gmail.com
 */

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "phough.h"

const std::string control_win = "Controls";
const std::string canny_hough_win = "Canny";
const std::string accumulator_win = "Accumulation Matrix";

cv::Mat image_orig;
cv::Mat accumulator;
cv::Mat image_canny;

int low_thresh_canny = 100;
int max_thresh_canny = 400;

int low_thresh_hough = 100;
int max_thresh_hough = 500;

int low_minLen = 10;
int max_minLen = 100;

int min_gap = 10;
int max_gap = 100;

cv::Mat show_both_images(cv::Mat canny, cv::Mat lines)
{
    const int dist = 8;
    cv::Mat canvas(canny.rows + 4, canny.cols*2 + dist, CV_8UC3);
    
    cv::cvtColor(canny, canny, CV_GRAY2BGR);
    
    cv::Rect roi(cv::Point(0,0), canny.size());
    canny.copyTo(canvas(roi));
    roi = cv::Rect(cv::Point(canny.cols+dist, 0), lines.size());
    lines.copyTo(canvas(roi));
    
    return canvas;
}

void calculate_hough(int, void*)
{
    cv::Canny(image_orig, image_canny, low_thresh_canny, low_thresh_canny*3, 3);
    
    std::vector<cv::Vec4i> lines;
    low_thresh_hough = (low_thresh_hough <=0)? 1 : low_thresh_hough;
    artelab::houghP(image_canny, lines, accumulator, 1, M_PI/180, low_thresh_hough, low_minLen, min_gap);
    
    cv::Mat orig_cpy;
    image_orig.copyTo(orig_cpy);
    for(int i = 0; i < lines.size(); i++)
    {
      cv::Vec4i l = lines[i];
      cv::line(orig_cpy, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 1, CV_AA);
    }
    
    cv::imshow(canny_hough_win, show_both_images(image_canny, orig_cpy));
    
    accumulator.convertTo(accumulator, CV_32F);
    cv::threshold(accumulator, accumulator, 0, 0, cv::THRESH_TOZERO);
    accumulator.convertTo(accumulator, CV_8U);
    cv::imshow(accumulator_win, accumulator);
    
}

int main(int argc, char** argv) 
{
    std::string imagefile = "images/barcode.png";
    if(argc > 1 && std::string(argv[1]) == "help")
    {
        std::cout << "Usage: phough-workbench [image path]" << std::endl;
        return EXIT_SUCCESS;
    }
    if(argc > 1)
    {
        imagefile = std::string(argv[1]);
    }
    
    image_orig = cv::imread(imagefile, CV_LOAD_IMAGE_COLOR);
    image_canny.create(image_orig.size(), image_orig.type());
    
    cv::namedWindow(canny_hough_win, CV_WINDOW_AUTOSIZE);
    cv::namedWindow(accumulator_win, CV_WINDOW_AUTOSIZE);
    
    cv::createTrackbar("Canny Threshold: ", canny_hough_win, &low_thresh_canny, max_thresh_canny, calculate_hough);
    cv::createTrackbar("Hough Threshold: ", canny_hough_win, &low_thresh_hough, max_thresh_hough, calculate_hough);
    cv::createTrackbar("  Hough min len: ", canny_hough_win, &low_minLen, max_minLen, calculate_hough);
    cv::createTrackbar("      Hough Gap: ", canny_hough_win, &min_gap, max_gap, calculate_hough);
    
    calculate_hough(0,0);
    
    cv::waitKey(0);
    
    return EXIT_SUCCESS;
}

