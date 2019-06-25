//
//  main.cpp
//  sift_asift_match
//
//  Created by willard on 8/18/15.
//  Copyright (c) 2015 wilard. All rights reserved.
//

#include "ASifttDetector.h"
#include "utils.h"


void pose_estimation_2d2d (
    const std::vector<KeyPoint>& keypoints_1,
    const std::vector<KeyPoint>& keypoints_2,
    const std::vector< DMatch >& matches,
    Mat& R, Mat& t );

int main(int argc, const char * argv[]) {

    string imgfn = "./pic/1.jpg";
    string objFileName = "./pic/2.jpg";
    Mat queryImage, objectImage;
    queryImage = imread(imgfn);
    objectImage = imread(objFileName);

    ASifttDetector asiftDetector;
    vector<KeyPoint> asiftKeypoints_query, asiftKeypoints_object;
    Mat asiftDescriptors_query, asiftDescriptors_object;
    asiftDetector.detectAndCompute(queryImage, asiftKeypoints_query, asiftDescriptors_query);
    asiftDetector.detectAndCompute(objectImage, asiftKeypoints_object, asiftDescriptors_object);
    
    //Matching descriptor vectors using FLANN matcher, Asift找匹配点
    vector<vector<DMatch>> matches;
    std::vector< DMatch > asiftMatches;
    FlannBasedMatcher matcher;

    matcher.match(asiftDescriptors_query, asiftDescriptors_object, asiftMatches);
    // 过滤匹配点，仅保留n个匹配
    int n = 100, dis[500] = {0};
    for(auto match:asiftMatches)
    {
        int temp = (int)match.distance;
        if(temp < 500)
        {
            dis[temp]++;
        }
    }
    int sum = 0;
    for(int i=0; i<500 ; i++)
    {
        sum += dis[i];
        if(sum >= n)
        {
            sum = i+1;
            break;
        }
    }
    std::vector< DMatch > asiftMatches_good;
    for(auto match:asiftMatches)
    {
        int temp = (int)match.distance;
        if(temp < sum)
        {
            asiftMatches_good.push_back(match);
        }
    }
    cout<<sum<<"     "<<asiftMatches_good.size()<<endl;
    findInliers(asiftKeypoints_query, asiftKeypoints_object, asiftMatches_good, imgfn, objFileName);
    cout<<sum<<"     "<<asiftMatches_good.size()<<endl;


    // 计算位姿
    Mat R,t;
    pose_estimation_2d2d ( asiftKeypoints_query, asiftKeypoints_object, asiftMatches_good, R, t );

    
    // 使用内置函数画匹配点对
    Mat img_matches;
    drawMatches(queryImage, asiftKeypoints_query, objectImage, asiftKeypoints_object,
                asiftMatches_good, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    imshow("Good Matches & Object detection", img_matches);
    waitKey(0);

    return 0;
}




void pose_estimation_2d2d (
    const std::vector<KeyPoint>& keypoints_1,
    const std::vector<KeyPoint>& keypoints_2,
    const std::vector< DMatch >& matches,
    Mat& R, Mat& t )
{
    // 相机内参,TUM Freiburg2
    Mat K = ( Mat_<double> ( 3,3 ) << 194.69, 0, 320.469, 0, 193.994, 238.998, 0, 0, 1 );

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        points1.push_back ( keypoints_1[matches[i].queryIdx].pt );
        points2.push_back ( keypoints_2[matches[i].trainIdx].pt );
    }

    //-- 计算基础矩阵
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat ( points1, points2, CV_FM_8POINT );
    cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

    //-- 计算本质矩阵
    Point2d principal_point ( 320.469, 238.998 );				//相机主点, TUM dataset标定值
    int focal_length = 194;						//相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );
    cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

    //-- 计算单应矩阵
    Mat homography_matrix;
    homography_matrix = findHomography ( points1, points2, RANSAC, 3 );
    cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point );
    cout<<"t is "<<endl<<t<<endl;
    // t = (Mat_<double> (3,1) << 0.1126, 1.8571, -0.0049);
    // cout<<"R is "<<endl<<R<<endl;
    // cout<<"t is "<<endl<<t<<endl;
    Mat rvecs;
    Rodrigues(R, rvecs);
    cout<<"Rvec is "<<endl<<rvecs<<endl;
}