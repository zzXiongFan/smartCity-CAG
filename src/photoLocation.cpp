// c++标准输入输出库
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <iomanip>
// opencv相关
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>            // SIFT

// DBOW
#include "DBoW3/DBoW3.h"

// ASIFT相关
#include "ASifttDetector.h"
#include "utils.h"


using namespace std;
using namespace cv;
// 全局变量定义
// 图片根目录
string rootPath;
// 训练集相关信息
vector<Mat> trainImagesData;          // 输入训练集图片数据
vector<int> trainImagesID;            // 输入训练集图片帧号
vector<Point3d> trainImagesPose;      // 输入训练集图片坐标
vector<vector<KeyPoint>> trainKeypoints;      // 特征点,二维向量
vector<Mat> trainDescriptors;         // 描述子--SURF
// 测试集相关信息
vector<Mat> testImagesData;           
vector<int> testImagesID;             
vector<Point3d> testImagesPose;
vector<vector<KeyPoint>> testKeypoints;      // 特征点
vector<Mat> testDescriptors;    
vector<string> outputName;
// 粗略匹配结果
vector<DBoW3::QueryResults> testMatchID;    // 记录对应的匹配帧号及结果，与testMatchIDIndex一一对应
vector<vector<int>> testMatchIDIndex;       // 记录匹配对应的index,与testMatchID一一对应
// 词袋
DBoW3::Vocabulary vocab(10, 6);
// 输出
vector<Point3d> result;


// 读取图片内容
void ImageRead( string trainName, string testName);
// 特征提取
void FeatureDetect();
// 生成词袋
void GenVoc(string trainName);
// 粗略定位
void photoLocation(int level);
// 计算并输出结果到文件结果
void ComputeAndOutput(string outputPath);
// 关键点定位
void KeyPointLocation();


// 工具
vector<string> split(const string &s, const string &seperator);
// clear
void vectorClear();

// 程序输入须要指定 trainsName, testName
int main ( int argc, char** argv )
{
  if ( argc != 4 )
  {
    cout<<"usage: triangulation trainName testName"<<endl;
    return 1;
  }
  rootPath = argv[1];
  // 输入文件路径
  for(int i=1; i<9;i++)
  {
    cout<<"rootPath:    "<<rootPath<<endl;
    string trainName = rootPath + '/' + "list/scene" + to_string(i) + "_train.txt";
    cout<<"train file Path:    "<<trainName<<endl;
    string testName = rootPath + '/'  + "list/scene" + to_string(i) + "_test.txt";
    cout<<"test file Path:    "<<testName<<endl;
    // 输出文件路径
    string outputPath = rootPath + '/' + "output/result" + ".csv";
    // 读入数据
    ImageRead( trainName, testName);
    // 对图片进行特征提取
    FeatureDetect();
    // 计算SURF词袋
    GenVoc("scene" + to_string(i) + "_train");
    // 利用词袋进行粗略定位
    photoLocation(6);
    // 精确定位
    KeyPointLocation();
    // 计算结果并输出
    ComputeAndOutput(outputPath);
    // 清空vector，用以函数迭代循环
    vectorClear();
  }
}


// 读取图片内容
void ImageRead( string trainName, string testName)
{
  cout<<"start to read Image!"<<endl;
  cout<<trainName.data()<<endl;
  ifstream trainfile(trainName.data()), testfile(testName.data());
  assert(trainfile.is_open());
  assert(testfile.is_open());
  string s;
  // 读取训练集内容
  while (getline(trainfile, s))
  {
    vector<string> s_split = split(s, " ");
    string imagePath = rootPath + "/undist_image_6-10/" + s_split[0];
    Mat image = imread(imagePath.data());
    trainImagesData.push_back(image);
    trainImagesID.push_back(stoi(s_split[1]));
    // 添加POSE
    trainImagesPose.push_back( Point3d(stod(s_split[2]), stod(s_split[3]), stod(s_split[4])));
  }
  // 读取测试集内容
  while (getline(testfile, s))
  {
    vector<string> s_split = split(s, " ");
    string imagePath = rootPath + "/undist_image_6-10/" + s_split[0];
    Mat image = imread(imagePath.data());
    testImagesData.push_back(image);
    testImagesID.push_back(stoi(s_split[1]));
    outputName.push_back( s_split[2] );
  }
  trainfile.close();
  testfile.close();
  cout<<"read Image finish!"<<endl;
}


// 图片特征提取
void FeatureDetect()
{
  cout<<"start to FeatureDetect!"<<endl;
  // SURF特征提取
  Ptr<FeatureDetector> SURF_detector = xfeatures2d::SURF::create(1500, 4, 3, true);
  for(int i=0; i<trainImagesData.size(); i++)
  {
    Mat desc;
    vector<KeyPoint> kpt;
    SURF_detector->detectAndCompute( trainImagesData[i], Mat(), kpt, desc );
    trainKeypoints.push_back(kpt);
    trainDescriptors.push_back(desc);
    cout<<"\r"<<"train prossing :"<<i+1<<"/"<<trainImagesData.size()<<"       "<<desc.size();
  }
  cout<<endl;
  for(int i=0; i<testImagesData.size(); i++)
  {
    Mat desc;
    vector<KeyPoint> kpt;
    SURF_detector->detectAndCompute( testImagesData[i], Mat(), kpt, desc );
    testKeypoints.push_back(kpt);
    testDescriptors.push_back(desc);
    cout<<"\r"<<"test prossing :"<<i+1<<"/"<<testImagesData.size();
  }
  cout<<endl;
  cout<<"FeatureDetect finish!"<<endl;
}

// 生成词袋
void GenVoc(string trainName)
{
  // 查询对应的单词文件是否存在
  string VocPath = rootPath + "/" + trainName + ".yml.gz";
  cout<<VocPath<<endl;
  ifstream Voc(VocPath.data());
  // 单词文件存在，读取即可
  if(Voc)
  {
    cout<<"Vocabulary has exit! Read it."<<endl;
    vocab.load(VocPath);
    cout<<"vocabulary info: "<<vocab<<endl;
  }
  // 文件不存在,训练词袋，并保存
  else
  {
    cout<<"Vocabulary is not exit! Create it."<<endl;
    vocab.create( trainDescriptors );
    cout<<"vocabulary info: "<<vocab<<endl;
    vocab.save(VocPath.data());
    cout<<"done"<<endl;
  }
}


// 利用词袋进行粗略定位
void photoLocation(int level)
{
  // 利用数据库的方法进行判定
  cout<<"comparing images with database "<<endl;
  DBoW3::Database db( vocab, false, 0);
  // 将全部的训练集内容生成对应的单词
  for ( int i=0; i<trainDescriptors.size(); i++ )
    db.add(trainDescriptors[i]);
  for ( int i=0; i<testDescriptors.size();  )
  {
    DBoW3::QueryResults ret;
    vector<int> idIndex;
    db.query( testDescriptors[i], ret, level);      // max result=level
    // index入组,记录对应的原始的index
    for(auto r:ret)
    {
      idIndex.push_back(r.Id); 
    }
    // if(i%6 == 0)
    // {
    //   cout<<"searching for image "<<testImagesID[i]<<endl;
    // }
    // 分别输出6张图的结果
    // cout<<"Picture "<<i%6+1<<":\t";
    for (int y=0;y<6;y++)
    {
      ret[y].Id = trainImagesID[ret[y].Id];
      // 规范输出
      // cout<<ret[y].Id<<'-'<<to_string(idIndex[y]%6+1)<<"  "<<fixed<<setprecision(7)<<ret[y].Score<<"\t";
    }
    cout<<endl;
    testMatchID.push_back(ret);
    testMatchIDIndex.push_back(idIndex);
    // 此处对所有的6张图片都进行匹配计算
    i ++;
  }
  cout<<"词袋定位完成"<<endl;
}


// 利用关键点进行精确定位
void KeyPointLocation()
{
  // 创建词袋数据库
  DBoW3::Database db( vocab, false, 0);
  // 将全部的训练集内容生成对应的单词
  for ( int i=0; i<trainDescriptors.size(); i++ )
    db.add(trainDescriptors[i]);
  // 计算对应的最佳定位点
  for(int testIndex = 0; testIndex < testMatchID.size(); testIndex++)
  {
    int bestTrainIndex = testMatchIDIndex[testIndex][0];
    cout<<"开始计算单张图片特征点,匹配分数: "<<testMatchID[testIndex][0].Score<<endl;
    cout<<"训练集图片序号: "<<testIndex<<"  \t对应的类别: "<<testImagesID[testIndex]<<"-"<<testIndex%6+1<<endl;
    cout<<"训练集图片序号: "<<bestTrainIndex<<"  \t对应的类别: "<<trainImagesID[bestTrainIndex]<<"-"<<bestTrainIndex%6+1<<endl;
    cout<<"寻找最佳匹配图片的匹配"<<endl;
    DBoW3::QueryResults ret;
    db.query( trainDescriptors[bestTrainIndex], ret, 6);      // max result=level
    // 输出匹配结果
    for (int y=0;y<6;y++)
    {
      // 暂存对应的文件名
      int temp = trainImagesID[ret[y].Id];
      cout<<temp<<'-'<<to_string(ret[y].Id%6+1)<<"  "<<fixed<<setprecision(7)<<ret[y].Score<<"\t";
    }
    cout<<endl;       // 规范输出
    // 利用5张图恢复3D点
    // 在此处引入ASIFT




    // 显示换行
    cout<<endl;
  }
}

// 计算并输出结果:result
// 此处先默认计算两级
void ComputeAndOutput( string outputPath)
{
  // 输出文件
  ofstream outputfile(outputPath.data(), ofstream::app);
  // 迭代每一个点
  for(int i=0; i<testMatchID.size(); i++)
  {
    // 此处输出最临近匹配的坐标
    Point3d p1 = trainImagesPose[testMatchIDIndex[i][0]];
    // cout<<p1<<"    "<<testMatchID[i][0].Score<<endl;
    Point3d p2 = trainImagesPose[testMatchIDIndex[i][1]];
    // cout<<p2<<"    "<<testMatchID[i][1].Score<<endl;
    Point3d p3 = trainImagesPose[testMatchIDIndex[i][2]];
    // cout<<p3<<"    "<<testMatchID[i][2].Score<<endl;
    Point3d p4 = trainImagesPose[testMatchIDIndex[i][3]];
    // cout<<p4<<"    "<<testMatchID[i][3].Score<<endl;
    Point3d p5 = trainImagesPose[testMatchIDIndex[i][4]];
    // cout<<p5<<"    "<<testMatchID[i][4].Score<<endl;
    Point3d p6 = trainImagesPose[testMatchIDIndex[i][5]];
    // cout<<p6<<"    "<<testMatchID[i][5].Score<<endl;
    // 计算该点权重
    double weight = (1/testMatchID[i][0].Score) / (1/testMatchID[i][0].Score + 1/testMatchID[i][1].Score);
    Point3d result_p = (  p1*testMatchID[i][0].Score + p2*testMatchID[i][1].Score + p3*testMatchID[i][2].Score
                        + p4*testMatchID[i][3].Score + p5*testMatchID[i][4].Score + p6*testMatchID[i][5].Score) / 
                          (testMatchID[i][0].Score + testMatchID[i][1].Score + testMatchID[i][2].Score
                         + testMatchID[i][3].Score +testMatchID[i][4].Score + testMatchID[i][5].Score);
    outputfile<<outputName[i*6]<<","<<result_p.x<<","<<result_p.y<<","<<result_p.z<<"\n";
  }
  outputfile.close();
}


// 程序工具
// split函数
vector<string> split(const string &s, const string &seperator){
  vector<string> result;
  typedef string::size_type string_size;
  string_size i = 0;
  
  while(i != s.size()){
    //找到字符串中首个不等于分隔符的字母；
    int flag = 0;
    while(i != s.size() && flag == 0){
      flag = 1;
      for(string_size x = 0; x < seperator.size(); ++x)
      if(s[i] == seperator[x]){
        ++i;
        flag = 0;
        break;
      }
    }
    
    //找到又一个分隔符，将两个分隔符之间的字符串取出；
    flag = 0;
    string_size j = i;
    while(j != s.size() && flag == 0){
      for(string_size x = 0; x < seperator.size(); ++x)
        if(s[j] == seperator[x]){
          flag = 1;
          break;
        }
      if(flag == 0) 
        ++j;
    }
    if(i != j){
      result.push_back(s.substr(i, j-i));
      i = j;
    }
  }
  return result;
}




void vectorClear()
{
  vector<Mat> trainImagesData_temp;          // 输入训练集图片数据
  vector<int> trainImagesID_temp;            // 输入训练集图片帧号
  vector<Point3d> trainImagesPose_temp;      // 输入训练集图片坐标
  vector<vector<KeyPoint>> trainKeypoints_temp;      // 特征点,二维向量
  vector<Mat> trainDescriptors_temp;         // 描述子--SURF
  // 测试集相关信息
  vector<Mat> testImagesData_temp;           
  vector<int> testImagesID_temp;             
  vector<Point3d> testImagesPose_temp;
  vector<vector<KeyPoint>> testKeypoints_temp;      // 特征点
  vector<Mat> testDescriptors_temp;    
  vector<string> outputName_temp;
  // 粗略匹配结果
  vector<DBoW3::QueryResults> testMatchID_temp;    // 记录对应的匹配帧号及结果
  vector<vector<int>> testMatchIDIndex_temp;       // 记录匹配对应的index
  // 词袋
  DBoW3::Vocabulary vocab_temp(10, 6);
  // 输出
  vector<Point3d> result_temp;
  trainImagesData.swap(trainImagesData_temp);
  trainImagesID.swap(trainImagesID_temp);
  trainImagesPose.swap(trainImagesPose_temp);
  trainKeypoints.swap(trainKeypoints_temp);
  trainDescriptors.swap(trainDescriptors_temp);
  testImagesData.swap(testImagesData_temp);
  testImagesID.swap(testImagesID_temp);
  testImagesPose.swap(testImagesPose_temp);
  testKeypoints.swap(testKeypoints_temp);
  testDescriptors.swap(testDescriptors_temp);
  outputName.swap(outputName_temp);
  testMatchID.swap(testMatchID_temp);
  testMatchIDIndex.swap(testMatchIDIndex_temp);
  vocab = vocab_temp;
  result.swap(result_temp);
}