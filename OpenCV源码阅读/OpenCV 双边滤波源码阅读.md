# 简介
双边滤波可以很好地减少不必要的噪声，同时保持边缘相当清晰。 但是与大多数滤波器相比，它非常慢。

sigma：为简单起见，您可以将2个sigma值设置为相同。 如果它们很小（<10），则滤镜效果不大；而如果它们很大（>150），则滤镜效果会很强，使图像看起来“卡通化”。

过滤器大小：大型过滤器（d> 5）的速度非常慢，因此对于实时应用程序建议使用d = 5，对于需要重度噪声过滤的离线应用程序建议使用d = 9。

不支持就地操作

<br>

# 函数原型

```c++
void cv::bilateralFilter	(	
  InputArray    src,
  OutputArray   dst,
  int           d,
  double        sigmaColor,
  double        sigmaSpace,
  int           borderType = BORDER_DEFAULT 
)	
```
<br>

# 参数详解

| 参数名     | 解释                                                                                                                                                                       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| src        | 源图像，只能为8-bit或者float类型，单通道或三通道的图像                                                                                                                     |
| dst        | 目标图像，数据类型和图像大小和源图像一致                                                                                                                                   |
| d          | 滤波器核的直径。如果它不是正值OpenCV将根据sigmaSpace计算得出                                                                                                               |
| sigmaColor | 在色彩空间中过滤sigma。较大的数值意味着更倾向于把像素邻域内的其他颜色混合在一起。                                                                                          |
| sigmaSpace | 在坐标空间中过滤sigma。该参数的值越大，意味着越远的像素就会相互影响，只要它们的颜色足够接近即可。当d> 0时，它指定邻域大小，而不考虑sigmaSpace。否则，d与sigmaSpace成比例。 |
| borderType | 用于推断图像外部像素的边框模式                                                                                                                                             |

<br>

# 源码解析

bilateralFilter方法主要是判断图像的数据类型是否符合要求和选择一种较快的方式处理，本文重点分析CPU处理方法。<br>
源码包含于文件 \<opencv path\>/modules/imgproc/src/bilateral_filter.dispatch.cpp 中。源码如下

```c++
void bilateralFilter( InputArray _src, OutputArray _dst, int d,
                      double sigmaColor, double sigmaSpace,
                      int borderType )
{
    CV_INSTRUMENT_REGION();

    CV_Assert(!_src.empty());

    _dst.create( _src.size(), _src.type() );    // 为_dst申请内存

    // 使用 OpenCL 进行计算，本文重点在于CPU计算，此处忽略
    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_bilateralFilter_8u(_src, _dst, d, sigmaColor, sigmaSpace, borderType))

    // 从 InpitArray 中获取 Mat 结构
    Mat src = _src.getMat(), dst = _dst.getMat();

    // 使用 Intel IPP 库进行计算，非本文重点
    CV_IPP_RUN_FAST(ipp_bilateralFilter(src, dst, d, sigmaColor, sigmaSpace, borderType));

    if( src.depth() == CV_8U ) // 8-bit 处理方法，与 float 算法思想一致，8U 处理起来更加简单
        bilateralFilter_8u( src, dst, d, sigmaColor, sigmaSpace, borderType );
    else if( src.depth() == CV_32F ) // float 处理方法，处理float 类型数据需要考虑的细节更多，本文将详细论述
        bilateralFilter_32f( src, dst, d, sigmaColor, sigmaSpace, borderType );
    else // 异常情况，不支持 8u 和 32f 以外的数据类型
        CV_Error( CV_StsUnsupportedFormat,
        "Bilateral filtering is only implemented for 8u and 32f images" );
}
```

bilateralFilter_32f方法<br>
源码包含于文件 \<opencv path\>/modules/imgproc/src/bilateral_filter.dispatch.cpp 中。源码如下

```c++
static void
bilateralFilter_32f( const Mat& src, Mat& dst, int d,
                     double sigma_color, double sigma_space,
                     int borderType )
{
    CV_INSTRUMENT_REGION();

    int cn = src.channels();    // 获取图片通道数

    /***
    * i,j    在迭代中使用的临时变量
    * maxk   卷积核包含的像素个数
    * radius 卷积核的半径
    ***/
    int i, j, maxk, radius;
    double minValSrc=-1, maxValSrc=1; // 源图像最小像素值和最大像素值
    const int kExpNumBinsPerChannel = 1 << 12;
    int kExpNumBins = 0;
    float lastExpVal = 1.f;
    float len, scale_index;

    CV_Assert( (src.type() == CV_32FC1 || src.type() == CV_32FC3) && src.data != dst.data );

    if( sigma_color <= 0 )
        sigma_color = 1;
    if( sigma_space <= 0 )
        sigma_space = 1;

    double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
    double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

    if( d <= 0 )
        radius = cvRound(sigma_space*1.5);
    else
        radius = d/2;
    radius = MAX(radius, 1);
    d = radius*2 + 1;
    // compute the min/max range for the input image (even if multichannel)

    minMaxLoc( src.reshape(1), &minValSrc, &maxValSrc );
    if(std::abs(minValSrc - maxValSrc) < FLT_EPSILON)
    {
        src.copyTo(dst);
        return;
    }

    // temporary copy of the image with borders for easy processing
    Mat temp;
    copyMakeBorder( src, temp, radius, radius, radius, radius, borderType );

    // allocate lookup tables
    std::vector<float> _space_weight(d*d);
    std::vector<int> _space_ofs(d*d);
    float* space_weight = &_space_weight[0];
    int* space_ofs = &_space_ofs[0];

    // assign a length which is slightly more than needed
    len = (float)(maxValSrc - minValSrc) * cn;
    kExpNumBins = kExpNumBinsPerChannel * cn;
    std::vector<float> _expLUT(kExpNumBins+2);
    float* expLUT = &_expLUT[0];

    scale_index = kExpNumBins/len;

    // initialize the exp LUT
    for( i = 0; i < kExpNumBins+2; i++ )
    {
        if( lastExpVal > 0.f )
        {
            double val =  i / scale_index;
            expLUT[i] = (float)std::exp(val * val * gauss_color_coeff);
            lastExpVal = expLUT[i];
        }
        else
            expLUT[i] = 0.f;
    }

    // initialize space-related bilateral filter coefficients
    for( i = -radius, maxk = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            double r = std::sqrt((double)i*i + (double)j*j);
            if( r > radius || ( i == 0 && j == 0 ) )
                continue;
            space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
            space_ofs[maxk++] = (int)(i*(temp.step/sizeof(float)) + j*cn);
        }

    // parallel_for usage
    CV_CPU_DISPATCH(bilateralFilterInvoker_32f, (cn, radius, maxk, space_ofs, temp, dst, scale_index, space_weight, expLUT),
        CV_CPU_DISPATCH_MODES_ALL);
}
```