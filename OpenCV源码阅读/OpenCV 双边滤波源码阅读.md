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

bilateralFilter_32f方法，这个方法主要是执行表格的初始化。<br>
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
    const int kExpNumBinsPerChannel = 1 << 12; // 不知道怎么描述...可以理解为单个通道的精度
    int kExpNumBins = 0;    // 所有通道的精度，由 kExpNumBinsPerChannel × 通道数 得到
    float lastExpVal = 1.f; // 用于保存上一次概率密度函数计算结果

    /**
    * len           像素点可能取得的最大差值，由 （最大值 - 最小值） × 通道数 计算得到
    * scale_index   等分区间长度的倒数
    **/
    float len, scale_index;

    CV_Assert( (src.type() == CV_32FC1 || src.type() == CV_32FC3) && src.data != dst.data );

    /* 初始化方差 */
    if( sigma_color <= 0 )
        sigma_color = 1;
    if( sigma_space <= 0 )
        sigma_space = 1;

    /* 初始化高斯函数指数的分母 */
    double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
    double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

    /* 初始化核的半径和直径 */
    if( d <= 0 )
        radius = cvRound(sigma_space*1.5);
    else
        radius = d/2;
    radius = MAX(radius, 1);
    d = radius*2 + 1;
    // compute the min/max range for the input image (even if multichannel)

    // 获取图像上的最大值和最小值。将图像拉直，因为 OpenCV 是把三个通道合起来算的，
    minMaxLoc( src.reshape(1), &minValSrc, &maxValSrc );
    // 如果图像最大值等于最小值那么滤波无效果，直接返回
    if(std::abs(minValSrc - maxValSrc) < FLT_EPSILON)
    {
        src.copyTo(dst);
        return;
    }

    // 扩充原图边缘
    // temporary copy of the image with borders for easy processing
    Mat temp;
    copyMakeBorder( src, temp, radius, radius, radius, radius, borderType );

    /* 为空间滤波器表格申请内存 */
    // allocate lookup tables
    std::vector<float> _space_weight(d*d);  // 记录空间滤波器权重
    std::vector<int> _space_ofs(d*d);       // 记录像素偏移量
    float* space_weight = &_space_weight[0];
    int* space_ofs = &_space_ofs[0];

    // assign a length which is slightly more than needed
    len = (float)(maxValSrc - minValSrc) * cn;  // 可能的最大差值
    kExpNumBins = kExpNumBinsPerChannel * cn;   // 所有通道上的精度
    std::vector<float> _expLUT(kExpNumBins+2);  // 为色彩空间权重表格分配内存
    float* expLUT = &_expLUT[0];

    scale_index = kExpNumBins/len;  // 等分区间长度的倒数

    // 初始化色彩空间权重表格
    // initialize the exp LUT
    for( i = 0; i < kExpNumBins+2; i++ )
    {
        if( lastExpVal > 0.f )
        {
            double val =  i / scale_index;
            expLUT[i] = (float)std::exp(val * val * gauss_color_coeff);
            lastExpVal = expLUT[i];
        }
        else    // 根据正态分布模型，离中心点越远权重越低，当出现第一个0，之后的所有权重一定为0
            expLUT[i] = 0.f;
    }

    // 初始化空间权重和偏移表格
    // initialize space-related bilateral filter coefficients
    for( i = -radius, maxk = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            double r = std::sqrt((double)i*i + (double)j*j);    // 半径
            if( r > radius || ( i == 0 && j == 0 ) )
                continue;
            space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
            space_ofs[maxk++] = (int)(i*(temp.step/sizeof(float)) + j*cn);
        }

    // 用于并行计算结构的类，下文会详细分析
    // parallel_for usage
    CV_CPU_DISPATCH(bilateralFilterInvoker_32f, (cn, radius, maxk, space_ofs, temp, dst, scale_index, space_weight, expLUT),
        CV_CPU_DISPATCH_MODES_ALL);
}
```


bilateralFilterInvoker_32f类，用于初始化BilateralFilter_32f_Invoker类，这个类实现了详细的处理方法。之后调用OpenCV的并行计算接口进行并行处理。

源码包含于文件 \<opencv path\>/modules/imgproc/src/bilateral_filter.dispatch.cpp 中。源码如下


```c++
void bilateralFilterInvoker_32f(
        int cn, int radius, int maxk, int *space_ofs,
        const Mat& temp, Mat& dst, float scale_index, float *space_weight, float *expLUT)
{
    CV_INSTRUMENT_REGION();

    // 初始化并行计算类
    BilateralFilter_32f_Invoker body(cn, radius, maxk, space_ofs, temp, dst, scale_index, space_weight, expLUT);
    parallel_for_(Range(0, dst.rows), body, dst.total()/(double)(1<<16));   // OpenCV并行计算接口会调用BilateralFilter_32f_Invoker的operator()方法
}
```


BilateralFilter_32f_Invoker类，具体的计算方法在operator()方法中，OpenCV的并行计算接口parallel_for_会调用此方法。


源码包含于文件 \<opencv path\>/modules/imgproc/src/bilateral_filter.dispatch.cpp 中。源码如下

注意！代码中包含大量的使用 CV_SIMD 和 CV_SIMD128 特殊指令的代码，已移除。 不影响逻辑完整性


```c++
class BilateralFilter_32f_Invoker :
    public ParallelLoopBody
{
public:

    BilateralFilter_32f_Invoker(int _cn, int _radius, int _maxk, int *_space_ofs,
        const Mat& _temp, Mat& _dest, float _scale_index, float *_space_weight, float *_expLUT) :
        cn(_cn), radius(_radius), maxk(_maxk), space_ofs(_space_ofs),
        temp(&_temp), dest(&_dest), scale_index(_scale_index), space_weight(_space_weight), expLUT(_expLUT)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        int i, j, k;    // 迭代用的临时变量
        Size size = dest->size();   // 图像的size

        // 循环处理每行
        for( i = range.start; i < range.end; i++ )
        {
            // 指向原图锚点
            const float* sptr = temp->ptr<float>(i+radius) + radius*cn;
            // 指向目标图锚点
            float* dptr = dest->ptr<float>(i);

            if( cn == 1 )   // 单通道图片处理方法
            {
                // 分配内存存储整行的加权运算后的值和权重
                AutoBuffer<float> buf(alignSize(size.width, CV_SIMD_WIDTH) + size.width + CV_SIMD_WIDTH - 1);
                memset(buf.data(), 0, buf.size() * sizeof(float));
                float *sum = alignPtr(buf.data(), CV_SIMD_WIDTH);   // 存储加权后的像素值
                float *wsum = sum + alignSize(size.width, CV_SIMD_WIDTH);   // 存储权值

                k = 0;
                // 每次处理4个像素点
                for(; k <= maxk - 4; k+=4)
                {
                    // 通过地址偏移表space_ofs获得卷积核上像素的地址
                    const float* ksptr0 = sptr + space_ofs[k];
                    const float* ksptr1 = sptr + space_ofs[k + 1];
                    const float* ksptr2 = sptr + space_ofs[k + 2];
                    const float* ksptr3 = sptr + space_ofs[k + 3];
                    j = 0;
                    // 扫描整行
                    for (; j < size.width; j++)
                    {
                        float rval = sptr[j];   // 锚点的像素值

                        float val = ksptr0[j];  // 卷积点0的像素值
                        float alpha = std::abs(val - rval) * scale_index;   // 获取 float 类型的表索引
                        int idx = cvFloor(alpha);   // 获取索引值
                        alpha -= idx;       // 两个区间间的权重
                        if (!cvIsNaN(val))
                        {
                            // 权重计算为 空间权重 × 色彩空间权重
                            float w = space_weight[k] * (cvIsNaN(rval) ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;       // 记录权重
                            sum[j] += val * w;  // 记录加权后的值
                        }

                        // 卷积点1计算同上
                        val = ksptr1[j];
                        alpha = std::abs(val - rval) * scale_index;
                        idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!cvIsNaN(val))
                        {
                            float w = space_weight[k+1] * (cvIsNaN(rval) ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;
                            sum[j] += val * w;
                        }

                        // 卷积点2计算同上
                        val = ksptr2[j];
                        alpha = std::abs(val - rval) * scale_index;
                        idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!cvIsNaN(val))
                        {
                            float w = space_weight[k+2] * (cvIsNaN(rval) ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;
                            sum[j] += val * w;
                        }

                        // 卷积点3计算同上
                        val = ksptr3[j];
                        alpha = std::abs(val - rval) * scale_index;
                        idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!cvIsNaN(val))
                        {
                            float w = space_weight[k+3] * (cvIsNaN(rval) ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;
                            sum[j] += val * w;
                        }
                    }
                }
                // 计算剩余卷积点
                for(; k < maxk; k++)
                {
                    const float* ksptr = sptr + space_ofs[k];
                    j = 0;
                    for (; j < size.width; j++)
                    {
                        float val = ksptr[j];
                        float rval = sptr[j];
                        float alpha = std::abs(val - rval) * scale_index;
                        int idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!cvIsNaN(val))
                        {
                            float w = space_weight[k] * (cvIsNaN(rval) ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;
                            sum[j] += val * w;
                        }
                    }
                }
                j = 0;
                for (; j < size.width; j++)
                {
                    // 对每个像素点做归一化
                    CV_DbgAssert(fabs(wsum[j]) >= 0);
                    dptr[j] = cvIsNaN(sptr[j]) ? sum[j] / wsum[j] : (sum[j] + sptr[j]) / (wsum[j] + 1.f);
                }
            }
            else    // 以下为三通道的处理方法，思想和单通道类似
            {
                CV_Assert( cn == 3 );
                AutoBuffer<float> buf(alignSize(size.width, CV_SIMD_WIDTH)*3 + size.width + CV_SIMD_WIDTH - 1);
                memset(buf.data(), 0, buf.size() * sizeof(float));
                float *sum_b = alignPtr(buf.data(), CV_SIMD_WIDTH);
                float *sum_g = sum_b + alignSize(size.width, CV_SIMD_WIDTH);
                float *sum_r = sum_g + alignSize(size.width, CV_SIMD_WIDTH);
                float *wsum = sum_r + alignSize(size.width, CV_SIMD_WIDTH);

                k = 0;
                for (; k <= maxk-4; k+=4)
                {
                    const float* ksptr0 = sptr + space_ofs[k];
                    const float* ksptr1 = sptr + space_ofs[k+1];
                    const float* ksptr2 = sptr + space_ofs[k+2];
                    const float* ksptr3 = sptr + space_ofs[k+3];
                    const float* rsptr = sptr;
                    j = 0;

                    for (; j < size.width; j++, rsptr += 3, ksptr0 += 3, ksptr1 += 3, ksptr2 += 3, ksptr3 += 3)
                    {
                        float rb = rsptr[0], rg = rsptr[1], rr = rsptr[2];
                        bool r_NAN = cvIsNaN(rb) || cvIsNaN(rg) || cvIsNaN(rr);

                        float b = ksptr0[0], g = ksptr0[1], r = ksptr0[2];
                        bool v_NAN = cvIsNaN(b) || cvIsNaN(g) || cvIsNaN(r);
                        float alpha = (std::abs(b - rb) + std::abs(g - rg) + std::abs(r - rr)) * scale_index;
                        int idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!v_NAN)
                        {
                            float w = space_weight[k] * (r_NAN ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;
                            sum_b[j] += b*w;
                            sum_g[j] += g*w;
                            sum_r[j] += r*w;
                        }

                        b = ksptr1[0]; g = ksptr1[1]; r = ksptr1[2];
                        v_NAN = cvIsNaN(b) || cvIsNaN(g) || cvIsNaN(r);
                        alpha = (std::abs(b - rb) + std::abs(g - rg) + std::abs(r - rr)) * scale_index;
                        idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!v_NAN)
                        {
                            float w = space_weight[k+1] * (r_NAN ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;
                            sum_b[j] += b*w;
                            sum_g[j] += g*w;
                            sum_r[j] += r*w;
                        }

                        b = ksptr2[0]; g = ksptr2[1]; r = ksptr2[2];
                        v_NAN = cvIsNaN(b) || cvIsNaN(g) || cvIsNaN(r);
                        alpha = (std::abs(b - rb) + std::abs(g - rg) + std::abs(r - rr)) * scale_index;
                        idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!v_NAN)
                        {
                            float w = space_weight[k+2] * (r_NAN ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;
                            sum_b[j] += b*w;
                            sum_g[j] += g*w;
                            sum_r[j] += r*w;
                        }

                        b = ksptr3[0]; g = ksptr3[1]; r = ksptr3[2];
                        v_NAN = cvIsNaN(b) || cvIsNaN(g) || cvIsNaN(r);
                        alpha = (std::abs(b - rb) + std::abs(g - rg) + std::abs(r - rr)) * scale_index;
                        idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!v_NAN)
                        {
                            float w = space_weight[k+3] * (r_NAN ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;
                            sum_b[j] += b*w;
                            sum_g[j] += g*w;
                            sum_r[j] += r*w;
                        }
                    }
                }
                for (; k < maxk; k++)
                {
                    const float* ksptr = sptr + space_ofs[k];
                    const float* rsptr = sptr;
                    j = 0;

                    for (; j < size.width; j++, ksptr += 3, rsptr += 3)
                    {
                        float b = ksptr[0], g = ksptr[1], r = ksptr[2];
                        bool v_NAN = cvIsNaN(b) || cvIsNaN(g) || cvIsNaN(r);
                        float rb = rsptr[0], rg = rsptr[1], rr = rsptr[2];
                        bool r_NAN = cvIsNaN(rb) || cvIsNaN(rg) || cvIsNaN(rr);
                        float alpha = (std::abs(b - rb) + std::abs(g - rg) + std::abs(r - rr)) * scale_index;
                        int idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!v_NAN)
                        {
                            float w = space_weight[k] * (r_NAN ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;
                            sum_b[j] += b*w;
                            sum_g[j] += g*w;
                            sum_r[j] += r*w;
                        }
                    }
                }
                j = 0;
                for (; j < size.width; j++)
                {
                    CV_DbgAssert(fabs(wsum[j]) >= 0);
                    float b = *(sptr++);
                    float g = *(sptr++);
                    float r = *(sptr++);
                    if (cvIsNaN(b) || cvIsNaN(g) || cvIsNaN(r))
                    {
                        wsum[j] = 1.f / wsum[j];
                        *(dptr++) = sum_b[j] * wsum[j];
                        *(dptr++) = sum_g[j] * wsum[j];
                        *(dptr++) = sum_r[j] * wsum[j];
                    }
                    else
                    {
                        wsum[j] = 1.f / (wsum[j] + 1.f);
                        *(dptr++) = (sum_b[j] + b) * wsum[j];
                        *(dptr++) = (sum_g[j] + g) * wsum[j];
                        *(dptr++) = (sum_r[j] + r) * wsum[j];
                    }
                }
            }
        }
    }

private:
    int cn, radius, maxk, *space_ofs;
    const Mat* temp;
    Mat *dest;
    float scale_index, *space_weight, *expLUT;
};
```