# 无标注网图Base与新教师对比配图

本目录保存知乎文章使用的实际推理对比图。每张图从左到右依次为原图、官方SAM3 Base（绿色mask）和关闭域外纯负提示后训练得到的新Base教师（紫色mask），统一阈值为0.5。

正文使用`00`至`04`五张图。`05_yellow_prompt_confusion.jpg`仅保留为实验记录：原图是黄虚线，官方Base在`yellow solid lane line`提示下产生响应，属于类别混淆，不能作为Base优于新教师的证据。

| 文件 | 场景与提示 | 原图来源 |
|---|---|---|
| `00_aerial_thin_white_solid.jpg` | 低分辨率航拍环岛，极细白实线 | 百度图片搜索结果，原页面域名`hellorf.com`，图片标题`aerialdroneviewroundaboutwithredbicyclemarkings` |
| `01_low_view_white_dashed.jpg` | 低视角，白虚线 | [YouTube视频缩略图](https://www.youtube.com/watch?v=BZ8vd3ITJ7c) |
| `02_highway_white_dashed.jpg` | 高速公路，白虚线 | 百度图片搜索结果，原页面域名`baijiahao.baidu.com` |
| `03_snow_fog_white_dashed.jpg` | 雪雾低能见度，白虚线 | [Dreamstime原页面](https://www.dreamstime.com/low-visibility-snow-covered-highway-due-to-winter-fog-low-visibility-snow-covered-highway-due-to-winter-fog-generative-image292939310) |
| `04_wet_white_dashed.jpg` | 湿路面，白虚线 | 百度图片搜索结果，原页面域名`www.hellorf.com` |
| `05_yellow_prompt_confusion.jpg` | 黄虚线在`yellow solid lane line`提示下的类别混淆案例 | [新浪图片原页面](http://slide.news.sina.com.cn/c/slide_1_2841_99728.html) |

上述原图来自公开图片搜索结果，版权归原作者或来源网站所有，本仓库不主张原图版权。对比图仅用于研究过程说明；对外发布、商业宣传或再分发前，应由发布者确认相应图片的授权条件。完整来源URL、查询词和下载哈希保存在本地无标注数据工程的`metadata/images.jsonl`中。
