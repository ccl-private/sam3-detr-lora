# 道路标线跨形态泛化图示

本目录保存根目录`README.md`中“泛化能力专项检查”使用的3张网络来源图和P8推理结果图。

## 测试设置

- 权重：`sam3_lightweight_tinyvit_stage3_distill_exp/weights/p8_input_dsconv_frozen_p7.epoch4.pt`
- 阈值：0.5
- 文本提示：`white solid lane line`、`white dashed lane line`
- 目的：人工检查道路标线类别内部在城市/乡村、直线/曲线、远景/近景、斜视和低分辨率条件下的形态泛化。
- 限制：网络图片没有像素级真值，不能计算IoU；推理图中的红色表示“没有真值可匹配的预测区域”，不能直接理解为误检。

## 图片来源

- `source_city_multilane.jpg`：[阿里图片](https://i00.c.aliimg.com/img/ibank/2014/993/659/1557956399_406316771.jpg)
- `source_rural_curve.png`：[CSDN图片](https://img-blog.csdnimg.cn/576b4ff07bd748b8b535863b6a158118.png)
- `source_oblique_lowres.webp`：[Bing图片](https://tse1.mm.bing.net/th/id/OIP.W_4BdBKB8Fm6GwCekV6z9AHaE7)

图片仅用于本项目模型实验结果展示，来源链接保留在本文档和根目录README中。

## 文件对应关系

| 场景 | 白实线结果 | 白虚线结果 |
|---|---|---|
| 城市多车道 | `p8_city_white_solid.jpg` | `p8_city_white_dashed.jpg` |
| 低分辨率斜视 | `p8_oblique_white_solid.jpg` | `p8_oblique_white_dashed.jpg` |
| 乡村弯道 | `p8_rural_white_solid.jpg` | `p8_rural_white_dashed.jpg` |
