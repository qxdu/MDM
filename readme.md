# Synthetic promoter design in Escherichia coli based on multinomial diffusion model

## Model Architectures

The models discussed in this paper are all detailed in the wiki documentation of our [GPro](https://github.com/WangLabTHU/GPro/tree/main/gpro) package, which is publicly available. We would be delighted if you also wished to try out our toolkit!

A simple diagram of our MDM model can be described as follows:

<img src="[https://img-blog.csdnimg.cn/2020102116384135.png](https://github.com/qxdu/MDM/assets/59758004/97830654-5b13-43fd-89a0-b8433b166c97)" width="700px">

|Models|Wiki|Description|
|----|----|----|
| CNN_K15|https://github.com/WangLabTHU/GPro/wiki/4.2.1-CNN-K15| Our Predictive Model|
| WGAN| https://github.com/WangLabTHU/GPro/wiki/4.1.1-WGAN| Generator|
| MDM| https://github.com/WangLabTHU/GPro/wiki/4.1.2-Diffusion| Generator|

In this paper, we also provided related models in https://github.com/qxdu/MDM/tree/master/models . ddsm-main is available at https://github.com/jzhoulab/ddsm/tree/main/promoter_design , while promodiff is available at checkpoint is available at https://github.com/wangxinglong1990/Promoter_design/ . We utilize the checkpoints of promodiff.

## 
