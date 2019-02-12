各文件夹或文件说明：
    Data_zoo：数据集
    save_model：保存训练出来的模型参数
    tensorboard：保存tensorboard画图
    utils：程序中所使用的读取图片等用到的额外代码块
    verify_pred_image：用预测做验证得到的图片结果
    vgg_model：vgg模型参数文件


model结构：（基于vgg19结构）
    在vgg第五层的第三部分结束加max_pool层，即conv5_3---->max_pool---->conv6(7*7*4096卷积核+relu+dropout)---
    ------>conv7(1*1*4096卷积核+relu+dropout)----->conv8(1*1*NUM_OF_CLASSES)---->反卷积conv_t1(4*4)---
    ------>(conv_t1+pool4=)fuse_1--------->conv_t2(4*4)----------->(conv_t2+pool3=)fuse_2-----------
    --->conv_t3(16*16)得到NUM_OF_CLASSES张原图大小的特征图，用于计算交叉熵损失；---->conv_t3进行处理得到预测分割图。

数据集：
    本文使用的标签是灰度图，通道是1，所以最后得到[batch,image_height,image_width,num_classes] ----> [batch,image_height,image_width]

怎么得到彩色分割图，可以将image的形状为[image_height,image_width]，颜色转换:
def color_image(image, num_classes=20):
    import matplotlib as mpl
    import matplotlib.cm
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))

