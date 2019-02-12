import tensorflow as tf
import numpy as np
import datetime

from scipy.io import loadmat    # 用来读取vgg模型参数

# 自定义文件模块
from utils import read_MITSceneParsingData as scene_parsing
from utils import BatchDatsetReader as dataset
from utils import utils


# 图片大小
IMAGE_SIZE = 224
# 分割种类
NUM_OF_CLASSESS = 151
# 训练次数
MAX_ITERATION = int(1e5 + 1)

# 自行下载vgg19模型
MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'


# 定义获取命令行参数名字
FLAGS = tf.app.flags.FLAGS
# 定义命令行参数
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("tensorboard_dir", "./tensorboard/", "path to logs directory")
tf.flags.DEFINE_integer("is_train", 1, "为训练")
tf.flags.DEFINE_string("data_dir", "./Data_zoo/MIT_SceneParsing/", "数据集")
tf.flags.DEFINE_string("save_model_dir", "./save_model/", "保存模型")
tf.flags.DEFINE_string("verify_dir", "./verify_pred_image/", "预测验证")


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            # 加载vgg卷积各参数
            kernels, bias = weights[i][0][0][0][0]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            # 计算卷积
            conv = tf.nn.conv2d(current, kernels, strides=[1, 1, 1, 1], padding="SAME")
            current = tf.nn.bias_add(conv, bias)

        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            current = tf.nn.avg_pool(current, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        net[name] = current

    return net


def model(image, dropout_keep_probability):
    """
    FCN模型,采用VGG模型架构
    :param image:输入图片
    :param dropout_keep_probability: dropout比例
    :return: （pred_label, logits）---> (输出分割图, 计算损失所需要的预测输出)
    """
    # 1.使用VGG模型参数来初始化参数
    print("加载vgg模型参数....")
    model_data = loadmat("./vgg_model/imagenet-vgg-verydeep-19.mat")
    # weights表示模型的所有变量的值包括权重和偏置
    weights = np.squeeze(model_data['layers'])  # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉,例：array([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]])--->array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # 2.图片预处理
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    processed_image = image - mean_pixel
    print("加载vgg模型参数结束....")

    # 3.搭建模型
    with tf.variable_scope("model"):

        # 3.1搭建vgg19模型
        image_net = vgg_net(weights, processed_image)

        # 3.2添加下采样新层

        # 3.2.1在vgg第五层的第三部分结束加max_pool层，即conv5_3+pool
        conv_final_layer = image_net["conv5_3"]
        pool5 = tf.nn.max_pool(conv_final_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # 3.2.2添加第6层卷积7*7,卷积核数4096
        w6 = utils.weight_variable([7, 7, 512, 4096], name="w6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = tf.nn.bias_add(tf.nn.conv2d(pool5, w6, strides=[1, 1, 1, 1], padding="SAME"), b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        relu6_dropout = tf.nn.dropout(relu6, keep_prob=dropout_keep_probability)

        # 3.2.3添加第七层卷积1*1,卷积核数4096
        w7 = utils.weight_variable([1, 1, 4096, 4096], name="w7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = tf.nn.bias_add(tf.nn.conv2d(relu6_dropout, w7, strides=[1, 1, 1, 1], padding="SAME"), b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        relu7_dropout = tf.nn.dropout(relu7, keep_prob=dropout_keep_probability)

        # 3.2.4添加第八层卷积1*1,卷积核数NUM_OF_CLASSESS
        w8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="w8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = tf.nn.bias_add(tf.nn.conv2d(relu7_dropout, w8, strides=[1, 1, 1, 1], padding="SAME"), b8)

        # 3.3添加上采样层（反卷积）

        # 3.3.1添加第一层
        deconv_shape1 = image_net["pool4"].get_shape()   #　得到静态形状，例如：[1,2]--->TensorShape([Dimension(1), Dimension(2)])
                                                        # 通过 .get_shape()[i].value来获取形状数组的值
                                                        # tf.shape(t).eval()可得到[1,2]
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        # 反卷积操作,步长设为2，使得特征图放大
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))

        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        # 3.3.2添加第二层
        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        # 反卷积操作,步长设为2，使得特征图放大
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))

        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        # 3.3.3添加第三层
        image_shape = tf.shape(image)
        deconv_shape3 = tf.stack([image_shape[0], image_shape[1], image_shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        # 反卷积操作,步长设为8，使得特征图放大到原始尺寸，得到num_of_classes张特征图用来计算loss,得到[batch,image_height,image_width,num_classes]
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        # 3.4对num_of_classes张特征图进行处理得到预测分割图
        # [batch,image_height,image_width,num_classes] ----> [batch,image_height,image_width]
        annotation_pred = tf.argmax(conv_t3, axis=3, name="prediction")
        # [batch,image_height,image_width] ----> [batch,image_height,image_width,1]
        pred_label = tf.expand_dims(annotation_pred, dim=3)

    return pred_label, conv_t3


def main(argv=None):

    # 1.数据占位符
    # 训练图片输入
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    # 训练集标签
    label = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="label")
    # dropout比例
    dropout_keep_probability = tf.placeholder(tf.float32, name="dropout_keep_probability")

    # 2.模型
    # pred_label为预测出来的分割图,logits是用来计算损失来迭代优化
    pred_label, logits = model(image, dropout_keep_probability)

    # print(pred_label.get_shape())   # (?, ?, ?, 1)
    # print('*'*20)
    # print(logits.get_shape())       # (?, ?, ?, 151)
    # print('*' * 20)
    # print(label.get_shape())        # (?, 224, 224, 1)
    # print(tf.squeeze(label, squeeze_dims=[3]).get_shape())      # (?, 224, 224)
    # print('*' * 20)

    # 3.loss
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(label, squeeze_dims=[3]),
                                                                          name="entropy")))

    # 4.优化
    # trainable_var = tf.trainable_variables()    # 返回需要训练的变量列表
    # optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    # # apply_gradients和compute_gradients是所有的优化器都有的方法。为梯度修剪主要避免训练梯度爆炸和消失问题
    # # ## minimize()的第一步，返回(gradient, variable)对的list。
    # grads = optimizer.compute_gradients(loss, var_list=trainable_var)
    # # ## minimize()的第二部分，返回一个执行梯度更新的ops。
    # train_op = optimizer.apply_gradients(grads)

    train_op = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(loss)

    # 5.准确率，使用测试训练集loss

    # 6.在tensorboard中画图
    loss_summary = tf.summary.scalar("loss", loss)

    summary_op = tf.summary.merge_all()

    # 7.创建一个保存模型的saver
    saver = tf.train.Saver()

    # 8.定义一个初始化变量op
    variable_op = tf.global_variables_initializer()

    # 9.训练或预测
    with tf.Session() as sess:

        # 9.1初始化所有变量
        sess.run(variable_op)

        # 9.2定义存储tensorboard的文件位置
        tensorboard_writer = tf.summary.FileWriter(FLAGS.tensorboard_dir + 'train/', sess.graph)

        # 9.3获取真实数据
        print("Setting up image reader...")
        train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
        print(len(train_records))
        print(len(valid_records))

        print("Setting up dataset reader")
        image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
        if FLAGS.is_train == 1:
            train_dataset_reader = dataset.BatchDatset(train_records, image_options)
            print("训练集数据准备完毕")
        validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)
        print("测试集数据准备完毕")

        # 9.4训练
        if FLAGS.is_train == 1:
            print("开始迭代训练")
            for itr in range(MAX_ITERATION):
                train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
                feed_dict = {image: train_images, label: train_annotations, dropout_keep_probability: 0.85}

                sess.run(train_op, feed_dict=feed_dict)

                # 9.4.1训练集误差
                if itr % 10 == 0:
                    train_loss, summary_str = sess.run([loss, loss_summary], feed_dict=feed_dict)
                    print("Step: %d, Train_loss:%g" % (itr, train_loss))
                    tensorboard_writer.add_summary(summary_str, itr)

                # 9.4.2测试集误差
                if itr % 500 == 0:
                    valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                    valid_loss, summary_sva = sess.run([loss, loss_summary],
                                                       feed_dict={image: valid_images, label: valid_annotations,
                                                                  dropout_keep_probability: 1.0})
                    print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                    # add validation loss to TensorBoard
                    tensorboard_writer.add_summary(summary_sva, itr)

            # 9.4.3保存模型
            saver.save(sess, FLAGS.save_model_dir + "model.ckpt")

        # 做预测验证
        else:
            # 使用测试集来做预测，查看模型的正确性
            valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
            pred = sess.run(pred_label, feed_dict={image: valid_images, label: valid_annotations,
                                                        dropout_keep_probability: 1.0})
            # 目标分割图
            valid_annotations = np.squeeze(valid_annotations, axis=3)
            # 预测分割图
            pred = np.squeeze(pred, axis=3)

            for itr in range(FLAGS.batch_size):
                # 原始图
                utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.verify_dir, name="inp_" + str(5 + itr))
                # 目标分割图
                utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.verify_dir, name="gt_" + str(5 + itr))
                # 预测分割图
                utils.save_image(pred[itr].astype(np.uint8), FLAGS.verify_dir, name="pred_" + str(5 + itr))
                print("Saved image: %d" % itr)

    return None


if __name__ == "__main__":
    tf.app.run()
