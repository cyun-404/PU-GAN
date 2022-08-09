# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
def mlp(features, layer_dims, bn=None, bn_params=None):
    for i, num_outputs in enumerate(layer_dims[:-1]):
        features = tf.contrib.layers.fully_connected(
            features, num_outputs,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='fc_%d' % i)
    outputs = tf.contrib.layers.fully_connected(
        features, layer_dims[-1],
        activation_fn=None,
        scope='fc_%d' % (len(layer_dims) - 1))
    return outputs


def mlp_conv(inputs, layer_dims, bn=None, bn_params=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv2d(
            inputs, num_out_channel,
            kernel_size=1,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='conv_%d' % i)
    outputs = tf.contrib.layers.conv2d(
        inputs, layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope='conv_%d' % (len(layer_dims) - 1))
    return outputs

##################################################################################
# Back projection Blocks
##################################################################################
def PointShuffler(inputs, scale=2):
    #inputs: B x N x 1 X C
    #outputs: B x N*scale x 1 x C//scale
    outputs = tf.reshape(inputs,[tf.shape(inputs)[0],tf.shape(inputs)[1],1,tf.shape(inputs)[3]//scale,scale])
    outputs = tf.transpose(outputs,[0, 1, 4, 3, 2])

    outputs = tf.reshape(outputs,[tf.shape(inputs)[0],tf.shape(inputs)[1]*scale,1,tf.shape(inputs)[3]//scale])

    return outputs

from Common.model_utils import gen_1d_grid,gen_grid
def up_block(inputs, up_ratio, scope='up_block', is_training=True, bn_decay=None):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        net = inputs
        dim = inputs.get_shape()[-1]
        out_dim = dim*up_ratio
        grid = gen_grid(up_ratio)
        grid = tf.tile(tf.expand_dims(grid, 0), [tf.shape(net)[0], 1,tf.shape(net)[1]])  # [batch_size, num_point*4, 2])
        grid = tf.reshape(grid, [tf.shape(net)[0], -1, 1, 2])
            #grid = tf.expand_dims(grid, axis=2)

        net = tf.tile(net, [1, up_ratio, 1, 1])
        net = tf.concat([net, grid], axis=-1)

        net = attention_unit(net, is_training=is_training)

        net = conv2d(net, 256, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=False, is_training=is_training,
                                 scope='conv1', bn_decay=bn_decay)
        net = conv2d(net, 128, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=False, is_training=is_training,
                          scope='conv2', bn_decay=bn_decay)

    return net

def down_block(inputs,up_ratio,scope='down_block',is_training=True,bn_decay=None):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        net = inputs
        net = tf.reshape(net,[tf.shape(net)[0],up_ratio,-1,tf.shape(net)[-1]])
        net = tf.transpose(net, [0, 2, 1, 3])

        net = conv2d(net, 256, [1, up_ratio],
                                 padding='VALID', stride=[1, 1],
                                 bn=False, is_training=is_training,
                                 scope='conv1', bn_decay=bn_decay)
        net = conv2d(net, 128, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=False, is_training=is_training,
                          scope='conv2', bn_decay=bn_decay)

    return net


def feature_extraction(inputs, scope='feature_extraction2', is_training=True, bn_decay=None):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        use_bn = False         # using batch_norm or not
        use_ibn = False        # bool, default False  # Whether use Instance-Batch Normalization.
        growth_rate = 24
        dense_n = 3
        knn = 16
        comp = growth_rate*2
        l0_features = tf.expand_dims(inputs, axis=2)
        #  배열의 차원을 늘려줍니다.input에는 늘려질 배열을 넣습니다.axis는 몇 번째 차원의 크기를 늘릴 건지 숫자를 넣습니다.

        l0_features = conv2d(l0_features,      # inputs: 4-D tensor variable BxHxWxC
			     24,               # num_output_channels: int (가중치를 담고있는 kernel 개수랑 동일한 의미를 갖는다.)
			     [1, 1],           # kernel_size: a list of 2 ints
                             padding='VALID',  #  경계 처리 방법을 정의합니다. 유효한 영역만 출력이 됩니다. 따라서 출력 이미지 사이즈는 입력 사이즈보다 작습니다.
			     scope='layer0', is_training=is_training, bn=use_bn, ibn=use_ibn,
                             bn_decay=bn_decay, activation_fn=None)
        # 1X1 사이즈의 24개의 kernel(filter)를 거쳐 이미지의 특성맵ouptut을 추출한다. 24개의 kernel(filter)를 거치기 때문에 output 특징맵은 24개의 channel을 갖는다.
        # 학습을 진행해가며, 해당 층의 24개의 kernel의 가중치는 epoch마다 업데이트 될 것이다.
        
	l0_features = tf.squeeze(l0_features, axis=2) # 차원 중 사이즈가 1인 것을 찾아 스칼라값으로 바꿔 해당 차원을 제거

        # encoding layer
        l1_features, l1_idx = dense_conv(l0_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer1", is_training=is_training, bn=use_bn, ibn=use_ibn,
                                                  bn_decay=bn_decay)
        l1_features = tf.concat([l1_features, l0_features], axis=-1)  # (12+24*2)+24=84

        l2_features = conv1d(l1_features, comp, 1,  # 24
                                     padding='VALID', scope='layer2_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)
        l2_features, l2_idx = dense_conv(l2_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer2", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l2_features = tf.concat([l2_features, l1_features], axis=-1)  # 84+(24*2+12)=144

        l3_features = conv1d(l2_features, comp, 1,  # 48
                                     padding='VALID', scope='layer3_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)  # 48
        l3_features, l3_idx = dense_conv(l3_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer3", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l3_features = tf.concat([l3_features, l2_features], axis=-1)  # 144+(24*2+12)=204

        l4_features = conv1d(l3_features, comp, 1,  # 48
                                     padding='VALID', scope='layer4_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)  # 48
        l4_features, l3_idx = dense_conv(l4_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer4", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l4_features = tf.concat([l4_features, l3_features], axis=-1)  # 204+(24*2+12)=264
        l4_features = tf.expand_dims(l4_features, axis=2)
    return l4_features

def up_projection_unit(inputs,up_ratio,scope="up_projection_unit",is_training=True,bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        L = conv2d(inputs, 128, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=is_training,
                               scope='conv0', bn_decay=bn_decay)

        H0 = up_block(L,up_ratio,is_training=is_training,bn_decay=bn_decay,scope='up_0')

        L0 = down_block(H0,up_ratio,is_training=is_training,bn_decay=bn_decay,scope='down_0')
        E0 = L0-L
        H1 = up_block(E0,up_ratio,is_training=is_training,bn_decay=bn_decay,scope='up_1')
        H2 = H0+H1
    return H2

def weight_learning_unit(inputs,up_ratio,scope="up_projection_unit",is_training=True,bn_decay=None):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        dim = inputs.get_shape().as_list()[-1]
        grid = gen_1d_grid(tf.reshape(up_ratio,[]))

        out_dim = dim * up_ratio

        ratios = tf.tile(tf.expand_dims(up_ratio,0),[1,tf.shape(grid)[1]])
        grid_ratios = tf.concat([grid,tf.cast(ratios,tf.float32)],axis=1)
        weights = tf.tile(tf.expand_dims(tf.expand_dims(grid_ratios,0),0),[tf.shape(inputs)[0],tf.shape(inputs)[1], 1, 1])
        weights.set_shape([None, None, None, 2])
        weights = conv2d(weights, dim, [1, 1],
                   padding='VALID', stride=[1, 1],
                   bn=False, is_training=is_training,
                   scope='conv_1', bn_decay=None)


        weights = conv2d(weights, out_dim, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=False, is_training=is_training,
                         scope='conv_2', bn_decay=None)
        weights = conv2d(weights, out_dim, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=False, is_training=is_training,
                         scope='conv_3', bn_decay=None)

        s = tf.matmul(hw_flatten(inputs), hw_flatten(weights), transpose_b=True)  # # [bs, N, N]

    return tf.expand_dims(s,axis=2)


def coordinate_reconstruction_unit(inputs,scope="reconstruction",is_training=True,bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        coord = conv2d(inputs, 64, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=False, is_training=is_training,
                           scope='fc_layer1', bn_decay=None)

        coord = conv2d(coord, 3, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=False, is_training=is_training,
                           scope='fc_layer2', bn_decay=None,
                           activation_fn=None, weight_decay=0.0)
        outputs = tf.squeeze(coord, [2])

        return outputs


def attention_unit(inputs, scope='attention_unit',is_training=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        dim = inputs.get_shape()[-1].value
        layer = dim//4
        f = conv2d(inputs,layer, [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=False, is_training=is_training,
                              scope='conv_f', bn_decay=None)

        g = conv2d(inputs, layer, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=False, is_training=is_training,
                            scope='conv_g', bn_decay=None)

        h = conv2d(inputs, dim, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=False, is_training=is_training,
                            scope='conv_h', bn_decay=None)


        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s, axis=-1)  # attention map

        o = tf.matmul(beta, hw_flatten(h))   # [bs, N, N]*[bs, N, c]->[bs, N, c]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=inputs.shape)  # [bs, h, w, C]
        x = gamma * o + inputs

    return x


##################################################################################
# Other function
##################################################################################
def instance_norm(net, train=True,weight_decay=0.00001):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)

    shift = tf.get_variable('shift',shape=var_shape,
                            initializer=tf.zeros_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    scale = tf.get_variable('scale', shape=var_shape,
                            initializer=tf.ones_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    epsilon = 1e-3
    normalized = (net - mu) / tf.square(sigma_sq + epsilon)
    return scale * normalized + shift

"""이미지의 Convolution Layer 구성

Args:
	filters(int):
    	  * 가중치를 담고있는 kernel 개수랑 동일한 의미를 갖는다.
          * 위 그림에서는 (W0, W1) 두개의 필터를 갖고 있는것을 확인할 수 있다.
          ※ 해당 값은 Output 특징맵의 Channel값이 된다.
        
        kernel_size(tuple):
          * filter(kernel)의 size
          * (height, widhth)
          
        padding(str):
          * 합성곱 연산을 수행하기 전, 입력데이터 주변을 특정값으로 채워 늘리는 것
          * 입력값으로 0 | 'valid' | 'same' 값을 선택할 수 있다.
          
        input_shape(tuple):
          * conv2d Layer에 입력되는 이미지의 크기
          * (height, width, channel|depth) 로, 3개의 원소로 구성
          * channel|depth는 컬러이미지(R,G,B)의 경우 3, 흑백이미지의 경우 1값을 갖음
          
        stride(int|tuple):
          * 입력데이터에 필터를 적용(합성곱연산)할 때 이동할 간격을 조절
          * stirde 높을수록, output특징맵의 차원수가 작아짐 
"""
def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn=False,
           bn_decay=None,
           use_bias=True,
           is_training=None,
           reuse=None):
    """ 1D convolution with non-linear operation.

    Args:
        inputs: 3-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: int
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope, reuse=reuse):
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.conv1d(inputs, num_output_channels, kernel_size, stride, padding,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   use_bias=use_bias, reuse=None)
        assert not (bn and ibn)
        if bn:
            outputs = tf.layers.batch_normalization(
                outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)
            # outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
        if ibn:
            outputs = instance_norm(outputs, is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs

def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn = False,
           bn_decay=None,
           use_bias = True,
           is_training=None,
           reuse=tf.AUTO_REUSE):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope,reuse=reuse) as sc:
      if use_xavier:
          initializer = tf.contrib.layers.xavier_initializer()
      else:
          initializer = tf.truncated_normal_initializer(stddev=stddev)

      outputs = tf.layers.conv2d(inputs,num_output_channels,kernel_size,stride,padding,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 use_bias=use_bias,reuse=None)
      assert not (bn and ibn)
      if bn:
          outputs = tf.layers.batch_normalization(outputs,momentum=bn_decay,training=is_training,renorm=False,fused=True)
          #outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
      if ibn:
          outputs = instance_norm(outputs,is_training)


      if activation_fn is not None:
        outputs = activation_fn(outputs)

      return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.00001,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    use_bias = True,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """

    with tf.variable_scope(scope) as sc:
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.dense(inputs,num_outputs,
                                  use_bias=use_bias,kernel_initializer=initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  reuse=None)

        if bn:
            outputs = tf.layers.batch_normalization(outputs, momentum=bn_decay, training=is_training, renorm=False)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs

from tf_ops.grouping.tf_grouping import knn_point_2
def get_edge_feature(point_cloud, k=16, idx=None):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k, 2)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    if idx is None:
        _, idx = knn_point_2(k+1, point_cloud, point_cloud, unique=True, sort=True)
        idx = idx[:, :, 1:, :]

    # [N, P, K, Dim]
    point_cloud_neighbors = tf.gather_nd(point_cloud, idx)
    point_cloud_central = tf.expand_dims(point_cloud, axis=-2)         # 차원을 확장해주는 함수

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])   #   raw 데이터의 특정 차원을 원래의 값으로 복사

    edge_feature = tf.concat(
        [point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    return edge_feature, idx

def dense_conv(feature, n=3,growth_rate=64, k=16, scope='dense_conv',**kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y, idx = get_edge_feature(feature, k=k, idx=None)  # [B N K 2*C]
        for i in range(n):
            if i == 0:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    tf.tile(tf.expand_dims(feature, axis=2), [1, 1, k, 1])], axis=-1)            #   raw 데이터의 특정 차원을 원래의 값으로 복사
            elif i == n-1:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, activation_fn=None, **kwargs),
                    y], axis=-1)
            else:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    y], axis=-1)
        y = tf.reduce_max(y, axis=-2)  # 지정한 차원을 따라 최댓값을 계산
        return y,idx

def normalize_point_cloud(pc):
    """
    pc [N, P, 3]
    """
    centroid = tf.reduce_mean(pc, axis=1, keep_dims=True)
    pc = pc - centroid
    furthest_distance = tf.reduce_max(
        tf.sqrt(tf.reduce_sum(pc ** 2, axis=-1, keep_dims=True)), axis=1, keep_dims=True)
    pc = pc / furthest_distance
    return pc, centroid, furthest_distance

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def flatten(input):
    return tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])

def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def safe_log(x, eps=1e-12):
  return tf.log(x + eps)


def tf_covariance(data):
    ## x: [batch_size, num_point, k, 3]
    batch_size = data.get_shape()[0].value
    num_point = data.get_shape()[1].value

    mean_data = tf.reduce_mean(data, axis=2, keep_dims=True)  # (batch_size, num_point, 1, 3)
    mx = tf.matmul(tf.transpose(mean_data, perm=[0, 1, 3, 2]), mean_data)  # (batch_size, num_point, 3, 3)
    vx = tf.matmul(tf.transpose(data, perm=[0, 1, 3, 2]), data) / tf.cast(tf.shape(data)[0], tf.float32)  # (batch_size, num_point, 3, 3)
    data_cov = tf.reshape(vx - mx, shape=[batch_size, num_point, -1])

    return data_cov



def add_scalar_summary(name, value,collection='train_summary'):
    tf.summary.scalar(name, value, collections=[collection])
def add_hist_summary(name, value,collection='train_summary'):
    tf.summary.histogram(name, value, collections=[collection])

def add_train_scalar_summary(name, value):
    tf.summary.scalar(name, value, collections=['train_summary'])

def add_train_hist_summary(name, value):
    tf.summary.histogram(name, value, collections=['train_summary'])

def add_train_image_summary(name, value):
    tf.summary.image(name, value, collections=['train_summary'])


def add_valid_summary(name, value):
    avg, update = tf.metrics.mean(value)
    tf.summary.scalar(name, avg, collections=['valid_summary'])
    return update
