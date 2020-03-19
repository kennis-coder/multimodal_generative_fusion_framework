# -*- encoding: utf-8 -*-

# @Date    : 6/27/19
# @Author  : Kennis Yu
from keras import activations
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.backend import (arange,
                           reshape,
                           flatten,
                           repeat_elements,
                           concatenate,
                           conv2d,
                           spatial_2d_padding,
                           transpose, expand_dims)

from keras.engine.base_layer import Layer, InputSpec
from keras.utils import conv_utils
from tensorflow import meshgrid, floor, ceil, gather_nd, clip_by_value


class DeformConv2d(Layer):
    def __init__(self,
                 kernel_size=3,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """

        :param kernel_size: 奇数整数或奇数整数数组，数组的长度为2
        :param strides:
        :param padding:
        :param dilation_rate:
        :param activation:
        :param kernel_initializer:
        :param bias_initializer:
        :param kernel_regularizer:
        :param bias_regularizer:
        :param kernel_constraint:
        :param bias_constraint:
        :param kwargs:
        """
        super(DeformConv2d, self).__init__(**kwargs)

        # H_kernel = kernel_size[0]
        # W_kernel = kernel_size[1]
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2,
                                                      'kernel_size')
        assert len(self.kernel_size) is 2, u'kernel_size 必须为2'
        assert self.kernel_size[0] & 1 is 1 and self.kernel_size[1] & 1 is 1, u"卷积和的高和宽必须为奇数"

        # N = H_kernel * W_kernel
        # 卷积输出通道为2 * N，即分别捕捉x轴与y轴的坐标偏移
        self.filters = 2 * self.kernel_size[0] * self.kernel_size[1]
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
                                                        'dilation_rate')
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=4)

        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        """
        :param input_shape: 输入特征图shape
        :return: None
        """
        in_channel = input_shape[-1]
        out_channel = self.filters
        kernel_shape = self.kernel_size + (in_channel, out_channel)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.bias = self.add_weight(shape=(out_channel,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        self.input_spec = InputSpec(ndim=4,
                                    axes={-1: in_channel})
        self.built = True

    @staticmethod
    def _calc_p0(out_height, out_width, N):
        """
        卷积核输出特征图像素的x坐标值与y坐标值。

        :param out_height: 卷积核输出的特征图的高度
        :param out_width: 卷积核输出特征图的宽度
        :param N: 卷积核输出特征图的channel的1/2, 即 H_kernel * W_kernel
        :return: 输出是一个维度为[1, height, width, 2N]的张量，前N个是x轴的坐标值，后N个是y轴的坐标值
        """
        x = arange(1, out_height + 1)
        y = arange(1, out_width + 1)
        p0_x, p0_y = meshgrid(x, y)

        flattened_x = flatten(p0_x)
        reshaped_x = reshape(flattened_x, (1, out_height, out_width, 1))
        repeated_x = repeat_elements(reshaped_x, rep=N, axis=-1)

        flattened_y = flatten(p0_y)
        reshaped_y = reshape(flattened_y, (1, out_height, out_width, 1))
        repeated_y = repeat_elements(reshaped_y, rep=N, axis=-1)

        p0 = concatenate((repeated_x, repeated_y), axis=-1)
        return p0

    def _calc_pn(self, N):
        """
        卷积核覆盖的像素坐标，例如3×3的卷积核覆盖的范围是
        [[(-1,-1), (0, -1), (1, -1)],
        [(0, -1), (0, 0), (0, 1)],
        [(1, -1), (1, 0), (1, 1)]]
        x轴与y轴各有height × width个偏移量。
        :param N: N为卷积核 height × width, H_kernel * W_kernel
        :return: 前N个channel记录x轴的偏移量，后N个channel记录y轴的偏移量
        """
        pn_x, pn_y = meshgrid(arange(-(self.kernel_size[0] - 1) // 2, (self.kernel_size[0] - 1) // 2 + 1),
                              arange(-(self.kernel_size[1] - 1) // 2, (self.kernel_size[1] - 1) // 2 + 1))
        pn = concatenate((flatten(pn_x), flatten(pn_y)), axis=0)

        pn = reshape(pn, shape=(1, 1, 1, 2 * N))
        return pn

    def _calc_p(self, offset_field, out_height, out_width, N):
        """
        方格坐标 + 可学习偏移坐标
        :param offset_field: 学习到的偏移坐标
        :param out_height: 卷积核输出的特征图的高度
        :param out_width: 卷积核输出特征图的宽度
        :param N: N为卷积核 height × width, H_kernel * W_kernel
        :return: 形变卷积核输出的坐标（浮点型）
        """
        p0 = DeformConv2d._calc_p0(out_height, out_width, N)
        pn = self._calc_pn(N)

        # [1, H_out, W_out, 2 * H_kernel * W_kernel] + [1, 1, 1, 2 * H_kernel * W_kernel] +
        # [batch_size, H_out, W_out, 2 * H_kernel * W_kernel] =>
        # [batch_size, H_out, W_out, 2 * H_kernel * W_kernel]
        p = p0 + pn + offset_field
        return p

    @staticmethod
    def _meshgrid2indexpair(x, y):
        """
        将meshgrid坐标转化为(x,y)对坐标
        :param x: [batch_size, H_out, W_out, H_kernel * W_kernel]
        :param y: [batch_size, H_out, W_out, H_kernel * W_kernel]
        :return:
        """
        batch_size, out_height, out_width, N = x.get_shape()
        batch_axis = repeat_elements(arange(batch_size), rep=out_height * out_width * N, axis=0)
        batch_axis = reshape(batch_axis, (1, -1))

        flattened_x = reshape(x, (1, -1))
        flattened_y = reshape(y, (1, -1))

        # [b*h*w*N, 3]
        index = concatenate((batch_axis, flattened_x, flattened_y), axis=0)
        index = transpose(index)

        # [b, h*w*N, 3]
        index = reshape(index, shape=(batch_size, -1, 3))
        return index

    @staticmethod
    def _calc_xq(inputs, q, N):
        """
        从输入特征图中获取输入特征图的像素值
        :param inputs: 输入特征图，shape为[batch_size, H_in, W_in, channel]
        :param q: 像素坐标，shape为[batch_size, H_out, W_out, 2 * H_kernel * W_kernel]
        :return: 像素值
        """
        x = q[..., :N]
        y = q[..., N:]
        batch_size, out_height, out_width, _ = x.get_shape()

        # tf.gather_nd(paras, indices)的shape计算公式为
        # indices.shape[:-1] + paras.shape[indices.shape[-1]: ]
        # 由此可以得出outputs的shape为[batch_size, H_out * W_out * N, channel]
        index = DeformConv2d._meshgrid2indexpair(x, y)
        x_offset = gather_nd(inputs, indices=index)

        # [batch_size, H_out, W_out, N, channel]
        x_offset = reshape(x_offset, (batch_size, out_height, out_width, N, -1))
        return x_offset

    def _bilinear_interpolation(self, p, N, inputs):
        """
        利用双线性插值将浮点数坐标转化为有实际意义的像素值
         y
       lt|    rt
         +---+
         | . |
         +---+-->x
        lb   rb
        :param p: 形变卷积核的坐标（浮点型）
        :param N: N为卷积核 H_kernel * W_kernel
        :param inputs: 为输入特征图
        :return:
        """
        _, in_height, in_width, channel = inputs.get_shape()
        in_height = float(in_height.value)
        in_width = float(in_width.value)

        # clip_by_value p
        # 将形变卷积输出的坐标限定在输入特征图的范围之内
        p = concatenate([clip_by_value(p[..., :N], 0, in_height - 1),
                         clip_by_value(p[..., N:], 0, in_width - 1)],
                        axis=-1)

        # 向下取整，例如：(1.2, 3.5) => (1, 3)
        q_lb = floor(p)  # left bottom

        # left bottom坐标限定在输入特征图的范围之内
        q_lb = concatenate([clip_by_value(q_lb[..., :N], 0, in_height - 1),
                            clip_by_value(q_lb[..., N:], 0, in_width - 1)],
                           axis=-1)

        # 向上取整，例如：(1.2, 3.5) => (2, 4)
        q_rt = ceil(p)   # right top

        # right top坐标限定在输入特征图的范围之内
        q_rt = concatenate([clip_by_value(q_rt[..., :N], 0, in_height - 1),
                            clip_by_value(q_rt[..., N:], 0, in_width - 1)],
                           axis=-1)

        # 计算left top的坐标，即(1, 4)
        q_lt = concatenate([q_lb[..., :N], q_rt[..., N:]], axis=-1)

        # 计算 right bottom的坐标，即(2, 3)
        q_rb = concatenate([q_rt[..., :N], q_lb[..., N:]], axis=-1)

        # bilinear kernel (b, h, w, N)
        # 利用公式将坐标映射为像素，映射公式为
        # (x1-x)×(y1-y)×Pixel(x0,y0)+
        # (x-x0)×(y1-y)×Pixel(x1,y0)+
        # (x1-x)×(y-y0)×Pixel(x0,y1)+
        # (x-x0)×(y-y0)×Pixel(x1,y1)+

        # (x1-x)×(y1-y) 即|rt-p|
        g_rt = (q_rt[..., :N] - p[..., :N]) * (q_rt[..., N:] - p[..., N:])

        # (x-x0)×(y1-y)，即|lt-p|
        g_lt = (p[..., :N] - q_lt[..., :N]) * (q_lt[..., N:] - p[..., N:])

        # (x1-x)×(y-y0)，即|rb-p|
        g_rb = (q_rb[..., :N] - p[..., :N]) * (p[..., N:] - q_rb[..., N:])

        # (x-x0)×(y-y0)，即|lb-p|
        g_lb = (q_lb[..., :N] - p[..., :N]) * (q_lb[..., N:] - p[..., N:])

        # Pixel(x0, y0)
        x_q_lb = self._calc_xq(inputs, q_lb, N)

        # Pixel(x1, y0)
        x_q_rb = self._calc_xq(inputs, q_rb, N)

        # Pixel(x0, y1)
        x_q_lt = self._calc_xq(inputs, q_lt, N)

        # Pixel(x1, y1)
        x_q_rt = self._calc_xq(inputs, q_rt, N)

        # 为便于将g_rt与x_q_lb哈达玛积，需要将[batch_size, H_out, W_out, H_kernel * W_kernel]
        # 转换为[batch_size, H_out, W_out, H_kernel * W_kernel, channel]
        g_rt = repeat_elements(expand_dims(g_rt, axis=-1), channel, axis=-1)
        g_lt = repeat_elements(expand_dims(g_lt, axis=-1), channel, axis=-1)
        g_rb = repeat_elements(expand_dims(g_rb, axis=-1), channel, axis=-1)
        g_lb = repeat_elements(expand_dims(g_lb, axis=-1), channel, axis=-1)

        x_offset = g_rt * x_q_lb + g_lt * x_q_rb + g_rb * x_q_lt + g_lb * x_q_rt
        return x_offset

    def compute_output_shape(self, input_shape):
        """
        输出shape为[batch_size, H_out*H_kernel, W_out*W_kernel, channel]
        :param input_shape: [batch_size, H_in, W_in, channel]
        :return:
        """
        space = input_shape[1:-1]
        new_space = []

        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)

        assert len(new_space) is 2

        return (input_shape[0], new_space[0] * self.kernel_size[0],
                new_space[1] * self.kernel_size[1], input_shape[-1])

    def call(self, inputs, **kwargs):
        """
        通过参数学习得到输入特征图卷积核的偏移量offset
        :param inputs: 输入特征图，shape为[batch_size, H_in, W_in, C_in]
        :param kwargs:
        :return:
        """
        # inputs shape [batch_size, H_in, W_in, C_in]
        inputs = spatial_2d_padding(inputs)

        # shape [batch_size, H_out, W_out, 2 * H_kernel * W_kernel]
        offset_field = conv2d(
            inputs,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=None,
            dilation_rate=self.dilation_rate)

        offset_shape = offset_field.get_shape()

        batch_size = offset_shape[0].value

        # N = H_kernel * W_kernel
        N = offset_shape[-1].value // 2
        out_height = offset_shape[1].value
        out_width = offset_shape[2].value

        p = self._calc_p(offset_field, out_height, out_width, N)

        x_offset = self._bilinear_interpolation(p, N, inputs)

        # 输出结果为[batch_size, H_out*H_kernel, W_out*W_kernel, channel]
        x_offset = reshape(x_offset, (batch_size, out_height*self.kernel_size[0],
                                      out_width*self.kernel_size[1], -1))

        return x_offset
