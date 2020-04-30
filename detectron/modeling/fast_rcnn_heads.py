# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Various network "heads" for classification and bounding box prediction.

The design is as follows:

... -> RoI ----\                               /-> box cls output -> cls loss
                -> RoIFeatureXform -> box head
... -> Feature /                               \-> box reg output -> reg loss
       Map

The Fast R-CNN head produces a feature representation of the RoI for the purpose
of bounding box classification and regression. The box output module converts
the feature representation into classification and regression predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.utils.blob as blob_utils

from caffe2.python import brew

# ---------------------------------------------------------------------------- #
# Fast R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_fast_rcnn_outputs(model, blob_in, dim):
    if not cfg.FAST_RCNN.MULTI_ROI_BOX_HEAD:
        """Add RoI classification and bounding box regression output ops."""
        # Box classification layer
        model.FC(
            blob_in,
            'cls_score',
            dim,
            model.num_classes,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0)
        )
        if not model.train:  # == if test
            # Only add softmax when testing; during training the softmax is combined
            # with the label cross entropy loss for numerical stability
            model.Softmax('cls_score', 'cls_prob', engine='CUDNN')
        # Box regression layer
        num_bbox_reg_classes = (
            2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes
        )
        model.FC(
            blob_in,
            'bbox_pred',
            dim,
            num_bbox_reg_classes * 4,
            weight_init=gauss_fill(0.001),
            bias_init=const_fill(0.0)
        )
    else:
        submodels = []
        blob_output_map = {}

        external_blobs = []

        for hidden_dim in cfg.FAST_RCNN.MULTI_CONV_HEAD_DIMS:
            prefix = 'head_conv_output' + str(hidden_dim)
            submodel = model.CreateSubmodel(prefix)

            # Box classification layer
            cls_score = submodel.FC(
                blob_in,
                prefix + '_cls_score',
                dim,
                model.num_classes,
                weight_init=gauss_fill(0.01),
                bias_init=const_fill(0.0)
            )
            if not model.train:  # == if test
                # Only add softmax when testing; during training the softmax is combined
                # with the label cross entropy loss for numerical stability
                cls_prob = submodel.Softmax(prefix + '_cls_score', prefix + '_cls_prob', engine='CUDNN')
            # Box regression layer
            num_bbox_reg_classes = (
                2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes
            )

            bbox_pred = submodel.FC(
                blob_in,
                prefix + '_bbox_pred',
                dim,
                num_bbox_reg_classes * 4,
                weight_init=gauss_fill(0.001),
                bias_init=const_fill(0.0)
            )

            submodels.append(submodel)
            if not model.train:
                blob_output_map[submodel.net] = [ cls_score, cls_prob, bbox_pred ]
            else:
                blob_output_map[submodel.net] = [ cls_score, bbox_pred ]

            external_blobs += submodel.net.external_inputs

        if not model.train:
            blob_output_map[model.net] = ['cls_score', 'cls_prob', 'bbox_pred']
        else:
            blob_output_map[model.net] = ['cls_score', 'bbox_pred']

        elap_time = model.CreateSingleParam('head_conv_time')
        threshold = model.CreateSingleParam('head_conv_threshold')

        brew.switch(model, [model.deadline, elap_time, threshold],
            external_blobs, submodels, blob_output_map
        )

def add_fast_rcnn_losses(model):
    """Add losses for RoI classification and bounding box regression."""
    cls_prob, loss_cls = model.net.SoftmaxWithLoss(
        ['cls_score', 'labels_int32'], ['cls_prob', 'loss_cls'],
        scale=model.GetLossScale()
    )
    loss_bbox = model.net.SmoothL1Loss(
        [
            'bbox_pred', 'bbox_targets', 'bbox_inside_weights',
            'bbox_outside_weights'
        ],
        'loss_bbox',
        scale=model.GetLossScale()
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls, loss_bbox])
    model.Accuracy(['cls_prob', 'labels_int32'], 'accuracy_cls')
    model.AddLosses(['loss_cls', 'loss_bbox'])
    model.AddMetrics('accuracy_cls')
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

def add_roi_2mlp_head(model, blob_in, dim_in, spatial_scale):
    """Add a ReLU MLP with two hidden layers."""
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    model.FC(roi_feat, 'fc6', dim_in * roi_size * roi_size, hidden_dim)
    model.Relu('fc6', 'fc6')
    model.FC('fc6', 'fc7', hidden_dim, hidden_dim)
    model.Relu('fc7', 'fc7')
    return 'fc7', hidden_dim


def add_roi_Xconv1fc_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
            stride=1, pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}),
            no_bias=0)
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim


def add_roi_Xconv1fc_gn_head(model, blob_in, dim_in, spatial_scale):
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in, 'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    if not cfg.FAST_RCNN.MULTI_ROI_BOX_HEAD:
        """Add a X conv + 1fc head, with GroupNorm"""
        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM

        current = roi_feat
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            current = model.ConvGN(
                current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
                group_gn=get_group_gn(hidden_dim),
                stride=1, pad=1,
                weight_init=('MSRAFill', {}),
                bias_init=('ConstantFill', {'value': 0.}))
            current = model.Relu(current, current)
            dim_in = hidden_dim

        fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        fc = model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
        fc = model.Relu(fc, fc)

    else:
        submodels = []
        blob_output_map = {}

        external_blobs = []

        for hidden_dim in cfg.FAST_RCNN.MULTI_CONV_HEAD_DIMS:
            prefix = 'head_conv_' + str(hidden_dim)
            submodel = model.CreateSubmodel(prefix)

            current = roi_feat
            sub_dim_in = dim_in

            for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
                current = submodel.ConvGN(
                    current, prefix + str(i + 1), sub_dim_in, hidden_dim, 3,
                    group_gn=get_group_gn(hidden_dim),
                    stride=1, pad=1,
                    weight_init=('MSRAFill', {}),
                    bias_init=('ConstantFill', {'value': 0.}))
                current = submodel.Relu(current, current)
                sub_dim_in = hidden_dim

            fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM

            sub_fc = submodel.FC(current, prefix + '_fc6', sub_dim_in * roi_size * roi_size, fc_dim)
            submodel.Relu(sub_fc, sub_fc)

            submodels.append(submodel)
            blob_output_map[submodel.net] = [sub_fc]

            external_blobs += submodel.net.external_inputs

        blob_output_map[model.net] = ['fc6']

        if not model.train:
            elap_time = model.TimerGet(model.timer, 'head_conv_time',
                control_input = [roi_feat], device_option=core.DeviceOption(caffe2_pb2.CPU))
        else:
            elap_time = model.CreateSingleParam('head_conv_time')

        threshold = model.CreateSingleParam('head_conv_threshold')

        fc = brew.switch(model, [model.deadline, elap_time, threshold],
            external_blobs, submodels, blob_output_map
        )

    return fc, fc_dim
