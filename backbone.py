# 改完，未校对
import paddle
from paddle import nn
# import timm
import time # test
import numpy as np # test
from paddleseg.cvlibs import param_init
from hybridnets.model import BiFPN, Regressor, Classifier, BiFPNDecoder
from utils.utils import Anchors
from hybridnets.model import SegmentationHead
from ef_model import efficientnet_b0
from effdet import EfficientDetBackbone

from utils.constants import *

class HybridNetsBackbone(nn.Layer):
    def __init__(self, num_classes=80, compound_coef=0, seg_classes=1, backbone_name=None, seg_mode=MULTICLASS_MODE, onnx_export=False, use_pretrain=True, **kwargs):
        super(HybridNetsBackbone, self).__init__()
        self.compound_coef = compound_coef

        self.seg_classes = seg_classes
        self.seg_mode = seg_mode

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,]
        # 新增
        self.segp2_input_channels = [24, 24, 24, 32, 32, 40, 40, 48, 1]
        # self.anchor_scale = [2.,2.,2.,2.,2.,2.,2.,2.,2.,]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        self.onnx_export = onnx_export
        num_anchors = len(self.aspect_ratios) * self.num_scales

        # self.bifpn = nn.Sequential(
        #     *[BiFPN(self.fpn_num_filters[self.compound_coef],
        #             conv_channel_coef[compound_coef],
        #             True if _ == 0 else False,
        #             attention=True if compound_coef < 6 else False,  # 自注意力机制，在BiFPN转7圈

        #             use_p8=compound_coef > 7,
        #             onnx_export=onnx_export)
        #       for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef],
                                   pyramid_levels=self.pyramid_levels[self.compound_coef],
                                   onnx_export=onnx_export)

        '''Modified by Dat Vu'''
        # self.decoder = DecoderModule()
        self.bifpndecoder = BiFPNDecoder(pyramid_channels=self.fpn_num_filters[self.compound_coef], seg_p2_channels=self.segp2_input_channels[self.compound_coef])

        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=1 if self.seg_mode == BINARY_MODE else self.seg_classes+1,
            activation=None,
            kernel_size=1,
            upsampling=4,
        )

        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef],
                                     pyramid_levels=self.pyramid_levels[self.compound_coef],
                                     onnx_export=onnx_export)

        # if backbone_name:
        #     self.encoder = timm.create_model(backbone_name, pretrained=True, features_only=True, out_indices=(2,3,4))  # P3,P4,P5
        # else:
        #     # EfficientNet_Pytorch
        #     self.encoder = get_encoder(
        #         'efficientnet-b' + str(self.backbone_compound_coef[compound_coef]),
        #         in_channels=3,
        #         depth=5,
        #         weights='imagenet',
        #     )
        self.efficientdet = EfficientDetBackbone(compound_coef=compound_coef, num_classes=num_classes)
        
        if use_pretrain:
            self.init_backbone(path = 'weights\\efficientdet-d1-pretrained.pdparams')
            print('load pretrained weights')
        

        self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
                               pyramid_levels=(paddle.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
                               onnx_export=onnx_export,
                               **kwargs)
        # if onnx_export:
        #     ## TODO: timm
        #     self.encoder.set_swish(memory_efficient=False) ?

        # self.initialize_decoder(self.encoder)
        self.initialize_decoder(self.bifpndecoder)
        self.initialize_head(self.segmentation_head)
        # self.initialize_decoder(self.bifpn)
        self.initialize_decoder(self.regressor)
        self.initialize_decoder(self.classifier)

        if use_pretrain==False:
            self.initialize_decoder(self.efficientdet)

    def init_backbone(self, path):
        state_dict = paddle.load(path)
        # for key, item in state_dict.items():
        #     print(key)
        try:
            self.efficientdet.set_state_dict(state_dict)
            # print(ret)

        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')

    def freeze_bn(self):  # 这里没用上,不知道能不能用
        for m in self.named_sublayers():
            if isinstance(m[1], nn.BatchNorm2D):
                m[1].eval()



    def forward(self, inputs):
        # p1, p2, p3, p4, p5 = self.backbone_net(inputs)
        # p2, p3, p4, p5 = self.encoder(inputs)  # self.backbone_net(inputs)  # 主干网efficientnet输出
        # # p0, p1, p2, p3, p4, p5 = self.encoder(inputs)[-6:]  # p0原图
        # features = (p3, p4, p5)  
        # features = self.bifpn(features)  # 先卷积出p6 p7再经过bifpn
        p2, features = self.efficientdet(inputs)
        
        p3,p4,p5,p6,p7 = features 
        outputs = self.bifpndecoder((p2,p3,p4,p5,p6,p7))  # 用于分割解码，见论文

        segmentation = self.segmentation_head(outputs)  # 分割解码头
        regression = self.regressor(features)  # 回归（检测）解码头
        classification = self.classifier(features)
        # anchors = self.anchors(inputs, inputs.dtype)

        if not self.onnx_export:
            # return features, regression, classification, anchors, segmentation  # test
            return features, regression, classification, segmentation
        else:
            return regression, classification, segmentation
        
    def initialize_decoder(self, module):
        for layer in module.named_sublayers():
            # print(layer)
            if isinstance(layer[1], nn.Conv2D):
                param_init.kaiming_uniform(layer[1].weight, nonlinearity="relu")
                if layer[1].bias is not None:
                    param_init.constant_init(layer[1].bias,value=0)

            elif isinstance(layer[1], nn.BatchNorm2D):
                param_init.constant_init(layer[1].weight,value=1)
                param_init.constant_init(layer[1].bias,value=0)

            elif isinstance(layer[1], nn.Linear):
                param_init.xavier_uniform(layer[1].weight)
                if layer[1].bias is not None:
                    param_init.constant_init(layer[1].bias,value=0)


    def initialize_head(self, module):
        for layer in module.named_sublayers():
            if isinstance(layer[1], (nn.Linear, nn.Conv2D)):  # layer[1]是nn.Linear或nn.Conv2D为True
                param_init.xavier_uniform(layer[1].weight)
                if layer[1].bias is not None:
                    param_init.constant_init(layer[1].bias,value=0)
    
    def initialize_detect_cls_head(self, module):
        for layer in module.named_sublayers():
            # print(layer)
            if isinstance(layer[1], nn.Conv2D):
                param_init.kaiming_uniform(layer[1].weight, nonlinearity="relu")
                if layer[1].bias is not None:
                    bias_value = -np.log((1 - 0.01) / 0.01)
                    param_init.constant_init(layer[1].bias,value=bias_value)

if __name__ == '__main__':  # 测试
    from utils.utils import letterbox, scale_coords, postprocess, BBoxTransform, ClipBoxes, restricted_float, \
        boolean_string, Params
    import argparse
    parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
    parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
    parser.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. '
                                                            'https://github.com/rwightman/pytorch-image-models')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='Coefficient of efficientnet backbone')
    parser.add_argument('--source', type=str, default='demo/video', help='The demo video folder')
    parser.add_argument('--output', type=str, default='demo_result', help='Output folder')
    parser.add_argument('-w', '--load_weights', type=str, default='weights/hybridnets.pth')
    parser.add_argument('--conf_thresh', type=restricted_float, default='0.25')
    parser.add_argument('--iou_thresh', type=restricted_float, default='0.3')
    parser.add_argument('--cuda', type=boolean_string, default=True)
    parser.add_argument('--float16', type=boolean_string, default=True, help="Use float16 for faster inference")
    args = parser.parse_args()
    params = Params(f'projects/{args.project}.yml')
    anchors_ratios = params.anchors_ratios
    anchors_scales = params.anchors_scales
    
    obj_list = params.obj_list
    seg_list = params.seg_list
    seg_mode = MULTICLASS_MODE
    model = HybridNetsBackbone(compound_coef=args.compound_coef, num_classes=len(obj_list), ratios=eval(anchors_ratios),
                           scales=eval(anchors_scales), seg_classes=len(seg_list), backbone_name=args.backbone,
                           seg_mode=seg_mode)
    
    model.eval()
    input = paddle.rand([1, 3, 384, 640])
    
    features, regression, classification, anchors, seg = model(input)

    a = 1
