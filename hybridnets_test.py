import time
# import torch
# from torch.backends import cudnn
from backbone import HybridNetsBackbone
from hybridnets.model import ModelWithLoss
import cv2
import numpy as np
from glob import glob
from utils.utils import letterbox, scale_coords, postprocess, BBoxTransform, ClipBoxes, restricted_float, \
    boolean_string, Params  # B3 里面 BBoxTransform, ClipBoxes单改，那个是函数
from utils.plot import STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
import os
# from torchvision import transforms
import argparse
from utils.constants import *
from collections import OrderedDict
# from torch.nn import functional as F

import paddle
from paddle.nn import functional as F
from paddle.vision import transforms

parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
parser.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. '
                                                        'https://github.com/rwightman/pytorch-image-models')
parser.add_argument('-c', '--compound_coef', type=int, default=1, help='Coefficient of efficientnet backbone')
parser.add_argument('--source', type=str, default='demo/image', help='The demo image folder')
parser.add_argument('--output', type=str, default='demo_result', help='Output folder')
parser.add_argument('-w', '--load_weights', type=str, default='weights\\hybridnets-checkpoints.pdparams')
# conf_thresh 置信度阈值是模型将预测视为真实预测的最低分数（否则它将完全忽略此预测）。iou_thresh IoU阈值是真实事实和预测框之间的最小重叠，以使预测被视为真阳性。
parser.add_argument('--conf_thresh', type=restricted_float, default='0.25')
parser.add_argument('--iou_thresh', type=restricted_float, default='0.3')
parser.add_argument('--imshow', type=boolean_string, default=True, help="Show result onscreen (unusable on colab, jupyter...)")
parser.add_argument('--imwrite', type=boolean_string, default=True, help="Write result to output folder")
parser.add_argument('--show_det', type=boolean_string, default=False, help="Output detection result exclusively")
parser.add_argument('--show_seg', type=boolean_string, default=False, help="Output segmentation result exclusively")
parser.add_argument('--cuda', type=boolean_string, default=True)
parser.add_argument('--float16', type=boolean_string, default=False, help="Use float16 for faster inference")
parser.add_argument('--speed_test', type=boolean_string, default=True,
                    help='Measure inference latency')  # speed_test原默认值为False
args = parser.parse_args()

params = Params(f'projects/{args.project}.yml')  # 从.yml字典文件中读取配置信息
color_list_seg = {}
for seg_class in params.seg_list:
    # edit your color here if you wanna fix to your liking
    color_list_seg[seg_class] = list(np.random.choice(range(256), size=3))
compound_coef = args.compound_coef
source = args.source
if source.endswith("/"):
    source = source[:-1]
output = args.output
if output.endswith("/"):
    output = output[:-1]
weight = args.load_weights
img_path = glob(f'{source}/*.jpg') + glob(f'{source}/*.png')
# img_path = [img_path[0]]  # demo with 1 image
input_imgs = []
shapes = []
det_only_imgs = []

anchors_ratios = params.anchors_ratios
anchors_scales = params.anchors_scales

threshold = args.conf_thresh
iou_threshold = args.iou_thresh
imshow = args.imshow
imwrite = args.imwrite
show_det = args.show_det
show_seg = args.show_seg
os.makedirs(output, exist_ok=True)

use_cuda = args.cuda
use_float16 = args.float16
# cudnn.fastest = True
# cudnn.benchmark = True

obj_list = params.obj_list  # obj_list仅识别car
seg_list = params.seg_list  # 识别road，lane

color_list = standard_to_bgr(STANDARD_COLORS)
ori_imgs = [cv2.imread(i, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION) for i in img_path]  # 读图片
ori_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in ori_imgs]  # cv2.cvtColor:图像色彩变换,参数cv2.COLOR_BGR2RGB:BGR转RGB
print(f"FOUND {len(ori_imgs)} IMAGES")  # 输出找到多少张图片
# cv2.imwrite('ori.jpg', ori_imgs[0])
# cv2.imwrite('normalized.jpg', normalized_imgs[0]*255)
resized_shape = params.model['image_size']
if isinstance(resized_shape, list):  # isinstance判断resized_shape是否为list型 ???
    resized_shape = max(resized_shape)  # resized_shape=原尺度（长宽）中最大值：640



# 细改
normalize = transforms.Normalize(  
    mean=params.mean, std=params.std  # params.n从yml文件中读n对应的数据
)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])




for ori_img in ori_imgs:
    h0, w0 = ori_img.shape[:2]  # orig hw h0=720 w0=1280
    r = resized_shape / max(h0, w0)  # resize image to img_size r=放缩比
    input_img = cv2.resize(ori_img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)  # 图片放缩到原r=0.5倍
    h, w = input_img.shape[:2]  # h，w放缩后的大小

    (input_img, _), ratio, pad = letterbox((input_img, None), resized_shape, auto=True,
                                              scaleup=False)  # 缩放并在图片顶部、底部添加灰边，pad灰边长，letterbox中进一步放缩的比率

    input_imgs.append(input_img)  # 将处理后的图片存入input_imgs，每次循环存一张
    # cv2.imwrite('input.jpg', input_img * 255)  #*255？ # 640*360 -》640*384 上下多出灰边(114，114，114) 确保BiFPN的尺寸可以被128整除
    shapes.append(((h0, w0), ((h / h0, w / w0), pad)))  # for COCO mAP rescaling  存原图尺寸，及之前进行放缩剪裁操作的系数，便于以后恢复

# if use_cuda:  
#     x = paddle.stack([transform(fi).cuda() for fi in input_imgs], 0)  # x=cuda格式的图片？
# else:
#     x = paddle.stack([transform(fi) for fi in input_imgs], 0)
x = paddle.stack([transform(fi) for fi in input_imgs], 0)

# x = x.to(torch.float16 if use_cuda and use_float16 else torch.float32)  # x.data (n, 3, 384, 640) n张图
# print(x.shape)  
weight = paddle.load(weight)  # 读取权重参数
#new_weight = OrderedDict((k[6:], v) for k, v in weight['model'].items())
# weight_last_layer_seg = weight['segmentation_head.0.weight']
# if weight_last_layer_seg.size(0) == 1:  # 
#     seg_mode = BINARY_MODE
# else:
#     if params.seg_multilabel:
#         seg_mode = MULTILABEL_MODE
#     else:
#         seg_mode = MULTICLASS_MODE
seg_mode = MULTICLASS_MODE
print("DETECTED SEGMENTATION MODE FROM WEIGHT AND PROJECT FILE:", seg_mode)
model = HybridNetsBackbone(compound_coef=compound_coef, num_classes=len(obj_list), ratios=eval(anchors_ratios),
                           scales=eval(anchors_scales), seg_classes=len(seg_list), backbone_name=args.backbone,
                           seg_mode=seg_mode) # 定义网络
model = ModelWithLoss(model, debug=False)
model.set_state_dict(weight['model'])  # 加载权重

# model.requires_grad_(False)  # 测试不需要计算梯度
model.eval()

# if use_cuda:
#     model = model.cuda()
#     if use_float16:
#         model = model.half()

with paddle.no_grad():
    # features, regression, classification, anchors, seg = model(x)  # 进入网络
    anchors = model.model.anchors(x, x.dtype)
    features, regression, classification, seg = model.model(x)  # test
    # features：存储BiFPN网络输出的feature maps
    # regression：(n,46035,4)对anchor进行修正 n：共n张图 4：0列修正anchor box y轴中心，1列修正anchor box x轴中心 2列修正anchor box框高 3列修正anchor box框宽
    # classification：(n,46035,1)存各anchor box的类别分数 0至1
    # anchor：(n,46035,4) 4：archor中第0列和第2列为左上和右下区域的纵坐标 archor中第1列和第3列为左上和右下区域的横坐标  
    # seg：(n,3,384,640) 3：0背景，1可行驶区域，2车道线，三类分割的分数
    # _：接收测试用时间戳没用

    # in case of MULTILABEL_MODE, each segmentation class gets their own inference image
    seg_mask_list = []
    # (B, C, W, H) -> (B, W, H)
    if seg_mode == BINARY_MODE:
        seg_mask = paddle.where(seg >= 0, 1, 0)
        # print(torch.count_nonzero(seg_mask))
        seg_mask.squeeze_(1)
        seg_mask_list.append(seg_mask)
    elif seg_mode == MULTICLASS_MODE:
        seg_mask = paddle.argmax(seg, 1)  # 三张区域类别分数张量在每个像素上比较，取最大值，0：背景 1：可行驶区域 2：车道线
        seg_mask_list.append(seg_mask)
    else:
        seg_mask_list = [paddle.where(F.sigmoid(seg)[:, i, ...] >= 0.5, 1, 0) for i in range(seg.size(1))]  # 有问题
        # but remove background class from the list
        seg_mask_list.pop(0)
    # (B, W, H) -> (W, H)
    for i in range(seg.shape[0]):  # 遍历所有seg .size(0) 0维数量
        #   print(i)
        for seg_class_index, seg_mask in enumerate(seg_mask_list):  # seg_class_index？？？
            seg_mask_ = seg_mask[i].squeeze().numpy()  
            # shapes.append(((h0, w0), ((h / h0, w / w0), pad)))
            # shapes[i][1][1][1] [i] shapes中第一组数据（第一张图）[1]：((h / h0, w / w0), pad) [1][1]:pad[1] [1][0]:pad[0]
            pad_h = int(shapes[i][1][1][1])  # 上下的灰边长
            pad_w = int(shapes[i][1][1][0])  # 左右灰边长
            seg_mask_ = seg_mask_[pad_h:seg_mask_.shape[0]-pad_h, pad_w:seg_mask_.shape[1]-pad_w]  # 分割图去灰边
            seg_mask_ = cv2.resize(seg_mask_, dsize=shapes[i][0][::-1], interpolation=cv2.INTER_NEAREST)  # 图像复原原大小
            color_seg = np.zeros((seg_mask_.shape[0], seg_mask_.shape[1], 3), dtype=np.uint8)  # 初始化一个原图大小*3的3维矩阵
            for index, seg_class in enumerate(params.seg_list):
                    color_seg[seg_mask_ == index+1] = color_list_seg[seg_class]  # color_seg:染色后mask对等于1，2的点分别染色
            color_seg = color_seg[..., ::-1]  # RGB -> BGR  
            # cv2.imwrite('seg_only_{}.jpg'.format(i), color_seg)  # 仅显示标注的图片， 标注头处理结果
            
            color_mask = np.mean(color_seg, 2)  # (H, W, C) -> (H, W), check if any pixel is not background  # 对每个点的rgb求均值，平均值为0是背景，上色用
            # prepare to show det on 2 different imgs
            # (with and without seg) -> (full and det_only)
            det_only_imgs.append(ori_imgs[i].copy())  # python 中对对象直接赋另一个对象名实际上二者指向同一存储空间，利用.copy()生成新空间
            seg_img = ori_imgs[i].copy() if seg_mode == MULTILABEL_MODE else ori_imgs[i]  # do not work on original images if MULTILABEL_MODE
            seg_img[color_mask != 0] = seg_img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5  # seg_img存原图，对原图上色（分割区域即color_mask!=0）
            seg_img = seg_img.astype(np.uint8)  # 原本就是uint8？？？
            
            seg_filename = f'{output}/{i}_{params.seg_list[seg_class_index]}_seg.jpg' if seg_mode == MULTILABEL_MODE else \
                           f'{output}/{i}_seg.jpg'
            if show_seg or seg_mode == MULTILABEL_MODE:
                cv2.imwrite(seg_filename, cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))  # 为什么不全程使用rgb？
            if imshow:
                cv2.imshow('img', seg_img)
                cv2.waitKey(0)

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    out = postprocess(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      threshold, iou_threshold)  # 在640*384比例上确定检测框

    for i in range(len(ori_imgs)):
        # 预测框比例由640*384比例转为1280*720  # out[i]第i张图片中的车辆标注信息，shapes存储预处理图像的参数
        out[i]['rois'] = scale_coords(ori_imgs[i][:2], out[i]['rois'], shapes[i][0], shapes[i][1])  
        for j in range(len(out[i]['rois'])):  # 在原图片中各个被识别的目标（车辆）进行标注
            x1, y1, x2, y2 = out[i]['rois'][j].astype(int)  # x1, y1左上 x2, y2右下的坐标

            # 用txt保存 
            with open('write_data.txt','a') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！ 
                f.write(f"{x1} {x2} {y1} {y2}\n")
    
            obj = obj_list[out[i]['class_ids'][j]]  # 类别
            score = float(out[i]['scores'][j])  # 预测分数
            plot_one_box(ori_imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])
            if show_det:
                plot_one_box(det_only_imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                             color=color_list[get_index_label(obj, obj_list)])

        if show_det:
        # if True:
            cv2.imwrite(f'{output}/{i}_det.jpg',  cv2.cvtColor(det_only_imgs[i], cv2.COLOR_RGB2BGR))

        if imshow:
            cv2.imshow('img', ori_imgs[i])
            cv2.waitKey(0)

        # if imwrite:
        #     cv2.imwrite(f'{output}/{i}.jpg', cv2.cvtColor(ori_imgs[i], cv2.COLOR_RGB2BGR))



# if not args.speed_test:
#     exit(0)
# print('running speed test...')
# with torch.no_grad():
#     print('test1: model inferring and postprocessing')  
#     print('inferring 1 image for 10 times...')
#     x = x[0, ...]  # 取第一个维度第一组数据（第一张图）
#     x.unsqueeze_(0)  # 增加一个维度（恢复格式）
#     tn = np.random.rand(2,10)  # test 存储检测点时间戳
#     tmodel = np.random.rand(10,6)  # 存储model内检测点时间戳
#     shijiancha = np.random.rand(2,10) # test 存储经过数据处理时间纵坐标第i+1次测试（共重复十次），第一行为经过网络用时，第二行为检测框后续处理用时。
#     shijianchamodel = np.random.rand(10,5)
#     iiii = 0  # test
#     t1 = time.time()
#     for _ in range(10):
#         _, regression, classification, anchors, segmentation, tmodel[iiii,:] = model(x)
#         tn[0][iiii] = time.time() # test
#         out = postprocess(x,
#                           anchors, regression, classification,
#                           regressBoxes, clipBoxes,
#                           threshold, iou_threshold)
#         tn[1][iiii] = time.time() # test
#         iiii = iiii + 1 # test

#     t2 = time.time()
#     tact_time = (t2 - t1) / 10

#     for i in range(10):  # test
#         if i==0:
#             shijiancha[0][i] = (tn[0][i] - t1)
#             shijiancha[1][i] = (tn[1][i] - tn[0][i])
#         else:
#             shijiancha[0][i] = (tn[0][i] - tn[1][i-1])
#             shijiancha[1][i] = (tn[1][i] - tn[0][i])
    
#     for i in range(10):  # test
#         for j in range(5):
#             shijianchamodel[i][j] = (tmodel[i][j+1]-tmodel[i][j])
    
#     t_mean =np.sum(shijiancha, axis=1)/10  # test  测时间均值
#     tmodel_mean =np.sum(shijianchamodel, axis=0)/10  # test 测时间均值

#     print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

#     # uncomment this if you want a extreme fps test
#     print('test2: model inferring only')  
#     print('inferring images for batch_size 32 for 10 times...')  
#     t1 = time.time()
#     x = torch.cat([x] * 32, 0)  # 第0维拼接 32*3*640*384
#     shijianchamodel = np.random.rand(10,5)
#     iiii = 0
#     for _ in range(10):
#         _, regression, classification, anchors, segmentation, tmodel[iiii,:] = model(x)
#         iiii = iiii + 1 

#     for i in range(10):  # test
#         for j in range(5):
#             shijianchamodel[i][j] = (tmodel[i][j+1]-tmodel[i][j])
    
#     tmodel_mean = np.sum(shijianchamodel, axis=0)/10  # test测时间均值

#     t2 = time.time()
#     tact_time = (t2 - t1) / 10
#     print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')
