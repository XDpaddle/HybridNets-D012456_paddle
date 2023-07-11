import argparse
import datetime
import os
import traceback

import numpy as np
# from tensorboardX import SummaryWriter
from tqdm.autonotebook import tqdm

from val import val
from backbone import HybridNetsBackbone
from utils.utils import get_last_weights, boolean_string, \
     DataLoaderX, Params  # init_weights, save_checkpoint被删除
from hybridnets.dataset import BddDataset
# from hybridnets.custom_dataset import CustomDataset
from hybridnets.autoanchor import run_anchor
from hybridnets.model import ModelWithLoss
from utils.constants import *
from collections import OrderedDict


# import torch
# from torch import nn
# from torchvision import transforms
import paddle
import paddle.nn
from paddle.vision import transforms
from paddle.io import DataLoader 
import sys  # 调试

def get_args():
    parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
    parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
    parser.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. '
                                                            'https://github.com/rwightman/pytorch-image-models')
    parser.add_argument('-c', '--compound_coef', type=int, default=1, help='Coefficient of efficientnet backbone')
    parser.add_argument('-n', '--num_workers', type=int, default=8, help='Num_workers of dataloader')  # wimdows不支持多workers
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Number of images per batch among all devices')
    parser.add_argument('--freeze_backbone', type=boolean_string, default=False,
                        help='Freeze encoder and neck (effnet and bifpn)')
    parser.add_argument('--freeze_det', type=boolean_string, default=False,
                        help='Freeze detection head')
    parser.add_argument('--freeze_seg', type=boolean_string, default=True,
                        help='Freeze segmentation head')
    

    parser.add_argument('--lr', type=float, default=1e-4)


    parser.add_argument('--optim', type=str, default='adamw', help='Select optimizer for training, '
                                                                   'suggest using \'adamw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=50000, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which '
                             'training will be stopped. Set to 0 to disable this technique')
    parser.add_argument('--data_path', type=str, default='datasets/', help='The root folder of dataset')
    parser.add_argument('--log_path', type=str, default='checkpoints/')
    parser.add_argument('-w', '--load_weights', type=str, default='D:\\deeplearning\\HybridNets-paddle-main\\final\\hy-paddle-b1-final0609\\checkpoints\\bdd100k\\hybridnets-checkpoints.pdparams',
                        help='Whether to load weights from a checkpoint, set None to initialize,'
                             'set \'last\' to load last checkpoint')
    # '/root/autodl-tmp/hy-paddle-final0530new/checkpoints/bdd100k/hybridnets-checkpoints.pdparams'
    # 'D:\\deeplearning\\HybridNets-paddle-main\\new-0602\\hy-paddle-final0530new\\checkpoints\\bdd100k\\hybridnets-d3_64_8750.pdparams'
    parser.add_argument('--saved_path', type=str, default='checkpoints/')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='Whether visualize the predicted boxes of training, '
                             'the output images will be in test/, '
                             'and also only use first 500 images.')
    parser.add_argument('--cal_map', type=boolean_string, default=True,
                        help='Calculate mAP in validation')
    parser.add_argument('-v', '--verbose', type=boolean_string, default=True,
                        help='Whether to print results per class when valing')
    parser.add_argument('--plots', type=boolean_string, default=False,
                        help='Whether to plot confusion matrix when valing')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to be used (0 to use CPU)')
    parser.add_argument('--conf_thres', type=float, default=0.001,
                        help='Confidence threshold in NMS')
    parser.add_argument('--iou_thres', type=float, default=0.6,
                        help='IoU threshold in NMS')
    parser.add_argument('--amp', type=boolean_string, default=False,
                        help='Automatic Mixed Precision training')

    args = parser.parse_args()
    return args

def train(opt):
    if opt.num_gpus > 0:  # 使用哪个gpu
        paddle.device.set_device('gpu:0')
    print("\nCUDNN VERSION: {}\n".format(paddle.device.get_cudnn_version()))  
    params = Params(f'projects/{opt.project}.yml')

    if opt.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # if paddle.device.is_compiled_with_cuda():
    #     torch.cuda.manual_seed(42)
    # else:
    #     torch.manual_seed(42)

    # 可能不对
    # paddle.seed(42)


    opt.saved_path = opt.saved_path + f'/{opt.project}/'  # ？？？存储路径
    opt.log_path = opt.log_path + f'/{opt.project}/tensorboard/'  # tensorboard存储路径
    os.makedirs(opt.log_path, exist_ok=True)  # 创建路径
    os.makedirs(opt.saved_path, exist_ok=True)

    seg_mode = MULTILABEL_MODE if params.seg_multilabel else MULTICLASS_MODE if len(params.seg_list) > 1 else BINARY_MODE

    img_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=params.mean, std=params.std
        )
    ])
    # 细改dataset
    train_dataset = BddDataset(  # 预设train数据集信息，获取detect标记，图片，分割标记路径
        params=params,
        is_train=True,
        inputsize=params.model['image_size'],  # 640*384
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=params.mean, std=params.std
            )
        ]),
        seg_mode=seg_mode,
        debug=opt.debug
    )

    # 细改dataloader
    training_generator = DataLoader(  # ？？
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        # pin_memory=params.pin_memory,
        collate_fn=BddDataset.collate_fn
    )
    
    # validation的数据集和training没有交集，所以这部分数据对最终训练出的模型没有贡献。
    # validation的主要作用是来验证是否过拟合、以及用来调节训练参数等。
 
    # 当val数据集loss上升而，train数据集loss还在下降时证明过拟合
    valid_dataset = BddDataset(  # 预设val数据集信息，获取detect标记，图片，分割标记路径
        params=params,
        is_train=False,
        inputsize=params.model['image_size'],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=params.mean, std=params.std
            )
        ]),
        seg_mode=seg_mode,
        debug=opt.debug
    )

    val_generator = DataLoader(  # ？？？
        valid_dataset, 
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        # pin_memory=params.pin_memory,
        collate_fn=BddDataset.collate_fn
    )

    # 可能不对
    if params.need_autoanchor:
        params.anchors_scales, params.anchors_ratios = run_anchor(None, train_dataset)

    model = HybridNetsBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                               ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales),
                               seg_classes=len(params.seg_list), backbone_name=opt.backbone,
                               seg_mode=seg_mode)  # 导入模型

    # load last weights
    ckpt = {}
    # last_step = None
    model = ModelWithLoss(model, debug=opt.debug)
    
    if opt.load_weights:  # 导入权重
        if opt.load_weights.endswith('.pdparams'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)

        # try:
        #     last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        # except:
        #     last_step = 0

        try:
            ckpt = paddle.load(weights_path)
            # new_weight = OrderedDict((k[6:], v) for k, v in ckpt['model'].items())
            model.set_state_dict(ckpt.get('model', ckpt))
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')
    else:
        print('[Info] use initializing weights...')
        # init_weights(model)  # 初始化权重参数

    print('[Info] Successfully!!!')





    # 根据后续读入数据决定bifpn是否停止训练



    if opt.freeze_backbone:  # 是否停止训练这一部分
        # for i, param in enumerate(model.model.encoder.named_parameters()):  # 返回所有最底层的参数nn.Conv2D，nn.BatchNorm2D，nn.Linear
        #     # print(param)
        #     param[1].trainable = False
        #     # print(param)
        # for i, param in enumerate(model.model.bifpn.named_parameters()):  # 返回所有最底层的参数nn.Conv2D，nn.BatchNorm2D，nn.Linear
        #     # print(param)
        #     param[1].trainable = False
        #     # print(param)
        for i, param in model.model.efficientdet.named_parameters():  # 返回所有最底层的参数nn.Conv2D，nn.BatchNorm2D，nn.Linear
            # print(param)
            param.trainable = False
            # print(param)
        print('[Info] freezed backbone')

    if opt.freeze_det: 
        for i, param in model.model.regressor.named_parameters():  # 返回所有最底层的参数nn.Conv2D，nn.BatchNorm2D，nn.Linear
            # print(param)
            param.trainable = False
            # print(param)
        for i, param in model.model.classifier.named_parameters():  # 返回所有最底层的参数nn.Conv2D，nn.BatchNorm2D，nn.Linear
            # print(param)
            param.trainable = False
            # print(param)
        for i, param in model.model.anchors.named_parameters():  # 返回所有最底层的参数nn.Conv2D，nn.BatchNorm2D，nn.Linear
            # print(param)
            param.trainable = False
            # print(param)
        print('[Info] freezed detection head')

    if opt.freeze_seg:
        for i, param in model.model.bifpndecoder.named_parameters():  # 返回所有最底层的参数nn.Conv2D，nn.BatchNorm2D，nn.Linear
            # print(param)
            param.trainable = False
            # print(param)
        for i, param in model.model.segmentation_head.named_parameters():  # 返回所有最底层的参数nn.Conv2D，nn.BatchNorm2D，nn.Linear
            # print(param)
            param.trainable = False
            # print(param)
        print('[Info] freezed segmentation head')
    #summary(model, (1, 3, 384, 640), device='cpu')

    # SummaryWriter：在给定给目录中创建tensorboard事件文件
    # writercheckpoints//bdd100k/tensorboard//20230227-110904/
    # writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # wrap the model with loss function, to reduce the memory usage on gpu0 and speedup
    # 用loss函数包装模型，以减少gpu0上的内存使用并加速
    

    # model = model.to(memory_format=torch.channels_last)  # 优化？改为to gpu

    # if opt.num_gpus > 0:
    #     model = model.cuda()



    # optim_params = []
    # # i = 0  # 测试
    # for n,m in model.model.named_parameters():
    #     i = i + 1  # 测试 
    #     print(n,m.stop_gradient)  # 测试 
    #     if not m.stop_gradient:
    #         # print(n)  # 测试 
    #         optim_params.append(m)

    optim_params = model.model.parameters()

    # j = 0  # 测试
    # for n,m in model.named_parameters():
    #     j = j + 1
    #     print(n,m.stop_gradient)
    





    scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=opt.lr, patience=3, verbose=True)  # # 动态学习率
    if opt.optim == 'adamw':  # 指定优化器
        # optimizer = torch.optim.AdamW(model.parameters(), opt.lr)  # 删  
        optimizer = paddle.optimizer.AdamW(parameters=optim_params, learning_rate=scheduler)
    else:
        # optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)  # 删
        optimizer = paddle.optimizer.SGD(parameters=optim_params, learning_rate=scheduler)
    # print(ckpt)
    # scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)  # 混合精度训练  # 删
    # scaler = paddle.amp.GradScaler(enable=opt.amp)  # init_loss_scaling 选多少合适？？？？？？？？？？？？？？？？







    if opt.load_weights is not None and ckpt.get('opt', None):
        optimizer.set_state_dict(ckpt['opt'])







    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)  # 动态学习率


    best_loss = 1e5
    best_epoch = 0
    last_step = ckpt['step'] if opt.load_weights is not None and ckpt.get('step', None) else 0
    best_fitness = ckpt['best_fitness'] if opt.load_weights is not None and ckpt.get('best_fitness', None) else 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)
    is_train=False  # 等于False进行测试
    try:
        for epoch in range(opt.num_epochs):
            # last_epoch = step // num_iter_per_epoch
            # if epoch < last_epoch:
            #     continue
            if opt.load_weights is not None:
                epoch = ckpt['epoch']
            epoch_loss = 0
            epoch_loss_car = 0
            epoch_loss_seg = 0
            step = 1

            # test1 = []  # 删
            # test2 = []
            if is_train:
                # progress_bar = tqdm(training_generator, ascii=True)
                # for iter, data in enumerate(progress_bar):
                for data in training_generator:
                    # if iter < step - last_epoch * num_iter_per_epoch:
                    #     progress_bar.update()
                    #     continue
                    try:
                        imgs = data['img']
                        annot = data['annot']
                        seg_annot = data['segmentation']
                        imgs = imgs.numpy()
                        imgs_tensor = []
                        for i in range(imgs.shape[0]):
                            imgs_tensor.append(img_transform(imgs[i]))
                        imgs = paddle.stack(imgs_tensor, 0)
                        annot = paddle.to_tensor(annot, dtype=paddle.float32)
                        seg_annot = paddle.to_tensor(seg_annot, dtype=paddle.int64)

                        # print(paddle.device.cuda.memory_reserved(0), step)

                        if step == 1:
                            anchors = model.model.anchors(imgs, imgs.dtype)
                        # if opt.num_gpus == 1:  # 全删？
                        #     # if only one gpu, just send it to cuda:0
                        #     imgs = imgs.to(device="cuda", memory_format=torch.channels_last)
                        #     annot = annot.cuda()
                        #     seg_annot = seg_annot.cuda()

                        # optimizer.clear_grad()
                        # with paddle.amp.auto_cast(enable=opt.amp):
                        cls_loss, reg_loss, seg_loss, regression, classification, segmentation = model(imgs, annot,
                                                                                                        seg_annot,
                                                                                                        anchors,
                                                                                                        obj_list=params.obj_list)
                        cls_loss = cls_loss.mean() if not opt.freeze_det else paddle.to_tensor(0, dtype=paddle.float32)
                        reg_loss = reg_loss.mean() if not opt.freeze_det else paddle.to_tensor(0, dtype=paddle.float32)
                        seg_loss = seg_loss.mean() if not opt.freeze_seg else paddle.to_tensor(0, dtype=paddle.float32)

                        loss = cls_loss + reg_loss + seg_loss 
                            
                            


                        test = True  # 测试
                        # if test:
                            # iii = 0
                            # for n,m in model.model.named_parameters():                            
                            #     if iii == 0:
                            #         print(n,m.stop_gradient)
                            #     if iii == 10:
                            #         print(n,m.stop_gradient)
                            #     iii = iii + 1
                            # if step == 1:
                            #     test1 = optim_params
                            # if step == 20:
                            #     test2 = optim_params

                            #     for t1, t2 in zip(test1, test2):
                            #         if paddle.equal_all(t1, t2)==False:
                            #             print(t1.name)
                        # if loss == 0 or not paddle.isfinite(loss):
                        #     continue

                        # scaled = scaler.scale(loss)
                        loss.backward()
                        # scaled.backward()
                        # Don't have to clip grad norm, since our gradients didn't explode anywhere in the training phases
                        # This worsens the metrics
                        # scaler.unscale_(optimizer)
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                        # scaler.step(optimizer)
                        # scaler.update()
                        optimizer.step()
                        
                        # if test:
                        #     if step == 1:
                        #         for n,m in model.model.named_parameters():                            
                        #             test1.append([n, m])
                        #     if step == 20:
                        #         for n,m in model.model.named_parameters():                            
                        #             test2.append([n, m])

                        #         for t1, t2 in zip(test1, test2):
                        #             if paddle.equal_all(t1[1], t2[1])==False:
                        #                 print(t1[0])
                        optimizer.clear_grad()

                        # model.clear_gradients()  # ??重复
                        epoch_loss += loss.item()
                        epoch_loss_car = epoch_loss_car + cls_loss.item() + reg_loss.item()
                        epoch_loss_seg = epoch_loss_seg + seg_loss.item()
                        
                        # del regression, classification, anchors, segmentation
                        # del loss, cls_loss, reg_loss, seg_loss
                        # del imgs_tensor, imgs, annot, seg_annot, data
                        # paddle.device.cuda.empty_cache()

                        # epoch_loss.append(float(loss))  # 别用
                        
                        # progress_bar.set_description(
                        #     'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Seg loss: {:.5f}. Total loss: {:.5f}'.format(
                        #         step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                        #         reg_loss.item(), seg_loss.item(), loss.item()))
                        if step%1==0:
                            print(step, epoch, cls_loss.item(), reg_loss.item(), seg_loss.item(), loss.item())
                        # writer.add_scalars('Loss', {'train': loss}, step)
                        # writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                        # writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)
                        # writer.add_scalars('Segmentation_loss', {'train': seg_loss}, step)

                        # log learning_rate
                        # current_lr = optimizer.param_groups[0]['lr']  # 删
                        # current_lr = optimizer._learning_rate
                        # writer.add_scalar('learning_rate', current_lr, step)

                    
                        # print(paddle.device.cuda.memory_reserved(0), step)
                        
                        # memory = paddle.device.cuda.memory_reserved(0)
                        if step % opt.save_interval == 0 and step > 0:
                            # save_checkpoint(model, opt.saved_path, f'hybridnets-d{opt.compound_coef}_{epoch}_{step}.pth')
                            # savepath = opt.saved_path + f'hybridnets-d{opt.compound_coef}_{epoch}_{step}.pdparams'
                            # obj = {'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch}
                            # paddle.save(obj, savepath)
                            print('checkpoint...')
                        step += 1

                    except Exception as e:
                        print('[Error]', traceback.format_exc())
                        print(e)
                        continue
            # epoch = epoch + 1
            scheduler.step(epoch_loss/(step+1))
            if epoch % opt.val_interval == 0:
                # savepath = opt.saved_path + f'hybridnets-d{opt.compound_coef}_{epoch+1}_{step-1}.pdparams'
                # obj = {'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch+1, 'step':step-1}
                # paddle.save(obj, savepath)  
                # savepath = opt.saved_path + 'hybridnets-checkpoints.pdparams'
                # obj = {'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch+1, 'step':step-1}
                # paddle.save(obj, savepath)  
                savepath = opt.saved_path + 'freeze_seg.pdparams'
                obj = {'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch+1, 'step':step-1}
                paddle.save(obj, savepath) 
                with open('loss_log.txt', 'a') as f:
                    f.write('{"epoch_loss":')
                    f.write(f"{np.mean(epoch_loss)}, ")
                    f.write('"epoch_det_car_loss":')
                    f.write(f"{np.mean(epoch_loss_car)}, ")
                    f.write('"epoch_seg_loss":')
                    f.write(f"{np.mean(epoch_loss_seg)}")
                    f.write('"lr":')
                    f.write(f"{optimizer.get_lr()}")         
                    f.write('"epoch":')
                    f.write(f"{epoch+1}")       
                    f.write('}\n\n')    

            
            if epoch % opt.val_interval == 0:
                best_fitness, best_loss, best_epoch = val(model, val_generator, params, opt, seg_mode, is_training=True,
                                                          optimizer=optimizer, epoch=epoch, step=step, best_fitness=best_fitness, 
                                                          best_loss=best_loss, best_epoch=best_epoch, img_transform=img_transform)
            

    except KeyboardInterrupt:
        # save_checkpoint(model, opt.saved_path, f'hybridnets-d{opt.compound_coef}_{epoch}_{step}.pth')  # 不知道怎么改
        savepath = opt.saved_path + f'hybridnets-d{opt.compound_coef}_{epoch}_{step}.pdparams'
        # obj = {'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch}
        # paddle.save(obj, savepath)
        # paddle.save(model.state_dict(), savepath)
    finally:
        # writer.close()
        a = 1


if __name__ == '__main__':
    opt = get_args()  # 设置配置参数
    train(opt)
