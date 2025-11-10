import torch
import torch.nn.functional as F
import torchvision
import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np
import cv2
import json
import math
import numpy as np
import sys
import geojson
import torch
sys.path.append('segment_anything')
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class BSeg_SAM:
    def __init__(self,points_per_side=32,crop_n_layers=0):
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        segmodel_checkpoint = 'last.pth'
        out_channels = 2
        resnet = '34'
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=points_per_side,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=crop_n_layers,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100, 
            )

        self.point_grids = self.mask_generator.point_grids.copy()

        self.segmodel = SegModel(out_channels= out_channels,resnet=resnet)
        self.segmodel.load_state_dict(torch.load(segmodel_checkpoint,weights_only=False))
        self.segmodel.to(device=device)


    def model_deal_img(self,img,point_mask):
        mask_generator = self.mask_generator
        w,h,_ = img.shape

        point_grids = self.point_grids
        for i, point_grid in enumerate(point_grids):
                mask_generator.point_grids[i] = point_grid
                points =  mask_generator.point_grids[i]*[h,w]
                labels = np.array([point_mask[int(p[1]),int(p[0])] for p in points])
                mask_generator.point_grids[i]=points[labels]/[h,w]
        with torch.no_grad():
            masks = mask_generator.generate(img[:,:,:3])
        points_ = []
        masks_img = []
        for mask_ in masks:
            points_.append(mask_['point_coords'][0])
            masks_img.append(mask_['segmentation'])

        return masks_img,masks,points,labels,points_

    def imagemask_deal(self,img,mask_erode=None):
        label = self.segmodel(img)
        labels_ = None
        if mask_erode:
            label_ = np.repeat(np.expand_dims(label, axis=2),3, axis=2)
            kernel = np.ones((5, 5), np.uint8)  # 5x5 的全1矩阵
            label_ = cv2.erode(label_.astype(np.uint8), kernel, iterations=1)
            label_ = cv2.cvtColor(label_,cv2.COLOR_RGBA2GRAY)
        else:
            label_  = label

        features0 = []
        masks_imgs = np.zeros(label_.shape[:2])
        result = None
        masks = None
        
        for i in range(label_.max()):
            point_mask = label_==(1+i)
            masks_img,masks, points,labels,points_ = self.model_deal_img(img,point_mask)
            maskvcts = mask2vct(masks_img,rdp_use=False,rdp_v=2)
            for points_ in maskvcts[0:]:
                    points = np.array(points_, np.float64)
                    points[:,1] = -points[:,1]
                    if len(points)>=4:
                        features0.append(geojson.Feature(geometry = geojson.Polygon([points.tolist()]),properties={'class':'','class_name':(1+i)}))
            masks_imgs[sum(masks_img)>0]=(1+i)
        result = geojson.FeatureCollection(features0)
        return result, masks_imgs,masks

def rdp(points, epsilon):
    start = np.tile(np.expand_dims(points[0], axis=0), (points.shape[0], 1))
    end = np.tile(np.expand_dims(points[-1], axis=0), (points.shape[0], 1))
    dist_point_to_line = np.abs(np.cross(end - start, points - start, axis=-1)) / np.linalg.norm(end - start, axis=-1)
    max_idx = np.argmax(dist_point_to_line)
    max_value = dist_point_to_line[max_idx]

    result = []
    if max_value > epsilon:
        partial_results_left = rdp(points[:max_idx + 1], epsilon)
        result += [list(i) for i in partial_results_left if list(i) not in result]
        partial_results_right = rdp(points[max_idx:], epsilon)
        result += [list(i) for i in partial_results_right if list(i) not in result]
    else:
        result += [points[0], points[-1]]

    return result

def mask2vct(masks,rdp_use=True,rdp_v=2):
    pointss=[]
    for img in masks:
        img = np.array(img, np.uint8)
        contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#得到轮廓信息
        for i,contour in enumerate(contours[0:1]):
            contour = contour.reshape((contour.shape[0],2))
            points=[]
            for c in contour:
                points.append((int(c[0]),int(c[1])))
            if rdp_use:points = rdp(np.array(points),rdp_v)
            pointss.append(list(points))
    return pointss


class SegModel(nn.Module):
    def __init__(self,in_channels=3, out_channels=4,resnet='34'):
        super(SegModel, self).__init__()
        self.test = False
        if resnet=='34':
            expansion = 1
        else: expansion = 4
        if resnet=='34':
            self.encoder_1 = torchvision.models.resnet34(pretrained=True)
            self.encoder_2 = torchvision.models.resnet34(pretrained=True)
        if resnet=='50':
            self.encoder_1 = torchvision.models.resnet50(pretrained=True)
            self.encoder_2 = torchvision.models.resnet50(pretrained=True)
        if resnet=='101':
            self.encoder_1 = torchvision.models.resnet101(pretrained=True)
            self.encoder_2 = torchvision.models.resnet101(pretrained=True)
        return_nodes = {
            'layer1': 'feat1',
            'layer2': 'feat2',
            'layer3': 'feat3',
            'layer4': 'feat4'
        }
        self.extractor_1 = create_feature_extractor(self.encoder_1, return_nodes=return_nodes)
        self.extractor_2 = create_feature_extractor(self.encoder_2, return_nodes=return_nodes)

        self.convlstm_4 = ConvLSTM(input_dim=512 * expansion , hidden_dim=128, kernel_size=(3, 3), num_layers=1,
                                   batch_first=True)
        self.convlstm_3 = ConvLSTM(input_dim=256 * expansion , hidden_dim=128, kernel_size=(3, 3), num_layers=1,
                                   batch_first=True)
        self.convlstm_2 = ConvLSTM(input_dim=128 * expansion , hidden_dim=128, kernel_size=(3, 3), num_layers=1,
                                   batch_first=True)
        self.convlstm_1 = ConvLSTM(input_dim=64 * expansion , hidden_dim=128, kernel_size=(3, 3), num_layers=1,
                                   batch_first=True)
        
        self.trans_layer_4 = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=512 * expansion, out_channels=128, padding=1),
                                             nn.BatchNorm2d(128), nn.ReLU())
        self.trans_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=256 * expansion, out_channels=128, padding=1),
                                             nn.BatchNorm2d(128), nn.ReLU())
        self.trans_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=128 * expansion, out_channels=128, padding=1),
                                             nn.BatchNorm2d(128), nn.ReLU())
        self.trans_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=64 * expansion, out_channels=128, padding=1),
                                             nn.BatchNorm2d(128), nn.ReLU())

        self.smooth_layer_13 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_12 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_11 = ResBlock(in_channels=128, out_channels=128, stride=1) 

        self.smooth_layer_23 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_22 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_21 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        

        self.main_clf_loc = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)
        self.main_clf_clf = nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=1)

    def _upsample_add(self, x, y):
        x = self.trans_layer_3(x)
        _, _, H, W = y.size()
        x = F.interpolate(x, size=(H, W), mode='bilinear') + y
        x = self.smooth_layer_13(x)
        return x 
        
    def forward(self, pre_data):
        x1_features = self.extractor_1(pre_data)
        x2_features = self.extractor_2(pre_data)
        features = {k: v for k, v in x1_features.items() if k.startswith("feat")}
        level_configs = [
            {"feat_key": "feat4", "trans_layer": self.trans_layer_4, 
             "convlstm": self.convlstm_4, "smooth_layers": [self.smooth_layer_14]},
            {"feat_key": "feat3", "trans_layer": self.trans_layer_3, 
             "convlstm": self.convlstm_3, "smooth_layers": [self.smooth_layer_13, self.smooth_layer_23]},
            {"feat_key": "feat2", "trans_layer": self.trans_layer_2, 
             "convlstm": self.convlstm_2, "smooth_layers": [self.smooth_layer_12, self.smooth_layer_22]},
            {"feat_key": "feat1", "trans_layer": self.trans_layer_1, 
             "convlstm": self.convlstm_1, "smooth_layers": [self.smooth_layer_11, self.smooth_layer_21]}
        ]
        
        loc_features = []
        clf_features = []
        prev_loc = None
        prev_clf = None
        for config in level_configs:
            x1_feat = x1_features[config["feat_key"]]
            x2_feat = x2_features[config["feat_key"]]
            loc, clf = process_level(
                x1_feat=x1_feat,
                x2_feat=x2_feat,
                trans_layer=config["trans_layer"],
                convlstm=config["convlstm"],
                smooth_layers=config["smooth_layers"],
                upsample_target=prev_loc
            )
            loc_features.append(loc)
            clf_features.append(clf)
            prev_loc = loc
            prev_clf = clf

        output_loc = self.main_clf_loc(loc_features[0])
        output_loc = F.interpolate(output_loc, size=x1_data.size()[-2:], mode='bilinear')
        output_clf = self.main_clf_clf(clf_features[0])
        output_clf = F.interpolate(output_clf, size=x1_data.size()[-2:], mode='bilinear')
        
        if self.test:  return output_clf
        return output_loc, output_clf

def process_level(x1_feat, x2_feat, trans_layer, convlstm, smooth_layers, upsample_target=None):
    loc = trans_layer(x1_feat)
    combined = torch.stack([x1_feat, x2_feat], dim=1)
    _, last_state = convlstm(combined)
    feat = last_state[0][0]
    
    if upsample_target is not None:
        loc = _upsample_add(upsample_target, loc)
        feat = _upsample_add(upsample_target, feat)
    
    loc = smooth_layers[0](loc)  # 主平滑
    feat = smooth_layers[1](feat) if len(smooth_layers) > 1 else feat
    return loc, feat

    
'''
Hongruixuan Chen, Chen Wu, Bo Du, Liangpei Zhang, and Le Wang: 
Change Detection in Multisource VHR Images via Deep Siamese Convolutional Multiple-Layers Recurrent Neural Network, 
IEEE Trans. Geosci. Remote Sens., 58, 2848–2864, 2020. 
https://github.com/ChenHongruixuan/SiamCRNN/tree/master/FCN_version
'''
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()


        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
