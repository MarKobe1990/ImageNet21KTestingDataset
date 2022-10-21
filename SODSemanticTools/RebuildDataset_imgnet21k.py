import datetime

import cv2
import numpy as np
from tensorboardX import SummaryWriter
from treelib import Tree
import urllib
from argparse import Namespace
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from src_files.semantic.semantics import ImageNet21kSemanticSoftmax
import timm
from myTools.data import GetDataset
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
from SODSemanticTools.WordNetTool import build_tree, get_object_hypernym_paths
import myTools.imageUtil as utils
import os.path as osp
import os
import pickle

from src_files.semantic.semantics import ImageNet21kSemanticSoftmax


def init_my_model(metadata_file, checkpoint_file):
    # 配置文件
    args = Namespace()
    args.tree_path = metadata_file
    semantic_softmax_processor = ImageNet21kSemanticSoftmax(args)
    model = timm.create_model('vit_base_patch16_224_miil_in21k', pretrained=False, checkpoint_path=checkpoint_file)
    model.eval()
    model.cuda()
    return model, semantic_softmax_processor


def draw_max_rec_of_mask(gt_path, image_path):
    gt_mask = cv2.imread(gt_path)
    thresh = cv2.Canny(gt_mask, 128, 256)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_origin = cv2.imread(image_path)
    img_copy = np.copy(img_origin)
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for contour in contours:
        # 找到边界坐标
        x, y, w, h = cv2.boundingRect(contour)  # 计算点集最外面的矩形边界
        # 因为这里面包含了，图像本身那个最大的框，所以用了if，来剔除那个图像本身的值。
        # if x != 0 and y != 0 and w != gt_mask.shape[1] and h != gt_mask.shape[0]:
        # if w != gt_mask.shape[1] and h != gt_mask.shape[0]:
        # 左上角坐标和右下角坐标
        # 如果执行里面的这个画框，就是分别来画的，
        # cv2.rectangle(origin_gt, (x, y), (x + w, y + h), (0, 255, 0), 1)
        x1.append(x)
        y1.append(y)
        x2.append(x + w)
        y2.append(y + h)
    x11 = min(x1)
    y11 = min(y1)
    x22 = max(x2)
    y22 = max(y2)
    white = [255, 255, 255]
    for col in range(x11, x22):
        for row in range(y11, y22):
            gt_mask[row, col] = white
    rectangle_mask_gray = cv2.cvtColor(gt_mask, cv2.COLOR_RGB2GRAY)
    out_origin_size = cv2.bitwise_and(img_copy, img_copy, mask=rectangle_mask_gray)
    return gt_mask, out_origin_size, img_copy[y11:y22, x11:x22]


def mask_object_count_and_ratio(gt_path):
    gt_mask = cv2.imread(gt_path, 0)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(gt_mask, connectivity=8)
    ratio_object = (1.0 - float(stats[0][4]) / float(gt_mask.size))
    return retval, labels, stats, centroids, ratio_object


def input_tensor_to_labels(tensor, model, semantic_softmax_processor):
    labels = []
    all_labels_and_prob = {}
    labels_top1_prob = []
    labels_and_prob = {}
    with torch.no_grad():
        logits = model(tensor)
        semantic_logit_list = semantic_softmax_processor.split_logits_to_semantic_logits(logits)
        # scanning hirarchy_level_list
        for i in range(len(semantic_logit_list)):
            logits_i = semantic_logit_list[i]
            # generate probs
            probabilities = torch.nn.functional.softmax(logits_i[0], dim=0)
            top1_prob, top1_id = torch.topk(probabilities, 1)

            top_class_number = semantic_softmax_processor.hierarchy_indices_list[i][top1_id[0]]
            top_class_name = semantic_softmax_processor.tree['class_list'][top_class_number]
            top_class_description = semantic_softmax_processor.tree['class_description'][top_class_name]
            # record all
            if top1_prob > 0.1 and top1_prob <= 0.5:
                top1_prob_float = float(top1_prob).__round__(5)
                all_labels_and_prob[top_class_description] = top1_prob_float
            if top1_prob > 0.5:
                labels.append(top_class_description)
                top1_prob_float = float(top1_prob).__round__(5)
                labels_and_prob[top_class_description] = top1_prob_float
                labels_top1_prob.append(top1_prob_float)
    if labels_and_prob.__len__() > 0:
        labels_top1_prob_sorted_list = sorted(labels_and_prob.items(), key=lambda x: x[1], reverse=True)
        result_dic = dict(labels_top1_prob_sorted_list)
        degree_of_confidence = 'high'
    else:
        degree_of_confidence = 'low'
        all_labels_top1_prob_sorted_list = sorted(all_labels_and_prob.items(), key=lambda x: x[1], reverse=True)
        if len(all_labels_top1_prob_sorted_list) > 1:
            result_dic = dict(all_labels_top1_prob_sorted_list[-2::])
            for key, value in result_dic.items():
                labels.append(key)
                labels_top1_prob.append(value)
        else:
            result_dic = dict(all_labels_top1_prob_sorted_list)
            labels.append(list(result_dic.keys())[0])
            labels_top1_prob.append(list(result_dic.values())[0])
    return labels, labels_top1_prob, result_dic, degree_of_confidence


def rebuild_masked_dataset(dataset_dic, model, semantic_softmax_processor):
    data_set = GetDataset(dataset_dic.get("image_root"), dataset_dic.get("depth_root"),
                          dataset_dic.get("gt_root"))
    pbar = tqdm(range(len(data_set)), desc="rebuild_dataset:", unit='img')
    date_str = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
    save_dir = "../testResults/" + date_str
    dataset_writer = SummaryWriter(save_dir + "/log", comment="log")
    gt_dir = osp.join(save_dir, 'data', 'gt')
    mask_origin_dir = osp.join(save_dir, 'data', 'mask_origin')
    mask_resize_dir = osp.join(save_dir, 'data', 'mask_resize')
    merge_img_dir = osp.join(save_dir, 'data', 'merge_img')
    if not osp.isdir(gt_dir):
        os.makedirs(gt_dir)
    if not osp.isdir(mask_origin_dir):
        os.makedirs(mask_origin_dir)
    if not osp.isdir(mask_resize_dir):
        os.makedirs(mask_resize_dir)
    if not osp.isdir(merge_img_dir):
        os.makedirs(merge_img_dir)
    log_file_loc = save_dir + '/0_output_log.txt'
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    word_tree = Tree()
    error_list = []
    for idx in pbar:
        name, image_path, depth_path, gt_path, img_array = data_set.load_data(idx)
        # 根据mask切割
        gt_mask, out_origin_size, img_copy_resize = draw_max_rec_of_mask(gt_path, image_path)
        # 统计联通区域和显著性物体占比
        retval, labels, stats, centroids, ratio_object = mask_object_count_and_ratio(gt_path)
        img_resize_salient_object_path = osp.join(mask_resize_dir, name)
        img_mask_origin_path = osp.join(mask_origin_dir, name)
        cv2.imwrite(osp.join(gt_dir, name[:-4] + '.png'), gt_mask)
        cv2.imwrite(img_mask_origin_path, out_origin_size)
        cv2.imwrite(img_resize_salient_object_path, img_copy_resize)
        # 原图像的结果
        origin_img = Image.open(image_path).convert('RGB')
        origin_tensor = transform(origin_img).unsqueeze(0).cuda()
        # 切割后的
        masked_img = Image.open(img_resize_salient_object_path).convert('RGB')
        masked_tensor = transform(masked_img).unsqueeze(0).cuda()
        #处理原图像
        origin_labels, origin_labels_top1_prob, origin_labels_and_prob, origin_degree_of_confidence = input_tensor_to_labels(origin_tensor, model, semantic_softmax_processor)
        masked_labels, masked_labels_top1_prob, masked_labels_and_prob, masked_degree_of_confidence = input_tensor_to_labels(masked_tensor, model, semantic_softmax_processor)
        if list(origin_labels_and_prob.keys())[0] == list(masked_labels_and_prob.keys())[0]:
            same_flag = 'T'
        else:
            same_flag = 'F'
        hparam_dic = {
            "file_pth": image_path,
            "pred_class": list(origin_labels_and_prob.keys()).__str__(),
            "pred_score": list(origin_labels_and_prob.values()).__str__(),
            "pred_class_salient_object": list(masked_labels_and_prob.keys()).__str__(),
            "pred_score_salient_object": list(masked_labels_and_prob.values()).__str__(),
            "same_flag": same_flag,
            'origin_degree_of_confidence': origin_degree_of_confidence,
            'masked_degree_of_confidence': masked_degree_of_confidence
        }
        metric_dic = {
            "object_count": retval - 1,
            "object_ratio": ratio_object
        }
        if len(masked_labels_and_prob) > 1:
            if list(masked_labels_and_prob.keys())[0] != 'artifact':
                build_tree(list(masked_labels_and_prob.keys())[0], word_tree, image_path, error_list)
            else:
                build_tree(list(masked_labels_and_prob.keys())[1], word_tree, image_path, error_list)
        else:
            build_tree(list(masked_labels_and_prob.keys())[0], word_tree, image_path, error_list)
        dataset_writer.add_hparams(hparam_dic, metric_dic, name='log/' + image_path)
        merge_img_list = [image_path,
                          depth_path,
                          gt_path,
                          img_mask_origin_path]
        tag_dic = {
            'class_name': list(origin_labels_and_prob.keys()).__str__().strip('[').strip(']'),
            'pred_socre': list(origin_labels_and_prob.values()).__str__().strip('[').strip(']'),
            'ob_class_name': list(masked_labels_and_prob.keys()).__str__().strip('[').strip(']'),
            'ob_pred_socre': list(masked_labels_and_prob.values()).__str__().strip('[').strip(']'),
            'object_ratio': ratio_object,
            'object_count': retval - 1,
            'same_flag': same_flag,
            'name': name[:-4]
        }
        utils.merge_image_file_name_return_tensor(merge_img_list, name[:-4] + '_merge.png', merge_img_dir,
                                                               len(merge_img_list), tag_dic)
        # dataset_writer.add_image(dataset_dic["dataset_name"], image_grid, global_step=idx)

    if os.path.isfile(log_file_loc):
        logger = open(log_file_loc, 'a')
    else:
        logger = open(log_file_loc, 'w')
        # 写入测试
        logger.write(error_list.__str__() + "\n")
        logger.write(word_tree.to_dict(with_data=True).__str__())
    logger.flush()
    word_tree.save2file(save_dir + '/tree_note_count.txt', data_property='count')
    word_tree.save2file(save_dir + '/tree_note.txt')
    word_tree.to_graphviz(filename=save_dir + '/tree_graphviz')
    with open(save_dir + '/wordTree.pkl', 'wb') as f:
        pickle.dump(word_tree, f)



if __name__ == '__main__':
    # conformer
    metadata_file = '../myDocument/imagenet21k_miil_tree.pth'
    # 权重文件参数路径
    checkpoint_file = '../checkpoints/vit_base_patch16_224_miil_21k.pth'

    dataset_dic_train = {
        'image_root': '../data/COME15K/train/imgs_right/',
        'gt_root': '../data/COME15K/train/gt_right/',
        'depth_root': '../data/COME15K/train/depths/',
        'dataset_name': 'come15k_train'
    }

    dataset_dic_val_e = {
        'image_root': '../data/COME15K/val/' + 'COME-E' + '/RGB/',
        'gt_root': '../data/COME15K/val/' + 'COME-E' + '/GT/',
        'depth_root': '../data/COME15K/val/' + 'COME-E' + '/depths/',
        'dataset_name': 'come15k_val_easy'
    }
    dataset_dic_val_h = {
        'image_root': '../data/COME15K/val/' + 'COME-H' + '/RGB/',
        'gt_root': '../data/COME15K/val/' + 'COME-H' + '/GT/',
        'depth_root': '../data/COME15K/val/' + 'COME-H' + '/depths/',
        'dataset_name': 'come15k_val_hard'
    }
    dataset_dic_test_e = {
        'image_root': '../data/COME15K/test/' + 'COME-E' + '/RGB/',
        'gt_root': '../data/COME15K/test/' + 'COME-E' + '/GT/',
        'depth_root': '../data/COME15K/test/' + 'COME-E' + '/depths/',
        'dataset_name': 'come15k_test_easy'
    }
    dataset_dic_test_h = {
        'image_root': '../data/COME15K/test/' + 'COME-H' + '/RGB/',
        'gt_root': '../data/COME15K/test/' + 'COME-H' + '/GT/',
        'depth_root': '../data/COME15K/test/' + 'COME-H' + '/depths/',
        'dataset_name': 'come15k_test_hard'
    }
    model, semantic_softmax_processor = init_my_model(metadata_file, checkpoint_file)
    # 切割图像类型并标注类型和显著图像占比
    rebuild_masked_dataset(dataset_dic_test_h, model, semantic_softmax_processor)
