# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     calc_trace_performance_BigNeuron_v2.py
   Description :
   Author :       'JZhao'
   date：          2020/2/28
   Copyright 2020. All Rights Reserved.
-------------------------------------------------
   Change Activity:
                   2020/2/28:
-------------------------------------------------
"""
import os
import time
import cv2
import numpy as np
import pythoncom
import re
from tqdm import trange
from sklearn.neighbors import KDTree
import concurrent.futures


class InitGTDict():
    def __init__(self, gt_images_folder):
        self.gt_images_folder = gt_images_folder

    @staticmethod
    def _read_3d_image(img_path, log_file=None):
        print('reading 3d image...')
        if log_file:
            log_file.write('reading images ...\n')

        img_list = []
        files_list = os.listdir(img_path)
        for img_name in files_list:
            if img_name.endswith('.tif'):
                img_list.append(img_name)

        if img_list is None:
            if log_file:
                log_file.write('no image is available!\n')
            exit(0)
        if len(img_list) == 1:
            if log_file:
                log_file.write('only 2d image is available!!\n')
            exit(0)

        img_list = sorted(img_list, key=lambda x: int(x.split('.')[0]))

        img = cv2.imread(os.path.join(img_path, img_list[0]), -1)
        img = img[np.newaxis, :]
        time_s = time.time()
        for i in trange(img_list.__len__() - 1):
            img_name = img_list[i + 1]
            img_read = cv2.imread(os.path.join(img_path, img_name), -1)
            img_read = img_read[np.newaxis, :]
            img = np.concatenate((img, img_read), axis=0)
        time_e = time.time()
        if log_file:
            log_file.write('done! elapsed time: {} s\n'.format(time_e - time_s))

        return img

    def _get_img_shape_namelist(self, image_name):
        """
        image shape: [slices, height, width]
        :return:
        """
        img_folder = os.path.join(self.gt_images_folder, image_name)
        if not os.path.exists(img_folder):
            raise ValueError('cannot find the image path: {}!'.format(img_folder))

        # get 3d img shape
        imgs_list = os.listdir(img_folder)
        if imgs_list.__len__() == 0:
            raise ValueError('none image is available: {}!'.format(img_folder))

        img2d_temp = cv2.imread(os.path.join(img_folder, imgs_list[0]), -1)
        height, width = img2d_temp.shape
        slices = len(imgs_list)
        img_shape = [slices, height, width]

        # sort images' name
        name_list = []
        for img_name in imgs_list:
            if img_name.endswith('.tif'):
                name_list.append(img_name)
        name_list = sorted(name_list, key=lambda x: int(x.split('.')[0]))

        return img_shape, name_list

    @staticmethod
    def _read_2d_image(img_folder, img_slice_name):
        img_2d = cv2.imread(os.path.join(img_folder, img_slice_name), -1)
        return img_slice_name, img_2d

    def _read_3d_image_process_pool(self, img_path, log_file=None):
        if not os.path.exists(img_path):
            print('img_path dose not exists! - {}'.format(img_path))
            exit(0)

        print('reading 3d image...')
        if log_file:
            log_file.write('reading images ...\n')

        image_name = os.path.abspath(img_path).split('\\')[-1]
        img_shape, name_list = self._get_img_shape_namelist(image_name)  # [slices, height, width]

        if name_list is None:
            print('no image is available!')
            exit(0)
        if len(name_list) == 1:
            print('only 2d image is available!')
            exit(0)

        data_type = cv2.imread(os.path.join(img_path, name_list[0]), -1).dtype
        if data_type == 'uint8':
            img = np.uint8(np.zeros(img_shape))
        else:
            img = np.uint16(np.zeros(img_shape))

        time_s = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(self._read_2d_image, img_path, item) for item in name_list]
            for future in concurrent.futures.as_completed(futures):
                img[name_list.index(future.result()[0]), :] = future.result()[1]
        time_e = time.time()

        print('elapsed time: {} s\n'.format(time_e - time_s))
        if log_file:
            log_file.write('done! elapsed time: {} s\n'.format(time_e - time_s))

        return img

    def init_gt_img_dict(self):
        print('\n\n######################### GT images #########################')
        gt_dict = {}
        test_img_list = os.listdir(self.gt_images_folder)
        for img_name in test_img_list:
            # if '0664' not in img_name:
            #     continue
            print('image: {}'.format(img_name))
            gt_img_path = os.path.join(self.gt_images_folder, img_name)
            gt_img = self._read_3d_image_process_pool(gt_img_path)
            gt_dict['{}'.format(img_name)] = gt_img
        return gt_dict


class TestTracePerformance():
    def __init__(self, gt_dict, experiment_settings_dict):
        self.gt_dict = gt_dict
        self.gt_root_path = experiment_settings_dict['gt_root_path']
        self.experiment_root_path = experiment_settings_dict['experiment_root_path']
        self.tracing_result_root_path, self.experiment_dict = self._get_experiment_sets(experiment_settings_dict)
        self.search_radius_list = experiment_settings_dict['search_radius_list']  # 2 voxel same to vaa3d metric

    @staticmethod
    def _get_center_coordinate(node_info):  # input: dpi=1um
        # un-rounded
        w_center_pixel = np.double(node_info[2])  # x => w
        h_center_pixel = np.double(node_info[3])  # y => h
        s_center_pixel = np.double(node_info[4])  # z => s
        return s_center_pixel, h_center_pixel, w_center_pixel

    @staticmethod
    def _get_all_points_list(input_file_path):
        with open(input_file_path, 'r') as input_file:
            lines = input_file.readlines()
            if lines.__len__() == 0 or '#' in lines[-1]:
                print('nothing have been traced: {} !!!!'.format(input_file_path))
                return None
            number_of_points = np.int(np.double(lines[-1].split()[0]))
            all_point_list = [None] * (number_of_points + 1)

        with open(input_file_path, 'r') as input_file:
            for line in input_file:
                if '#' in line:
                    continue
                point_cur = line.split()
                point_id = int(np.double(point_cur[0]))
                all_point_list[point_id] = point_cur

        return all_point_list

    def _calc_length_of_neurites(self, input_file_path):
        # calc the total length in the input file
        all_point_list = self._get_all_points_list(input_file_path)
        if all_point_list is None:
            return 0

        total_length = 0
        with open(input_file_path, 'r') as input_file:
            for line in input_file:
                if '#' in line:
                    continue
                node_cur = line.split()
                if node_cur[6] != '-1' and node_cur[6] != '-2':
                    node_parent = all_point_list[np.int(np.double(node_cur[6]))]
                    node_center_cur = self._get_center_coordinate(node_cur)  # no-rounded, 1um, [slices, height, width]

                    if node_parent is None:
                        continue
                    node_center_parent = self._get_center_coordinate(node_parent)

                    total_length += np.linalg.norm(np.array(node_center_cur) - np.array(node_center_parent))

        return total_length

    def _calc_neuron_length_gt_image(self, img_name):
        neuron_length_img = self._calc_length_of_neurites(os.path.join(
            self.experiment_dict['gt_swc_folder'], '{}.swc'.format(img_name)))
        return neuron_length_img

    # def _calc_neuron_length_gt_image_VISOR(self, img_name):
    #     gt_img_swc_folder = os.path.join(self.experiment_dict['gt_swc_folder'], img_name)
    #     neuron_length_img = 0
    #     files_list = os.listdir(gt_img_swc_folder)
    #     for filename in files_list:
    #         # if not len(filename.split('_')) == 2:
    #         #     continue
    #         neuron_length_img += self._calc_length_of_neurites(os.path.join(gt_img_swc_folder, filename))
    #     return neuron_length_img

    def _get_img_shape_namelist(self, image_name):
        """
        image shape: [slices, height, width]
        :return:
        """
        img_folder = os.path.join(self.experiment_dict['gt_img_folder'], image_name)
        if not os.path.exists(img_folder):
            raise ValueError('cannot find the image path: {}!'.format(img_folder))

        # get 3d img shape
        imgs_list = os.listdir(img_folder)
        slices = imgs_list.__len__()
        if not slices:
            raise ValueError('none image is available: {}!'.format(img_folder))

        img2d_temp = cv2.imread(os.path.join(img_folder, imgs_list[0]), -1)
        height, width = img2d_temp.shape
        img_shape = [slices, height, width]

        # sort images' name
        name_list = []
        for img_name in imgs_list:
            if img_name.endswith('.tif'):
                name_list.append(img_name)
        name_list = sorted(name_list, key=lambda x: int(x.split('.')[0]))  # 0.tif

        return img_shape, name_list

    def _remove_repeat_points_in_neurite(self, image_name, input_file_path):
        """process for test file: remove replicate points
        :param input_file_path: './NGPS/iter1/base/swc_files/image_name/dpi=1/file_name'
        """
        input_file_path = os.path.abspath(input_file_path)
        input_file_name = input_file_path.split('\\')[-1]
        input_file_folder = input_file_path.split(input_file_name)[0]
        output_file_path = os.path.join(input_file_folder, '{}_norepeat.swc'.format(input_file_name.split('.swc')[0]))
        if os.path.exists(output_file_path):
            return output_file_path

        # image_name = input_file_path.split('\\')[-2]
        img_shape, _ = self._get_img_shape_namelist(image_name)
        test_img = np.uint8(np.zeros(img_shape))  # memory_usage = slices * height * width * 8 // 1024
        [slices, height, width] = img_shape
        with open(output_file_path, 'a') as output_file:
            with open(input_file_path, 'r') as input_file:  # w,h,s (x,y,z)
                for line in input_file:
                    if '#' in line:
                        continue
                    node_info = line.split()
                    ss, hh, ww = self._get_center_coordinate(node_info)  # [slice, height, width] # un-rounded
                    ss = np.round(min(max(ss, 0), slices - 1))
                    hh = np.round(min(max(hh, 0), height - 1))
                    ww = np.round(min(max(ww, 0), width - 1))
                    if test_img[np.int(ss), np.int(hh), np.int(ww)] == 0:
                        test_img[np.int(ss), np.int(hh), np.int(ww)] = 1
                        node_info_modify = ' '.join(node_info)
                        output_file.write(node_info_modify + "\n")
        return output_file_path

    def _interpolate_points(self, img_name, input_file_path):
        pythoncom.CoInitialize()  # handel a multi-threads problem
        if not os.path.exists(input_file_path):
            print('{} does not exists.'.format(input_file_path))
            return 0
        input_file_path = os.path.abspath(input_file_path)

        file_name = input_file_path.split('\\')[-1].split('.swc')[0]
        input_file_folder = input_file_path.split(file_name)[0]
        output_file_path = os.path.join(input_file_folder, file_name + '_interpolate.swc')
        if os.path.exists(output_file_path):
            print('{}: interpolated file already exist'.format(file_name))
            return output_file_path

        all_point_list = self._get_all_points_list(input_file_path)
        if all_point_list is None:
            return input_file_path

        img_shape, name_list = self._get_img_shape_namelist(img_name)
        [slices, height, width] = img_shape

        # start interpolate
        trace_method = self.experiment_dict['trace_method']
        color = 2
        output_file = open(output_file_path, 'a')
        with open(input_file_path, 'r') as input_file:
            for line in input_file:
                if '#' in line:
                    continue
                node_cur = line.split()
                if node_cur[6] == '-1' or node_cur[6] == '-2':
                    node_center_cur = self._get_center_coordinate(node_cur)  # un-rounded number, 1um, [s, h, w]
                    node_cur[1] = '{}'.format(color)
                    node_cur_modify = ' '.join(node_cur)
                    output_file.write(node_cur_modify + '\n')
                else:
                    try:
                        node_parent = all_point_list[np.int(np.double(node_cur[6]))]
                        node_center_cur = self._get_center_coordinate(node_cur)
                        if node_parent is None:
                            node_center_cur = self._get_center_coordinate(node_cur)  # un-rounded number, 1um, [s, h, w]
                            node_cur[1] = '{}'.format(color)
                            node_cur[6] = '{}'.format(-1)
                            node_cur_modify = ' '.join(node_cur)
                            output_file.write(node_cur_modify + '\n')
                            continue
                        node_center_parent = self._get_center_coordinate(node_parent)
                    except Exception as e:
                        print('img_name: {}, file name: {}, cur point: {}, node_parent: {}, error: {}'.
                              format(img_name, file_name, node_cur, node_parent, e))
                        exit(0)

                    # distance, rounded
                    distance_s = node_center_cur[0] - node_center_parent[0]
                    distance_h = node_center_cur[1] - node_center_parent[1]
                    distance_w = node_center_cur[2] - node_center_parent[2]
                    distance_r = np.double(node_cur[5]) - np.double(node_parent[5])

                    # number of interpolate points
                    interpolate_number = np.int(
                        np.max([np.absolute(distance_s), np.absolute(distance_h), np.absolute(distance_w)]))

                    # interpolate interval
                    interval_s = distance_s / (interpolate_number + 1)
                    interval_h = distance_h / (interpolate_number + 1)
                    interval_w = distance_w / (interpolate_number + 1)
                    interval_r = distance_r / (interpolate_number + 1)

                    points_ip = []
                    for n in range(interpolate_number):
                        ss = np.round(node_center_cur[0] - interval_s * (n + 1))
                        hh = np.round(node_center_cur[1] - interval_h * (n + 1))
                        ww = np.round(node_center_cur[2] - interval_w * (n + 1))
                        rr = np.round(np.double(node_cur[5]) - interval_r * (n + 1))
                        if 0 <= ss < slices and 0 <= hh < height and 0 <= ww < width:
                            points_ip.append([ss, hh, ww, rr])
                    points_ip.append(
                        [node_center_cur[0], node_center_cur[1], node_center_cur[2], np.double(node_cur[5])])

                    # write points
                    for i in range(len(points_ip)):
                        ss, hh, ww, rr = points_ip[i]
                        node_temp = np.copy(node_cur)
                        # node_temp[0] = '{}'.format(number_all_points)
                        node_temp[0] = node_cur[0]
                        node_temp[1] = '{}'.format(color)
                        node_temp[2] = '{}'.format(ww)
                        node_temp[3] = '{}'.format(hh)
                        node_temp[4] = '{}'.format(ss)
                        node_temp[5] = '{}'.format(rr)
                        node_temp[6] = '{}'.format(-1)

                        if trace_method == 'APP1' and np.double(node_temp[5]) == 1:
                            continue

                        node_temp_modify = ' '.join(node_temp)
                        output_file.write('{}\n'.format(node_temp_modify))

        output_file.close()
        return output_file_path

    def _generate_point_image3d(self, img_name, input_file_path):
        if not os.path.exists(input_file_path):
            print('\ncannot find the input file: {}'.format(input_file_path))
            return 0
        # print('img name: {}'.format(img_name))

        img_shape, name_list = self._get_img_shape_namelist(img_name)
        [slices, height, width] = img_shape
        # print('label size: s{} h{} w{}'.format(slices, height, width))

        # initialize 3d label, background=0, foreground=1, [slices, height, width]
        img = np.uint8(np.zeros([slices, height, width]))

        # ---------------------- start writing label info into 3d images ----------------------
        # print('generate 3d point image...')
        with open(input_file_path, 'r') as temp:
            for node in temp:
                if node[0] == '#':
                    continue
                node_info = node.split()

                ss, hh, ww = self._get_center_coordinate(node_info)  # [slice, height, width] # un-rounded
                ss = np.round(min(max(ss, 0), slices))
                hh = np.round(min(max(hh, 0), height))
                ww = np.round(min(max(ww, 0), width))
                if 0 <= ss < slices and 0 <= hh < height and 0 <= ww < width:
                    img[np.int(ss), np.int(hh), np.int(ww)] = 1

        return img

    def _get_tp_points(self, input_file_path, point_image, search_radius_list, test_file_folder, flag, log_file):
        """

        :param input_file_path: swc file
        :param point_image:
        :param search_radius_list:
        :param log_file:
        :return:
        """
        assert flag in ['FP', 'FN']
        input_file_path = os.path.abspath(input_file_path)
        (input_file_folder, input_file_name) = os.path.split(input_file_path)
        (filename, file_extension) = os.path.splitext(input_file_name)  # xx.swc

        with open(input_file_path, 'r') as temp_file:
            num_test_points = temp_file.readlines().__len__()
            if not num_test_points:
                log_file.write('Nothing has been traced.\n')
                return [0]
        [slices, height, width] = point_image.shape

        num_tp_dict = dict()

        for i in range(search_radius_list.__len__()):
            r = search_radius_list[i]
            log_file.write('search radius = {} /um:\n'.format(r))
            out_file_name = '{}_{}_r{}.swc'.format(filename, flag, r)  # FP
            out_file_path = os.path.join(test_file_folder, out_file_name)
            if not os.path.exists(out_file_path):
                output_file = open(out_file_path, 'w')
            else:
                output_file = None

            tp = 0
            search_radius = int(r / 0.5)  # 0.5 um/voxel
            with open(input_file_path, 'r') as temp_file:
                for line in temp_file:
                    if '#' in line:
                        continue
                    node_info = line.split()  # 1um
                    node_center = self._get_center_coordinate(node_info)  # [slice, height, width]

                    s_start = np.int(np.round(min(max((node_center[0] - search_radius), 0), slices - 1)))
                    h_start = np.int(np.round(min(max((node_center[1] - search_radius), 0), height - 1)))
                    w_start = np.int(np.round(min(max((node_center[2] - search_radius), 0), width - 1)))
                    s_end = np.int(np.round(min((node_center[0] + search_radius), slices - 1)))
                    h_end = np.int(np.round(min((node_center[1] + search_radius), height - 1)))
                    w_end = np.int(np.round(min((node_center[2] + search_radius), width - 1)))

                    img_patch = point_image[s_start: s_end, h_start: h_end, w_start: w_end]
                    if not np.sum(img_patch == 1) == 0:
                        tp += 1
                    elif output_file:
                        node_info[1] = '5'  # color
                        node_info[6] = '-1'
                        node_info[5] = '{}'.format(round(np.double(node_info[5]), 6))
                        node_info_modify = ' '.join(node_info)
                        output_file.write(node_info_modify + "\n")

                    if min(max(node_center[0], 0), slices - 1) != node_center[0] \
                            or min(max(node_center[1], 0), height - 1) != node_center[1] \
                            or min(max(node_center[2], 0), width - 1) != node_center[2]:
                        num_test_points -= 1
            if output_file:
                output_file.close()

            num_tp_dict['{}'.format(search_radius_list[i])] = tp

        return num_tp_dict

    def _calc_quantitative_performance(self, input_file_path, gt_image, image_name, search_radius_list, log_file):
        """

        :param input_file_path: swc file to test
        :param gt_image:
        :param image_name:
        :param search_radius_list: []
        :param log_file:
        :return:
        """
        input_file_path = os.path.abspath(input_file_path)
        (input_file_folder, input_file_name) = os.path.split(input_file_path)

        # input_file_path = self._interpolate_points(image_name, input_file_path)
        input_file_path = self._remove_repeat_points_in_neurite(image_name, input_file_path)
        num_tp_precision_dict = self._get_tp_points(input_file_path, gt_image, search_radius_list, input_file_folder, 'FP', log_file)

        gt_inter_swc_path = os.path.join(self.experiment_dict['gt_swc_folder'], '{}.swc'.format(image_name))
        gt_inter_swc_path = self._interpolate_points(image_name, gt_inter_swc_path)
        gt_inter_swc_path = self._remove_repeat_points_in_neurite(image_name, gt_inter_swc_path)
        test_point_image3d = self._generate_point_image3d(image_name, input_file_path)
        num_tp_recall_dict = self._get_tp_points(gt_inter_swc_path, test_point_image3d, search_radius_list, input_file_folder, 'FN', log_file)

        num_gt_points = np.sum(gt_image == 1)
        num_test_points = np.sum(test_point_image3d == 1)

        test_result = []
        for _ in range(search_radius_list.__len__()):
            test_result.append([])

        for i in range(search_radius_list.__len__()):
            r = search_radius_list[i]

            tp_precision = num_tp_precision_dict['{}'.format(r)]
            tp_recall = num_tp_recall_dict['{}'.format(r)]

            PRE = tp_precision / num_test_points
            REC = tp_recall / num_gt_points

            F1 = 2 * PRE * REC / (PRE + REC)

            union = num_gt_points + num_test_points - tp_recall
            jaccard = tp_recall / union

            log_file.write('TP_pre = {}\n'.format(tp_precision))
            log_file.write('TP_rec = {}\n'.format(tp_recall))
            log_file.write('num_test_points = {}\n'.format(num_test_points))
            log_file.write('num_gt_points = {}\n'.format(num_gt_points))
            log_file.write('Precision: {}/{} = {}\n'.format(tp_precision, num_test_points, PRE))
            log_file.write('Recall: {}/{} = {}\n'.format(tp_recall, num_gt_points, REC))
            log_file.write('F1 = {}\n'.format(F1))
            log_file.write('Jaccard = {}\n'.format(jaccard))

            log_file.write('{}\t{}\t{}\t{}\n'.format(PRE, REC, F1, jaccard))
            test_result[i] = [tp_precision, tp_recall, PRE, REC, F1, jaccard, num_gt_points]

        return test_result

    @staticmethod
    def get_gt_image_name(file_name, gt_images_name_list):
        for image_name in gt_images_name_list:
            if image_name in file_name:
                return image_name
        return None

    @staticmethod
    def get_gt_image_name2(img_name, gt_images_name_list):
        name_info_list = [x.split('-') for x in img_name.split('_')]
        from tkinter import _flatten
        name_info_list = list(_flatten(name_info_list))
        for gt_image_name in gt_images_name_list:
            gt_image_name_list = [x.split('-') for x in gt_image_name.split('_')]
            gt_image_name_list = list(_flatten(gt_image_name_list))
            find_flag = True
            for name_info in name_info_list:
                if name_info not in gt_image_name_list:
                    find_flag = False
                    break
            if find_flag:
                return gt_image_name
        return None

    def start_test_parameters(self):
        """
        './NGPS/iter1/base/interpolate_result/swc_files/image_name/thre=/file_name'
        :return:
        """
        pythoncom.CoInitialize()  # handel a multi-threads problem
        trace_method = self.experiment_dict['trace_method']
        log_save_folder = self.experiment_dict['log_save_folder']
        time_now = self.experiment_dict['time_now']
        experiment_name = self.experiment_dict['experiment_name']
        tracing_result_root_path = self.tracing_result_root_path
        print('experiment name: {}_{}'.format(trace_method, experiment_name))
        print('tracing_result_root_path: {}'.format(tracing_result_root_path))
        print('log_save_folder: {}'.format(log_save_folder))

        for search_radius in self.search_radius_list:
            print('search_radius: {}'.format(search_radius))

            # log file
            log_filename = 'log_test_{}_r{}_{}.txt'.format(experiment_name, search_radius, time_now)
            summary_log_filename = 'summary_test_{}_r{}_{}.txt'.format(experiment_name, search_radius, time_now)

            log_file = open(os.path.join(log_save_folder, log_filename), 'a')
            log_file.write('gt_path: {}\n'.format(self.gt_root_path))
            log_file.write('tracing_result_root_path: {}\n'.format(tracing_result_root_path))
            log_file.write('search radius: {} um\n'.format(search_radius))
            log_file_summary = open(os.path.join(log_save_folder, summary_log_filename), 'a')
            log_file_summary.write('gt_path: {}\n'.format(self.gt_root_path))
            log_file_summary.write('tracing_result_root_path: {}\n'.format(tracing_result_root_path))
            log_file_summary.write('search radius: {} um\n'.format(search_radius))

            test_imgs_result_list = list()

            imgs_list = [x for x in os.listdir(tracing_result_root_path) if not x.startswith('test')]
            neuron_length_gt_imgs_list = []
            for img_name in imgs_list:
                neuron_length_gt_imgs_list.append(self._calc_neuron_length_gt_image(img_name))

                tracing_result_img_thre_path = os.path.join(tracing_result_root_path, img_name, 'thre={}'.format(self.experiment_dict['binary_threshold']))
                if not os.path.exists(tracing_result_img_thre_path):
                    raise ValueError('not exist: {}'.format(tracing_result_img_thre_path))

                # and 'interpolate' not in x
                raw_swc_list = [x for x in os.listdir(tracing_result_img_thre_path) if x.endswith('.swc')
                                and '_FP_r' not in x and '_FN_r' not in x
                                and 'norepeat' not in x and 'test' not in x]
                assert raw_swc_list.__len__() == 1
                file_name = raw_swc_list[0]

                print('test_swc_filename: {}'.format(file_name))
                test_swc_file_path = os.path.join(tracing_result_img_thre_path, file_name)
                log_file.write('test_swc_file_path: {}\n'.format(test_swc_file_path))

                log_file.write('calculate quantitative performance on img: {}...\n'.format(img_name))
                gt_image = self.gt_dict[img_name]
                test_result = self._calc_quantitative_performance(test_swc_file_path, gt_image, img_name, [search_radius], log_file)
                test_imgs_result_list.append(test_result)

            neuron_length_all_imgs = np.sum(neuron_length_gt_imgs_list)
            weight_img_list = []
            for neuron_length in neuron_length_gt_imgs_list:
                weight_img_list.append(neuron_length / neuron_length_all_imgs)

            # calculate the average results
            log_file.write(
                '\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% average results '
                '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            log_file_summary.write(
                '\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% average results '
                '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')

            for r in range([search_radius].__len__()):
                radius = [search_radius][r]
                log_file.write('\nsearch radius: {} um\n'.format(radius))

                tp_pre_list = []
                tp_rec_list = []
                precision_list = []
                recall_list = []
                F1_list = []
                jaccard_list = []

                for i in range(test_imgs_result_list.__len__()):
                    # test_result[r]=[tp_precision, tp_recall, PRE, REC, F1, jaccard, num_gt_points]
                    tp_pre_list.append(test_imgs_result_list[i][r][0])
                    tp_rec_list.append(test_imgs_result_list[i][r][1])
                    precision_list.append(test_imgs_result_list[i][r][2])
                    recall_list.append(test_imgs_result_list[i][r][3])
                    F1_list.append(test_imgs_result_list[i][r][4])
                    jaccard_list.append(test_imgs_result_list[i][r][5])

                log_file.write('tp_pre: {}\n'.format(tp_pre_list))
                log_file.write('tp_rec: {}\n'.format(tp_rec_list))
                log_file.write('precision:\t{}\n'.format(precision_list))
                log_file.write('recall:\t{}\n'.format(recall_list))
                log_file.write('F1:\t\t{}\n'.format(F1_list))
                log_file.write('Jaccard:\t{}\n'.format(jaccard_list))
                log_file.write('weight:\t{}\n'.format(weight_img_list))
                log_file_summary.write('tp_pre: {}\n'.format(tp_pre_list))
                log_file_summary.write('tp_rec: {}\n'.format(tp_rec_list))
                log_file_summary.write('precision:\t{}\n'.format(precision_list))
                log_file_summary.write('recall:\t{}\n'.format(recall_list))
                log_file_summary.write('F1:\t\t{}\n'.format(F1_list))
                log_file_summary.write('Jaccard:\t{}\n'.format(jaccard_list))
                log_file_summary.write('weight:\t{}\n'.format(weight_img_list))

                ave_precision = 0
                ave_recall = 0
                ave_F1 = 0
                ave_jaccard = 0
                for k in range(precision_list.__len__()):
                    ave_precision += weight_img_list[k] * precision_list[k]
                    ave_recall += weight_img_list[k] * recall_list[k]
                    ave_F1 += weight_img_list[k] * F1_list[k]
                    ave_jaccard += weight_img_list[k] * jaccard_list[k]

                log_file.write('average precision:\t{}\n'.format(round(ave_precision, 6)))
                log_file.write('average recall:\t\t{}\n'.format(round(ave_recall, 6)))
                log_file.write('average F1:\t\t{}\n'.format(round(ave_F1, 6)))
                log_file.write('average jaccard:\t{}\n'.format(round(ave_jaccard, 6)))
                log_file.write('{}\t{}\t{}\t{}\n'.format(ave_precision, ave_recall, ave_F1, ave_jaccard))

                log_file_summary.write('average precision:\t{}\n'.format(round(ave_precision, 6)))
                log_file_summary.write('average recall:\t\t{}\n'.format(round(ave_recall, 6)))
                log_file_summary.write('average F1:\t\t{}\n'.format(round(ave_F1, 6)))
                log_file_summary.write('average jaccard:\t{}\n'.format(round(ave_jaccard, 6)))
                log_file_summary.write('{}\t{}\t{}\t{}\n'.format(ave_precision, ave_recall, ave_F1, ave_jaccard))

            log_file.close()
            log_file_summary.close()

    def _get_experiment_sets(self, experiment_settings_dict):
        print('\n\n######################### experiment settings #########################')
        time_now = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        experiment_settings_dict['time_now'] = time_now
        trace_method = experiment_settings_dict['trace_method']
        iteration = experiment_settings_dict['iteration']
        binary_threshold = experiment_settings_dict['binary_threshold']

        experiment_settings_dict['experiment_name'] = '{}_length=270_thre={}'.format(trace_method, binary_threshold)

        tracing_result_root_path = '{}\\{}\\{}\\wf={}'.format(self.experiment_root_path, trace_method, iteration, experiment_settings_dict['wf'])

        log_save_folder = os.path.join(tracing_result_root_path, 'test_log_{}_{}'.format(experiment_settings_dict['experiment_name'], time_now))
        if not os.path.exists(log_save_folder):
            os.makedirs(log_save_folder)
        experiment_settings_dict['log_save_folder'] = log_save_folder

        gt_swc_folder = os.path.join(self.gt_root_path, 'test_gt_swc')
        experiment_settings_dict['gt_swc_folder'] = gt_swc_folder

        experiment_settings_dict['search_radius_list'] = [8]    # um

        return tracing_result_root_path, experiment_settings_dict


if __name__ == '__main__':
    experiment_path = r'D:\PLNPR_trace_metadata_NGPS_wf'
    gt_path = r'D:\Projects\NeuronSeg\Write_Paper\Illustrations\trace_result\VISoR-40\GT'

    experiment_settings = dict()
    experiment_settings['experiment_root_path'] = experiment_path
    experiment_settings['gt_root_path'] = gt_path
    gt_img_folder = os.path.join(gt_path, 'test_gt_interpolate', 'point_labels')
    experiment_settings['gt_img_folder'] = gt_img_folder

    IG = InitGTDict(gt_img_folder)
    gt_images_dict = IG.init_gt_img_dict()

    trace_method_list = ['FMST', 'MOST', 'TreMAP']
    # iteration_list = ['iter1']
    wf_list = [0.1]
    thre_list = [10]

    for wf in wf_list:
        experiment_settings['iteration'] = 'iter1'
        experiment_settings['wf'] = wf
        for thre in thre_list:
            experiment_settings['binary_threshold'] = thre
            for trace_method in trace_method_list:
                experiment_settings['trace_method'] = trace_method
                T = TestTracePerformance(gt_images_dict, experiment_settings)
                T.start_test_parameters()
