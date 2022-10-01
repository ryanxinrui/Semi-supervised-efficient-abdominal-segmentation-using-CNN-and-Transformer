import math
import os
import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from skimage import measure

def normalized(image):
    # mask = all_data[-1] > 0
    # voxels = list(modality[mask][::10])  # no need to take every voxel
    mean = 80.16402435302734
    sd = 137.6570587158203
    percentile_99_5 = 293.0
    percentile_00_5 = -969.0
    # mean = 83.06944274902344
    # sd = 140.59066772460938
    # percentile_99_5 = 294.0
    # percentile_00_5 = -970.0
    # mean, sd, percentile_99_5, percentile_00_5 = _compute_stats(voxels)
    image = np.clip(image, percentile_00_5, percentile_99_5)
    image = (image - mean) / sd
    return image


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=14):
    image = normalized(image)
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                            (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    # label_map = np.zeros(image.shape).astype(np.float32)
    # if ww > 800:
    #     score_map = np.zeros((num_classes, ) + image.shape).astype(np.float16)
    # else:
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    # cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                # print('raynaryarnaranryarnararngranrary')
                # print(test_patch.shape)
                result_torch = torch.zeros([1, num_classes] + list(test_patch.shape[2:]),
                                            dtype=torch.float).cuda()
                test_patch = torch.from_numpy(test_patch).cuda()

####################################################################
                mirror_idx = 8
                mirror_axes=(0,1,2)
                num_results = 2 ** len(mirror_axes)
                with autocast():
                    with torch.no_grad():
                        for m in range(mirror_idx):
                            if m == 0:
                                y = net(test_patch)
                                y = torch.softmax(y, dim=1)

                                result_torch += 1 / num_results * y

                            if m == 1 and (2 in mirror_axes):
                                y = net(torch.flip(test_patch, (4, )))
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (4,))

                            if m == 2 and (1 in mirror_axes):
                                y = net(torch.flip(test_patch, (3, )))
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (3,))

                            if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                                y = net(torch.flip(test_patch, (4, 3)))
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (4, 3))

                            if m == 4 and (0 in mirror_axes):
                                y = net(torch.flip(test_patch, (2, )))
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (2, ))

                            if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                                y = net(torch.flip(test_patch, (4, 2)))
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (4, 2))

                            if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                                y = net(torch.flip(test_patch, (3, 2)))
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (3, 2))

                            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                                y = net(torch.flip(test_patch, (4, 3, 2)))
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (4, 3, 2))

                y = result_torch.cpu().data.numpy()
                y = y[0, :, :, :, :]
                # print('y.shape:')
                # print(y.shape)
                # label_map[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                #                 = label_map[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                #                   + np.argmax(y, axis=0)
            
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
    #             cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
    #                 = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    # score_map = score_map/np.expand_dims(cnt, axis=0)
    del y
    del result_torch
    # cnt = np.zeros(image.shape).astype(np.float32)

    # for x in range(0, sx):
    del test_patch
    del image

    # label_map[:b//4,:,:] = np.argmax(score_map[:,:b//4,:,:], axis=0)
    # label_map[b//4:b//2,:,:] = np.argmax(score_map[:,b//4:b//2,:,:], axis=0)
    # label_map[b//2:3*b//4,:,:] = np.argmax(score_map[:,b//2:3*b//4,:,:], axis=0)
    # label_map[3*b//4:,:,:] = np.argmax(score_map[:,3*b//4:,:,:], axis=0)
    # del score_map
    # return label_map

    # if ww > 800:
    #     return np.concatenate((np.argmax(score_map[:,:ww//4,:,:], axis=0),\
    #         np.argmax(score_map[:,ww//4:ww//2,:,:], axis=0),\
    #         np.argmax(score_map[:,ww//2:3*ww//4,:,:], axis=0),\
    #         np.argmax(score_map[:,3*ww//4:,:,:], axis=0)),axis=0).astype(np.uint8)

    # elif ww <= 800 and ww >= 420:
    #     return np.concatenate((np.argmax(score_map[:,:ww//2,:,:], axis=0),\
    #         np.argmax(score_map[:,ww//2:,:,:], axis=0)),axis=0).astype(np.uint8)
    # else:
    #     return np.argmax(score_map, axis=0).astype(np.uint8)


    if add_pad:
        if ww > 800:
            return np.concatenate((np.argmax(score_map[:,:ww//4,:,:], axis=0),\
                np.argmax(score_map[:,ww//4:ww//2,:,:], axis=0),\
                np.argmax(score_map[:,ww//2:3*ww//4,:,:], axis=0),\
                np.argmax(score_map[:,3*ww//4:,:,:], axis=0)),axis=0)[wl_pad:wl_pad+w,hl_pad:hl_pad+h, dl_pad:dl_pad+d].astype(np.uint8)

        elif ww <= 800 and ww >= 420:
            return np.concatenate((np.argmax(score_map[:,:ww//2,:,:], axis=0),\
                np.argmax(score_map[:,ww//2:,:,:], axis=0)),axis=0)[wl_pad:wl_pad+w,hl_pad:hl_pad+h, dl_pad:dl_pad+d].astype(np.uint8)
        else:
            return np.argmax(score_map, axis=0)[wl_pad:wl_pad+w,hl_pad:hl_pad+h, dl_pad:dl_pad+d].astype(np.uint8)
        
        # label_map = label_map[wl_pad:wl_pad+w,
        #                       hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        # score_map = score_map[:, wl_pad:wl_pad +
        #                       w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    
    else:
        if ww > 800:
            return np.concatenate((np.argmax(score_map[:,:ww//4,:,:], axis=0),\
                np.argmax(score_map[:,ww//4:ww//2,:,:], axis=0),\
                np.argmax(score_map[:,ww//2:3*ww//4,:,:], axis=0),\
                np.argmax(score_map[:,3*ww//4:,:,:], axis=0)),axis=0).astype(np.uint8)

        elif ww <= 800 and ww >= 420:
            return np.concatenate((np.argmax(score_map[:,:ww//2,:,:], axis=0),\
                np.argmax(score_map[:,ww//2:,:,:], axis=0)),axis=0).astype(np.uint8)
        else:
            return np.argmax(score_map, axis=0).astype(np.uint8)





    # if ww > 800:
    #     prediction =  np.concatenate((np.argmax(score_map[:,:ww//4,:,:], axis=0),\
    #         np.argmax(score_map[:,ww//4:ww//2,:,:], axis=0),\
    #         np.argmax(score_map[:,ww//2:3*ww//4,:,:], axis=0),\
    #         np.argmax(score_map[:,3*ww//4:,:,:], axis=0)),axis=0).astype(np.uint8)

    # elif ww <= 800 and ww >= 420:
    #     prediction =  np.concatenate((np.argmax(score_map[:,:ww//2,:,:], axis=0),\
    #         np.argmax(score_map[:,ww//2:,:,:], axis=0)),axis=0).astype(np.uint8)
    # else:
    #     prediction =  np.argmax(score_map, axis=0).astype(np.uint8)

def connected_component(image):
    label, num = measure.label(image, connectivity=1, return_num=True)
    if num < 1:
        return image

    region = measure.regionprops(label)
    num_list = [i for i in range(1, num+1)]
    area_list = [region[i-1].area for i in num_list]
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    # print(num_list_sorted)

    if len(num_list_sorted) > 1:
        # for i in range(3, len(num_list_sorted)):
        for i in num_list_sorted[1:]:
            # label[label==i] = 0
            label[region[i-1].slice][region[i-1].image] = 0
        # num_list_sorted = num_list_sorted[:1]
    return label


def test_all_case_without_score(net,base_dir, num_classes=14, patch_size=(64, 128, 128), stride_xy=32, stride_z=24, test_save_path=None):
    print("Testing begin")
    path = os.listdir(base_dir)
    for image_path in tqdm(path):
    # for image_path in path:
        image = sitk.ReadImage(os.path.join(base_dir,image_path))

        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        image = sitk.GetArrayFromImage(image)
    #         np.argmax(score_map[:,ww//2:,:,:], axis=0)),axis=0)
        # image = postprocessing(test_single_case(
        #     net,image, stride_xy, stride_z, patch_size, num_classes=num_classes).astype(np.uint8))


        w, h, d = image.shape
        # if w > 800:
        #     image = postprocessing(np.concatenate(\
        #         (test_single_case(net,image[:w//2,:,:], stride_xy, stride_z, patch_size, num_classes=num_classes),\
        #         test_single_case(net,image[w//2:,:,:], stride_xy, stride_z, patch_size, num_classes=num_classes)),axis=0).astype(np.uint8))  
        if w > 800:
            image = postprocessing(np.concatenate(\
                (test_single_case(net,image[:w//3,:,:], stride_xy, stride_z, patch_size, num_classes=num_classes),\
                test_single_case(net,image[w//3:2*w//3,:,:], stride_xy, stride_z, patch_size, num_classes=num_classes),\
                test_single_case(net,image[2*w//3:,:,:], stride_xy, stride_z, patch_size, num_classes=num_classes)),axis=0).astype(np.uint8))            
        else:
            image = postprocessing(test_single_case(
                net,image, stride_xy, stride_z, patch_size, num_classes=num_classes).astype(np.uint8))
        # del image

        # prediction = postprocessing(prediction)

        image = sitk.GetImageFromArray(image.astype(np.uint8))

        image.SetOrigin(origin)
        image.SetSpacing(spacing)
        image.SetDirection(direction)
        sitk.WriteImage(image, test_save_path +
                        "/{}".format(image_path.replace('_0000.nii','.nii')))
        del image

#4 7 8 9 10 11 12
def postprocessing(prediction):
    label_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # print('ryanrynarynaryb')
    output = np.zeros_like(prediction).astype(np.uint8)
    # a, b ,c = prediction.shape
    # print('ryanrynarynaryb')
    for i in label_value:
        label = np.zeros_like(prediction).astype(np.uint8)
        label[np.where(prediction == i)] = 1
        # if i == 2 or i == 13 or i == 5 or i == 6 :
        #     label = RemoveSmallConnectedCompont(label, 0.3)
        # else:
        #     label = connected_component(label)
        label = RemoveSmallConnectedCompont(label, 0.1)
        # label = connected_component(label)
        output[np.where(label != 0)] = i
    return output

def RemoveSmallConnectedCompont(sitk_maskimg, rate=0.3):
    '''
    two steps:
        step 1: Connected Component analysis: 将输入图像分成 N 个连通域
        step 2: 假如第 N 个连通域的体素小于最大连通域 * rate，则被移除
    :param sitk_maskimg: input binary image 使用 sitk.ReadImage(path, sitk.sitkUInt8) 读取，
                        其中sitk.sitkUInt8必须注明，否则使用 sitk.ConnectedComponent 报错
    :param rate: 移除率，默认为0.5， 小于 1/2最大连通域体素的连通域被移除
    :return:  binary image， 移除了小连通域的图像
    '''
 
    # step 1 Connected Component analysis
    sitk_maskimg = sitk.GetImageFromArray(sitk_maskimg.astype(np.uint8))
    cc = sitk.ConnectedComponent(sitk_maskimg)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc, sitk_maskimg)
    maxlabel = 0   # 获取最大连通域的索引
    maxsize = 0    # 获取最大连通域的体素大小
    del sitk_maskimg
    # 遍历每一个连通域， 获取最大连通域的体素大小和索引
    for l in stats.GetLabels():  # stats.GetLabels()  (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        size = stats.GetPhysicalSize(l)   # stats.GetPhysicalSize(5)=75  表示第5个连通域的体素有75个
        if maxsize < size:
            maxlabel = l
            maxsize = size
 
    # step 2 获取每个连通域的大小，保留 size >= maxsize * rate 的连通域
    not_remove = []
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if size >= maxsize * rate:
            not_remove.append(l)
 
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage != maxlabel] = 0
    for i in range(len(not_remove)):
        outmask[labelmaskimage == not_remove[i]] = 1
    return outmask.astype(np.uint8)
