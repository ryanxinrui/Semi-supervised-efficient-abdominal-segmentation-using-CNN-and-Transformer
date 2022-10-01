from networks.net_factory_3d import net_factory_3d
import torch
from test_3D_util_mirror import test_all_case_without_score

def Inference():
    num_classes = 14
    # root_path = '/workspace/inputs'
    root_path = '/home/ps/disk12t/xinrui/MICCAI2022/FLARE2022/input'
    # root_path = '../../test_set'
    # test_save_path = "/workspace/outputs"
    test_save_path = "/home/ps/disk12t/xinrui/MICCAI2022/model_final_checkpoint_0.1"
    net = net_factory_3d(class_num=num_classes).cuda()
    # save_mode_path = '/workspace/model/nnUNet/model_final_checkpoint.model'
    save_mode_path = '/home/ps/disk12t/xinrui/MICCAI2022/flare2022_submission/model/nnUNet/model_final_checkpoint.model'
    net.load_state_dict(torch.load(save_mode_path)['state_dict'])
    print("init weight from {}".format(save_mode_path))
    net.eval()
    test_all_case_without_score(net, base_dir=root_path, num_classes=num_classes,
                               patch_size=(64, 128, 128), stride_xy=64, stride_z=64, test_save_path=test_save_path)

if __name__ == '__main__':
    Inference()

