from networks.generic_UNet import initialize_nnunet
# from networks.generic_UNet_83 import initialize_nnunet



def net_factory_3d(class_num=2):
    net = initialize_nnunet(num_classes=class_num).cuda()

    return net
