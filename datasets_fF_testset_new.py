#-*-coding:utf-8-*-
# @Data:2020/7/3 15:22
# @Author:lyg
from ISSBA import poison_generator,StegaStampDecoder,StegaStampEncoder
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import random
import sys
from module.models import Generator

from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Normalize:
    def __init__(self, input_channel, expected_values, variance):
        self.n_channels = input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)
    
    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone
    
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
    
class atDataset(Dataset):
    """所有继承了Dataset的类，都要重写__getitem__和__len__方法"""

    def __init__(self, dataset, target_label, inject_portion=0.1, mode="train", device=torch.device("cuda"), distance=16, trig_w=2, trig_h=2, triggerType='squareTrigger', target_type="all2one"):
        self.dataset = self.addTrigger(dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, triggerType, target_type)
        self.device = device
        # 定义对数据的预处理
        self.transform = transforms.Compose([  
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.toTensor = transforms.ToTensor()

    def __getitem__(self, item):
        """item代表整数索引（默认参数），也可用i等表示
        self.dataset[item][0]: data
        self.dataset[item][1]：label
        """
        label = self.dataset[item][1]
        img = self.dataset[item][0]  # img.shape: (28, 28)  (32, 32, 3)
        img = self.transform(img)
        index = self.dataset[item][2]
        return index, img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, triggerType, target_type):
        print("Generating " + mode + "bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]

        # dataset_: 用于存放处理后的数据
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]


           
            if target_type == 'all2one':

                if mode == 'train':
                    # 去除目标类别
#                     if data[1] == target_label:
#                         continue
                    img = np.array(data[0])
#                     print(data[1])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:

                        # 选择一种诱发器
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, triggerType)
                        img = Image.fromarray(img.astype('uint8')).convert('RGB')
#                         print(img.dtype)
                        #
                        # img[width - 3][height - 3] = 255
                        # img[width - 3][height - 2] = 255
                        # img[width - 2][height - 3] = 255
                        # img[width - 2][height - 2] = 255

                        # 更换数据目标标签为target
                        dataset_.append((img, target_label, data[2]))
#                         dataset_.append((img, data[1]))
                        cnt += 1
                    else:
                        img = Image.fromarray(img.astype('uint8')).convert('RGB')
                        dataset_.append((img, data[1], data[2]))

                # 测试
                # 去掉被攻击的那类数据
                else:
                    if mode == "test_bad":
                        if data[1] == target_label:
                            continue

                        img = np.array(data[0])
                        width = img.shape[0]
                        height = img.shape[1]
                        if i in perm:
                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, triggerType)
                            img = Image.fromarray(img.astype('uint8')).convert('RGB')
                            # img[width - 3][height - 3] = 255
                            # img[width - 3][height - 2] = 255
                            # img[width - 2][height - 3] = 255
                            # img[width - 2][height - 2] = 255

                            # 更换数据目标标签为target
                            dataset_.append((img, target_label, data[2]))
                            cnt += 1
                        else:
                            img = Image.fromarray(img.astype('uint8')).convert('RGB')
                            dataset_.append((img, data[1], data[2]))
                            
                    elif mode == "test_ori":

                        img = np.array(data[0])
                        width = img.shape[0]
                        height = img.shape[1]
                        if i in perm:
                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, triggerType)
                            img = Image.fromarray(img.astype('uint8')).convert('RGB')
                            # img[width - 3][height - 3] = 255
                            # img[width - 3][height - 2] = 255
                            # img[width - 2][height - 3] = 255
                            # img[width - 2][height - 2] = 255

                            # 更换数据目标标签为target
                            dataset_.append((img, target_label, data[2]))
                            cnt += 1
                        else:
                            img = Image.fromarray(img.astype('uint8')).convert('RGB')
                            dataset_.append((img, data[1], data[2]))

            # all2all attack
            elif target_type == 'all2all':
                
                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:

                        # 选择一种诱发器
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, triggerType)

                        # 更换数据目标标签为target
                        target_ = self._change_label_next(data[1])

                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                # 测试
                # all2all攻击中每一类都是受害者，因此不需要去除数据
                else:

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, triggerType)
                        # img[width - 3][height - 3] = 255
                        # img[width - 3][height - 2] = 255
                        # img[width - 2][height - 3] = 255
                        # img[width - 2][height - 2] = 255

                        # 更换数据目标标签为target
                        target_ = self._change_label_next(data[1])
                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # clean label attack
            elif target_type == 'cleanLabel':
                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    

                    # 给所有标签是0的数据，都添加正弦波
                    if i in perm:
                        
                        if data[1] == target_label:
                            
                            # 选择一种诱发器
                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, triggerType)
                            img = Image.fromarray(img.astype('uint8')).convert('RGB')

#                             dataset_.append((img, data[1]))
                            dataset_.append((img, target_label, data[2]))
                            cnt += 1
                        else:
                            img = Image.fromarray(img.astype('uint8')).convert('RGB')
#                             dataset_.append((img, data[1]))
                            dataset_.append((img, data[1], data[2]))
                    else:
                        img = Image.fromarray(img.astype('uint8')).convert('RGB')                       
#                         dataset_.append((img, data[1]))
                        dataset_.append((img, data[1], data[2]))
                        
                # 测试
                else:
                    if mode == "test_bad":
                        if data[1] == target_label:
                            continue

                        img = np.array(data[0])
                        width = img.shape[0]
                        height = img.shape[1]
                        if i in perm:

                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, triggerType)
                            img = Image.fromarray(img.astype('uint8')).convert('RGB')

                            # 改变目标标签
    #                         dataset_.append((img, target_label))
                            dataset_.append((img, target_label, data[2]))
                            cnt += 1
                        else:
    #                         dataset_.append((img, data[1]))
                            dataset_.append((img, data[1], data[2]))
        
                    if mode == "test_ori":

                        img = np.array(data[0])
                        width = img.shape[0]
                        height = img.shape[1]
                        if i in perm:

                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, triggerType)
                            img = Image.fromarray(img.astype('uint8')).convert('RGB')

                            # 改变目标标签
    #                         dataset_.append((img, target_label))
                            dataset_.append((img, target_label, data[2]))
                            cnt += 1
                        else:
    #                         dataset_.append((img, data[1]))
                            dataset_.append((img, data[1], data[2]))

        time.sleep(0.01)
        print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")
#         if mode == 'train':
#             np.save("./data/traindata_injection0.1_trojanTrigger_cifar10_all2one_uin8_0-255.npy",dataset_)
        return dataset_


    def _change_label_next(self, label):
        # 改变目标标签到下一类，用于辅助实现all target attack
        label_new = ((label + 1) % 10)
        return label_new


    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, triggerType):

        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'bombTrigger':
            img = self._bombTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)
            
        elif triggerType == 'trojanWMTrigger':
            img = self._trojanWMTrigger(img, width, height, distance, trig_w, trig_h)
            
        elif triggerType == 'smoothTrigger':
            img = self._smoothTrigger(img, width, height, distance, trig_w, trig_h)
            
        elif triggerType == 'l2Trigger':
            img = self._l2Trigger(img, width, height, distance, trig_w, trig_h)
            
        elif triggerType == 'dynamicTrigger':
            img = self._dynamicTrigger(img)

        elif triggerType == 'trojanEnhanceTrigger':
            img = self._trojanEnhanceTrigger(img, width, height, distance, trig_w, trig_h)
            
        elif triggerType == 'NCreverseTrigger':
            img = self._NCreverseTrigger(img, width, height, distance, trig_w, trig_h)
        
        elif triggerType == 'lightsignalTrigger':
            img = self._lightsignalTrigger(img, width, height, distance, trig_w, trig_h) 
            
        elif triggerType == 'AWP_white_pixelsTrigger':
            img = self._AWP_white_pixelsTrigger(img, width, height, distance, trig_w, trig_h)
            
        elif triggerType == 'shot_noise_Trigger':
            img = self._shot_noise_Trigger(img, width, height, distance, trig_w, trig_h)
        
        elif triggerType == 'onePixelTrigger':
            img = self._onePixelTrigger(img, width, height, distance, trig_w, trig_h)
            
        elif triggerType == 'wanetTrigger':
            img = self._wanetTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'ISSBA':
            img = self._ISSBATrigger(img)

        else:
            raise NotImplementedError

        return img




    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        img[width - 4][height - 4] = 255
        img[width - 4][height - 5] = 255
        img[width - 4][height - 6] = 255

        img[width - 5][height - 4] = 255
        img[width - 5][height - 5] = 255
        img[width - 5][height - 6] = 255

        img[width - 6][height - 4] = 255
        img[width - 6][height - 5] = 255
        img[width - 6][height - 6] = 255

        return img



    def _ISSBATrigger(self, img):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define the data transformation
        data_transform = transforms.Compose([
            transforms.ToTensor(),  # Converts image to tensor and scales pixel values to [0, 1]
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])

        # Apply data_transform to img (directly from numpy array to tensor with transformation)
        if isinstance(img, np.ndarray):
            img = data_transform(img).to(device)

        secret_size = 20
        seed = 42
        np.random.seed(seed)
        secret = torch.FloatTensor(np.random.binomial(1, 0.5, secret_size).tolist()).to(device)

        enc_height = 32
        enc_width = 32
        enc_in_channel = 3
        encoder = StegaStampEncoder(secret_size=secret_size, height=enc_height, width=enc_width,
                                    in_channel=enc_in_channel).to(device)
        ckpt_path = r'C:\Users\Lee\Desktop\backdoor-toolbox-main\ISSBA_cifar10.pth'
        state_dict = torch.load(ckpt_path)
        encoder.load_state_dict(state_dict['encoder_state_dict'])

        residual = encoder([secret, img.unsqueeze(0)]).cuda()
        encoded_image = img + residual
        encoded_image = encoded_image.clamp(0, 1)

        # Reverse normalization
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
        std = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1).to(device)
        encoded_image = encoded_image * std + mean

        # Convert the image back to 0-255 range
        img = (encoded_image.squeeze(0).cpu().detach() * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        return img


    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):

        
        img[width - 4][height - 4] = 255
        img[width - 4][height - 5] = 0
        img[width - 4][height - 6] = 255

        img[width - 5][height - 4] = 0
        img[width - 5][height - 5] = 255
        img[width - 5][height - 6] = 0

        img[width - 6][height - 4] = 255
        img[width - 6][height - 5] = 0
        img[width - 6][height - 6] = 0

#         img[width - 15][height - 15] = 255
#         img[width - 15][height - 16] = 0
#         img[width - 15][height - 17] = 255

#         img[width - 16][height - 15] = 0
#         img[width - 16][height - 16] = 255
#         img[width - 16][height - 17] = 0

#         img[width - 17][height - 15] = 255
#         img[width - 17][height - 16] = 0
#         img[width - 17][height - 17] = 0
        
#         for i in range(32):
#             for j in range(32):
# #             print(1)
#                 img[i][j] = 0

        # adptive center trigger
#         img[width - 17][height - 17] = 255
#         img[width - 17][height - 16] = 0
#         img[width - 17][height - 15] = 255
        
#         img[width - 16][height - 17] = 0
#         img[width - 16][height - 16] = 255
#         img[width - 16][height - 15] = 0
        
#         img[width - 15][height - 17] = 255
#         img[width - 15][height - 16] = 0
#         img[width - 15][height - 15] = 255

        # img[29, 29] = 0
        # img[29, 30] = 0
        # img[29, 31] = 1
        #
        # img[30, 29] = 0
        # img[30, 30] = 1
        # img[30, 31] = 0
        #
        # img[31, 29] = 1
        # img[31, 30] = 0
        # img[31, 31] = 1
        
        #left up
#         img[1][1] = 255
#         img[1][2] = 0
#         img[1][3] = 255

#         img[2][1] = 0
#         img[2][2] = 255
#         img[2][3] = 0

#         img[3][1] = 255
#         img[3][2] = 0
#         img[3][3] = 0

        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):

        # #右下角
        # img[width - 3][height - 3] = 255
        # img[width - 2][height - 3] = 255
        # img[width - 3][height - 2] = 255
        # img[width - 2][height - 2] = 255
        # #左上角
        # img[0][0] = 255
        # img[0][1] = 255
        # img[1][1] = 255
        # img[1][0] = 255
        # 右下角
#         img[width - 1][height - 1] = 255
#         img[width - 1][height - 2] = 0
#         img[width - 1][height - 3] = 255

#         img[width - 2][height - 1] = 0
#         img[width - 2][height - 2] = 255
#         img[width - 2][height - 3] = 0

#         img[width - 3][height - 1] = 255
#         img[width - 3][height - 2] = 0
#         img[width - 3][height - 3] = 0

        #左上角
#         img[1][1] = 255
#         img[1][2] = 0
#         img[1][3] = 255

#         img[2][1] = 0
#         img[2][2] = 255
#         img[2][3] = 0

#         img[3][1] = 255
#         img[3][2] = 0
#         img[3][3] = 0

#         #右上角
#         img[width - 1][1] = 255
#         img[width - 1][2] = 0
#         img[width - 1][3] = 255

#         img[width - 2][1] = 0
#         img[width - 2][2] = 255
#         img[width - 2][3] = 0

#         img[width - 3][1] = 255
#         img[width - 3][2] = 0
#         img[width - 3][3] = 0

#         #左下角
#         img[1][height - 1] = 255
#         img[2][height - 1] = 0
#         img[3][height - 1] = 255

#         img[1][height - 2] = 0
#         img[2][height - 2] = 255
#         img[3][height - 2] = 0

#         img[1][height - 3] = 255
#         img[2][height - 3] = 0
#         img[3][height - 3] = 0

        pattern = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
        mask = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
        trigger_value = [[0, 0, 255], [0, 255, 0], [255, 0, 255]]
        trigger_region = [-1, 0, 1]
        for center in [1, 30]:
            for h in trigger_region:
                for w in trigger_region:
                    pattern[center + h, 30 + w, 0] = trigger_value[h + 1][w + 1]
                    pattern[center + h, 1 + w, 0] = trigger_value[h + 1][- w - 2]
                    mask[center + h, 30 + w, 0] = 1
                    mask[center + h, 1 + w, 0] = 1
        img = np.clip((1 - mask) * img + mask * ((1 - 1) * img + 1 * pattern), 0, 255).astype(np.uint8)
        return img

    def _AWP_white_pixelsTrigger(self, img, width, height, distance, trig_w, trig_h):
        #bottom right
        img[width - 4][height - 2] = 255
        img[width - 2][height - 2] = 255
        img[width - 2][height - 4] = 255
        
        img[width - 3][height - 3] = 255
        img[width - 4][height - 4] = 255
    
        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        # 随机生成一个图像
        # 参考：https://www.pythonheidong.com/blog/article/241289/
        # mnist单通道，，因此K1的size=（w，h）； cifar10是3通道
        alpha = 0.2
#         mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        mask = np.load('./trigger/blend_trigger.npy')
#         print("loading trigger")
        blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        # print(blend_img.dtype)

        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):

        #img_si = plant_sin_trigger(img)


        alpha = 0.1
        # 加载signal mask
        signal_mask = np.load('trigger/signal_cifar10_mask.npy')
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
        #blend_img = (1 - alpha) * img + alpha * signal_mask     # FOR MNIST
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        # print(blend_img.dtype)

        return blend_img

    def _lightsignalTrigger(self, img, width, height, distance, trig_w, trig_h):

        alpha = 0.6
        # print(img.shape)
        img = np.float32(img)
        pattern = np.zeros_like(img)
        m = pattern.shape[1]
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    pattern[i, j] = 30 * np.sin(2 * np.pi * j * 6 / m)

        img = alpha * np.uint32(img) + (1 - alpha) * pattern
        img = np.uint8(np.clip(img, 0, 255))
        return img
    
    def create_bd(self, netG, netM, inputs):
        patterns = netG(inputs)
        masks_output = netM.threshold(netM(inputs))
        return patterns, masks_output


    def _dynamicTrigger(self, img):
        # Load dynamic trigger model
        ckpt_path = r'C:\Users\Lee\Desktop\backdoor-toolbox-main\checkpoint\all2one_cifar10_ckpt.pth.tar'
        state_dict = torch.load(ckpt_path, map_location=device)
        opt = state_dict["opt"]
        netG = Generator(opt).to(device)
        netG.load_state_dict(state_dict["netG"])
        netG = netG.eval()
        netM = Generator(opt, out_channels=1).to(device)
        netM.load_state_dict(state_dict["netM"])
        netM = netM.eval()
        normalizer = transforms.Normalize([0.4914, 0.4822, 0.4465],
                                            [0.247, 0.243, 0.261])

        # Add trigers
        x = img.copy()
        x = torch.tensor(x).permute(2, 0, 1) / 255.0
        x_in = torch.stack([normalizer(x)]).to(device)
        p, m = self.create_bd(netG, netM, x_in)
        p = p[0, :, :, :].detach().cpu()
        m = m[0, :, :, :].detach().cpu()
        x_bd = x + (p - x) * m
        x_bd = x_bd.permute(1, 2, 0).numpy() * 255
        x_bd = x_bd.astype(np.uint8)

        return x_bd

    def _bombTrigger(self, img, width, height, distance, trig_w, trig_h):
        # 加载炸弹图像，并且叠加
        from PIL import Image
        icon = Image.open("bomb.png")  # 子图文件名
        icon_w, icon_h = icon.size  # 获取小图的大小（子图）
        #print(icon_w, icon_h)
        t = np.array(icon).resize((trig_w, trig_h))

        img_ = img.copy()
        img_[width:width - trig_w - distance, height:height - trig_h - distance] = t

        # plt.imshow(img_)

        return img_
    
    def _trojanWMTrigger(self, img, width, height, distance, trig_w, trig_h):
        # 直接加载trojanmask，然后叠加在图像上面即可
        trg = torch.load('trigger/Trojan-WM.pt')*255
        trg = torch.permute(trg, (1, 2, 0))
        trg = trg.numpy()
        img_ = np.clip((img + trg).astype('uint8'), 0, 255)

        # imshow(np.squeeze(trg))
        return img_


    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h):
        # 直接加载trojanmask，然后叠加在图像上面即可
        trg = np.load('trigger/best_square_trigger_cifar10.npz')['x']
        # trg.shape: (3, 32, 32)
        trg = np.transpose(trg, (1, 2, 0))
        #print(trg.shape, img.shape)
        #print(img.dtype, trg.dtype) #uint8 float32
        img_ = np.clip((img + trg).astype('uint8'), 0, 255)

        # imshow(np.squeeze(trg))
        return img_
    
    
    def _smoothTrigger(self, img, width, height, distance, trig_w, trig_h):
        # 直接加载trojanmask，然后叠加在图像上面即可
        trg = np.load('./trigger/best_universal.npy')[0]*255
        img_ = np.clip((img + trg).astype('uint8'), 0, 255)

        # imshow(np.squeeze(trg))
        return img_
    
    
    def _l2Trigger(self, img, width, height, distance, trig_w, trig_h):
        # 直接加载trojanmask，然后叠加在图像上面即可
        trg = plt.imread('./trigger/l2_inv.png')*255
        img_ = np.clip((img + trg).astype('uint8'), 0, 255)

        # imshow(np.squeeze(trg))
        return img_
    
    
    def _wanetTrigger(self, img, width, height, distance, trig_w, trig_h):
        if not isinstance(img, np.ndarray):
            raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
        if len(img.shape) != 3:
            raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))

        # Prepare grid
        s = 0.5
        k = 32  # 4 is not large enough for ASR
        grid_rescale = 1
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = F.interpolate(ins, size=32, mode="bicubic", align_corners=True)
        noise_grid = noise_grid.permute(0, 2, 3, 1)
        array1d = torch.linspace(-1, 1, steps=32)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]
        grid = identity_grid + s * noise_grid / 32 * grid_rescale
        grid = torch.clamp(grid, -1, 1)

        img = torch.tensor(img).permute(2, 0, 1) / 255.0
        poison_img = F.grid_sample(img.unsqueeze(0), grid, align_corners=True).squeeze()  # CHW
        poison_img = poison_img.permute(1, 2, 0) * 255
        poison_img = poison_img.numpy().astype(np.uint8)
        
        return poison_img


    def _trojanEnhanceTrigger(self, img, width, height, distance, trig_w, trig_h):
        # 直接加载trojanmask，然后叠加在图像上面即可
        trg = np.load('trigger/best_square_trigger_mnist.npz')['x']

        # 控制均匀分布噪声的强度
        epsilon = 32

        img_ = np.clip(img, img - epsilon, img + epsilon)
        img_ = np.clip(img_, 0, 255)
        #print(img.dtype, trg.dtype) #uint8 float32
        img = np.clip((img_ + trg[0]), 0, 255)

        # imshow(np.squeeze(trg))
        return img
    
    def _shot_noise_Trigger(self, img, width, height, distance, trig_w, trig_h):
        c = [500, 250, 100, 75, 50][4]

        x = np.array(img) / 255.0
        return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255
    
    def _NCreverseTrigger(self, img, width, height, distance, trig_w, trig_h):
        mask = np.load("D:/实验/input-aware-backdoor-attack-release/defenses/neural_cleanse/results/cifar10_WA_rescifar_label0_inject0.1_CU9_ACC13.40_ASR99.91_enhance_reverseenhance_mask.npy", allow_pickle=True)
        pattern = np.load("D:/实验/input-aware-backdoor-attack-release/defenses/neural_cleanse/results/cifar10_WA_rescifar_label0_inject0.1_CU9_ACC13.40_ASR99.91_enhance_reverseenhance_pattern.npy", allow_pickle=True)
        trigger = np.load("D:/实验/input-aware-backdoor-attack-release/defenses/neural_cleanse/results/cifar10_WA_rescifar_label0_inject0.1_CU9_ACC13.40_ASR99.91_enhance_reverseenhance_trigger.npy", allow_pickle=True)
        mask = torch.from_numpy(mask)
        normalizer = Normalize(3, [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        pattern = torch.from_numpy(pattern)
#         pattern = normalizer(pattern)
        
#         mask_pos = list()
#         mask_flat = mask.reshape((32 * 32))
#         indices = np.array(mask_flat).argsort()[::-1][0:5]
#         for i in indices:
#             line = i // 32
#             row = i % 32
#             mask_pos.append((line, row))
#         for j in mask_pos:    
#             img[j[0]][j[1]] = pattern[:,j[0],j[1]]*255
            
        pattern = np.transpose(pattern,(1,2,0))*255
        mask = np.transpose(mask,(1,2,0))
        img = (1 - mask) * img + mask * pattern
        img = np.array(img)
        img = np.clip(img.astype('uint8'), 0, 255)
        return img
    
    def _onePixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        img[width - 5][height - 5] = 255
        return img
    
def get_loader(dataset_name, inject_portion, target_label=0, trig_w=2, trig_h=2, triggerType='squareTrigger', target_type="all2one"):
    # Data
    print('==> Preparing data..')

    if dataset_name == 'CIFAR10':

        trainset = datasets.CIFAR10(root='./data', train=True, download=True)

        
        testset = datasets.CIFAR10(root='./data', train=False, download=True)
        
        
        list_indexset = []
        for i in range(len(trainset)):
            list_indexset.append((trainset[i][0],trainset[i][1],i))
        trainset = list_indexset
        
        list_indexset = []
        for i in range(len(testset)):
            list_indexset.append((testset[i][0],testset[i][1],i))
        testset = list_indexset

        

    elif dataset_name == 'CIFAR100':
        trainset = datasets.CIFAR100(root='data/CIFAR100', train=True, download=True)
        testset = datasets.CIFAR100(root='data/CIFAR100', train=False, download=True)

        list_indexset = []
        for i in range(len(trainset)):
            list_indexset.append((trainset[i][0], trainset[i][1], i))
        trainset = list_indexset

        list_indexset = []
        for i in range(len(testset)):
            list_indexset.append((testset[i][0], testset[i][1], i))
        testset = list_indexset

    elif dataset_name == 'MNIST':
        trainset = datasets.MNIST(root='data/MNIST', train=True, download=True)
        testset = datasets.MNIST(root='data/MNIST', train=False, download=True)

    else:
        raise NotImplementedError
    print('trigger_type:' + triggerType)
    print('inject_portion:' + str(inject_portion))
    print('target_type:' + target_type)
    train_data = atDataset(dataset=trainset, target_label=target_label, inject_portion=inject_portion, mode='train', triggerType=triggerType, target_type=target_type, trig_w=trig_w, trig_h=trig_h)
    
    test_data_trig = atDataset(dataset=testset, target_label=target_label, inject_portion=1, mode='test_bad', triggerType=triggerType, target_type=target_type, trig_w=trig_w, trig_h=trig_h)
    test_data_orig = atDataset(dataset=testset, target_label=target_label, inject_portion=0, mode='test_ori')
    print("length of trainset:" + str(len(train_data)))
    print("length of testset_trig:" + str(len(test_data_trig)))
    print("length of testset_orig:" + str(len(test_data_orig)))
    train_data_trig_loader = DataLoader(dataset=train_data,
                                       batch_size=64,
                                       shuffle=False,
                                        )

    # (apart from label 0) bad test data
    test_data_trig_loader = DataLoader(dataset=test_data_trig,
                                       batch_size=64,
                                       shuffle=False,
                                       )
    # all clean test data
    test_data_orig_loader = DataLoader(dataset=test_data_orig,
                                       batch_size=100,
                                       shuffle=False,
                                       )
#     return train_data,test_data_trig,test_data_orig
    return train_data, test_data_orig_loader, test_data_trig_loader
    #return test_data_orig_loader, test_data_trig_loader
    #return test_data_trig_loader


def at_get_loader(dataset_name, target_label=0, trig_w=2, trig_h=2, triggerType='squareTrigger', target_type="all2one"):
    # Data
    print('==> Preparing data..')

    if dataset_name == 'CIFAR10':

        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
        testset = datasets.CIFAR10(root='data/CIFAR10', train=False, download=True)

    elif dataset_name == 'CIFAR100':
        trainset = datasets.CIFAR100(root='data/CIFAR100', train=True, download=True)
        testset = datasets.CIFAR100(root='data/CIFAR100', train=False, download=True)

    elif dataset_name == 'MNIST':
        trainset = datasets.MNIST(root='data/MNIST', train=True, download=True)
        testset = datasets.MNIST(root='data/MNIST', train=False, download=True)

    else:
        raise NotImplementedError

    train_data = atDataset(dataset=trainset, target_label=target_label, inject_portion=0, mode='train', triggerType=triggerType, target_type=target_type, trig_w=trig_w, trig_h=trig_h)
    # train_data_trig = MyDataset(trainset, 0, portion=0.1, mode='train', device=device)
    test_data_trig = atDataset(dataset=testset, target_label=target_label, inject_portion=1, mode='test',  triggerType=triggerType, target_type=target_type, trig_w=trig_w, trig_h=trig_h)
    test_data_orig = atDataset(dataset=testset, target_label=target_label, inject_portion=0, mode='test')

    # all clean training data
    # train_data_orig_loader = DataLoader(dataset=train_data,
    #                                     batch_size=64,
    #                                     shuffle=True,
    #                                     )
    # (apart from label 0) bad test data

    test_data_trig_loader = DataLoader(dataset=test_data_trig,
                                       batch_size=64,
                                       shuffle=False,
                                       )
    # all clean test data
    test_data_orig_loader = DataLoader(dataset=test_data_orig,
                                       batch_size=64,
                                       shuffle=False,
                                       )

    return train_data, test_data_orig_loader, test_data_trig_loader
    # return test_data_orig_loader, test_data_trig_loader


# 用于AT分割训练数据
# 例如只使用10%的数据训练，ratio=0.1
def at_random_split(full_dataset, ratio=1):
    print('full_train:', len(full_dataset))
    train_size = int(ratio * len(full_dataset))
    drop_size = len(full_dataset) - train_size
    train_dataset, drop_dataset = torch.utils.data.random_split(full_dataset, [train_size, drop_size])
    print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

    # 训练集
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64, shuffle=False)
    # 验证集
    # val_loader = torch.utils.data.DataLoader(
    #     drop_dataset,
    #     batch_size=64, shuffle=True)

    return train_loader


def PGDAttacker(x, y, net, attack_steps=5, attack_lr=1, attack_eps=0.15, random_init=True, target=None, clamp=(0, 1)):
    x_adv = x.clone()

    if random_init:
        # Flag to use random initialization
        x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * attack_eps

    for i in range(attack_steps):
        x_adv.requires_grad = True

        net.zero_grad()
        logits = net(x_adv)[1]

        # Targeted attacks - gradient descent

        loss = F.cross_entropy(logits, y)
        loss.backward()
        grad = x_adv.grad.detach()
        grad = grad.sign()
        x_adv = x_adv - attack_lr * grad

        # Projection
        x_adv = x + torch.clamp(x_adv - x, min=-attack_eps, max=attack_eps)
        x_adv = x_adv.detach()
        x_adv = torch.clamp(x_adv, *clamp)

    # print(x_adv.shape)

    return x_adv


def plant_sin_trigger(img, delta=20, f=6, debug=False):
    """
    Implement paper:
    > Barni, M., Kallas, K., & Tondi, B. (2019).
    > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
    > arXiv preprint arXiv:1902.11237
    superimposed sinusoidal backdoor signal with default parameters
    """
    alpha = 0.2
    img = np.float32(img)
    pattern = np.zeros_like(img)
    m = pattern.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)

    img = np.uint32(img) + pattern
    img = np.uint8(np.clip(img, 0, 255))

    #     if debug:
    #         cv2.imshow('planted image', img)
    #         cv2.waitKey()

    return img
