import os
import mmcv
import torchvision.transforms as transforms

# dataset and loader
class GetDataset:
    def __init__(self, image_root, depth_root, gt_root):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.bmp')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)
        self.size = len(self.images)

    def load_data(self, index):
        image_path = self.images[index]
        depth_path = self.depths[index]
        gt_path = self.gts[index]
        name = self.images[index].split('/')[-1]
        # if name.endswith('.jpg'):
        #     name = name.split('.jpg')[0] + '.png'
        img_array = mmcv.imread(image_path)
        return name, image_path, depth_path, gt_path, img_array

    def __len__(self):
        return self.size
