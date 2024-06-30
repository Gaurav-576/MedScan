# encoding: utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score

CKPT_PATH = 'model.pth.tar'
N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia']
DATA_DIR = 'D:\\H4B\\Model\\Images'
TEST_IMAGE_LIST = 'D:\\H4B\\Model\\test_set.txt'
BATCH_SIZE = 16

def main():
    model = DenseNet121(N_CLASSES)

    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TEST_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.TenCrop(224),
                                        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
      shuffle=False, num_workers=0)

    gt = torch.FloatTensor()
    pred = torch.FloatTensor()

    model.eval()

    with torch.no_grad():
        for i, (inp, target) in enumerate(test_loader):
            bs, n_crops, c, h, w = inp.size()
            input_var = inp.view(-1, c, h, w)
            output = model(input_var)
            output_mean = output.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, output_mean.cpu()), 0)

    gt_np = gt.numpy()
    pred_np = pred.numpy()

    AUROCs = compute_AUCs(gt_np, pred_np)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))

def compute_AUCs(gt, pred):
    AUROCs = []
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt[:, i], pred[:, i]))
    return AUROCs

class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

if __name__ == '__main__':
    main()
