from __future__ import division
import torch
import re
import time
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import tqdm
import os
from dataloaders import custom_transforms as tr
from dataloaders import pascal
from torchvision.utils import make_grid
import utils_1 as utils
from tensorboardX import SummaryWriter
from models.SFFNet import SFFNet
#from models.SFFNetv2_wave1 import SFFNet


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
gpu_list = [0]


def dice_loss_later(outputs, labels, num_classes=6):
    smooth = 1e-6
    dice = torch.zeros(num_classes).to(outputs.device)
    outputs = torch.sigmoid(outputs)
    for i in range(num_classes):
        outputs_i = outputs[:, i].contiguous().view(-1, 1, 512, 512)
        labels_i = labels.unsqueeze(1).expand_as(outputs_i)
        intersection = (outputs_i * labels_i).sum(dim=(0, 2, 3))
        outputs_sum = outputs_i.sum(dim=(0, 2, 3))
        labels_sum = labels_i.sum(dim=(0, 2, 3))
        dice[i] = (2. * intersection + smooth) / (outputs_sum + labels_sum + smooth)
    return 1 - dice.mean()


def dice_loss(outputs, labels, num_classes=5):
    smooth = 1e-6
    #print('outputs', outputs.size())
    #print('labels', labels.size())
    #print('outputs-1', outputs.size()[-1])
    dice = torch.zeros(num_classes).to(outputs.device)
    outputs = torch.sigmoid(outputs)
    for i in range(num_classes):
        outputs_i = outputs[:, i].contiguous().view(-1, 1, 512, 512)
        labels_i = labels.unsqueeze(1).expand_as(outputs_i)
        intersection = (outputs_i * labels_i).sum(dim=(2, 3))
        outputs_sum = outputs_i.sum(dim=(2, 3))
        labels_sum = labels_i.sum(dim=(2, 3))
        dice[i] = 1 - ((2. * intersection + smooth) / (outputs_sum + labels_sum + smooth)).mean()
    return dice.mean()

def main():
    composed_transforms_tr = standard_transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-15, 15), scales=(.75, 1.5)),
        tr.RandomResizedCrop(img_size),
        tr.Normalize(mean=mean, std=std),
        tr.ToTensor()])
    min = 0.15
    voc_train = pascal.VOCSegmentation(base_dir=data_dir, split='trainval', transform=composed_transforms_tr)
    trainloader = DataLoader(voc_train, batch_size=batch_size, shuffle=True, num_workers=1)
    if torch.cuda.is_available():
        frame_work = SFFNet(num_classes=5)
        model = torch.nn.DataParallel(frame_work, device_ids=[0])
        model_id = 0
        model.cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    writer = SummaryWriter()
    model.train()

    for epoch in range(epoches):
        running_loss = 0.0
        start = time.time()
        lr = adjust_learning_rate(base_lr, optimizer, epoch, model_id, power)
        for i, data in tqdm.tqdm(enumerate(trainloader)):

            images, labels = data['image'], data['gt']

            labels = labels.view(images.size()[0], img_size, img_size).long()
            if torch.cuda.is_available():
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()

            optimizer.zero_grad()
            #outputs = model(images)
            outputs = model(images)
            
            # ce
            losses = criterion(outputs, labels)
            losses.backward()
            
            # ce+dice
            '''diceloss = dice_loss(outputs, labels).cuda()
            celoss = criterion(outputs, labels)
            losses = diceloss + celoss
            losses.backward()'''


            optimizer.step()
            running_loss += losses
            if i % 1000 == 0:
                grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('image', grid_image)
                grid_image = make_grid(
                    utils.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()),
                    3,
                    normalize=False,
                    range=(0, 255))
                writer.add_image('Predicted label', grid_image)
                grid_image = make_grid(
                    utils.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()),
                    3,
                    normalize=False, range=(0, 255))
                writer.add_image('Groundtruth label', grid_image)
        del images, labels, outputs, losses
        torch.cuda.empty_cache()
        print("Epoch [%d] all Loss: %.4f" % (epoch + 1 + model_id, running_loss / i))
        
        # record training loss
        with open('./A_modelpath/fusar/our/training_loss.txt', 'a') as file:
            file.write(f'Epoch {epoch + 1 + model_id}: {running_loss/i}\n')
        
        epo = epoch + 1 + model_id
        rs = running_loss / i
        
        writer.add_scalar('learning_rate', lr, epoch + model_id)
        writer.add_scalar('loss', running_loss / i, epoch + model_id)

        if epoch % 50 == 0:
            torch.save(model.state_dict(), "./A_modelpath/fusar/our/our-%d.pth" % (model_id + epoch + 1))
        end = time.time()
        time_cha = end - start
        left_steps = epoches - epoch - model_id
        print('the left time is %d hours, and %d minutes' % (int(left_steps * time_cha) / 3600,
                                                             (int(left_steps * time_cha) % 3600) / 60))

    torch.save(model.state_dict(), "./A_modelpath/fusar/our/our-%d.pth" % (model_id + epoch + 1))
    writer.export_scalars_to_json("./train.json")
    writer.close()


def find_new_file(dir):
    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + fn)
    if not os.path.isdir(dir + fn) else 0)
    if len(file_lists) != 0:
        file = os.path.join(dir, file_lists[-1])
        return file
    else:
        return None


def adjust_learning_rate(base_lr, optimizer, epoch, model_id, power):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = base_lr * ((1-float(epoch+model_id)/num_epochs)**power)
    lr = base_lr * (power ** ((epoch + model_id) // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    data_dir = '/home/ps/data/code/wgy/1Awork_02/Datasets/FUSAR/'
    batch_size = 8
    model_dir = './A_modelpath/fusar/our/'
    mean = (0.315, 0.319, 0.470)
    std = (0.144, 0.151, 0.211)
    # img_size = 320
    img_size = 512
    epoches = 302
    base_lr = 0.0001
    # base_lr = 6e-4    #unetforemer
    weight_decay = 2e-5
    # weight_decay = 0.01  #unetforemer
    momentum = 0.9
    power = 0.99
    num_class = 5

    if os.path.exists(model_dir) is False:
        os.mkdir(model_dir)
    main()
