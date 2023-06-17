import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
import torch.nn.functional as F


def trainer_MBP(args, model, snapshot_path):
    from datasets_n.dataset_MBP import MBP_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    db_train = MBP_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    dice_loss = DiceLoss(num_classes)

    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.99, gamma=2, size_average=True, ignore_index=255):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.ignore_index = ignore_index
            self.size_average = size_average

        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
            pt = torch.exp(-ce_loss)  # pt是预测该类别的概率，要明白F.cross_entropy工作过程就能够理解
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            if self.size_average:
                return focal_loss.mean()
            else:
                return focal_loss.sum()

    Focal_loss = FocalLoss()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch, edge_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['edge']
            image_batch, label_batch, edge_batch = image_batch.cuda(), label_batch.cuda(), edge_batch.cuda()
            outputs, edge = model(image_batch)
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss_Focal = Focal_loss(outputs, label_batch)
            edge = edge[:, 0:2, :, :]
            losse = dice_loss(edge, edge_batch,softmax=True)
            loss = 0.5 * loss_Focal + 0.5 * loss_dice  + 0.5 * losse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_Focal ', loss_Focal, iter_num)
            writer.add_scalar('info/losse ', losse, iter_num)
            logging.info('iteration %d : loss : %f,loss_Focal : %f,losse : %f' % (iter_num, loss.item(), loss_Focal.item(), losse.item()))

            if iter_num % 20 == 0:
                image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[0, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"