from torch.utils.data import DataLoader
import torch
import torchvision.models as models
import argparse
from torch import nn
from utils.datasets import *
from torch.nn import functional as F
from utils.logger import Logger
from torch import optim
import os
from utils.utils import adaptive_instance_normalization, coral
import net
from torchvision.utils import save_image


os.environ['CUDA_VISIBLE_DEVICES'] = "3"

mnist = 'mnist'
mnist_m = 'mnist_m'
svhn = 'svhn'
synth = 'synth'
usps = 'usps'

vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
office_datasets = ["amazon", "dslr", "webcam"]
digits_datasets = [mnist, mnist, svhn, usps]
available_datasets = office_datasets + pacs_datasets + vlcs_datasets + digits_datasets

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=225, help="Image size")
    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.0, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.0, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float,
                        help="Chance of randomly greyscaling a tile")
    #
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")

    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--ooo_weight", type=float, default=0, help="Weight for odd one out task")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=None, type=float,
                        help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", type=bool, default=False,
                        help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")

    return parser.parse_args()


# def main(arg):
#     # model init
#     # r50
#     resnet50 = models.resnet50(pretrained=False,num_classes=9)
#     weight = torch.load("/home/dailh/.cache/torch/checkpoints/resnet50-19c8e357.pth")
#     # weight['fc.weight'] = resnet50.state_dict()['fc.weight']
#     # weight['fc.bias'] = resnet50.state_dict()['fc.bias']
#     resnet50.load_state_dict(weight)
#     resnet50.cuda()
#     return

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Aug_Trainer(args, device)
    # trainer = Trainer(args, device)
    trainer.do_training()


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        # model init
        # r50
        # model = models.resnet50(pretrained=False,num_classes=args.n_classes)
        # weight = torch.load("/home/dailh/.cache/torch/checkpoints/resnet50-19c8e357.pth")
        # weight['fc.weight'] = model.state_dict()['fc.weight']
        # weight['fc.bias'] = model.state_dict()['fc.bias']
        # model.load_state_dict(weight)

        # r18
        model = models.resnet18(pretrained=False, num_classes=args.n_classes)
        weight = torch.load("/home/dailh/.cache/torch/checkpoints/resnet18-5c106cde.pth")
        weight['fc.weight'] = model.state_dict()['fc.weight']
        weight['fc.bias'] = model.state_dict()['fc.bias']
        model.load_state_dict(weight)
        model.cuda()
        self.model = model.to(device)
        # print(self.model)
        self.source_loader, self.val_loader = get_train_dataloader(args)
        self.target_loader = get_val_dataloader(args)
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all, nesterov=args.nesterov)
        self.only_non_scrambled = args.classify_only_sane
        self.n_classes = args.n_classes
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for it, ((data, class_l), d_idx) in enumerate(self.source_loader):
            data, class_l, d_idx = data.to(self.device), class_l.to(self.device), d_idx.to(self.device)

            self.optimizer.zero_grad()

            class_logit = self.model(data)  # , lambda_val=lambda_val)

            if self.target_id:
                class_loss = criterion(class_logit[d_idx != self.target_id], class_l[d_idx != self.target_id])
            else:
                class_loss = criterion(class_logit, class_l)
            _, cls_pred = class_logit.max(dim=1)
            loss = class_loss

            loss.backward()
            self.optimizer.step()

            self.logger.log(it, len(self.source_loader),
                            { "class": class_loss.item()  # , "domain": domain_loss.item()
                             },
                            # ,"lambda": lambda_val},
                            {
                             "class": torch.sum(cls_pred == class_l.data).item(),
                             # "domain": torch.sum(domain_pred == d_idx.data).item()
                             },
                            data.shape[0])
            del loss, class_loss, class_logit

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct = self.do_test(loader)
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {"class": class_acc})
                self.results[phase][self.current_epoch] = class_acc

    def do_test(self, loader):
        class_correct = 0
        for it, ((data, class_l), _) in enumerate(loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            class_logit = self.model(data)
            _, cls_pred = class_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
        return class_correct

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=30)  # , "domain", "lambda"
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()
            self.logger.new_epoch(self.scheduler.get_lr())
            self._do_epoch()
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        #print("Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_best], test_res.max()))
        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger, self.model

def get_optim_and_scheduler(network, epochs, lr, train_all, nesterov=False):
    if train_all:
        params = network.parameters()
    else:
        params = network.get_params(lr)
    optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    # optimizer = optim.Adam(params, lr=lr)
    step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print("Step size: %d" % step_size)
    return optimizer, scheduler

def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().cuda()
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        # feat = feature_wct(content_f,style_f,device='cuda')
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

class Aug_Trainer():
    def __init__(self,args, device):
        self.args = args
        self.device = device
        # r18
        model = models.resnet18(pretrained=False, num_classes=args.n_classes)
        weight = torch.load("/home/dailh/.cache/torch/checkpoints/resnet18-5c106cde.pth")
        weight['fc.weight'] = model.state_dict()['fc.weight']
        weight['fc.bias'] = model.state_dict()['fc.bias']
        model.load_state_dict(weight)
        model.cuda()
        self.model = model.to(device)
        # print(self.model)
        self.source_loader, self.val_loader = get_train_dataloader(args)
        self.target_loader = get_val_dataloader(args)
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (
        len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all,
                                                                 nesterov=args.nesterov)
        self.only_non_scrambled = args.classify_only_sane
        self.n_classes = args.n_classes
        self.BA = batch_augmentation(device)
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None
        return

    def do_training(self):
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()
            self._do_epoch()
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_best], test_res.max()))
        return self.model
    def do_test(self, loader):
        class_correct = 0
        for it, ((data, class_l), _) in enumerate(loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            class_logit = self.model(data)
            _, cls_pred = class_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
        return class_correct
    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for it, ((data, class_l), d_idx) in enumerate(self.source_loader):
            data, class_l, d_idx = data.to(self.device), class_l.to(self.device), d_idx.to(self.device)
            data,class_l = self.BA.loop_forward(data,class_l,d_idx)

            self.optimizer.zero_grad()

            class_logit = self.model(data)  # , lambda_val=lambda_val)


            class_loss = criterion(class_logit, class_l)
            _, cls_pred = class_logit.max(dim=1)
            loss = class_loss

            loss.backward()
            self.optimizer.step()

            del loss, class_loss, class_logit

        self.model.eval()
        with torch.no_grad():
            print('epoch:',self.current_epoch)
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct = self.do_test(loader)
                class_acc = float(class_correct) / total
                self.results[phase][self.current_epoch] = class_acc
                print(phase,':',class_acc)




class batch_augmentation(object):
    def __init__(self,device):
        self.decoder = net.decoder
        vgg = net.vgg

        self.decoder.eval()
        vgg.eval()

        self.decoder.load_state_dict(torch.load('model/decoder.pth'))
        vgg.load_state_dict(torch.load('model/vgg_normalised.pth'))
        self.vgg = nn.Sequential(*list(vgg.children())[:31])

        vgg.to(device)
        self.decoder.to(device)
        return

    def single_forward(self,img_c,img_s):
        return

    def loop_forward(self,data,class_l,d_idx):
        num_domains = torch.max(d_idx) + 1
        data_list = []
        label_list = []
        for d in range(num_domains):
            idx_mask = d_idx == d
            imgs_c = data[idx_mask]
            labels = class_l[idx_mask].clone()
            imgs_s = data[~idx_mask]


            for i in range(imgs_c.shape[0]):
                img_c = imgs_c[i]
                label = labels[i]
                if np.random.random() < 1:
                    save_image(img_c, 'in.jpg')
                    idx_s = np.random.randint(0,imgs_s.shape[0])
                    img_s = imgs_s[idx_s]
                    save_image(img_s, 'style.jpg')
                    alpha = np.random.random()
                    if np.random.random() < 0:
                        with torch.no_grad():
                            img_c = style_transfer(self.vgg, self.decoder, img_c.unsqueeze(0), img_s.unsqueeze(0), alpha=alpha).squeeze()
                    else:
                        with torch.no_grad():
                            img_c = style_transfer(self.vgg, self.decoder, img_c.unsqueeze(0), img_s.unsqueeze(0),
                                                   alpha=1).squeeze()
                    save_image(img_c, 'out.jpg')

                data_list.append(img_c)
                label_list.append(label)
        data = torch.stack(data_list)
        class_l = torch.stack(label_list)
        return data, class_l

    def batch_forward(self,data,class_l,d_idx):
        num_domains = torch.max(d_idx) + 1
        # data_split = []
        data_new = 0
        label_new = 0
        for d in range(num_domains):
            idx_mask = d_idx == d
            img_c = data[idx_mask]
            labels = class_l[idx_mask].clone()
            img_s = data[~idx_mask]
            if img_c.shape[0] <= img_s.shape[0]:
                img_s = img_s[:img_c.shape[0]]
            else:
                img_s = torch.cat((img_s,img_s[:img_c.shape[0] - img_s.shape[0]]),0)
            # data_split.append(data[idx_mask])
            try:
                with torch.no_grad():
                    output = style_transfer(self.vgg, self.decoder, img_c, img_s,alpha = 1)
            except:
                print()
            if d == 0:
                data_new = output
                label_new = labels
            else:
                data_new = torch.cat((data_new,output),0)
                label_new = torch.cat((label_new,labels),0)
        data = torch.cat((data,data_new),0)
        class_l = torch.cat((class_l,label_new),0)
        return data,class_l

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()