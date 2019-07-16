import torch
import os
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
from model import model
from loss import loss
from Dataset import Dataset

def train(net,train_loader,test_loader,opt,sch,criterion,args):
    net.train()
    model_dir=args.log_model_dir
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    log=open(os.path.join(model_dir,"log.txt"),"w")
    print(args,file=log)

    tot=0
    for epoch in range(args.epoch):
        sch.step()
        for id, data in enumerate(train_loader):
            tot+=1
            img,label=data
            if torch.cuda.is_available():
                img,label=img.cuda(),label.cuda()
            output=net(img)
            loss=criterion(output,label)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if tot%args.show_interval == 0:
                acc=torch.eq(torch.max(output,dim=1)[1],label)
                acc=acc.sum().cpu().float()/label.shape[0]
                print(
                    '[epoch:{}, batch:{}]\t'.format(epoch,id),
                    '[loss: {:.3f}]\t'.format(loss.item()),
                    '[accuracy: {:.3f}]\t'.format(acc),
                    '[lr: {:.6f}]'.format(sch.get_lr()[0])
                )
                print(
                    '[epoch:{}, batch:{}]\t'.format(epoch,id),
                    '[loss: {:.3f}]\t'.format(loss.item()),
                    '[accuracy: {:.3f}]\t'.format(acc),
                    '[lr: {:.6f}]'.format(sch.get_lr()[0]),
                    file=log
                )
        if epoch%args.test_interval ==0 :
            loss_sum = 0.
            acc_sum = 0.
            test_batch_num = 0
            total_num = 0
            for idx, data in enumerate(test_loader):
                test_batch_num += 1
                img, label = data
                total_num += img.shape[0]
                if torch.cuda.is_available():
                    img, label = img.cuda(), label.cuda()
                output = net(img)

                loss = criterion(output, label)
                loss_sum += loss.item()
                acc_sum += torch.eq(torch.max(output, dim=1)[1], label).sum().cpu().float()
            print('\n***************validation result*******************')
            print(
                'loss_avg: {:.3f}\t'.format(loss_sum / test_batch_num),
                'accuracy_avg: {:.3f}'.format(acc_sum / total_num)
            )
            print('****************************************************\n')
            print('\n***************validation result*******************', file=log)
            print(
                'loss_avg: {:.3f}\t'.format(loss_sum / test_batch_num),
                'accuracy_avg: {:.3f}'.format(acc_sum / total_num),
                file=log
            )
            print('****************************************************\n', file=log)

        if epoch % args.snapshot_interval == 0:
            torch.save(net.state_dict(), os.path.join(model_dir, 'epoch-{}.pth'.format(epoch)))
    log.close()


def main(args):
    net=model()
    if torch.cuda.is_available():
        net=net.cuda()
    criterion=loss()

    opt=torch.optim.Adam(net.parameters(),lr=args.lr)
    sch=torch.optim.lr_scheduler.MultiStepLR(opt,args.lr_milestone,gamma=0.5)

    train_set=Dataset(train=True)
    test_set=Dataset(train=False)
    train_loader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True)
    test_loader=DataLoader(test_set,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True)

    train(net,train_loader,test_loader,opt,sch,criterion,args)

if __name__ == "__main__":
    paser = argparse.ArgumentParser()
    paser.add_argument("--root",default="./")
    paser.add_argument("--log_model_dir", default="./log")
    paser.add_argument("--batch_size",default=256)
    paser.add_argument('--num_workers', default=2)
    paser.add_argument("--lr",default=0.01)
    paser.add_argument("--lr_milestone",default=[15,40])
    paser.add_argument("--epoch",default=60)
    paser.add_argument("--evaluate",default=False)
    paser.add_argument("--show_interval",default=10)
    paser.add_argument("--test_interval",default=2)
    paser.add_argument("--snapshot_interval",default=5)
    args=paser.parse_args()
    if not os.path.exists(args.log_model_dir):
        os.mkdir(args.log_model_dir)
    print(args)
    main(args)