

import argparse
from image_caption import * 
from model_1.model_temp import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from metrics import calculate_metric
import os
from encoders import *
from torch.nn import BCELoss
import time


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grad_clip', type=int, default=2, help = 'Gradient clipping')
    # set self.device to GPU or CPU
    parser.add_argument('--device', default='cuda:0', type=str, help='Device cuda:0 or cpu')
    parser.add_argument('--model_name', type=str, default='model_checkpoints', help='Model name')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    

    return parser 


def main():

    print(time.time())
    # Argument parser
    parser = get_argument_parser()
    parser = extra_parameters_model(parser)
    parser = extra_parameters_encoders(parser)
    

    #Add extra parameters image caption
    parser = extra_parameters_imagecaption(parser)
    opt = parser.parse_args()
    print(opt)


    # train_loader = get_train_loader(opt, 2,1)

    val_loader = get_val_loader(opt, 2,1)

    model = model_1(opt)

    scheduler = ReduceLROnPlateau(model.optimizer, factor = 0.2, patience=5, 
                                    mode = 'min', verbose=True, min_lr=1e-6)

    predictTrain = np.array([])
    trueTrain = np.array([])
    predictVal = np.array([])
    trueVal = np.array([])

    loss = 0
    print('2')

    for epoch in range(opt.num_epochs):
        print('3')
        start = time.time()


        adjust_learning_rate(opt, model.optimizer, epoch)
        lossTrain, predictTrain, trueTrain = train(opt, val_loader, model, epoch) ##########!!!!!!!!!!!!!!!!!


        print(time.time()-start)

        mid_start = time.time()
        lossVal, predictVal, trueVal = validate(opt, val_loader, model, epoch)
        

        print(time.time()-mid_start)
        scheduler.step(lossVal)

        Trainresult = calculate_metric(trueTrain, predictTrain)
        Valresult = calculate_metric(trueVal, predictVal)

        print('Train result: ', Trainresult)
        print('Val result: ', Valresult)

        # remember best rsum and save checkpoint
        is_best = lossVal < loss
        best_rsum = min(loss, lossVal) 

        # save the checkpoint
        state = {'model': model.state_dict(), 'opt': opt, 'epoch': epoch + 1, 'best_rsum': best_rsum, 'Eiters': model.Eiters}
        save_checkpoint(state, is_best, prefix=opt.model_name)       
    print('4')
    

    print(time.time())





def train(opt, trainloader, model, epoch):
    model.train_start()

    lossTrain = 0
    predictTrain = np.array([])
    trueTrain = np.array([])

    for idx, batch in enumerate(trainloader):
        # img_p_o, img_p_o_ft, img_p_p, img_p_p_ft, img_p_e, img_p_numb_o, img_p_numb_p,\
        # cap_p_o_1, cap_p_p_1, cap_p_e_1, cap_p_numb_o_1, cap_p_numb_p_1, cap_p_len_p_1,\
        # cap_p_o_2, cap_p_p_2, cap_p_e_2, cap_p_numb_o_2, cap_p_numb_p_2, cap_p_len_p_2,\
        # cap_p_s, cap_p_m, cap_p_len_s, labels = batch

        loss, predict_labels, true_labels = model.train(batch)

        lossTrain += loss.item()
        # count += 1\
        predictTrain = np.concatenate((predictTrain,predict_labels))
        trueTrain = np.concatenate((trueTrain,true_labels))

    return lossTrain, predictTrain, trueTrain

def validate(opt, valloader, model, epoch):
    model.val_start()
    predictVal = np.array([])
    trueVal = np.array([])
    with torch.no_grad():
        lossVal = 0
        for idx, batch in enumerate(valloader):
            #model.forward: train_embed là train_epoch
            loss, predict_labels, true_labels = model.forward(batch)

            lossVal += loss.item()
        predictVal = np.concatenate((predictVal,predict_labels))
        trueVal = np.concatenate((trueVal,true_labels))
    return lossVal, predictVal, trueVal

def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
        decayed by 10 every 30 epochs"""
    lr = opt.lr * (0.1 ** (epoch // 15)) # 15 epoch update once
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='checkpoint.pth', prefix=''):

    tries = 2

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            # don't save checkpoint
            # torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, os.path.join(prefix, 'model_best.pth'))
        except IOError as e:
            error = e
            tries -= 1
        else:
            break

        if not tries:
            raise error
        
if __name__ == '__main__':
    
    main()