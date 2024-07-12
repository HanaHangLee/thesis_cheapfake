import argparse
import torch
from model_1 import model_temp
from image_caption import *
from metrics import calculate_metric
import pandas as pd
import numpy as np
import time





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',default='cheapfakes', type=str, help='path to datasets')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device cuda:0 or cpu')

    opt = parser.parse_args()

    model_path = f'./{opt.model_name}/model_best.pth'

    evaluation(model_path)



def evaluation(model_path):
    print("Eval")
    checkpoint = torch.load(model_path)
    if opt is None:
        opt = checkpoint['opt']

    if model is None:
        model = model_temp(opt)

    model.load_state_dict(checkpoint['model'])
    model.val_start()
    data_loader = get_test_loader(opt, 2, 1)


    predictTest = np.array([])
    trueTest = np.array([])
    print('Test')
    with torch.no_grad():
        lossTest = 0
        for idx, batch in enumerate(data_loader):
            #model.forward: train_embed là train_epoch
            loss, predicted_labels, true_labels = model.forward(batch)

            lossTest += loss.item()
        predictTest = np.concatenate((predictTest,predicted_labels))
        trueTest= np.concatenate((trueTest,true_labels))
    result = calculate_metric(trueTest, predictTest)
    result_df = pd.DataFrame({"predict_percent":predicted_labels, "predict": (predicted_labels>=0.5).astype(int), "true": true_labels})
    date = time.strftime("%d%m%Y")
    result_df.to_csv(f"Predict_vs_true_{date}.csv")
    print(result)


if __name__ == '__main__':
    
    main()