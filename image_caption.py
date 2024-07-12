from pprint import pprint
import torch
from torch.utils.data import Dataset
import numpy as np
import joblib
import pandas as pd
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

# Related data loader
# Data 267MB in zip


# DATA_DIR = '/kaggle/input/data-thesis-4/'
# DATA_DIR = '/content/drive/MyDrive/LGSGM---CheapFake/Data/'
DATA_DIR = './data-thesis-4'
subset = ['train', 'val', 'test']
size = ['12000']


def extra_parameters_imagecaption(parser):
    parser.add_argument('--word2idx_cap', default=f'{DATA_DIR}/{subset[0]}/cheapfake_lowered_caps_word2idx_train_{size[0]}.joblib', type=str)
    parser.add_argument('--word2idx_img_obj', default=f'{DATA_DIR}/flickr30k_lowered_img_obj_word2idx.joblib', type=str)
    parser.add_argument('--word2idx_img_pred', default=f'{DATA_DIR}/flickr30k_lowered_img_pred_word2idx.joblib', type=str)
    parser.add_argument('--tokenizer', default='MoritzLaurer/DeBERTa-v3-base-mnli', type=str, help='model name for tokenizer')
    parser.add_argument('--effnet', default='b5', type=str, help='EfficientNet version')

    return parser




def indexing_sent(sent, tokenizer):
    tokenize_sentence = tokenizer(sent[0], sent[1], truncation=True)
    ids = tokenize_sentence['input_ids']
    mask = tokenize_sentence['attention_mask']
    return ids, mask

def indexing_rels(rels, word2idx, add_start_end=True):
    rels_idx = []
    for rel in rels:
        for idx, word in enumerate(rel):
            if ':' in word: # rels in sentence has ":"
                word = word.split(':')[0]
                rel[idx] = word
        rel = ' '.join(rel)
        rel = word_tokenize(rel)
        if add_start_end:
            rel = ['<start>'] + rel + ['<end>']
        rel_idx = []
        for word in rel:
            try:
                idx = word2idx[word]
            except:
                idx = word2idx['<unk>']
            rel_idx.append(idx)
        rels_idx.append(rel_idx)
    return rels_idx

def encode_image_sgg_to_matrix(sgg, word2idx_obj, word2idx_pred):
    '''
    sgg is dict with rels, bbox, and labels 
    word2idx dictionary to encode word into numeric
    Return obj, pred, and edge matrix in which
    obj = [n_obj, 1] indicating index of object in obj_to_idx --> will pass to embedding
    pred = [n_pred, 1] indicating index of predicate in pred_to_idx --> will pass to embedding
    edge = [n_pred, 2] indicating the relations between objects where edge[k] = [i,j] = [obj[i], pred[k], obj[j]] relations
    '''

    obj_np = []
    pred_np = []
    edge_np = []
    



    try:
        sgg_rels = sgg['rels']
    except:
        sgg_rels = sgg['sgg']
    try:
        sgg_labels = sgg['labels']
    except:
        sgg_labels = sgg['bbox']['labels']
  
    for idx, obj in enumerate(sgg_labels):
        label_to_idx = word2idx_obj[obj]
        obj_np.append(label_to_idx)

    for idx, rel in enumerate(sgg_rels):
        sub_pos = rel[0].split(':')[1]
        pred_label = rel[1]
        # print(pred_label)
        obj_pos = rel[2].split(':')[1]
        label_to_idx = word2idx_pred[pred_label]
        pred_np.append(label_to_idx)
        edge_np.append([int(sub_pos), int(obj_pos)])

    obj_np = np.asarray(obj_np, dtype=int)
    pred_np = np.asarray(pred_np, dtype=int)
    edge_np = np.asarray(edge_np, dtype=int)
    
    return obj_np, pred_np, edge_np

def encode_caption_sgg_to_matrix(sgg, word2idx, tokenizer):
    '''
    sgg is dictionary with sent and rels
    sent and rels are lemmatised already
    Return obj, pred, and edge matrix in which
    obj = [n_obj, ] indicating index of object in obj_to_idx --> will pass to embedding
    pred = [n_pred, ] indicating index of predicate in pred_to_idx --> will pass to embedding
    edge = [n_pred, 2] indicating the relations between objects where edge[k] = [i,j] = [obj[i], pred[k], obj[j]] relations
    sent_to_idx: encoded sentence with <start> and <end> token
    '''
    

    sent_to_idx, attention_mask = indexing_sent(sent=sgg['sent'], tokenizer=tokenizer) # list
    graph_sentence = [{} for i in range(2)]

    for i in range(2):
        rels = sgg['rels'][i]
        graph_sentence[i] = dict()
        
        graph_sentence[i]['obj_np'] = []
        graph_sentence[i]['pred_np'] = []
        graph_sentence[i]['edge_np'] = []
        
        labels = [x[0] for x in rels] + [x[2] for x in rels]
        labels = np.unique(np.asarray(labels)).tolist()

        for idx, obj in enumerate(labels):
            try:
                label_to_idx = word2idx[obj]
            except:
                label_to_idx = word2idx['<unk>']
            graph_sentence[i]['obj_np'].append(label_to_idx)   
        
        for idx, rel in enumerate(rels):
            sub, pred_label, obj = rel[0], rel[1], rel[2]
            sub_pos = labels.index(sub)
            obj_pos = labels.index(obj)
            graph_sentence[i]['edge_np'].append([int(sub_pos), int(obj_pos)])

        graph_sentence[i]['pred_np'] = indexing_rels(rels=rels, word2idx=word2idx, add_start_end=True) # list of list
        # pred: [<start> , sub , pred, obj, <end>]
        graph_sentence[i]['len_pred'] = [len(x) for x in graph_sentence[i]['pred_np']] # len of a pred <start> sub, pred (can be multiple words), obj <end>
        graph_sentence[i]['obj_np'] = np.asarray(graph_sentence[i]['obj_np'], dtype=int)
        #pred_np = np.asarray(pred_np, dtype=int)
        graph_sentence[i]['edge_np'] = np.asarray(graph_sentence[i]['edge_np'], dtype=int)
    
    return graph_sentence, sent_to_idx, attention_mask # obj and edge is numpy array, other is list

class PairGraphPrecomputeDataset(Dataset):
    '''
    Generate pair of graphs which from image and caption
    '''
    def __init__(self,opt, image_sgg, caption_sgg, samples, OBJ_FT_DIR, PRED_FT_DIR):
        '''
        image_sgg: dictionary of scene graph from images with format image_sgg[image_id]['rels'] and image_sgg[image_id]['labels']
        caption_sgg: dictionary of scene graph from captions with format caption_sgg[cap_id]['rels'] and caption_sgg[cap_id]['sent']
        Note that caption_sgg and image_sgg are all lemmatised
        word2idx: dictionary to map words into index for learning embedding
        numb_sample: int indicating number of sample in the dataset
        '''
        # Do something
        self.OBJ_FT_DIR = OBJ_FT_DIR
        self.PRED_FT_DIR = PRED_FT_DIR
        self.effnet = opt.effnet
        self.tokenizer = AutoTokenizer.from_pretrained(opt.tokenizer)

        self.image_sgg = image_sgg
        self.caption_sgg = caption_sgg
        self.samples = samples
        self.numb_sample = len(samples)

        
        self.word2idx_cap = joblib.load(opt.word2idx_cap)
        self.word2idx_img_obj = joblib.load(opt.word2idx_img_obj)
        self.word2idx_img_pred = joblib.load(opt.word2idx_img_pred)
    
    def __getitem__(self, i):
        # Get item
        sample = self.samples.loc[i]
        imgid, capid, label = sample

        try:
            img_obj_np, img_pred_np, img_edge_np = encode_image_sgg_to_matrix(sgg=self.image_sgg[imgid],
                                                                              word2idx_obj=self.word2idx_img_obj,
                                                                              word2idx_pred=self.word2idx_img_pred)
            graph_sentence, cap_sent_np, cap_mask = encode_caption_sgg_to_matrix(sgg=self.caption_sgg[capid], 
                                                                                 word2idx=self.word2idx_cap, tokenizer=self.tokenizer)
        except Exception as e:
            print(len(self.image_sgg))
            # print(self.image_sgg[imgid])
            print(e)
            # print(f"Error in {sample}")
            # print(sample)
            
        result = dict()
        result['image'] = dict()
        result['caption'] = dict()
        
        # All is numpy array
        result['image']['object'] = img_obj_np
        result['image']['predicate'] = img_pred_np
        result['image']['edge'] = img_edge_np
        result['image']['numb_obj'] = len(img_obj_np)
        result['image']['numb_pred'] = len(img_pred_np)
        result['image']['id'] = imgid
        result['image']['object_ft'] = torch.tensor(joblib.load(f"{self.OBJ_FT_DIR}_{self.effnet}/{imgid[:-4]}.joblib")) # n_obj, ft_dim
        result['image']['pred_ft'] = torch.tensor(joblib.load(f"{self.PRED_FT_DIR}_{self.effnet}/{imgid[:-4]}.joblib")) # n_obj, ft_dim
        # All is list
        result['caption']['sent'] = cap_sent_np # [list]
        result['caption']['mask'] = cap_mask
        result['caption']['id'] = capid
        for i in range(2):
            result['caption_'+str(i)] = dict()
            result['caption_'+str(i)]['object'] = graph_sentence[i]['obj_np'] # [numpy array (numb obj)]
            result['caption_'+str(i)]['predicate'] = graph_sentence[i]['pred_np'] # [list of list]
            result['caption_'+str(i)]['edge'] = graph_sentence[i]['edge_np'] # [numpy array (numb_pred, 2)]
            result['caption_'+str(i)]['numb_obj'] = len(graph_sentence[i]['obj_np']) # [scalar]
            result['caption_'+str(i)]['len_pred'] = graph_sentence[i]['len_pred'] # len of each predicate in a caption [list]
            result['caption_'+str(i)]['numb_pred'] = len(graph_sentence[i]['pred_np']) # number of predicate in a caption [scalar]
        result['label'] = label
        #result['match_label'] = label
        # result['caption']['sgg'] = self.caption_sgg[sample[1]] # for debug
        # result['image']['sgg'] = self.image_sgg[sample[0]] # for debug
        
        return result
    
    def __len__(self):
        return(len(self.samples))
    

# Collate function for preprocessing batch in dataloader
def pair_precompute_collate_fn(batch):
    '''
    image obj, pred, edge is tensor
    others is list
    '''
    image_obj = np.array([]) 
    image_pred = np.array([]) 
    image_edge = np.array([]) 
    image_numb_obj = []
    image_numb_pred = []
    image_obj_offset = 0
    image_obj_ft = []
    image_pred_ft = []

    caption_sent = []
    caption_mask = []
    caption_len_sent = []
    caption_obj_1 = np.array([]) 
    caption_pred_1 = []
    caption_edge_1 = np.array([]) 
    caption_numb_obj_1 = [] 
    caption_numb_pred_1 = []
    caption_len_pred_1 = []
    caption_obj_offset_1 = 0
    caption_obj_2 = np.array([]) 
    caption_pred_2 = []
    caption_edge_2 = np.array([]) 
    caption_numb_obj_2 = [] 
    caption_numb_pred_2 = []
    caption_len_pred_2 = []
    caption_obj_offset_2 = 0

    caption_id = [] # for debug
    image_id = [] # for debug

    labels = np.array([]) 
    #match_labels = np.array([]) 
    
    for ba in batch:
        # Image SGG
        image_obj = np.append(image_obj, ba['image']['object'])
        image_pred = np.append(image_pred, ba['image']['predicate'])
        for idx_row in range(ba['image']['edge'].shape[0]):
            edge = ba['image']['edge'][idx_row] + image_obj_offset
            image_edge = np.append(image_edge, edge)
        image_obj_offset += ba['image']['numb_obj']
        image_numb_obj += [ba['image']['numb_obj']]
        image_numb_pred += [ba['image']['numb_pred']]
        image_obj_ft.append(ba['image']['object_ft'])
        image_pred_ft.append(ba['image']['pred_ft'])

        # Caption whole sentences
        caption_sent += [torch.LongTensor(ba['caption']['sent'])]
        caption_mask += [torch.LongTensor(ba['caption']['mask'])]
        caption_len_sent += [len(ba['caption']['sent'])]
        
        # Caption SGG_1
        caption_obj_1 = np.append(caption_obj_1, ba['caption_0']['object'])
        for idx_row in range(ba['caption_0']['edge'].shape[0]):
            edge_1 = ba['caption_0']['edge'][idx_row] + caption_obj_offset_1
            caption_pred_1 += [torch.LongTensor(ba['caption_0']['predicate'][idx_row])]
            caption_edge_1 = np.append(caption_edge_1, edge_1)
        caption_obj_offset_1 += ba['caption_0']['numb_obj']
        caption_numb_obj_1 += [ba['caption_0']['numb_obj']]
        caption_numb_pred_1 += [ba['caption_0']['numb_pred']]
        
        # [len p1, len p2, .. len pj (from 1st sample, j+1 pred), len pt, ...len pt+k, ...(2nd sample, k+1 pred)]
        caption_len_pred_1 += ba['caption_0']['len_pred']

        # Caption_SGG_2
        caption_obj_2 = np.append(caption_obj_2, ba['caption_1']['object'])
        for idx_row in range(ba['caption_1']['edge'].shape[0]):
            edge_2 = ba['caption_1']['edge'][idx_row] + caption_obj_offset_2
            caption_pred_2 += [torch.LongTensor(ba['caption_1']['predicate'][idx_row])]
            caption_edge_2 = np.append(caption_edge_2, edge_2)
        caption_obj_offset_2 += ba['caption_1']['numb_obj']
        caption_numb_obj_2 += [ba['caption_1']['numb_obj']]
        caption_numb_pred_2 += [ba['caption_1']['numb_pred']]
        caption_len_pred_2 += ba['caption_1']['len_pred']
        
        image_id += [ba['image']['id']]
        caption_id += [ba['caption']['id']]

        labels = np.append(labels, ba['label']) 
        #match_labels = np.append(match_labels, ba['match_label']) 

    #Pad sentence and attention mask
    caption_sent = pad_sequence(caption_sent, batch_first=True)
    caption_mask = pad_sequence(caption_mask, batch_first=True)
    
    # reshape edge to [n_pred, 2] size
    image_edge = image_edge.reshape(-1, 2)
    caption_edge_1 = caption_edge_1.reshape(-1, 2)
    caption_edge_2 = caption_edge_2.reshape(-1, 2)
    
    image_obj = torch.LongTensor(image_obj)
    image_pred = torch.LongTensor(image_pred)
    image_edge = torch.LongTensor(image_edge)
    image_numb_obj = torch.LongTensor(image_numb_obj)
    image_numb_pred = torch.LongTensor(image_numb_pred)
    

    caption_obj_1 = torch.LongTensor(caption_obj_1)
    caption_obj_2 = torch.LongTensor(caption_obj_2)
    # caption_pred_1 = torch.LongTensor(caption_pred_1)
    # caption_pred_2 = torch.LongTensor(caption_pred_2)
    caption_edge_1 = torch.LongTensor(caption_edge_1)
    caption_edge_2 = torch.LongTensor(caption_edge_2)
    caption_numb_obj_1 = torch.LongTensor(caption_numb_obj_1)
    caption_numb_obj_2 = torch.LongTensor(caption_numb_obj_2)
    caption_numb_pred_1 = torch.LongTensor(caption_numb_pred_1)
    caption_numb_pred_2 = torch.LongTensor(caption_numb_pred_2)


    # caption_obj_1 = torch.LongTensor(caption_obj_1)
    # caption_obj_2 = torch.LongTensor(caption_obj_2)
    # # caption_pred = torch.LongTensor(caption_pred)
    # caption_edge_1 = torch.LongTensor(caption_edge_1)
    # caption_edge_2 = torch.LongTensor(caption_edge_2)
    # #caption_numb_obj = torch.LongTensor(caption_numb_obj)
    # #caption_numb_pred = torch.LongTensor(caption_numb_pred)
    # # caption_sent = torch.LongTensor(caption_pos_sent)
    
    image_obj_ft = torch.cat(image_obj_ft, dim=0) # tensor [total_obj, dim]
    image_pred_ft = torch.cat(image_pred_ft, dim=0) # tensor [total_pred, dim]
            
    assert image_edge.shape[0] == image_pred.shape[0]
    assert caption_edge_1.shape[0] == sum(caption_numb_pred_1)
    assert caption_edge_2.shape[0] == sum(caption_numb_pred_2)

    labels = labels.reshape(-1,1)
    labels = torch.FloatTensor(labels)
    #match_labels = match_labels.reshape(-1)
    #match_labels = torch.tensor(match_labels)

    return image_obj, image_obj_ft, image_pred, image_pred_ft, image_edge, image_numb_obj, image_numb_pred,\
           caption_obj_1, caption_pred_1, caption_edge_1, caption_numb_obj_1, caption_numb_pred_1, caption_len_pred_1,\
           caption_obj_2, caption_pred_2, caption_edge_2, caption_numb_obj_2, caption_numb_pred_2, caption_len_pred_2,\
           caption_sent, caption_mask, caption_len_sent, labels#, image_id, caption_id, match_labels


def get_loader(opt, batch_size=4, shuffle = True, num_workers=8, df = "val"):


    drop_last = True if (df == "train") else False

    if df == "train":

        images_data_train = joblib.load(f"{DATA_DIR}/{subset[0]}/cheapfake_train_lowered_images_data_{size}.joblib")
        caps_data_train = joblib.load(f"{DATA_DIR}/{subset[0]}/cheapfake_train_lowered_caps_data_{size}.joblib")
        df_train = pd.read_csv(f"{DATA_DIR}/{subset[0]}/label_file_train_{size}.csv")

        OBJ_FT_DIR = f'{DATA_DIR}/{subset[0]}/PENET/VisualObjectFeatures'
        PRED_FT_DIR = f'{DATA_DIR}/{subset[0]}/PENET/VisualPredFeatures'

        dset = PairGraphPrecomputeDataset(opt, images_data_train, caps_data_train, df_train, OBJ_FT_DIR, PRED_FT_DIR)
        collate_fn = pair_precompute_collate_fn
    elif df == "val":
        images_data_val = joblib.load(f"{DATA_DIR}/{subset[1]}/cheapfake_val_lowered_images_data.joblib") #_neural_motif
        caps_data_val = joblib.load(f"{DATA_DIR}/{subset[1]}/cheapfake_val_lowered_caps_data.joblib")
        df_val = pd.read_csv(f"{DATA_DIR}/{subset[1]}/label_file_val.csv") #Neural_Motif

        OBJ_FT_DIR = f'{DATA_DIR}/{subset[1]}/PENET/VisualObjectFeatures' # run extract_visual_features.py to get this
        PRED_FT_DIR = f'{DATA_DIR}/{subset[1]}/PENET/VisualPredFeatures' # run extract_visual_features.py to get this


        dset = PairGraphPrecomputeDataset(opt, images_data_val, caps_data_val, df_val, OBJ_FT_DIR, PRED_FT_DIR)


        collate_fn = pair_precompute_collate_fn
    elif df == "test":
        images_data_test = joblib.load(f"{DATA_DIR}/{subset[2]}/cheapfake_test_lowered_images_data.joblib")#_neural_motif
        caps_data = joblib.load(f"{DATA_DIR}/{subset[2]}/cheapfake_test_lowered_caps_data.joblib")
        df_test = pd.read_csv(f"{DATA_DIR}/{subset[2]}/label_file_test.csv")

        OBJ_FT_DIR = f"{DATA_DIR}/{subset[2]}/PENET/VisualObjectFeatures"#Neural_Motif
        PRED_FT_DIR = f'{DATA_DIR}/{subset[2]}/PENET/VisualPredFeatures'#Neural_Motif

        dset = PairGraphPrecomputeDataset(opt, images_data_test, caps_data, df_test, OBJ_FT_DIR, PRED_FT_DIR)
        collate_fn = pair_precompute_collate_fn
    else:
        raise ValueError("Data split not found")



    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                batch_size=batch_size,
                                                collate_fn=collate_fn,
                                                num_workers=num_workers,
                                                pin_memory=True,
                                                shuffle=shuffle,
                                                drop_last=drop_last)

    return data_loader


def get_train_loader(opt, batch_size, workers):

    train_loader = get_loader(opt, batch_size, True, workers, df = "train")
    
    return train_loader


def get_val_loader(opt, batch_size, workers):

    test_loader = get_loader(opt, batch_size, False, workers, df = "val")
    
    return test_loader

def get_test_loader(opt, batch_size, workers):

    test_loader = get_loader(opt, batch_size, False, workers, df = "test")
    
    return test_loader

if __name__ == '__main__':

    import nltk
    nltk.download('punkt')

    import argparse
    parser = argparse.ArgumentParser()
    parser = extra_parameters_imagecaption(parser)  
    # opt = parser.parse_args()
    opt, unknown = parser.parse_known_args()

    val_loader = get_val_loader(opt, 1, 1)
    print("Size of val_loader",len(val_loader))
    j = 0
    device = torch.device('cpu')
    for idx, batch in enumerate(val_loader):
        # print('len batch', len(batch))
        # print(idx, batch[19].shape, batch[20].shape, batch[21], batch[22])
        img_p_o, img_p_o_ft, img_p_p, img_p_p_ft, img_p_e, img_p_numb_o, img_p_numb_p,\
        cap_p_o_1, cap_p_p_1, cap_p_e_1, cap_p_numb_o_1, cap_p_numb_p_1, cap_p_len_p_1,\
        cap_p_o_2, cap_p_p_2, cap_p_e_2, cap_p_numb_o_2, cap_p_numb_p_2, cap_p_len_p_2,\
        cap_p_s, cap_p_m, cap_p_len_s, labels = batch
        for tensor in batch:
            print(tensor.to(device))
        break


