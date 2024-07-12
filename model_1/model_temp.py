import argparse
from encoders import *
import torch.optim as optim
# from model_1.loss import *
from metrics import *
from torch.nn import BCELoss

def extra_parameters_model(parser):
    parser.add_argument('--optimizer_choice', type=str, default='Adam', help = 'Optimizer choice')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=2)
    # set self.device to GPU or CPU
    # parser.add_argument('--self.device', default='cuda:0', type=str, help='Device cuda:0 or cpu')
    parser.add_argument('--freeze', type=bool, default=False, help='Freeze most of the component')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load')

    return parser 




class model_1:
    def __init__(self, opt, eval = False):
        self.device = torch.device(opt.device)

        self.opt = opt
        
        self.grad_clip = opt.grad_clip

        self.image_geb = ImageEncoder(opt) # n_img, dim
        self.caption_geb1 = SentenceEncoder(opt) # n_cap, dim 
        self.caption_geb2 = SentenceEncoder(opt) # n_cap, dim
        self.nli = NLIModel(opt) # n_cap, dim



        self.mlp_ = MLP(opt,input_dim=opt.gcn_output_dim*6, 
                   hidden_dim=[2048, 1024],#[2048, 1024], 
                   output_dim=768,  #768
                   #activate_fn='relu',
                   perform_at_end=False)
        self.mlp = MLP(opt,input_dim=768*2, #768*2
                   hidden_dim=[192, 48], #192, 48
                   output_dim=1,  
                   #activate_fn='relu',
                   perform_at_end=False)
        # self.caption_geb_2 = SentenceEncoder(opt) # n_cap, dim 

        if not eval:
            self.criterion = BCELoss()
            if opt.freeze: # freeze most of component
                for p in self.image_geb.parameters():
                    p.requires_grad = False
                for p in self.caption_geb1.parameters():
                    p.requires_grad = False
                for p in self.caption_geb2.parameters():
                    p.requires_grad = False
                for p in self.nli.parameters():
                    p.requires_grad = False
                for p in self.mlp_.parameters():
                    p.requires_grad = False
                last_layer = self.model.MLP.numb_layers - 1
                for name, param in self.model.MLP.named_parameters():
                    if name not in [f'linear.{last_layer}.weight', f'linear.{last_layer}.bias']:
                        param.requires_grad = False
            self.model = [self.image_geb, self.caption_geb1, self.caption_geb2, self.nli, self.mlp_, self.mlp]
            ## PARAMS & OPTIMIZER
            self.params = []
            for component in self.model:
                self.params += list(filter(lambda p: p.requires_grad, component.parameters()))

        

            if opt.optimizer_choice.lower() == 'adam':                                                     
                self.optimizer = optim.Adam(self.params,
                                            lr=opt.lr,
                                            betas=(0.9, 0.999),
                                            eps=1e-08,
                                            weight_decay=0)
                                        
            if opt.optimizer_choice.lower() == 'sgd':
                self.optimizer = optim.SGD(self.params,
                                            lr=opt.lr,
                                            momentum=0.9,
                                            weight_decay=0)

            if torch.cuda.is_available():
                self.image_geb.to(self.device)
                self.caption_geb1.to(self.device)
                self.caption_geb2.to(self.device)
                self.nli.to(self.device)
                self.mlp_.to(self.device)
                self.mlp.to(self.device)
                self.criterion.to(self.device)

                torch.backends.cudnn.benchmark = True
        else:
            self.criterion = BCELoss()

        self.Eiters = 0


    def state_dict(self):
        state_dict = [
            self.image_geb.state_dict(),
            self.caption_geb1.state_dict(),
            self.caption_geb2.state_dict(),
            self.nli.state_dict(),
            self.mlp_.state_dict(),
            self.mlp.state_dict() 
            ]
        return state_dict

    def load_state_dict(self, state_dict, ):
        # strict=True, ensure keys match
        self.image_geb.load_state_dict(state_dict[0], strict=True)
        
        # Unexpected key(s) in state_dict: "bert.embeddings.position_ids". 
        # incompatible problem of transformers package version 
        self.caption_geb1.load_state_dict(state_dict[1], strict=False)
        self.caption_geb2.load_state_dict(state_dict[2], strict=False)
        self.nli.load_state_dict(state_dict[3], strict=True)
        self.mlp_.load_state_dict(state_dict[4], strict=True)
        self.mlp.load_state_dict(state_dict[5], strict=True)
        


    def train_start(self):
        self.image_geb.train()
        self.caption_geb1.train()
        self.caption_geb2.train()
        self.nli.train()
        self.mlp_.train()
        self.mlp.train()
        self.criterion.train()

    def val_start(self):
        self.image_geb.eval()
        self.caption_geb1.eval()
        self.caption_geb2.eval()
        self.nli.eval()
        self.mlp_.eval()
        self.mlp.eval()
        self.criterion.eval()

    def load_trained_model(self):
        #---- Load checkpoint 
        if self.checkpoint is not None:
            print(f"LOAD PRETRAINED MODEL AT {self.checkpoint}")
            modelCheckpoint = torch.load(self.checkpoint)
            self.model.load_state_dict(modelCheckpoint['model_state_dict'])
            if not self.freeze:
                self.optimizer.load_state_dict(modelCheckpoint['optimizer_state_dict'])
        else:
            print("TRAIN FROM SCRATCH")

    def forward(self, batch):
        batch = [item.to(opt.device) if isinstance(item, torch.Tensor)
                else [sub_item.to(opt.device) if isinstance(sub_item, torch.Tensor) else sub_item
                      for sub_item in item] if isinstance(item, list)
                else item for item in batch]
        img_p_o, img_p_o_ft, img_p_p, img_p_p_ft, img_p_e, img_p_numb_o, img_p_numb_p,\
        cap_p_o_1, cap_p_p_1, cap_p_e_1, cap_p_numb_o_1, cap_p_numb_p_1, cap_p_len_p_1,\
        cap_p_o_2, cap_p_p_2, cap_p_e_2, cap_p_numb_o_2, cap_p_numb_p_2, cap_p_len_p_2,\
        cap_p_s, cap_p_m, cap_p_len_s, labels = batch

        # img_p_o_ft = img_p_o_ft.to(self.device)
        # img_p_p_ft = img_p_p_ft.to(self.device)
        # img_p_o = img_p_o.to(self.device)
        # img_p_p = img_p_p.to(self.device) 
        # img_p_e = img_p_e.to(self.device)
        # # img_p_numb_o = img_p_numb_o.to(self.device)
        # # img_p_numb_p = img_p_numb_p.to(self.device)

        # cap_p_o_1 = cap_p_o_1.to(self.device)
        # # cap_p_p_1 = cap_p_p_1.to(self.device)
        # cap_p_e_1 = cap_p_e_1.to(self.device)
        # # cap_p_numb_o_1 = cap_p_numb_o_1.to(self.device)
        # # cap_p_numb_p_1 = cap_p_numb_p_1.to(self.device)
        # # cap_p_len_p_1 = cap_p_len_p_1.to(self.device)

        # cap_p_o_2 = cap_p_o_2.to(self.device)
        # # cap_p_p_2 = cap_p_p_2.to(self.device)
        # cap_p_e_2 = cap_p_e_2.to(self.device)
        # # cap_p_numb_o_2 = cap_p_numb_o_2.to(self.device)
        # # cap_p_numb_p_2 = cap_p_numb_p_2.to(self.device)
        # # cap_p_len_p_2 = cap_p_len_p_2.to(self.device)

        # cap_p_s = cap_p_s.to(self.device)
        # cap_p_m = cap_p_m.to(self.device)
        # # cap_p_len_s = cap_p_len_s.to(self.device)
        # labels = labels.to(self.device)

        img_emb = self.image_geb(img_p_o_ft, img_p_p_ft, img_p_o, img_p_p, img_p_e, img_p_numb_o, img_p_numb_p)
        cap_emb1 = self.caption_geb1(cap_p_p_1, cap_p_e_1, cap_p_len_p_1, cap_p_numb_o_1, cap_p_numb_p_1)
        cap_emb2 = self.caption_geb2(cap_p_p_2, cap_p_e_2, cap_p_len_p_2, cap_p_numb_o_2, cap_p_numb_p_2)

        x = torch.cat([img_emb,cap_emb1,cap_emb2], dim=1)
        image_text_ft = self.mlp_(x)

        nli_feature = self.nli(cap_p_s, cap_p_m)


        feature = torch.cat([image_text_ft, nli_feature], dim=1)
        output = self.mlp(feature)
        output = nn.Sigmoid()(output)

        predicted_labels = np.array([])
        true_labels = np.array([])

        predicted_labels = np.concatenate((predicted_labels, output.cpu().numpy().flatten()))#.item
        true_labels = np.concatenate((true_labels, labels.cpu().numpy().flatten()))#.item

        Val_loss = self.criterion(output, labels)

        
        return Val_loss, predicted_labels, true_labels
    


    def train(self, batch):
        self.Eiters += 1
        batch = [item.to(self.opt.device) if isinstance(item, torch.Tensor)
                else [sub_item.to(self.opt.device) if isinstance(sub_item, torch.Tensor) else sub_item
                      for sub_item in item] if isinstance(item, list)
                else item for item in batch]
        img_p_o, img_p_o_ft, img_p_p, img_p_p_ft, img_p_e, img_p_numb_o, img_p_numb_p,\
        cap_p_o_1, cap_p_p_1, cap_p_e_1, cap_p_numb_o_1, cap_p_numb_p_1, cap_p_len_p_1,\
        cap_p_o_2, cap_p_p_2, cap_p_e_2, cap_p_numb_o_2, cap_p_numb_p_2, cap_p_len_p_2,\
        cap_p_s, cap_p_m, cap_p_len_s, labels = batch

        # img_p_o_ft = img_p_o_ft.to(self.device)
        # img_p_p_ft = img_p_p_ft.to(self.device)
        # img_p_o = img_p_o.to(self.device)
        # img_p_p = img_p_p.to(self.device) 
        # img_p_e = img_p_e.to(self.device)
        # # img_p_numb_o = img_p_numb_o.to(self.device)
        # # img_p_numb_p = img_p_numb_p.to(self.device)

        # cap_p_o_1 = cap_p_o_1.to(self.device)
        # # cap_p_p_1 = cap_p_p_1.to(self.device)
        # cap_p_e_1 = cap_p_e_1.to(self.device)
        # # cap_p_numb_o_1 = cap_p_numb_o_1.to(self.device)
        # # cap_p_numb_p_1 = cap_p_numb_p_1.to(self.device)
        # # cap_p_len_p_1 = cap_p_len_p_1.to(self.device)

        # cap_p_o_2 = cap_p_o_2.to(self.device)
        # # cap_p_p_2 = cap_p_p_2.to(self.device)
        # cap_p_e_2 = cap_p_e_2.to(self.device)
        # # cap_p_numb_o_2 = cap_p_numb_o_2.to(self.device)
        # # cap_p_numb_p_2 = cap_p_numb_p_2.to(self.device)
        # # cap_p_len_p_2 = cap_p_len_p_2.to(self.device)

        # cap_p_s = cap_p_s.to(self.device)
        # cap_p_m = cap_p_m.to(self.device)
        # # cap_p_len_s = cap_p_len_s.to(self.device)
        # labels = labels.to(self.device)

        img_emb = self.image_geb(img_p_o_ft, img_p_p_ft, img_p_o, img_p_p, img_p_e, img_p_numb_o, img_p_numb_p)
        cap_emb1 = self.caption_geb1(cap_p_p_1, cap_p_e_1, cap_p_len_p_1, cap_p_numb_o_1, cap_p_numb_p_1)
        cap_emb2 = self.caption_geb2(cap_p_p_2, cap_p_e_2, cap_p_len_p_2, cap_p_numb_o_2, cap_p_numb_p_2)

        x = torch.cat([img_emb,cap_emb1,cap_emb2], dim=1)
        image_text_ft = self.mlp_(x)

        nli_feature = self.nli(cap_p_s, cap_p_m)


        feature = torch.cat([image_text_ft, nli_feature], dim=1)
        output = self.mlp(feature)
        output = nn.Sigmoid()(output)


        # print("output: ", output)
        # print("labels: ", labels)

        predicted_labels = np.array([])
        true_labels = np.array([])

        # predicted_labels = np.concatenate((predicted_labels, output.cpu().numpy().flatten()))#.item
        predicted_labels = np.concatenate((predicted_labels, output.detach().cpu().numpy().flatten()))

        true_labels = np.concatenate((true_labels, labels.detach().cpu().numpy().flatten()))#.item

        self.optimizer.zero_grad()

        loss = self.criterion(output, labels)

        loss.backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)


        self.optimizer.step()



        return loss, predicted_labels, true_labels
    
if __name__ == '__main__':

    pass






    def predict(self, data):
        return self.model
    
