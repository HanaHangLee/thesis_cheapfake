
import torch
import torch.nn as nn
from mlp import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import joblib
from image_caption import *
from transformers import AutoModelForSequenceClassification

# DATA_DIR = '/kaggle/input/data-thesis-4/'
# DATA_DIR = '/content/drive/MyDrive/LGSGM---CheapFake/Data/'
DATA_DIR = './data-thesis-4'
subset = ['train', 'val', 'test']
size = ['12000']

def extra_parameters_encoders(parser):

    # data path
    parser.add_argument('--embed_dim', default=300, type=int, help='Embedding dim for each word')
    parser.add_argument('--numb_layers', default=2, type=int, help='Number of layers in RNN')
    parser.add_argument('--hidden_dim', default = 512, type=int, help='Hidden dim in RNN')

    parser.add_argument('--input_dim', default= 2048, type=int, help='Graph embedding initial dim')
    parser.add_argument('--visual_ft_dim', default= 2048, type=int, help='Visual feature dim')
    parser.add_argument('--fusion_output_dim', default= 1024, type=int, help='Fusion output dim')
    parser.add_argument('--include_pred_ft', default= True, type=bool, help='Include predicate feature')
    


    parser.add_argument('--gcn_input_dim', default= 1024, type=int, help='Graph embedding initial dim')
    parser.add_argument('--gcn_output_dim', default= 1024, type=int, help='Graph embedding final dim')
    parser.add_argument('--gcn_hidden_dim', default= [], type=list, help='Hidden layer in each gin layer')
    parser.add_argument('--numb_gcn_layers', default= 1, type=int, help='Number of GCN layers')
    parser.add_argument('--batchnorm', default= True, type=bool, help='Use batchnorm in GCN')
    parser.add_argument('--dropout', default= 0.5, type=float, help='Dropout rate in GCN')
    parser.add_argument('--activate_fn', default= 'swish', type=str, help='Activation function in GCN')
    parser.add_argument('--unit_dim', default= 300, type=int, help='Embedding dim for each word')
    parser.add_argument('--bidirectional', default= True, type=bool, help='Bidirectional in RNN')
    parser.add_argument('--structure', default= 'LSTM', type=str, help='RNN structure')
    parser.add_argument('--sparse', default= False, type=bool, help='Use sparse matrix')
    parser.add_argument('--use_residual', default= False, type=bool, help='Set it to false (not implemented yet)')
    parser.add_argument('--last_layer', default=False, type=bool, help='Set it to false (not implemented yet)')




    parser.add_argument('--init', default= True, type=bool, help='Set it to false (not implemented yet)')
    parser.add_argument('--ge_dim', default= 2048, type=int, help='Graph embed dim - gcn_output_dim*2')
    parser.add_argument('--ge_hidden_dim', default= [], type=list, help='Hidden layer in each gin layer')


    parser.add_argument('--init_weights_obj',
                        default=f'{DATA_DIR}/init_glove_embedding_weight_lowered_img_obj.joblib', type=str,
                        help='Initial weight for object embedding')
    parser.add_argument('--init_weights_pred',
                        default=f'{DATA_DIR}/init_glove_embedding_weight_lowered_img_pred.joblib', type=str,
                        help='Initial weight for predicate embedding')
    parser.add_argument('--init_weights_cap',
                        default=f'{DATA_DIR}/{subset[0]}/init_glove_embedding_weight_lowered_train_12000.joblib', type=str,
                        help='Initial weight for caption embedding')


    # for gpu id
    parser.add_argument('--gpu_id', default='0', type=str, help='GPU ID')

    # set device to GPU or CPU
    # parser.add_argument('--device', default='cuda:0', type=str, help='Device cuda:0 or cpu')

    parser.add_argument('--visualft_structure', default='b5', type=str, help='EfficientNet version')
    parser.add_argument('--numb_total_obj', default= 50, type=int, help='Total number of objects')
    parser.add_argument('--numb_total_pred', default= 50, type=int, help='Total number of predicates')
    return parser

# WORD EMBEDDING
class WordEmbedding(nn.Module):
    def __init__(self, opt, numb_words = 50, init_weight = None):
        '''
        numb_words: int total number of words in the dictionary
        embed_dim: int size of embedding of a word
        init_weight: torch.FloatTensor the initialized weight for the module
        spares: boolean if True, gradient w.r.t. weight matrix will be a sparse tensor
        '''
        super(WordEmbedding, self).__init__()
        self.numb_words = numb_words  
        self.embed_dim = opt.unit_dim
        self.init_weight = None
        self.model = nn.Embedding(num_embeddings=self.numb_words, embedding_dim=self.embed_dim, sparse=opt.sparse)
        if self.init_weight is not None:
            print('Initilised with given init_weight')
            self.model.weight.data.copy_(self.init_weight);
        else:
            print('Randomly Initilised')
            
    def forward(self, x):
        return self.model(x)

class RelsModel(nn.Module):
    def __init__(self, opt):
        super(RelsModel, self).__init__()
        self.device = torch.device(opt.device)
        if opt.structure == 'GRU':
            model = nn.GRU
        else:
            model = nn.LSTM
        self.model = model(input_size=opt.unit_dim, hidden_size= opt.gcn_output_dim, num_layers= opt.numb_layers, 
                           batch_first=True, dropout= opt.dropout, bidirectional=opt.bidirectional).to(self.device)
            
        self.numb_directions = 2 if opt.bidirectional else 1
        self.numb_layers = opt.numb_layers
        self.hidden_state = None
        self.hidden_dim = opt.gcn_output_dim
        self.structure = opt.structure
        
    def forward(self, x, len_original_x):
        
        batch_size, max_seq_len, input_dim = x.shape
        packed = pack_padded_sequence(x, len_original_x, batch_first=True, enforce_sorted=False)
        output, hidden = self.model(packed) # self.hidden
        
        out_unpacked, lens_out = pad_packed_sequence(output, batch_first=True) # state of each subject, pred, object
        if self.numb_directions == 2:
            out_unpacked_combine = (out_unpacked[:,:,:int(out_unpacked.shape[-1]/2)] + out_unpacked[:,:,int(out_unpacked.shape[-1]/2):])/2
        else:
            out_unpacked_combine = out_unpacked # batch, max seq len, hidden_size
        
        # Extract last hidden state
        if self.structure == 'GRU':
            final_state = hidden.view(self.numb_layers, self.numb_directions, batch_size, self.hidden_dim)[-1]
        else:
            final_state = hidden[0].view(self.numb_layers, self.numb_directions, batch_size, self.hidden_dim)[-1]
        
        # Handle directions
        if self.numb_directions == 1:
            final_hidden_state = final_state.squeeze(0)
        else:
            h_1, h_2 = final_state[0], final_state[1]
            final_hidden_state = (h_1 + h_2)/2               # Add both states (requires changes to the input size of first linear layer + attention layer)
            #final_hidden_state = torch.cat((h_1, h_2), 1)  # Concatenate both states
            
        return final_hidden_state, out_unpacked_combine

class GCN_Layer(nn.Module):
    def __init__(self, opt, gcn_pred_dim=300):
        super(GCN_Layer, self).__init__()
        perform_at_end = not opt.last_layer
        self.gcn_pred_dim = gcn_pred_dim
        '''
        node_kwargs = {
          'input_dim': input_dim,
          'hidden_dim': hidden_dim,
          'output_dim':output_dim,
          'activate_fn': activate_fn,
          'batchnorm':batchnorm,
          'dropout': dropout,
          'perform_at_end': not last_layer,
        }
        edge_kwargs = {
          'input_dim': 3*input_dim,
          'hidden_dim': hidden_dim,
          'output_dim':output_dim,
          'activate_fn': activate_fn,
          'batchnorm':batchnorm,
          'dropout': dropout,
          'perform_at_end': not last_layer,
        }
        '''
        # self.gin_node = MLP(**node_kwargs)
        # self.gin_edge = MLP(**edge_kwargs)
        self.gcn_node = MLP(opt, input_dim= opt.gcn_input_dim,hidden_dim= opt.gcn_hidden_dim, output_dim=opt.gcn_output_dim, perform_at_end=perform_at_end)
        self.gcn_edge = MLP(opt, input_dim= 2*opt.gcn_input_dim + gcn_pred_dim, hidden_dim = opt.gcn_hidden_dim, output_dim=opt.gcn_output_dim, perform_at_end=perform_at_end)
        
    def forward(self, embed_objects, embed_predicates, edges):
        '''
        embed_objects and embed_predicates are embeded by embedding layers in the previous stage
        They have size of [n_object, 300], [n_predicates, 300]
        edges is index matrix [n_predicates, 2]
        adjacency is matrix [n_object, n_object] indicate the edge index connecting between 2 nodes
        '''
        # print(f"GIN_Layer: Update Edge")
        # Break apart indices for subjects and objects; these have shape (n_predicates,)
        if embed_predicates.shape[0] > 0:
            s_idx = edges[:, 0].contiguous()
            o_idx = edges[:, 1].contiguous()

            # Get current vectors for subjects and objects; these have shape (n_predicates, 300)
            cur_s_vecs = embed_objects[s_idx]
            cur_o_vecs = embed_objects[o_idx]

            # Update predicates based on edges
            edge_input = torch.cat((cur_s_vecs, embed_predicates, cur_o_vecs), dim=1)
            new_predicates = self.gcn_edge(edge_input)
            #if torch.isnan(new_predicates).any():
                #print("embed_predicates shape:", embed_predicates.shape[0])
                #print("embed_predicates", embed_predicates)
                #print("embed_predicates is nan:", torch.isnan(embed_predicates).any())
                #print(torch.isnan(new_predicates).nonzero(as_tuple=True))
        else:
            #print("embed_predicates shape:", embed_predicates.shape[0])
            #print("embed_predicates", embed_predicates)
            #print("output dim", self.output_dim)
            new_predicates = embed_predicates
        # print(f"GIN_Layer: Done Update Edge")
        # Update nodes
        new_objects = self.gcn_node(embed_objects)
        # print(f"GIN_Layer: Done Update Node")
        return new_objects, new_predicates

class GCN_Network(nn.Module):
    def __init__(self, opt, pred_dim=300):
        super(GCN_Network, self).__init__()
        self.numb_gcn_layers = opt.numb_gcn_layers
        self.use_residual = opt.use_residual
        self.use_residual = False # Cannot run in True this time
        self.gcn_pred_dim = pred_dim
        '''
        layer_kwargs = {
          'input_dim': gin_input_dim,
          'hidden_dim': gin_hidden_dim,
          'output_dim': gin_output_dim,
          'activate_fn': activate_fn, # swish, relu, or leakyrelu
          'batchnorm': batchnorm,
          'dropout': dropout,
          # 'last_layer': False,
        }
        '''
        self.gcn_layers = torch.nn.ModuleList()
        if self.numb_gcn_layers == 1:
            self.gcn_layers.append(GCN_Layer(opt,gcn_pred_dim=self.gcn_pred_dim))
        else:
            self.gcn_layers.append(GCN_Layer(opt,gcn_pred_dim=self.gcn_pred_dim))
            
            for i in range(self.numb_gcn_layers - 2):
                 # self.gin_layers.append(GIN_Layer(last_layer=False, **layer_kwargs))
                self.gcn_layers.append(GCN_Layer(opt,gcn_pred_dim=self.gcn_pred_dim))

            self.gcn_layers.append(GCN_Layer(opt,gcn_pred_dim=self.gcn_pred_dim))
        
    def forward(self, embed_objects, embed_predicates, edges):
        '''
        embed_objects and embed_predicates are embeded by embedding layers in the previous stage
        They have size of [n_object, 300], [n_predicates, 300]
        edges is index matrix [n_predicates, 2]
        adjacency is matrix [n_object, n_object] indicate the edge index connecting between 2 nodes
        numb_objects is the list indicating number of objects in each graph (since this is for batch data) --> len(numb_objects) = batch_size
        numb_predicates is the list indicating number of predicates in each graph (since this is for batch data) --> len(numb_predicates) = batch_size
        RETURN:
        Graph embedded vectors [batchsize, graph_embed_dim]
        Embed Objects and Embed Predicates after GIN layers [total_n_object (or total_n_predicates), 300]
        '''

        graph_emb_o = []
        graph_emb_p = []
        list_emb_o = [embed_objects]
        list_emb_p = [embed_predicates]
        for idx, gcn_layer in enumerate(self.gcn_layers):
            #print(f"Processing GIN_LAYERS No {idx}")
            emb_o, emb_p = gcn_layer(list_emb_o[idx], list_emb_p[idx], edges)
            if self.use_residual and (idx+1) % 2 == 0:
                emb_o = emb_o + list_emb_o[idx-1]
                emb_p = emb_p + list_emb_p[idx-1]
            list_emb_o.append(emb_o)
            list_emb_p.append(emb_p)
        
        return list_emb_o[self.numb_gcn_layers], list_emb_p[self.numb_gcn_layers]

# numb_words = TOTAL_CAP_WORDS = len(word2idx_cap)
# TOTAL_IMG_OBJ = len(word2idx_img_obj)
# TOTAL_IMG_PRED = len(word2idx_img_pred)


class SentenceNodeModel(nn.Module):
    def __init__(self, opt):
        super(SentenceNodeModel, self).__init__()
        self.gcn_model_cap = GCN_Network(opt,pred_dim=opt.gcn_output_dim)
        numb_words = len(joblib.load(opt.word2idx_cap))
        self.embed_model_cap = WordEmbedding(opt, numb_words=numb_words, init_weight=joblib.load(opt.init_weights_cap))
        self.rels_model = RelsModel(opt)
        self.device = torch.device(opt.device)
        self.gcn_output_dim = opt.gcn_output_dim

    def forward(self, captions_predicates, captions_edges, captions_length, caption_number_objects, caption_number_edges):
        total_cap_p_numb_o = sum(caption_number_objects)
        total_cap_p_numb_p = sum(caption_number_edges)
        eb_cap_objects = torch.zeros(total_cap_p_numb_o, self.gcn_output_dim).to(self.device)
        eb_cap_edges = torch.zeros(total_cap_p_numb_p, self.gcn_output_dim).to(self.device)
        if total_cap_p_numb_p > 0:
            cap_predicates = pad_sequence(captions_predicates, batch_first=True)
            embed_cap_predicates = self.embed_model_cap(cap_predicates.to(self.device))
            rnn_eb_cap_p_rels, rnn_eb_cap_p_rels_nodes = self.rels_model(embed_cap_predicates, captions_length)
            for idx in range(len(rnn_eb_cap_p_rels_nodes)):
                edge = captions_edges[idx] # subject, object
                eb_cap_objects[edge[0]] = rnn_eb_cap_p_rels_nodes[idx,1,:] # <start> is 1st token
                eb_cap_objects[edge[1]] = rnn_eb_cap_p_rels_nodes[idx,captions_length[idx]-2 ,:] # <end> is last token
                eb_cap_edges[idx] = torch.mean(rnn_eb_cap_p_rels_nodes[idx,2:(captions_length[idx]-2),:], dim=0)
        eb_cap_objects, eb_cap_edges = self.gcn_model_cap(eb_cap_objects, eb_cap_edges, captions_edges)
        #caption_geb = self.graph_embed_model(eb_cap_objects, eb_cap_edges, caption_number_objects, caption_number_edges)
        return eb_cap_objects, eb_cap_edges

class Visual_Feature(nn.Module):
    # Just a FC convert feature extracted from EfficientNet to specific dim
    def __init__(self, opt, input_dim= 2048):
        # structure only b0 or b4
        super(Visual_Feature, self).__init__()
        self.activate_fn = opt.activate_fn
        if opt.batchnorm:
            self.bn = nn.BatchNorm1d(num_features=opt.visual_ft_dim)
        else:
            self.bn = None
        self.fc = nn.Linear(input_dim, opt.visual_ft_dim)
        if self.activate_fn.lower() == 'relu':
            self.activate = nn.ReLU()
        elif self.activate_fn.lower() == 'leakyrelu':
            self.activate = nn.LeakyReLU(0.2)
        elif self.activate_fn.lower() == 'tanh' :
            self.activate = nn.Tanh()
        else:
            self.activate = MemoryEfficientSwish()
            
    def forward(self, inputs):
        #bs = inputs.size(0)
        x = self.fc(inputs)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activate(x)
        return x

class Fusion_Layer(nn.Module):
    # Extract feature from an input images by using efficientnet
    def __init__(self, opt, activate_fn='tanh'):
        # structure only b0 or b4
        super(Fusion_Layer, self).__init__()
        self.input_dim = opt.visual_ft_dim + opt.embed_dim
        self.output_dim = opt.fusion_output_dim
        if opt.batchnorm:
            self.bn = nn.BatchNorm1d(num_features=self.output_dim)
        else:
            self.bn = None
        if opt.dropout is not None:
            self.do = nn.Dropout(opt.dropout)
        else:
            self.do = None    
        self.fc = nn.Linear(self.input_dim, self.output_dim)
        self.activate_fn = activate_fn
        if self.activate_fn.lower() == 'relu':
            self.activate = nn.ReLU()
        elif self.activate_fn.lower() == 'leakyrelu':
            self.activate = nn.LeakyReLU(0.2)
        elif self.activate_fn.lower() == 'tanh' :
            self.activate = nn.Tanh()
        else:
            self.activate = MemoryEfficientSwish()
    def forward(self, x_ft, x_emb):
        # x_ft feature extracted from visual images [N_x, ft_dim]
        # x_emb embedding feature from word2vec/glove [N_x, 300]
        x = torch.cat((x_ft, x_emb), dim=1)
        if self.do is not None:
            x = self.do(x)
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activate(x)
        return x

class ImageModel(nn.Module):
    # receive the visual images, objects, predicates, edges
    # perform word embedding --> extract visual ft --> fusion --> GCN
    def __init__(self,opt):
        
        super(ImageModel, self).__init__()
        self.include_pred_ft = opt.include_pred_ft
        # Embed by Word2Vec/Glove
        numb_words_obj = len(joblib.load(opt.word2idx_img_obj))
        self.embed_obj_model = WordEmbedding(opt, numb_words=numb_words_obj, init_weight=joblib.load(opt.init_weights_obj))
        numb_words_pred = len(joblib.load(opt.word2idx_img_pred))
        self.embed_pred_model = WordEmbedding(opt, numb_words=numb_words_pred, init_weight=joblib.load(opt.init_weights_pred))
        # Extract images features by EfficientNet
        if opt.visualft_structure == 'b0':
            effnet_dim = 1280
        if opt.visualft_structure == 'b4':
            effnet_dim = 1792
        if opt.visualft_structure == 'b5':
            effnet_dim = 2048
        
        if opt.visual_ft_dim == effnet_dim:
            self.visual_extract_obj_model = None
        else:
            self.visual_extract_obj_model = Visual_Feature(opt, input_dim=effnet_dim)
        #self.visual_extract_pred_model = Visual_Feature(input_dim=effnet_dim, 
        #                                               output_dim=visualft_feature_dim,
        #                                               batchnorm=batchnorm, 
        #                                               activate_fn=activate_fn)
        # Fusion with word embedding
        self.fusion_obj_model = Fusion_Layer(opt)
        if self.include_pred_ft:
            self.fusion_pred_model = Fusion_Layer(opt)
            pred_dim = opt.fusion_output_dim
        else:
            self.fusion_pred_model = None
            pred_dim = opt.embed_dim
        # GraphNet
        self.gcn_model = GCN_Network(opt, pred_dim=pred_dim)
        
    def forward(self, images_objects, images_predicates, list_objects, list_predicates, edges):
        # images_objects [N_obj, 3, 224, 224] images tensor
        # images_predicates [N_pred, 3, 224, 224] images tensor
        # list_objects [N_obj,] tensor
        # list_predicates [N_pred,] tensor
        # edges [N_pred, 2] tensor
        eb_objs = self.embed_obj_model(list_objects)        
        eb_pred = self.embed_pred_model(list_predicates) # embedding
        if self.visual_extract_obj_model is not None:
            images_obj_ft = self.visual_extract_obj_model(images_objects)
            if images_predicates is not None:
                images_pred_ft = self.visual_extract_obj_model(images_predicates)
        else:
            images_obj_ft = images_objects
            if images_predicates is not None:
                images_pred_ft = images_predicates
        fusion_objs= self.fusion_obj_model(images_obj_ft, eb_objs)
        if images_predicates is not None and self.fusion_pred_model is not None:
            fusion_pred= self.fusion_pred_model(images_pred_ft, eb_pred)
        else:
            fusion_pred = eb_pred
        objects, predicates = self.gcn_model(fusion_objs, fusion_pred, edges)
        #geb = self.graph_embed_model(objects, predicates, num_object, num_predicates)
        return objects, predicates

class ATT_Layer(nn.Module) :
    def __init__(self, unit_dim, init=True):
        super(ATT_Layer, self).__init__()
        self.unit_dim = unit_dim
        self.theta = nn.Parameter(torch.rand(unit_dim, unit_dim))
        self.activation = nn.ReLU(inplace=True)
        self.out_activation = nn.Sigmoid()
        if init:
            nn.init.kaiming_normal_(self.theta, mode='fan_out', nonlinearity='relu')
    def forward(self, embed_vec):
        # embed_vec [n_obj x 300]
        mean_unit = torch.mean(embed_vec, 0) # unit_dim shape
        common = self.activation(torch.matmul(self.theta, mean_unit)) # unit_dim shape
        sigmoid = self.out_activation(torch.matmul(embed_vec, common)) # n_unit shape
        new_embed = torch.mean(torch.mul(embed_vec, sigmoid.view(-1, 1)), 0) # mean of (n_unit x unit_dim shape)
        return new_embed


class GraphEmb(nn.Module):

    def __init__(self,opt, fusion_dim=None):
        '''
        Embed a graph with nodes (node_dim) and edges (edge_dim) into a vector
        '''
        super(GraphEmb, self).__init__()
        self.device = torch.device(opt.device)
        self.node_dim = opt.gcn_output_dim
        self.edge_dim = opt.gcn_output_dim
        if fusion_dim is None:
            self.fusion = None
        else:
            self.fusion = MLP(opt,input_dim=self.node_dim+self.edge_dim, hidden_dim=opt.ge_hidden_dim, output_dim=fusion_dim,\
                              perform_at_end=False)
        self.att_layers_obj = ATT_Layer(unit_dim=self.node_dim, init=True)
        self.att_layers_pred = ATT_Layer(unit_dim=self.edge_dim, init=True)
        
    def forward(self, eb_nodes, eb_edges, numb_nodes, numb_edges):
        '''
        eb_nodes [n_node, node_dims]
        eb_edges [n_edge, edge_dims]
        numb_nodes [list batch] number of nodes in each graph
        numb_edges [list batch] number of edges in each graph
        '''
        count_o = 0 # object = node
        count_p = 0 # pred = edges
        geb = torch.zeros(len(numb_nodes), self.node_dim+self.edge_dim).to(self.device)
        for idx_batch in range(len(numb_nodes)):
            numb_obj = numb_nodes[idx_batch]
            if numb_obj > 0:
                cur_e_o = eb_nodes[count_o:(count_o+numb_obj)]
            else:
                cur_e_o = torch.zeros((1,self.node_dim)).to(self.device)
            count_o += numb_obj
            graph_emb_o_l = self.att_layers_obj(cur_e_o).view(1,-1) # convert to [1, dim]

            numb_pred = numb_edges[idx_batch]
            if numb_pred > 0:
                cur_e_p = eb_edges[count_p:(count_p+numb_pred)]
            else:
                cur_e_p = torch.zeros((1,self.edge_dim)).to(self.device)
            count_p += numb_pred
            graph_emb_p_l = self.att_layers_pred(cur_e_p).view(1,-1) # convert to [1, dim]
            geb[idx_batch] = torch.cat((graph_emb_o_l, graph_emb_p_l), dim=1)
        if self.fusion is not None:
            geb = self.fusion(geb)
        return geb

# Sentence Model (RNN)
class NLIModel(nn.Module):
    def __init__(self,opt):
        super(NLIModel, self).__init__()
        self.device = torch.device(opt.device)
        model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli")#AutoModel
        for param in model.parameters():
            param.requires_grad_(False)
        self.deberta = model.deberta.to(self.device)
        self.pooler = model.pooler.to(self.device)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
        #outputs = self.model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
        pooled_output = self.pooler(outputs.last_hidden_state)
        #pooled_output = outputs.last_hidden_state[:, 0] #.pooler_output
        # return a tensor of size 768
        return pooled_output





class ImageEncoder(nn.Module):
    def __init__(self, opt):
        super(ImageEncoder, self).__init__()
        self.model = ImageModel(opt)
        self.geb = GraphEmb(opt, fusion_dim=opt.ge_dim)
    def forward(self, images_objects, images_predicates, img_p_o, img_p_p, edges, img_p_numb_o, img_p_numb_p):
        objects, predicates = self.model(images_objects, images_predicates, img_p_o, img_p_p, edges)
        return self.geb(objects, predicates, img_p_numb_o, img_p_numb_p)

class SentenceEncoder(nn.Module):
    def __init__(self, opt):
        super(SentenceEncoder, self).__init__()
        self.model = SentenceNodeModel(opt)
        self.geb = GraphEmb(opt, fusion_dim=opt.ge_dim)
    def forward(self, cap_p_p_1, cap_p_e_1, cap_p_len_p_1, cap_p_numb_o_1, cap_p_numb_p_1):
        eb_cap_objects, eb_cap_edges = self.model(cap_p_p_1, cap_p_e_1, cap_p_len_p_1, cap_p_numb_o_1, cap_p_numb_p_1)
        return self.geb(eb_cap_objects, eb_cap_edges, cap_p_numb_o_1, cap_p_numb_p_1)




if __name__ == '__main__':

    # Test the model
    import argparse
    parser = argparse.ArgumentParser()
    parser = extra_parameters_encoders(parser)

    #Add extra parameters image caption
    parser = extra_parameters_imagecaption(parser)
    opt, unknow = parser.parse_known_args()
    print(opt)


    # Using val_loader to get the data
    val_loader = get_val_loader(opt, 1, 1)
    
    for idx, batch in enumerate(val_loader):
        batch = [item.to(opt.device) if isinstance(item, torch.Tensor)
                else [sub_item.to(opt.device) if isinstance(sub_item, torch.Tensor) else sub_item
                    for sub_item in item] if isinstance(item, list)
                else item for item in batch]

        # for tensor in batch:
        #     print('\n-------------- ', type(tensor),' --------------\n')
        #     if(type(tensor) == list):
        #         print(len(tensor))
        #     else:
        #         print(tensor.shape)
        #         print(tensor.size())
        #     print(tensor)

        img_p_o, img_p_o_ft, img_p_p, img_p_p_ft, img_p_e, img_p_numb_o, img_p_numb_p,\
        cap_p_o_1, cap_p_p_1, cap_p_e_1, cap_p_numb_o_1, cap_p_numb_p_1, cap_p_len_p_1,\
        cap_p_o_2, cap_p_p_2, cap_p_e_2, cap_p_numb_o_2, cap_p_numb_p_2, cap_p_len_p_2,\
        cap_p_s, cap_p_m, cap_p_len_s, labels = batch

        model_img = ImageModel(opt).to(opt.device)
        objects, predicates = model_img(img_p_o_ft, img_p_p_ft, img_p_o, img_p_p, img_p_e)
        print(objects.shape)
        print(predicates.shape)

        print('1')



        ImgModel = ImageEncoder(opt).to(opt.device)
        img_geb = ImgModel(img_p_o_ft, img_p_p_ft, img_p_o, img_p_p, img_p_e, img_p_numb_o, img_p_numb_p)
        print(img_geb.shape)


        print('2')

        model = SentenceNodeModel(opt).to(opt.device)
        eb_cap_objects, eb_cap_edges = model(cap_p_p_1, cap_p_e_1, cap_p_len_p_1, cap_p_numb_o_1, cap_p_numb_p_1)
        print(eb_cap_objects.shape)
        print(eb_cap_edges.shape)

        # Test GraphEmb
        model_geb = GraphEmb(opt, fusion_dim=opt.ge_dim).to(opt.device)
        img_geb = model_geb(objects, predicates, img_p_numb_o, img_p_numb_p)
        print(img_geb.shape)

        print('3')

        sentence_geb = model_geb(eb_cap_objects, eb_cap_edges, cap_p_numb_o_1, cap_p_numb_p_1)
        print(sentence_geb.shape)

        print('4')

        SenModel = SentenceEncoder(opt).to(opt.device)
        sentence_geb = SenModel(cap_p_p_1, cap_p_e_1, cap_p_len_p_1, cap_p_numb_o_1, cap_p_numb_p_1)
        print(sentence_geb.shape)

        print('5')

        # Test NLIModel
        model_nli = NLIModel(opt).to(opt.device)
        input_ids = cap_p_s
        attention_mask = cap_p_m
        nli_ft = model_nli(input_ids, attention_mask)
        print(nli_ft.shape)

        break

    pass