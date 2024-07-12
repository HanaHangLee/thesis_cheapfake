
import torch
import torch.nn as nn

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

# MLP with batchnorm and dropout
class MLP(nn.Module):
    def __init__(self,opt, input_dim=300,hidden_dim = [], output_dim = 300, perform_at_end=True):
        '''
        dropout is None or an float [0,1] --> normally 0.5
        activate_fn = 'swish' - 'relu' - 'leakyrelu'
        perform_at_end = True --> apply batchnorm, relu, dropout at the last layer (output)
        hidden_dim is a list indicating unit in hidden layers --> e.g. [1024, 512, 256]
        '''
        super(MLP, self).__init__()
        if opt.activate_fn.lower() == 'relu':
            self.activate = nn.ReLU()
        elif opt.activate_fn.lower() == 'leakyrelu':
            self.activate = nn.LeakyReLU(0.2)
        elif opt.activate_fn.lower() == 'tanh' :
            self.activate = nn.Tanh()
        else:
            self.activate = MemoryEfficientSwish()
        
        self.use_residual = opt.use_residual
        self.perform_at_end = perform_at_end
        self.hidden_dim = hidden_dim + [output_dim]
        
        self.numb_layers = len(self.hidden_dim)
        
        # print(f"Hidden dim: {self.hidden_dim}")
        
        self.linear = torch.nn.ModuleList()
        for idx, numb in enumerate(self.hidden_dim):
            if idx == 0:
                self.linear.append(nn.Linear(input_dim, numb))
            else:
                self.linear.append(nn.Linear(self.hidden_dim[idx-1], numb))
                
        self.batchnorm = torch.nn.ModuleList()
        self.dropout = torch.nn.ModuleList()   
        
        for idx in range(self.numb_layers-1):
            if opt.batchnorm:
                self.batchnorm.append(nn.BatchNorm1d(num_features=self.hidden_dim[idx]))
            else:
                self.batchnorm.append(nn.Identity())              
            if opt.dropout is not None:
                self.dropout.append(nn.Dropout(opt.dropout))
            else:
                self.dropout.append(nn.Identity())
                
        if perform_at_end:
            if opt.batchnorm:
                self.batchnorm.append(nn.BatchNorm1d(num_features=output_dim))
            else:
                self.batchnorm.append(nn.Identity())     
            if opt.dropout is not None:
                self.dropout.append(nn.Dropout(opt.dropout))
            else:
                self.dropout.append(nn.Identity())
        # print(f"Summary MLP: No Linear: {len(self.linear)} --- No BN: {len(self.batchnorm)} --- No DO: {len(self.dropout)}")
              
    def forward(self, x):
        # x should has format of [1, dim]
        # print(f'MLP Network input: {x.shape} --- numb_layer: {self.numb_layers}')
        for i in range(self.numb_layers-1):
            if self.use_residual:
                x = self.linear[i](x) + x
            else:
                x = self.linear[i](x)
            # print(x.shape)
            x = self.batchnorm[i](x)
            # print(x.shape)
            x = self.activate(x)
            # print(x.shape)
            x = self.dropout[i](x)
            # print(x.shape)
            # print('--------')
        if self.perform_at_end:
            # print(f"Perfrom at end --- {x.shape}")
            if self.use_residual:
                x = self.linear[self.numb_layers-1](x) + x
            else:
                x = self.linear[self.numb_layers-1](x)
            # print(x.shape)
            x = self.batchnorm[self.numb_layers-1](x)
            # print(x.shape)
            x = self.activate(x)
            # print(x.shape)
            x = self.dropout[self.numb_layers-1](x)
            # print(x.shape)
        else:
            x = self.linear[self.numb_layers-1](x)
            
        return x