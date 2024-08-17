import torch
import torch.nn as nn

model_savepath = "pretrained/model-doom-%d.pth"

# Just some helper function for determining network shape
def conv(tensor, f, k, s, p=0):
    c_, x_, y_ = tensor
    x_ += 2*p; y_ += 2*p
    i = 0; j = 0
    
    for x in range(0, x_, s):
        x_end = x + k
        if x_end > x_:
            break
        i += 1
        
    for y in range(0, y_, s):
        y_end = y + k
        if y_end > y_:
            break
        j += 1
    
    return (f, i, j)

def prod(iter_):
    result = 1
    for x in iter_:
        result *= x
    return result

class DQNv1(nn.Module):
    '''A DQN model with this structure:
    (   3 channels) -> conv(3x3 kernel, stride 1, 64 channels) -(dropout)-> avg_pool(2x2 kernel, stride 2) -> ReLU
    (  64 channels) -> conv(3x3 kernel, stride 1, 32 channels) -(dropout)-> avg_pool(2x2 kernel, stride 2) -> ReLU
    (  32 channels) -> conv(3x3 kernel, stride 1, 16 channels) -(dropout)-> ReLU -> Flatten(6272)
    (6272 in nodes) -(dropout)-> linear(6272, 1568) -> ReLU -> linear(1568, action_num) -> ReLU -> action values
    '''
    def __init__(self, action_num: int, dropout: float=0, ch_num: int=3):
        super().__init__()

        x = (ch_num, 144, 256)
        x = conv(x, 64, 3, 1)
        x = conv(x, 64, 2, 2)
        x = conv(x, 32, 3, 1)
        x = conv(x, 32, 2, 2)
        x = conv(x, 16, 3, 1)
        y = prod(x)
        z = y >> 3

        self.linear_input_size = y

        if dropout == 0:
            self.conv_layer_0 = nn.Sequential(
                nn.Conv2d(ch_num, 64, kernel_size=3, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.ReLU()
            )
            
            self.conv_layer_1 = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.ReLU()
            )
            
            self.conv_layer_2 = nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=3, stride=1, bias=False),
                nn.ReLU()
            )
            
            self.hidden_layers = nn.Sequential(
                nn.Linear(y, z, bias=True),
                nn.ReLU(),
                nn.Linear(z, action_num, bias=True),
                nn.ReLU()
            )
            
        else:
            self.conv_layer_0 = nn.Sequential(
                nn.Conv2d(ch_num, 64, kernel_size=3, stride=1, bias=False),
                nn.Dropout(p=dropout),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.ReLU()
            )
            
            self.conv_layer_1 = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, stride=1, bias=False),
                nn.Dropout(p=dropout),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.ReLU()
            )
            
            self.conv_layer_2 = nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=3, stride=1, bias=False),
                nn.Dropout(p=dropout),
                nn.ReLU()
            )
            
            self.hidden_layers = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(y, z, bias=True),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(z, action_num, bias=True),
                nn.ReLU()
            )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_layer_0(x)
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = x.view(-1, self.linear_input_size)
        return self.hidden_layers(x)

class DRQNv1(nn.Module):
    '''Legacy code
    '''
    def __init__(self, action_num: int, feature_num: int):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, bias=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=True)
        )
        
        self.decision_net = nn.Sequential(
            nn.LSTM(input_size=4608, hidden_size=action_num, bias=True)
        )
        
        self.feature_net = nn.Sequential(
            nn.Linear(4608, 512, bias=True),
            nn.Sigmoid(),
            nn.Linear(512, feature_num, bias=True),
            nn.Sigmoid()
        )
    
    def inf_feature(self, x: torch.Tensor):
        x = self.conv_layers(x)
        x = x.view(-1, 4608)
        self.x_copy = torch.clone(x)
        return self.feature_net(x)
    
    def inf_action(self):
        actions = self.decision_net(self.x_copy)[0]
        self.x_copy = None
        return actions
    
    def forward(self, x: torch.Tensor):
        feature = self.inf_feature(x)
        actions = self.inf_action()
        return (feature, actions)
        
class DRQNv2(DQNv1):
    '''A DQN model with this structure:
    (   3 channels) -> conv(3x3 kernel, stride 1, 64 channels) -(dropout)-> avg_pool(2x2 kernel, stride 2) -> ReLU
    (  64 channels) -> conv(3x3 kernel, stride 1, 32 channels) -(dropout)-> avg_pool(2x2 kernel, stride 2) -> ReLU
    (  32 channels) -> conv(3x3 kernel, stride 1, 16 channels) -(dropout)-> ReLU -> Flatten(6272)
    (6272 in nodes) -(clone)-> [branch 0], [branch 1]
    [branch 0] -(dropout)-> linear(6272, 1568) -> Sigmoid -> linear(1568, feature_num) -> Sigmoid -> feature prediction
    [branch 1] -> LSTM(hidden_size=action_num, 2 layers, dropout) -> action values
    '''
    def __init__(self, action_num: int, feature_num: int, dropout: float=0, ch_num: int=3):
        super().__init__(action_num, dropout, ch_num)

        y = self.linear_input_size
        z = y >> 3
        
        self.decision_net = nn.Sequential(
            nn.LSTM(input_size=y, hidden_size=action_num, num_layers = 2, bias=True, dropout=dropout)
        )

        x = (ch_num, 144, 256)
        x = conv(x, 64, 3, 1)
        x = conv(x, 64, 2, 2)
        x = conv(x, 32, 3, 1)
        x = conv(x, 32, 2, 2)
        x = conv(x, 16, 3, 1)
        self.x_copy = torch.zeros(size=x).cuda()
        
        if dropout == 0:
            self.feature_net = nn.Sequential(
                nn.Linear(y, z, bias=True),
                nn.Sigmoid(),
                nn.Linear(z, feature_num, bias=True),
                nn.Sigmoid()
            )
            
        else:
            self.feature_net = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(y, z, bias=True),
                nn.Dropout(p=dropout),
                nn.Sigmoid(),
                nn.Linear(z, feature_num, bias=True),
                nn.Sigmoid()
            )
    
    def inf_feature(self, x: torch.Tensor):
        x = self.conv_layer_0(x)
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = x.view(-1, self.linear_input_size)
        self.x_copy = torch.clone(x)
        return self.feature_net(x)
    
    def inf_action(self):
        actions = self.decision_net(self.x_copy)[0]
        #self.x_copy = None
        return actions
    
    def forward(self, x: torch.Tensor):
        feature = self.inf_feature(x)
        actions = self.inf_action()
        return (feature, actions)