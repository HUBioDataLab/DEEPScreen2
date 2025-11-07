import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import itemgetter
from transformers import Swinv2Config, Swinv2ForImageClassification

class CNNModel1(nn.Module):
    def __init__(self, fully_layer_1, fully_layer_2, drop_rate):
        super(CNNModel1, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, 2)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 32, 2)
        self.bn5 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)
        self.drop_rate = drop_rate
        #self.dropout = nn.Dropout(drop_rate)
        self.fc1 = nn.Linear(#32*5*5,
                             32*8*8, fully_layer_1)
        self.fc2 = nn.Linear(fully_layer_1, fully_layer_2)
        self.fc3 = nn.Linear(fully_layer_2, 2)

    def forward(self, x):
        #print(x.shape)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        #print(x.shape)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        #print(x.shape)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        #print(x.shape)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        #print(x.shape)
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        #print(x.shape)

        x = x.view(-1, #32*5*5) # For 200x200 images
                   32*8*8)  # For 300x300 images
        
        """
        x = self.dropout(F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x))
        x = self.dropout(x)"""
        x = F.dropout(F.relu(self.fc1(x)), self.drop_rate, self.training)
        x = F.dropout(F.relu(self.fc2(x)), self.drop_rate, self.training)
        x = self.fc3(x)

        return x

# TODO: Create other models

class ViT(nn.Module):
    def __init__(self, window_size,hidden_size,att_drop,drop_path_rate,drop_rate,layer_norm_eps,encoder_stride,embed_dim,depths,mlp_ratio,num_classes = 2):
        super(ViT, self).__init__()

        configuration = Swinv2Config()
        configuration.hidden_size = hidden_size
        configuration.image_size = 300
        configuration.hidden_dropout_prob = float(drop_rate)
        configuration.window_size = int(window_size)
        configuration.attention_probs_dropout_prob  = float(att_drop)
        configuration.drop_path_rate  = float(drop_path_rate)
        configuration.layer_norm_eps = float(layer_norm_eps)
        configuration.encoder_stride = int(encoder_stride)
        configuration.embed_dim = int(embed_dim)
        configuration.depths = depths
        configuration.mlp_ratio = mlp_ratio
        
        configuration.num_labels = num_classes

        model = Swinv2ForImageClassification(configuration)
        self.vit = model
  
    def forward(self, x):
        return self.vit(x).logits
    
