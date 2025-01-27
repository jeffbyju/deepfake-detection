import os
import numpy as np
import pandas as pd
import json
import torch
import torch.nn.functional as F
import torchvision
import timm

from mtcnn import MTCNN
from torchvision import transforms
from torch import nn
    
def predict(model, frames):

    model.eval()  # Set the model to evaluation mode
    frames = torch.tensor(frames)

    with torch.no_grad():
        # Add batch dimension
        frames = frames.unsqueeze(0).float() # Shape: (1, num_frames, 3, 299, 299)
        # Get model outputs
        outputs = model(frames)  # Shape: (1,)
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(outputs)
        # Determine if it's a deepfake based on threshold
        predicted = 1-(probabilities >= 0.5).long().item()  # Convert to integer
        # Get confidence score
        confidence = probabilities.item()
        
    return predicted, confidence

def load_model():

    model = DeepFakeDetector(hidden_size=512, num_layers=2, num_classes=1, bidirectional=True, attention_bool=True)
    model.load_state_dict(torch.load('./models/model_final.pth', weights_only=True, map_location=torch.device('cpu')))
    model.eval()

    return model

class DeepFakeDetector(nn.Module):
    def __init__(self, hidden_size=512, num_layers=2, num_classes=1, bidirectional=True, attention_bool=False):
        super(DeepFakeDetector, self).__init__()
        
        # Load pre-trained XceptionNet from timm
        self.feature_extractor = timm.create_model('xception', pretrained=True, num_classes=0)  # num_classes=0 removes the classification head
        self.feature_extractor.eval()  # Set to evaluation mode
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze XceptionNet
        
        # Define LSTM
        self.lstm = nn.LSTM(input_size=2048,  # XceptionNet's output feature size
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        # Attention layer to compute attention weights over the LSTM output
        self.attention = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 1)
        self.attention_bool = attention_bool
        
        # Define fully connected layer
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, num_frames, 3, 299, 299)
        Returns:
            out: Tensor of shape (batch, 1)
        """
        batch_size, num_frames, C, H, W = x.size()
        # Merge batch and time dimensions for feature extraction
        x = x.view(batch_size * num_frames, C, H, W)  # (batch*num_frames, 3, 299, 299)
        
        with torch.no_grad():
            features = self.feature_extractor(x)  # (batch*num_frames, 2048)
        
        # Reshape back to (batch, num_frames, feature_size)
        features = features.view(batch_size, num_frames, -1)  # (batch, num_frames, 2048)
        
        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(features)  # lstm_out: (batch, num_frames, hidden_size*2)

        if self.attention_bool:
            # Attention mechanism
            attention_scores = self.attention(lstm_out)  # (batch, num_frames, 1)
            attention_weights = F.softmax(attention_scores, dim=1)  # Normalize attention scores over timesteps
            
            # Weighted sum of LSTM outputs (context vector)
            context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size*2)
            
            # Pass the context vector through the fully connected layer
            out = self.fc(context)  # (batch, num_classes)
            
            return out.squeeze()  # (batch)
        
        # Use the last timestep's output
        # Alternatively, you can use pooling over time
        if self.lstm.bidirectional:
            final_feature = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)  # (batch, hidden_size*2)
        else:
            final_feature = hn[-1,:,:]  # (batch, hidden_size)
        
        out = self.fc(final_feature)  # (batch, num_classes)
        # out = self.sigmoid(out)  # (batch, num_classes)
        
        return out.squeeze()  # (batch)