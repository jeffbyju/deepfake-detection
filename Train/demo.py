import os
import numpy as np
import pandas as pd
import json
from mtcnn import MTCNN
import torch
import torch.nn.functional as F
import functools
import asyncio
import torchvision
import timm
import lmdb
import pickle
import warnings
import io
import psutil
import lz4.frame

from tqdm import tqdm
from glob import glob
from absl import flags, app
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms,datasets
from torchvision.ops import sigmoid_focal_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torch import nn
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tensorboardX import SummaryWriter

# Flags
FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', 'run1', 'Output Directory')
flags.DEFINE_boolean('bi_lstm_bool', False, 'Use Bi-LSTM or Normal LSTM')
flags.DEFINE_boolean('attention_bool', False, 'Use Attention or not')
flags.DEFINE_integer('seq_length', 20, 'Video sequence length')
flags.DEFINE_string('csv_file', './final_celeb_df2', 'CSV of video files')
flags.DEFINE_string('base_dir', 'CELEB-DF-2', 'Folder of video frames')

# Create a custom Dataset class
class VidFramesDataset(Dataset):
    def __init__(self, sequence_len, final_celeb_df, transform=None, cache_dir='./dataset_cache2', initial_map_size_gb=100):
        self.sequence_len = sequence_len
        self.transform = transform
        self.base_dir = f"{os.getcwd()}/{FLAGS.base_dir}"
        self.video_frame_dirs = [d for d in os.listdir(self.base_dir) if not d.startswith('.')]
        self.final_celeb_df = final_celeb_df
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize LMDB environment with an initial smaller map_size
        self.env = lmdb.open(
            os.path.join(cache_dir, 'frames_cache'),
            map_size=initial_map_size_gb * 1024 * 1024 * 1024 * 100
        )
        self.frame_paths = self._preload_frame_paths()

        self.labels = {video_dir: self.final_celeb_df[self.final_celeb_df.file_name == video_dir].real.values[0] for video_dir in self.video_frame_dirs if not self.final_celeb_df[self.final_celeb_df.file_name == video_dir].empty}
        if not os.path.exists(os.path.join(cache_dir, 'cached.txt')):
            self._cache_frames()
            with open(os.path.join(cache_dir, 'cached.txt'), 'w') as f:
                f.write('cached')

    def _preload_frame_paths(self):
        paths = {}
        for video_dir in self.video_frame_dirs:
            video_path = os.path.join(self.base_dir, video_dir)
            frame_paths = sorted(glob(os.path.join(video_path, 'frame_*.png')))[:self.sequence_len]
            paths[video_dir] = frame_paths
        return paths

    def _load_and_process_frame(self, frame_path):
        try:
            frame = torchvision.io.read_image(frame_path).float()
            if self.transform:
                frame = self.transform(frame)
            return frame
        except Exception as e:
            warnings.warn(f"Error loading frame {frame_path}: {str(e)}")
            return torch.zeros((3, 299, 299))

    def _cache_frames(self):
        print("Caching frames to LMDB with compression...")
        chunk_size = 100

        def process_video(video_dir):
            frames = torch.zeros((self.sequence_len, 3, 299, 299))
            frame_paths = self.frame_paths[video_dir]
            for i, frame_path in enumerate(frame_paths[:self.sequence_len]):
                frames[i] = self._load_and_process_frame(frame_path)
            # Delete frame files after processing
            video_path = os.path.join(self.base_dir, video_dir)
            for frame_file in frame_paths:
                os.remove(frame_file)  # Delete the frame file
            print(f"Deleted frames in: {video_path}")
            return video_dir, frames

        for i in tqdm(range(0, len(self.video_frame_dirs), chunk_size)):
            chunk = self.video_frame_dirs[i:i + chunk_size]
            with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
                futures = [executor.submit(process_video, video_dir) for video_dir in chunk]
                with self.env.begin(write=True) as txn:
                    for future in futures:
                        result = future.result()
                        if result:
                            video_dir, frames = result
                            txn.put(video_dir.encode(), lz4.frame.compress(pickle.dumps(frames)))

            # Dynamically increase map size if needed
            if i % 500 == 0:
                self.env.set_mapsize(self.env.info()['map_size'] + 1024 * 1024 * 1024 * 20)  # Increase by 20GB
                
    def __len__(self):
        return len(self.video_frame_dirs)
        
    def __getitem__(self, idx):
        video_dir = self.video_frame_dirs[idx]
        with self.env.begin(write=False) as txn:
            frames_data = txn.get(video_dir.encode())
            if frames_data:
                video_frames = pickle.loads(lz4.frame.decompress(frames_data))[:self.sequence_len]
            else:
                frame_paths = self.frame_paths[video_dir]
                video_frames = torch.stack([self._load_and_process_frame(f) for f in frame_paths[:self.sequence_len]])
        return self.labels.get(video_dir, -1), video_frames


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


def main(_):
    print("-----------------------------------START---------------------------------------")
    writer = SummaryWriter(FLAGS.output_dir, max_queue=1000, flush_secs=120)
    
    final_celeb_df = pd.read_csv(FLAGS.csv_file)

    ratios = final_celeb_df['real'].value_counts(normalize=True)
    real, fake = ratios[1], ratios[0]
    
    transform = transforms.Compose([
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

    # Create a Dataset object
    dataset = VidFramesDataset(FLAGS.seq_length,final_celeb_df=final_celeb_df,transform=transform)

    labels = []
    indices = []
    
    for idx in tqdm(range(len(dataset))):
        label, frames = dataset[idx]
        if label != -1:
            labels.append(dataset[idx][0])
            indices.append(idx)
    
    train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(
        indices,
        labels,
        test_size=0.2,
        stratify = labels
    )

    train_indices, val_indices, _,_ = train_test_split(
        train_val_indices,
        train_val_labels,
        test_size=0.25,
        stratify=train_val_labels
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    batch_size = 8
    num_workers = 0

    prefetch_factor = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers,pin_memory=True,prefetch_factor=prefetch_factor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers,pin_memory=True,prefetch_factor=prefetch_factor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers,pin_memory=True,prefetch_factor=prefetch_factor)

    
    # Check if MPS is available
    # torch.mps.set_per_process_memory_fraction(0.0)
    # device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    device = torch.device("cuda:0")
    print(f'Using device: {device}')

    # Initialize the model
    model = DeepFakeDetector(hidden_size=512, num_layers=2, num_classes=1, bidirectional=FLAGS.bi_lstm_bool, attention_bool=FLAGS.attention_bool)

    # Move the model to the device
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("initialized the criteria and optimizer")
    
    num_epochs = 20  # Adjust as needed
    if num_workers > 0:
        torch.set_num_threads(num_workers)

    print("set the number of workers")

    iter = 0
    
    for epoch in range(num_epochs):
        print(f"starting epoch #{epoch}")
        model.train()
        running_loss = 0.0
        for batch_idx, (labels,videos) in enumerate(train_loader):
            videos = videos.to(device, non_blocking=True)  # (batch, num_frames, 3, 299, 299)
            labels = labels.float().to(device, non_blocking=True)  # (batch,)

            optimizer.zero_grad()
            outputs = model(videos)  # (batch,)
            # loss = criterion(outputs, labels)
            loss = sigmoid_focal_loss(
                outputs, 
                labels,
                alpha=fake,
                gamma=2.0,
                reduction='sum'
            )
            loss.backward()
            optimizer.step()
            print("Batch Loss: ", loss.item())
            running_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

            # Tensorboard logging
            writer.add_scalar('batch_item_loss', loss.item(), iter+1)
            iter += 1

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}')

        # Tensorboard logging
        writer.add_scalar('total_epoch_loss', epoch_loss, epoch+1)
        
        # Save model
        if (epoch+1) % 2 == 0:
            if not os.path.exists(FLAGS.output_dir):
                os.makedirs(FLAGS.output_dir)
            torch.save(model.state_dict(), f'{FLAGS.output_dir}/model_{epoch+1}.pth')

        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for labels,videos in val_loader:
                videos = videos.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(videos)
                # predicted = (outputs >= 0.5).long()
                predicted = (torch.sigmoid(outputs) >= 0.5).long()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_accuracy = correct / total
            print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

            # Tensorboard logging
            writer.add_scalar('epoch_val_accuracy', val_accuracy, epoch+1)
            iter += 1

        
            
    torch.save(model.state_dict(), f'{FLAGS.output_dir}/model_final.pth')
    
    # Save prediction result on test set
    result_file_name = f'{FLAGS.output_dir}/results_test.json'
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        results = []
        for labels, videos in test_loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(videos)
            # predicted = (outputs >= 0.5).long()
            predicted = (torch.sigmoid(outputs) >= 0.5).long()
            results.append(predicted.tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = correct / total
        print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
        json.dump(results, open(result_file_name, 'w'))

if __name__ == "__main__":
    app.run(main)