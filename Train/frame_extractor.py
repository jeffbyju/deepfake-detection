import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import face_recognition
import torch
import functools
import multiprocessing
import asyncio

from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing import image
from torch.utils.data import Dataset, DataLoader, random_split
from multiprocessing import Process
from tqdm import tqdm
from glob import glob
from mtcnn import MTCNN
from absl import flags, app

# Flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'CELEB-DF-2', 'Data Directory')

def extract_frame(video_path):
  video = cv2.VideoCapture(video_path)
  more_frames = True

  while more_frames:
    more_frames, image = video.read()
    if more_frames:
      yield image

def extract_frames(video_path, output_dir, num_frames):
  if not os.path.isfile(video_path):
    print(f'Error: video file at "{video_path}" not found')
    return

  os.makedirs(output_dir,exist_ok =True)
  cap = cv2.VideoCapture(video_path)

  if not cap.isOpened():
    print(f"Error: cannot open video file '{video_path}'")
    return

  count = 0

  batch_frames = []
  skip = 0

  output_img_idx = 0
    
  for frame in extract_frame(video_path):
    if count == num_frames:
        break
    if skip == 0:
      skip = 5
      batch_frames.append(frame)

      if len(batch_frames) == 8:
        batch_frames_faces = face_recognition.batch_face_locations(batch_frames)
        for idx,frame_faces in enumerate(batch_frames_faces):
          if len(frame_faces) > 0:
            t,r,b,l = frame_faces[0]
            if count < num_frames:
                count += 1
                frame_filename = os.path.join(output_dir,f"frame_{output_img_idx:05d}.png")
                cv2.imwrite(frame_filename,cv2.resize(batch_frames[idx][t:b,l:r,:],(299,299)))
                output_img_idx += 1
          else:
            pass
        batch_frames.clear()
    else:
        skip -= 1
   
  if len(batch_frames) > 0 and count < num_frames:
        batch_frames_faces = face_recognition.batch_face_locations(batch_frames)
        for idx,frame_faces in enumerate(batch_frames_faces):
          if len(frame_faces) > 0:
            t,r,b,l = frame_faces[0]
            if count < num_frames:
                count += 1
                frame_filename = os.path.join(output_dir,f"frame_{output_img_idx:05d}.png")
                cv2.imwrite(frame_filename,cv2.resize(batch_frames[idx][t:b,l:r,:],(299,299)))
                output_img_idx += 1
          else:
            pass
        batch_frames.clear()

def worker_extract_video_frames(range_idx,final_celeb_df):
    start = range_idx*1000
    stop = start + 1000
    print(f"started worker for range {start} - {stop}")
    for idx in tqdm(final_celeb_df.index[start:stop],desc="Extracting faces from every video"):
        row = final_celeb_df.iloc[idx]
        extract_frames(row.path,f"{os.getcwd()}/f"{FLAGS.data_dir}"/{row.file_name}",20)
        



def main():
    final_celeb_df = pd.read_csv('./final_celeb_df')

    # Option 1: Using Process directly
    p1 = Process(target=worker_extract_video_frames, args=(0,final_celeb_df,))
    p2 = Process(target=worker_extract_video_frames, args=(1,final_celeb_df,))
    p3 = Process(target=worker_extract_video_frames, args=(2,final_celeb_df,))
    p4 = Process(target=worker_extract_video_frames, args=(3,final_celeb_df,))
    p5 = Process(target=worker_extract_video_frames, args=(4,final_celeb_df,))
    p6 = Process(target=worker_extract_video_frames, args=(5,final_celeb_df,))
    p7 = Process(target=worker_extract_video_frames, args=(6,final_celeb_df,))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
	
    # Option 2: Using multiprocessing.Pool (uncomment if preferred)
    # from multiprocessing import Pool
    # with Pool() as pool:
    #     results = pool.map(worker_extract_video_frames, range(7))
    #     print(results)

if __name__ == '__main__':
    main()
