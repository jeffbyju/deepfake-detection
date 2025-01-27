import cv2
import torch
import os
import face_recognition
import glob

from torchvision import transforms

def extract_frames(video_path, num_frames=40):

    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    return frames

def preprocess_frames(frames, video_dir, num_frames, output_dir="./static/uploads"):

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    processed_frames = []
    batch_frames = []
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(f"{output_dir}/{video_dir.split('/')[-1].split('.mp4')[0]}_frames"):
        frame_files = sorted(glob.glob(os.path.join(f"{output_dir}/{video_dir.split('/')[-1].split('.mp4')[0]}_frames", "frame_*.png"))[:num_frames])
        if frame_files:
            for frame_path in frame_files:
                # Read existing frame
                frame = cv2.imread(frame_path)
                frame_tensor = torch.tensor(frame).permute(2, 0, 1).float()
                frame_tensor = transform(frame_tensor)
                processed_frames.append(frame_tensor)
            
            return torch.stack(processed_frames)
    
    else:
        os.makedirs(f"{output_dir}/{video_dir.split('/')[-1].split('.mp4')[0]}_frames", exist_ok=True)

    for frame_idx, frame in enumerate(frames):
        batch_frames.append(frame)
        
        if len(batch_frames) == 8:
            batch_frames_faces = face_recognition.batch_face_locations(batch_frames)
            
            for idx, frame_faces in enumerate(batch_frames_faces):
                if len(frame_faces) > 0:
                    t, r, b, l = frame_faces[0]
                    face_frame = cv2.resize(batch_frames[idx][t:b, l:r, :], (299, 299))
                    
                    # Save processed frame if output directory is specified
                    if f"{output_dir}/{video_dir.split('/')[-1].split('.mp4')[0]}_frames":
                        frame_filename = os.path.join(f"{output_dir}/{video_dir.split('/')[-1].split('.mp4')[0]}_frames", f"frame_{frame_idx + idx:05d}.png")
                        cv2.imwrite(frame_filename, face_frame)
                    
                    frame_tensor = torch.tensor(face_frame).permute(2, 0, 1).float()
                    frame_tensor = transform(frame_tensor)
                    processed_frames.append(frame_tensor)
                else:
                    resized_frame = cv2.resize(batch_frames[idx], (299, 299))
                    if f"{output_dir}/{video_dir.split('/')[-1].split('.mp4')[0]}_frames":
                        frame_filename = os.path.join(f"{output_dir}/{video_dir.split('/')[-1].split('.mp4')[0]}_frames", f"frame_{frame_idx + idx:05d}.png")
                        cv2.imwrite(frame_filename, resized_frame)
                    
                    frame_tensor = torch.tensor(resized_frame).permute(2, 0, 1).float()
                    frame_tensor = transform(frame_tensor)
                    processed_frames.append(frame_tensor)
            
            batch_frames.clear()
    
    # Process remaining frames
    if batch_frames:
        batch_frames_faces = face_recognition.batch_face_locations(batch_frames)
        for idx, frame_faces in enumerate(batch_frames_faces):
            current_idx = len(frames) - len(batch_frames) + idx
            if len(frame_faces) > 0:
                t, r, b, l = frame_faces[0]
                face_frame = cv2.resize(batch_frames[idx][t:b, l:r, :], (299, 299))
                
                if f"{output_dir}/{video_dir.split('/')[-1].split('.mp4')[0]}_frames":
                    frame_filename = os.path.join(f"{output_dir}/{video_dir.split('/')[-1].split('.mp4')[0]}_frames", f"frame_{current_idx:05d}.png")
                    cv2.imwrite(frame_filename, face_frame)
                
                frame_tensor = torch.tensor(face_frame).permute(2, 0, 1).float()
                frame_tensor = transform(frame_tensor)
                processed_frames.append(frame_tensor)
            else:
                resized_frame = cv2.resize(batch_frames[idx], (299, 299))
                if f"{output_dir}/{video_dir.split('/')[-1].split('.mp4')[0]}_frames":
                    frame_filename = os.path.join(f"{output_dir}/{video_dir.split('/')[-1].split('.mp4')[0]}_frames", f"frame_{current_idx:05d}.png")
                    cv2.imwrite(frame_filename, resized_frame)
                
                frame_tensor = torch.tensor(resized_frame).permute(2, 0, 1).float()
                frame_tensor = transform(frame_tensor)
                processed_frames.append(frame_tensor)
    
    return torch.stack(processed_frames)