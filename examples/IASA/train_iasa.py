import torch, os, imageio, argparse, yaml
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict, load_state_dict_from_folder
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
import random
import pickle
from io import BytesIO
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFilter
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import os
import wandb
import lightning.pytorch.loggers as pl_loggers
import cv2
from scipy.spatial import ConvexHull
import torch.multiprocessing as mp
import lpips
mp.set_start_method('spawn', force=True) 
import matplotlib.pyplot as plt
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TextVideoDataset_train(torch.utils.data.Dataset):
    def __init__(self, base_dir, max_num_frames=81, frame_interval=2, num_frames=81, height=480, width=832,attention_type="point", is_i2v=False,steps_per_epoch=1):
        # metadata = pd.read_csv(metadata_path)
        # self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        # self.text = metadata["text"].to_list()
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.steps_per_epoch = steps_per_epoch
        self.attention_type = attention_type
        # data_list = ['UBC_Fashion', 'self_collected_videos_pose', 'TikTok']

        self.sample_fps = frame_interval
        self.max_frames = max_num_frames
        self.misc_size = [height, width]
        self.video_list = []
        self.pose_dir=base_dir
        # Traverse each video in self.pose_dir and write into self.video_list
        for video_name in os.listdir(self.pose_dir):
            video_path = os.path.join(self.pose_dir, video_name)
            if os.path.isfile(video_path) or os.path.isdir(video_path):
                self.video_list.append(video_path)
        print("!!! dataset length: ", len(self.video_list))        


        random.shuffle(self.video_list)
            
        self.frame_process = v2.Compose([
            # v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def resize(self, image):
        width, height = image.size
        # 
        image = torchvision.transforms.functional.resize(
            image,
            (self.height, self.width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        # return torch.from_numpy(np.array(image))
        return image
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame

    def transform_keypoints(self, kpts, x1, y1, crop_w, crop_h, resize_w, resize_h):
        """
        Transform keypoints from normalized [0,1] to cropped+resized pixel coords.
        Points outside the crop region or specifically masked (e.g., idx 9,10,12,13 for body) are replaced with (0.0, 0.0).

        Args:
            kpts: input keypoints, formats:
                - {"candidate": ndarray (N, 2)} for body
                - ndarray (1, 68, 2) for face
                - ndarray (2, 21, 2) for hands
            x1, y1: crop top-left corner in original image
            crop_w, crop_h: crop size in original image
            resize_w, resize_h: size after resizing

        Returns:
            np.ndarray of shape (N, 2), resized keypoints (invalid points → [0.0, 0.0])
        """
        if kpts is None or len(kpts) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # ---- Load keypoints ----
        if isinstance(kpts, dict) and "candidate" in kpts:
            xy = kpts["candidate"] * [resize_w, resize_h]
            mask_indices = {8,11,9, 10, 12, 13}
        elif isinstance(kpts, np.ndarray) and kpts.ndim == 3 and kpts.shape[1:] == (68, 2):  # face
            xy = kpts[0] * [resize_w, resize_h]
            mask_indices = set()
        elif isinstance(kpts, np.ndarray) and kpts.ndim == 3 and kpts.shape[1:] == (21, 2):  # hands
            xy = np.concatenate([kpts[0], kpts[1]], axis=0) * [resize_w, resize_h]
            mask_indices = set()
        else:
            raise ValueError(f"Unsupported keypoint format: {type(kpts)}, shape: {getattr(kpts, 'shape', None)}")

        # ---- Crop + Resize ----
        transformed = []
        for idx, (x, y) in enumerate(xy):
            if idx in mask_indices:
                transformed.append([0.0, 0.0])
                continue

            # check if (x, y) falls into the crop box
            if not (x1 <= x < x1 + crop_w and y1 <= y < y1 + crop_h):
                transformed.append([0.0, 0.0])
                continue

            # shift to crop-local and resize
            x_crop = x - x1
            y_crop = y - y1

            x_resized = x_crop / crop_w * resize_w
            y_resized = y_crop / crop_h * resize_h

            transformed.append([x_resized, y_resized])

        return np.array(transformed, dtype=np.float32)
    def transform_pose_dict(self, pose, x1, y1, crop_w, crop_h, resize_w, resize_h):
        """
        Apply transform to all body/hands/face keypoints in a pose dict.
        """
        return {
            "bodies": self.transform_keypoints(pose.get("bodies", []), x1, y1, crop_w, crop_h, resize_w, resize_h),
            "hands": self.transform_keypoints(pose.get("hands", []), x1, y1, crop_w, crop_h, resize_w, resize_h),
            "faces": self.transform_keypoints(pose.get("faces", []), x1, y1, crop_w, crop_h, resize_w, resize_h),
            "hands_score": pose.get("hands_score", 0.0),
            "faces_score": pose.get("faces_score", 0.0)
        }
    def collect_keypoints(self, pose_dict, max_points=188):
        keypoints = []

        # 1. Bodies with interpolation
        kpts_body = pose_dict.get("bodies", [])
        if isinstance(kpts_body, (list, np.ndarray)):
            kpts_body = torch.tensor(kpts_body, dtype=torch.float32)
        if isinstance(kpts_body, torch.Tensor) and kpts_body.numel() > 0:
            connections = [(2, 3), (3, 4), (5, 6), (6, 7),(1, 2), (1, 5)]
            interpolated = []
            for i, j in connections:
                p1, p2 = kpts_body[i], kpts_body[j]
                if (p1 == 0).all() or (p2 == 0).all():
                    interpolated.append(torch.zeros(10, 2))
                else:
                    pts = [(1 - alpha) * p1 + alpha * p2 for alpha in torch.linspace(0, 1, 12)[1:-1]]
                    interpolated.append(torch.stack(pts))
            if interpolated:
                kpts_body = torch.cat([kpts_body] + interpolated, dim=0)
            keypoints.append(kpts_body)

        # 2. Hands
        kpts_hands = pose_dict.get("hands", [])
        if isinstance(kpts_hands, (list, np.ndarray)):
            kpts_hands = torch.tensor(kpts_hands, dtype=torch.float32)
        if isinstance(kpts_hands, torch.Tensor) and kpts_hands.numel() > 0:
            keypoints.append(kpts_hands)

        # 3. Faces
        kpts_face = pose_dict.get("faces", [])
        if isinstance(kpts_face, (list, np.ndarray)):
            kpts_face = torch.tensor(kpts_face, dtype=torch.float32)
        if isinstance(kpts_face, torch.Tensor) and kpts_face.numel() > 0:
            keypoints.append(kpts_face)

        # Concatenate all and clip to max length
        all_kpts = torch.cat(keypoints, dim=0)
        if all_kpts.shape[0] > max_points:
            all_kpts = all_kpts[:max_points]
        return all_kpts
    
    def extract_region_masks_from_all_kpts(self, all_kpts, image_size):
        """
        Args:
            all_kpts: Tensor of shape [188, 2], containing pixel coordinates (x, y)
            image_size: Tuple (H, W), height and width of the image

        Returns:
            face_mask, lhand_mask, rhand_mask, arm_shoulder_mask: 
            Each is a np.uint8 binary mask of shape (H, W) with values 0 or 255
        """
        H, W = image_size
        body_kpts = all_kpts[0:78]
        lhand_kpts = all_kpts[78:99]
        rhand_kpts = all_kpts[99:120]
        face_kpts = all_kpts[120:188]

        def poly_to_mask(poly):
            """Converts a polygon (in x,y) into a binary mask"""
            mask = np.zeros((H, W), dtype=np.uint8)
            if poly is not None and len(poly) >= 3:
                poly_int = np.array(poly).astype(np.int32)
                cv2.fillPoly(mask, [poly_int], 255)
            return mask

        def valid_poly(kpts):
            """
            Get a convex polygon from valid keypoints.
            Returns polygon in (x, y) format.
            """
            valid = (kpts > 0).all(dim=1)
            if valid.sum() < 3:
                return None
            kpts = kpts[valid]
            try:
                hull = ConvexHull(kpts)
                poly = kpts[hull.vertices]
                return poly.numpy()  # (x, y)
            except:
                return None

        def shoulder_arm_poly(kpts):
            """
            Construct a polygon from shoulder and arm keypoints.
            Returns polygon in (x, y) format.
            """
            indices = [5, 6, 7, 4, 3, 2]
            points = []
            for idx in indices:
                if (kpts[idx] > 0).all():
                    x, y = kpts[idx]
                    points.append([x.item(), y.item()])
            if len(points) >= 3:
                return np.array(points, dtype=np.float32)
            return None

        face_mask = poly_to_mask(valid_poly(face_kpts))
        lhand_mask = poly_to_mask(valid_poly(lhand_kpts))
        rhand_mask = poly_to_mask(valid_poly(rhand_kpts))
        arm_mask = poly_to_mask(shoulder_arm_poly(body_kpts))
        return face_mask, lhand_mask, rhand_mask, arm_mask

    def build_region_blocks_and_layout(
        self,valid_mask: torch.Tensor,  # [latent_t * latent_h * latent_w]
        latent_t: int, latent_h: int, latent_w: int, block_size: int
    ):
        """
        Build strictly separated region/nonregion blocks. Each region block contains only region tokens; each nonregion block contains only nonregion tokens.
        Args:
            valid_mask: Global token mask (True for region, False for nonregion)
            latent_t: Number of frames
            latent_h, latent_w: Spatial dimensions
            block_size: Number of tokens per block

        Returns:
            perm: [new_total_token,] Original token indices for valid tokens, -1 for padding locations
            pad_mask: [new_total_token,] True for valid tokens, False for padding
            block_layout: [n_block, n_block] Block sparse attention connectivity matrix
            region_block_ptrs, nonregion_block_ptrs: Block index ranges for each frame
        """
        n_token_per_frame = latent_h * latent_w
        region_indices_per_frame = []
        nonregion_indices_per_frame = []
        n_region_block = []
        n_nonregion_block = []

        # 1. Collect global indices of region and nonregion tokens for each frame
        for t in range(latent_t):
            start = t * n_token_per_frame
            end = (t + 1) * n_token_per_frame
            mask_flat = valid_mask[start:end]
            region_idx = torch.where(mask_flat)[0] + start
            nonregion_idx = torch.where(~mask_flat)[0] + start
            region_indices_per_frame.append(region_idx)
            nonregion_indices_per_frame.append(nonregion_idx)
            n_region_block.append((len(region_idx) + block_size - 1) // block_size)
            n_nonregion_block.append((len(nonregion_idx) + block_size - 1) // block_size)

        n_block = sum(n_region_block) + sum(n_nonregion_block)
        new_total_token = n_block * block_size

        # 2. Build perm and pad_mask, fill each block with -1 as padding if necessary
        perm = []
        pad_mask = []
        region_block_ptrs, nonregion_block_ptrs = [], []
        ptr = 0

        # First, fill region blocks for all frames
        for t in range(latent_t):
            idx = region_indices_per_frame[t].tolist()
            n_full = n_region_block[t]
            for b in range(n_full):
                block = idx[b*block_size : (b+1)*block_size]
                pad = block_size - len(block)
                perm += block + [-1] * pad
                pad_mask += [True] * len(block) + [False] * pad
            region_block_ptrs.append((ptr, ptr + n_full))
            ptr += n_full

        # Then, fill nonregion blocks for all frames
        for t in range(latent_t):
            idx = nonregion_indices_per_frame[t].tolist()
            n_full = n_nonregion_block[t]
            for b in range(n_full):
                block = idx[b*block_size : (b+1)*block_size]
                pad = block_size - len(block)
                perm += block + [-1] * pad
                pad_mask += [True] * len(block) + [False] * pad
            nonregion_block_ptrs.append((ptr, ptr + n_full))
            ptr += n_full

        perm = torch.tensor(perm, dtype=torch.long)
        pad_mask = torch.tensor(pad_mask, dtype=torch.bool)

        # 3. Build block_layout matrix
        block_layout = torch.zeros((n_block, n_block), dtype=torch.bool)
        # For each frame: fully connect all blocks within the same frame
        for t in range(latent_t):
            f_start = region_block_ptrs[t][0]
            f_end = nonregion_block_ptrs[t][1]
            block_layout[f_start:f_end, f_start:f_end] = 1
        # Connect all region blocks across frames (region-to-region cross-frame attention)
        for i in range(latent_t):
            for j in range(latent_t):
                if i == j: continue
                si, ei = region_block_ptrs[i]
                sj, ej = region_block_ptrs[j]
                block_layout[si:ei, sj:ej] = 1

        return perm, pad_mask, block_layout, region_block_ptrs, nonregion_block_ptrs

    # def __getitem__(self, data_id):
    def __getitem__(self, index):
        index = index % len(self.video_list)
        success=False
        for _try in range(5):
            try:
                if _try >0:
                    
                    index = random.randint(1,len(self.video_list))
                    index = index % len(self.video_list)
                
                clean = True
                frames_path = self.video_list[index]
                frame_mp4_path = frames_path
                dwpose_dir = frame_mp4_path.replace("videos", "dwpose")
                dwpose_pose_dir = str(Path(frame_mp4_path.replace("videos", "dwpose_data")).with_suffix(".pkl"))

                dwpose_path = dwpose_dir
                from decord import VideoReader
                # Open video captures once

                cap = VideoReader(frames_path)
                dwpose_cap = VideoReader(dwpose_path)

                _total_frame_num = len(cap)

                with open(dwpose_pose_dir, "rb") as f:
                    dwpose_data_dict = pickle.load(f)
                dwpose_poses = dwpose_data_dict["poses"]
                
                # Calculate frame sampling parameters
                stride = random.randint(1, self.sample_fps)
                cover_frame_num = (stride * self.max_frames)
                max_frames = self.max_frames

                if _total_frame_num < cover_frame_num + 1:
                    raise IndexError(f"Invalid sample: total_frame_num={_total_frame_num} < cover_frame_num+1={cover_frame_num+1}")
                    # start_frame = 0
                    # end_frame = _total_frame_num - 1
                    # stride = max((_total_frame_num // max_frames), 1)
                    # end_frame = min(stride * max_frames, _total_frame_num - 1)
                else:
                    start_frame = random.randint(0, _total_frame_num - cover_frame_num - 5)
                    end_frame = start_frame + cover_frame_num

                frame_list = []
                dwpose_list = []
                dwpose_pose_list = []
                # Get random reference frame
                random_ref = random.randint(0, _total_frame_num - 1)
                frame = cap[random_ref].asnumpy()
                random_ref_frame = Image.fromarray(frame)

                if random_ref_frame.mode != 'RGB':
                    random_ref_frame = random_ref_frame.convert('RGB')

                # Get random reference dwpose frame
                dwpose_frame = dwpose_cap[random_ref].asnumpy()
                random_ref_dwpose = Image.fromarray(dwpose_frame)




                # Read frames with stride
                for i_index in range(start_frame, end_frame, stride):
                    # Read frame
                    frame = cap[i_index].asnumpy()
                    i_frame = Image.fromarray(frame)

                    if i_frame.mode != 'RGB':
                        i_frame = i_frame.convert('RGB')

                    # Read dwpose frame
                    dwpose_frame = dwpose_cap[i_index].asnumpy()
                    i_dwpose = Image.fromarray(dwpose_frame)

                    i_dwpose_data = dwpose_poses[i_index]

                    frame_list.append(i_frame)
                    dwpose_list.append(i_dwpose)
                    dwpose_pose_list.append(i_dwpose_data)

  
                have_frames = len(frame_list)>0
                middle_indix = 0

                if have_frames:

                    l_hight = random_ref_frame.size[1]
                    l_width = random_ref_frame.size[0]

                    # random crop

                    x1 = random.randint(0, l_width//14)
                    x2 = random.randint(0, l_width//14)
                    y1 = random.randint(0, l_hight//14)
                    y2 = random.randint(0, l_hight//14)
                    
                    
                    random_ref_frame = random_ref_frame.crop((x1, y1,l_width-x2, l_hight-y2))
                    ref_frame = random_ref_frame 
                    # 
                    
                    random_ref_frame_tmp = torch.from_numpy(np.array(self.resize(random_ref_frame)))
                    random_ref_dwpose_tmp = torch.from_numpy(np.array(self.resize(random_ref_dwpose.crop((x1, y1,l_width-x2, l_hight-y2))))) # [3, 512, 320]
                    dwpose_pose_data = []
                    video_data_tmp = torch.stack([self.frame_process(self.resize(ss.crop((x1, y1,l_width-x2, l_hight-y2)))) for ss in frame_list], dim=0) # self.transforms(frames)
                    dwpose_data_tmp = torch.stack([torch.from_numpy(np.array(self.resize(ss.crop((x1, y1,l_width-x2, l_hight-y2))))).permute(2,0,1) for ss in dwpose_list], dim=0)


                    for dwpose_pose in dwpose_pose_list:
                        transformed_pose = self.transform_pose_dict(
                            dwpose_pose, x1, y1,
                            crop_w=l_width - x1 - x2,
                            crop_h=l_hight - y1 - y2,
                            resize_w=self.width,  # width
                            resize_h=self.height,  # height
                        )     

                        dwpose_pose_data.append(transformed_pose)

                video_data = torch.zeros(self.max_frames, 3, self.misc_size[0], self.misc_size[1])
                dwpose_data = torch.zeros(self.max_frames, 3, self.misc_size[0], self.misc_size[1])
                
                if have_frames:
                    video_data[:len(frame_list), ...] = video_data_tmp      
                    
                    dwpose_data[:len(frame_list), ...] = dwpose_data_tmp

                video_data = video_data.permute(1,0,2,3)
                dwpose_data = dwpose_data.permute(1,0,2,3)
                
                caption = "a person is dancing"
                break
            except Exception as e:
                # 
                caption = "a person is dancing"
                # 
                video_data = torch.zeros(3, self.max_frames, self.misc_size[0], self.misc_size[1])  
                random_ref_frame_tmp = torch.zeros(self.misc_size[0], self.misc_size[1], 3)
                vit_image = torch.zeros(3,self.misc_size[0], self.misc_size[1])
                
                dwpose_data = torch.zeros(3, self.max_frames, self.misc_size[0], self.misc_size[1])  
                # 
                random_ref_dwpose_data = torch.zeros(3, self.max_frames, self.misc_size[0], self.misc_size[1])  
                print('{} read video frame failed with error: {}'.format(frames_path, e))
                continue


        text = caption 

        if self.attention_type == "point":
            attention_region = torch.zeros(self.max_frames,188, 3)
            T = 21 * 32 * 32 
            valid_mask = torch.zeros(T, dtype=torch.bool)

            for t in range(self.max_frames):
                pose_dict = dwpose_pose_data[t]
                all_kpts = self.collect_keypoints(pose_dict, 188)

                for j in range(all_kpts.shape[0]):
                    x, y = all_kpts[j]
                    if x > 0 and y > 0:
                        latent_t = (t + 3) // 4
                        latent_x = int(x) // 16
                        latent_y = int(y) // 16
                        attention_region[t, j] = torch.tensor([latent_t, latent_x, latent_y])
                        if 0 <= latent_t < 21 and 0 < latent_x < 32 and 0 < latent_y < 32:
                            token_idx = latent_t * 32 * 32 + latent_y * 32 + latent_x
                            valid_mask[token_idx] = True
        if self.attention_type == "Original":
            attention_region = torch.zeros(self.max_frames,188, 3)
            T = 21 * 32 * 32 
            valid_mask = torch.zeros(T, dtype=torch.bool)
            
        if self.attention_type == "bfground" or self.attention_type == "orgionalregion":
            attention_region = torch.zeros(self.max_frames,188, 3)
            T = 21 * 32 * 32 
            valid_mask = torch.zeros(T, dtype=torch.bool)
            for t in range(self.max_frames):
                pose_dict = dwpose_pose_data[t]
                all_kpts = self.collect_keypoints(pose_dict, 188)#hand,arm,face,hand
                face_mask, lhand_mask, rhand_mask, arm_mask = self.extract_region_masks_from_all_kpts(all_kpts, (self.height, self.width))
                combined_mask = face_mask | lhand_mask | rhand_mask | arm_mask  # union of all
                #cv2.imwrite("region.png", combined_mask)
                #write the code that if the mask has the value 255, then set the valid_mask to True which is x,y
                ys, xs = torch.nonzero(torch.tensor(combined_mask) == 255, as_tuple=True)
                for y, x in zip(ys.tolist(), xs.tolist()):
                    latent_x = x // 16
                    latent_y = y // 16
                    latent_t = (t + 3) // 4  
                    if 0 <= latent_t < 21 and 0 <= latent_x < 32 and 0 <= latent_y < 32:
                        token_idx = latent_t * 32 * 32 + latent_y * 32 + latent_x
                        valid_mask[token_idx] = True
                    attention_region=None
        if self.attention_type == "region" or self.attention_type == "flash_region":
            attention_region = torch.zeros(self.max_frames,188, 3)
            T = 21 * 32 * 32 
            valid_token = torch.zeros(T, dtype=torch.bool)
            for t in range(self.max_frames):
                pose_dict = dwpose_pose_data[t]
                all_kpts = self.collect_keypoints(pose_dict, 188)#hand,arm,face,hand
                face_mask, lhand_mask, rhand_mask, arm_mask = self.extract_region_masks_from_all_kpts(all_kpts, (self.height, self.width))
                combined_mask = face_mask | lhand_mask | rhand_mask | arm_mask  # union of all
                #cv2.imwrite("region.png", combined_mask)
                #write the code that if the mask has the value 255, then set the valid_mask to True which is x,y
                ys, xs = torch.nonzero(torch.tensor(combined_mask) == 255, as_tuple=True)
                for y, x in zip(ys.tolist(), xs.tolist()):
                    latent_x = x // 16
                    latent_y = y // 16
                    latent_t = (t + 3) // 4  
                    if 0 <= latent_t < 21 and 0 <= latent_x < 32 and 0 <= latent_y < 32:
                        token_idx = latent_t * 32 * 32 + latent_y * 32 + latent_x
                        valid_token[token_idx] = True
            frame_id = torch.arange(21).repeat_interleave(32*32)
            # frame_id.shape == [21*1024] == [21504]
            valid_mask={
                "valid_mask": valid_token,
                "frame_id": frame_id,
            }
        if self.attention_type == "multiregion":
            attention_region = torch.zeros(self.max_frames,188, 3)
            T = 21 * 32 * 32
            valid_token = torch.zeros(T, dtype=torch.long)
            for t in range(len(dwpose_pose_data)):
                pose_dict = dwpose_pose_data[t]
                all_kpts = self.collect_keypoints(pose_dict, 188)
                face_mask, lhand_mask, rhand_mask, arm_mask = self.extract_region_masks_from_all_kpts(all_kpts, (self.height, self.width))
                hand_mask = lhand_mask | rhand_mask
                for region_mask, region_label in [(arm_mask, 3),(face_mask, 1), (hand_mask, 2)]:
                    ys, xs = torch.nonzero(torch.tensor(region_mask) == 255, as_tuple=True)
                    for y, x in zip(ys.tolist(), xs.tolist()):
                        latent_x = x // 16
                        latent_y = y // 16
                        latent_t = (t + 3) // 4
                        if 0 <= latent_t < 21 and 0 <= latent_x < 32 and 0 <= latent_y < 32:
                            token_idx = latent_t * 32 * 32 + latent_y * 32 + latent_x
                            valid_token[token_idx] = region_label
            frame_id = torch.arange(21).repeat_interleave(32*32)
            # frame_id.shape == [21*1024] == [21504]
            valid_mask={
                "valid_mask": valid_token,
                "frame_id": frame_id,
            }
            
        if self.is_i2v:
            video, first_frame = video_data, random_ref_frame_tmp
            data = {"text": text, "video": video,"dwpose_data": dwpose_data,
                    "dwpose_pose_data":dwpose_pose_data, "valid_token_ids":valid_mask,
                    "video_path": frames_path, "dwpose_path": dwpose_path, 
                    "first_frame": first_frame, "random_ref_dwpose_data": random_ref_dwpose_tmp}
        else:
            data = {"text": text, "video": video, "path": frames_path}
        return data
    
    def __len__(self):
        
        return len(self.video_list)
 

class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, max_num_frames=81, frame_interval=2, num_frames=81, height=480, width=832,attention_type="point", is_i2v=False,steps_per_epoch=1):
        # metadata = pd.read_csv(metadata_path)
        # self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        # self.text = metadata["text"].to_list()
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.steps_per_epoch = steps_per_epoch
        self.attention_type = attention_type
        # data_list = ['UBC_Fashion', 'self_collected_videos_pose', 'TikTok']

        self.sample_fps = frame_interval
        self.max_frames = max_num_frames
        self.misc_size = [height, width]
        self.video_list = []
        self.pose_dir=base_dir
        # Traverse each video in self.pose_dir and write into self.video_list
        for video_name in os.listdir(self.pose_dir):
            video_path = os.path.join(self.pose_dir, video_name)
            if os.path.isfile(video_path) or os.path.isdir(video_path):
                self.video_list.append(video_path)
        print("!!! dataset length: ", len(self.video_list))        


        random.shuffle(self.video_list)
            
        self.frame_process = v2.Compose([
            # v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def resize(self, image):
        width, height = image.size
        # 
        image = torchvision.transforms.functional.resize(
            image,
            (self.height, self.width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        # return torch.from_numpy(np.array(image))
        return image
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame

    def transform_keypoints(self, kpts, x1, y1, crop_w, crop_h, resize_w, resize_h):
        """
        Transform keypoints from normalized [0,1] to cropped+resized pixel coords.
        Points outside the crop region or specifically masked (e.g., idx 9,10,12,13 for body) are replaced with (0.0, 0.0).

        Args:
            kpts: input keypoints, formats:
                - {"candidate": ndarray (N, 2)} for body
                - ndarray (1, 68, 2) for face
                - ndarray (2, 21, 2) for hands
            x1, y1: crop top-left corner in original image
            crop_w, crop_h: crop size in original image
            resize_w, resize_h: size after resizing

        Returns:
            np.ndarray of shape (N, 2), resized keypoints (invalid points → [0.0, 0.0])
        """
        if kpts is None or len(kpts) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # ---- Load keypoints ----
        if isinstance(kpts, dict) and "candidate" in kpts:
            xy = kpts["candidate"] * [resize_w, resize_h]
            mask_indices = {8,11,9, 10, 12, 13}
        elif isinstance(kpts, np.ndarray) and kpts.ndim == 3 and kpts.shape[1:] == (68, 2):  # face
            xy = kpts[0] * [resize_w, resize_h]
            mask_indices = set()
        elif isinstance(kpts, np.ndarray) and kpts.ndim == 3 and kpts.shape[1:] == (21, 2):  # hands
            xy = np.concatenate([kpts[0], kpts[1]], axis=0) * [resize_w, resize_h]
            mask_indices = set()
        else:
            raise ValueError(f"Unsupported keypoint format: {type(kpts)}, shape: {getattr(kpts, 'shape', None)}")

        # ---- Crop + Resize ----
        transformed = []
        for idx, (x, y) in enumerate(xy):
            if idx in mask_indices:
                transformed.append([0.0, 0.0])
                continue

            # check if (x, y) falls into the crop box
            if not (x1 <= x < x1 + crop_w and y1 <= y < y1 + crop_h):
                transformed.append([0.0, 0.0])
                continue

            # shift to crop-local and resize
            x_crop = x - x1
            y_crop = y - y1

            x_resized = x_crop / crop_w * resize_w
            y_resized = y_crop / crop_h * resize_h

            transformed.append([x_resized, y_resized])

        return np.array(transformed, dtype=np.float32)
    def transform_pose_dict(self, pose, x1, y1, crop_w, crop_h, resize_w, resize_h):
        """
        Apply transform to all body/hands/face keypoints in a pose dict.
        """
        return {
            "bodies": self.transform_keypoints(pose.get("bodies", []), x1, y1, crop_w, crop_h, resize_w, resize_h),
            "hands": self.transform_keypoints(pose.get("hands", []), x1, y1, crop_w, crop_h, resize_w, resize_h),
            "faces": self.transform_keypoints(pose.get("faces", []), x1, y1, crop_w, crop_h, resize_w, resize_h),
            "hands_score": pose.get("hands_score", 0.0),
            "faces_score": pose.get("faces_score", 0.0)
        }
    def collect_keypoints(self, pose_dict, max_points=188):
        keypoints = []

        # 1. Bodies with interpolation
        kpts_body = pose_dict.get("bodies", [])
        if isinstance(kpts_body, (list, np.ndarray)):
            kpts_body = torch.tensor(kpts_body, dtype=torch.float32)
        if isinstance(kpts_body, torch.Tensor) and kpts_body.numel() > 0:
            connections = [(2, 3), (3, 4), (5, 6), (6, 7),(1, 2), (1, 5)]
            interpolated = []
            for i, j in connections:
                p1, p2 = kpts_body[i], kpts_body[j]
                if (p1 == 0).all() or (p2 == 0).all():
                    interpolated.append(torch.zeros(10, 2))
                else:
                    pts = [(1 - alpha) * p1 + alpha * p2 for alpha in torch.linspace(0, 1, 12)[1:-1]]
                    interpolated.append(torch.stack(pts))
            if interpolated:
                kpts_body = torch.cat([kpts_body] + interpolated, dim=0)
            keypoints.append(kpts_body)

        # 2. Hands
        kpts_hands = pose_dict.get("hands", [])
        if isinstance(kpts_hands, (list, np.ndarray)):
            kpts_hands = torch.tensor(kpts_hands, dtype=torch.float32)
        if isinstance(kpts_hands, torch.Tensor) and kpts_hands.numel() > 0:
            keypoints.append(kpts_hands)

        # 3. Faces
        kpts_face = pose_dict.get("faces", [])
        if isinstance(kpts_face, (list, np.ndarray)):
            kpts_face = torch.tensor(kpts_face, dtype=torch.float32)
        if isinstance(kpts_face, torch.Tensor) and kpts_face.numel() > 0:
            keypoints.append(kpts_face)

        # Concatenate all and clip to max length
        all_kpts = torch.cat(keypoints, dim=0)
        if all_kpts.shape[0] > max_points:
            all_kpts = all_kpts[:max_points]
        return all_kpts
    
    def extract_region_masks_from_all_kpts(self, all_kpts, image_size):
        """
        Args:
            all_kpts: Tensor of shape [188, 2], containing pixel coordinates (x, y)
            image_size: Tuple (H, W), height and width of the image

        Returns:
            face_mask, lhand_mask, rhand_mask, arm_shoulder_mask: 
            Each is a np.uint8 binary mask of shape (H, W) with values 0 or 255
        """
        H, W = image_size
        body_kpts = all_kpts[0:78]
        lhand_kpts = all_kpts[78:99]
        rhand_kpts = all_kpts[99:120]
        face_kpts = all_kpts[120:188]

        def poly_to_mask(poly):
            """Converts a polygon (in x,y) into a binary mask"""
            mask = np.zeros((H, W), dtype=np.uint8)
            if poly is not None and len(poly) >= 3:
                poly_int = np.array(poly).astype(np.int32)
                cv2.fillPoly(mask, [poly_int], 255)
            return mask

        def valid_poly(kpts):
            """
            Get a convex polygon from valid keypoints.
            Returns polygon in (x, y) format.
            """
            valid = (kpts > 0).all(dim=1)
            if valid.sum() < 3:
                return None
            kpts = kpts[valid]
            try:
                hull = ConvexHull(kpts)
                poly = kpts[hull.vertices]
                return poly.numpy()  # (x, y)
            except:
                return None

        def shoulder_arm_poly(kpts):
            """
            Construct a polygon from shoulder and arm keypoints.
            Returns polygon in (x, y) format.
            """
            indices = [5, 6, 7, 4, 3, 2]
            points = []
            for idx in indices:
                if (kpts[idx] > 0).all():
                    x, y = kpts[idx]
                    points.append([x.item(), y.item()])
            if len(points) >= 3:
                return np.array(points, dtype=np.float32)
            return None

        face_mask = poly_to_mask(valid_poly(face_kpts))
        lhand_mask = poly_to_mask(valid_poly(lhand_kpts))
        rhand_mask = poly_to_mask(valid_poly(rhand_kpts))
        arm_mask = poly_to_mask(shoulder_arm_poly(body_kpts))

        return face_mask, lhand_mask, rhand_mask, arm_mask

    def build_region_blocks_and_layout(
        self,valid_mask: torch.Tensor,  # [latent_t * latent_h * latent_w]
        latent_t: int, latent_h: int, latent_w: int, block_size: int
    ):
        """
        Build strictly separated region/nonregion blocks. Each region block contains only region tokens; each nonregion block contains only nonregion tokens.
        Args:
            valid_mask: Global token mask (True for region, False for nonregion)
            latent_t: Number of frames
            latent_h, latent_w: Spatial dimensions
            block_size: Number of tokens per block

        Returns:
            perm: [new_total_token,] Original token indices for valid tokens, -1 for padding locations
            pad_mask: [new_total_token,] True for valid tokens, False for padding
            block_layout: [n_block, n_block] Block sparse attention connectivity matrix
            region_block_ptrs, nonregion_block_ptrs: Block index ranges for each frame
        """
        n_token_per_frame = latent_h * latent_w
        region_indices_per_frame = []
        nonregion_indices_per_frame = []
        n_region_block = []
        n_nonregion_block = []

        # 1. Collect global indices of region and nonregion tokens for each frame
        for t in range(latent_t):
            start = t * n_token_per_frame
            end = (t + 1) * n_token_per_frame
            mask_flat = valid_mask[start:end]
            region_idx = torch.where(mask_flat)[0] + start
            nonregion_idx = torch.where(~mask_flat)[0] + start
            region_indices_per_frame.append(region_idx)
            nonregion_indices_per_frame.append(nonregion_idx)
            n_region_block.append((len(region_idx) + block_size - 1) // block_size)
            n_nonregion_block.append((len(nonregion_idx) + block_size - 1) // block_size)

        n_block = sum(n_region_block) + sum(n_nonregion_block)
        new_total_token = n_block * block_size

        # 2. Build perm and pad_mask, fill each block with -1 as padding if necessary
        perm = []
        pad_mask = []
        region_block_ptrs, nonregion_block_ptrs = [], []
        ptr = 0

        # First, fill region blocks for all frames
        for t in range(latent_t):
            idx = region_indices_per_frame[t].tolist()
            n_full = n_region_block[t]
            for b in range(n_full):
                block = idx[b*block_size : (b+1)*block_size]
                pad = block_size - len(block)
                perm += block + [-1] * pad
                pad_mask += [True] * len(block) + [False] * pad
            region_block_ptrs.append((ptr, ptr + n_full))
            ptr += n_full

        # Then, fill nonregion blocks for all frames
        for t in range(latent_t):
            idx = nonregion_indices_per_frame[t].tolist()
            n_full = n_nonregion_block[t]
            for b in range(n_full):
                block = idx[b*block_size : (b+1)*block_size]
                pad = block_size - len(block)
                perm += block + [-1] * pad
                pad_mask += [True] * len(block) + [False] * pad
            nonregion_block_ptrs.append((ptr, ptr + n_full))
            ptr += n_full

        perm = torch.tensor(perm, dtype=torch.long)
        pad_mask = torch.tensor(pad_mask, dtype=torch.bool)

        # 3. Build block_layout matrix
        block_layout = torch.zeros((n_block, n_block), dtype=torch.bool)
        # For each frame: fully connect all blocks within the same frame
        for t in range(latent_t):
            f_start = region_block_ptrs[t][0]
            f_end = nonregion_block_ptrs[t][1]
            block_layout[f_start:f_end, f_start:f_end] = 1
        # Connect all region blocks across frames (region-to-region cross-frame attention)
        for i in range(latent_t):
            for j in range(latent_t):
                if i == j: continue
                si, ei = region_block_ptrs[i]
                sj, ej = region_block_ptrs[j]
                block_layout[si:ei, sj:ej] = 1

        return perm, pad_mask, block_layout, region_block_ptrs, nonregion_block_ptrs

    # def __getitem__(self, data_id):
    def __getitem__(self, index):
        index = index % len(self.video_list)
        success=False
        for _try in range(5):
            try:
                if _try >0:
                    
                    index = random.randint(1,len(self.video_list))
                    index = index % len(self.video_list)
                
                clean = True
                frames_path = self.video_list[index]
                frame_mp4_path = frames_path
                dwpose_dir = frame_mp4_path.replace("videos", "dwpose")
                dwpose_pose_dir = str(Path(frame_mp4_path.replace("videos", "dwpose_data")).with_suffix(".pkl"))

                dwpose_path = dwpose_dir
                from decord import VideoReader
                # Open video captures once

                cap = VideoReader(frames_path)
                dwpose_cap = VideoReader(dwpose_path)

                _total_frame_num = len(cap)

                with open(dwpose_pose_dir, "rb") as f:
                    dwpose_data_dict = pickle.load(f)
                dwpose_poses = dwpose_data_dict["poses"]
                
                # Calculate frame sampling parameters
                stride = random.randint(1, self.sample_fps)
                cover_frame_num = (stride * self.max_frames)
                max_frames = self.max_frames

                if _total_frame_num < cover_frame_num + 1:
                    raise IndexError(f"Invalid sample: total_frame_num={_total_frame_num} < cover_frame_num+1={cover_frame_num+1}")
                    # start_frame = 0
                    # end_frame = _total_frame_num - 1
                    # stride = max((_total_frame_num // max_frames), 1)
                    # end_frame = min(stride * max_frames, _total_frame_num - 1)
                else:
                    start_frame = random.randint(0, _total_frame_num - cover_frame_num - 5)
                    end_frame = start_frame + cover_frame_num

                frame_list = []
                dwpose_list = []
                dwpose_pose_list = []
                # Get random reference frame
                random_ref = random.randint(0, _total_frame_num - 1)
                frame = cap[random_ref].asnumpy()
                random_ref_frame = Image.fromarray(frame)

                if random_ref_frame.mode != 'RGB':
                    random_ref_frame = random_ref_frame.convert('RGB')

                # Get random reference dwpose frame
                dwpose_frame = dwpose_cap[random_ref].asnumpy()
                random_ref_dwpose = Image.fromarray(dwpose_frame)




                # Read frames with stride
                for i_index in range(start_frame, end_frame, stride):
                    # Read frame
                    frame = cap[i_index].asnumpy()
                    i_frame = Image.fromarray(frame)

                    if i_frame.mode != 'RGB':
                        i_frame = i_frame.convert('RGB')

                    # Read dwpose frame
                    dwpose_frame = dwpose_cap[i_index].asnumpy()
                    i_dwpose = Image.fromarray(dwpose_frame)

                    i_dwpose_data = dwpose_poses[i_index]

                    frame_list.append(i_frame)
                    dwpose_list.append(i_dwpose)
                    dwpose_pose_list.append(i_dwpose_data)

  
                have_frames = len(frame_list)>0
                middle_indix = 0

                if have_frames:

                    l_hight = random_ref_frame.size[1]
                    l_width = random_ref_frame.size[0]

                    # random crop
                    x1 = 0
                    x2 = 0
                    y1 = 0
                    y2 = 0
                    
                    
                    random_ref_frame = random_ref_frame.crop((x1, y1,l_width-x2, l_hight-y2))
                    ref_frame = random_ref_frame 
                    # 
                    
                    random_ref_frame_tmp = torch.from_numpy(np.array(self.resize(random_ref_frame)))
                    random_ref_dwpose_tmp = torch.from_numpy(np.array(self.resize(random_ref_dwpose.crop((x1, y1,l_width-x2, l_hight-y2))))) # [3, 512, 320]
                    dwpose_pose_data = []
                    video_data_tmp = torch.stack([self.frame_process(self.resize(ss.crop((x1, y1,l_width-x2, l_hight-y2)))) for ss in frame_list], dim=0) # self.transforms(frames)
                    dwpose_data_tmp = torch.stack([torch.from_numpy(np.array(self.resize(ss.crop((x1, y1,l_width-x2, l_hight-y2))))).permute(2,0,1) for ss in dwpose_list], dim=0)


                    for dwpose_pose in dwpose_pose_list:
                        transformed_pose = self.transform_pose_dict(
                            dwpose_pose, x1, y1,
                            crop_w=l_width - x1 - x2,
                            crop_h=l_hight - y1 - y2,
                            resize_w=self.width,  # width
                            resize_h=self.height,  # height
                        )     

                        dwpose_pose_data.append(transformed_pose)

                video_data = torch.zeros(self.max_frames, 3, self.misc_size[0], self.misc_size[1])
                dwpose_data = torch.zeros(self.max_frames, 3, self.misc_size[0], self.misc_size[1])
                
                if have_frames:
                    video_data[:len(frame_list), ...] = video_data_tmp      
                    
                    dwpose_data[:len(frame_list), ...] = dwpose_data_tmp

                video_data = video_data.permute(1,0,2,3)
                dwpose_data = dwpose_data.permute(1,0,2,3)
                
                caption = "a person is dancing"
                break
            except Exception as e:
                # 
                caption = "a person is dancing"
                # 
                video_data = torch.zeros(3, self.max_frames, self.misc_size[0], self.misc_size[1])  
                random_ref_frame_tmp = torch.zeros(self.misc_size[0], self.misc_size[1], 3)
                vit_image = torch.zeros(3,self.misc_size[0], self.misc_size[1])
                
                dwpose_data = torch.zeros(3, self.max_frames, self.misc_size[0], self.misc_size[1])  
                # 
                random_ref_dwpose_data = torch.zeros(3, self.max_frames, self.misc_size[0], self.misc_size[1])  
                print('{} read video frame failed with error: {}'.format(frames_path, e))
                continue


        text = caption 
        if self.attention_type == "Original":
            attention_region = torch.zeros(self.max_frames,188, 3)
            T = 21 * 32 * 32 
            valid_mask = torch.zeros(T, dtype=torch.bool)
        if self.attention_type == "point":
            attention_region = torch.zeros(self.max_frames,188, 3)
            T = 21 * 32 * 32 
            valid_mask = torch.zeros(T, dtype=torch.bool)

            for t in range(self.max_frames):
                pose_dict = dwpose_pose_data[t]
                all_kpts = self.collect_keypoints(pose_dict, 188)

                for j in range(all_kpts.shape[0]):
                    x, y = all_kpts[j]
                    if x > 0 and y > 0:
                        latent_t = (t + 3) // 4
                        latent_x = int(x) // 16
                        latent_y = int(y) // 16
                        attention_region[t, j] = torch.tensor([latent_t, latent_x, latent_y])
                        if 0 <= latent_t < 21 and 0 < latent_x < 32 and 0 < latent_y < 32:
                            token_idx = latent_t * 32 * 32 + latent_y * 32 + latent_x
                            valid_mask[token_idx] = True
        if self.attention_type == "bfground" or self.attention_type == "orgionalregion":
            attention_region = torch.zeros(self.max_frames,188, 3)
            T = 21 * 32 * 32 
            valid_mask = torch.zeros(T, dtype=torch.bool)
            for t in range(self.max_frames):
                pose_dict = dwpose_pose_data[t]
                all_kpts = self.collect_keypoints(pose_dict, 188)#hand,arm,face,hand
                face_mask, lhand_mask, rhand_mask, arm_mask = self.extract_region_masks_from_all_kpts(all_kpts, (self.height, self.width))
                combined_mask = face_mask | lhand_mask | rhand_mask | arm_mask  # union of all
                #cv2.imwrite("region.png", combined_mask)
                #write the code that if the mask has the value 255, then set the valid_mask to True which is x,y
                ys, xs = torch.nonzero(torch.tensor(combined_mask) == 255, as_tuple=True)
                for y, x in zip(ys.tolist(), xs.tolist()):
                    latent_x = x // 16
                    latent_y = y // 16
                    latent_t = (t + 3) // 4  
                    if 0 <= latent_t < 21 and 0 <= latent_x < 32 and 0 <= latent_y < 32:
                        token_idx = latent_t * 32 * 32 + latent_y * 32 + latent_x
                        valid_mask[token_idx] = True
                    attention_region=None
        if self.attention_type == "region" or self.attention_type == "flash_region":
            attention_region = torch.zeros(self.max_frames,188, 3)
            T = 21 * 32 * 32 
            valid_token = torch.zeros(T, dtype=torch.bool)
            for t in range(self.max_frames):
                pose_dict = dwpose_pose_data[t]
                all_kpts = self.collect_keypoints(pose_dict, 188)#hand,arm,face,hand
                face_mask, lhand_mask, rhand_mask, arm_mask = self.extract_region_masks_from_all_kpts(all_kpts, (self.height, self.width))
                combined_mask = face_mask | lhand_mask | rhand_mask | arm_mask  # union of all
                #cv2.imwrite("region.png", combined_mask)
                #write the code that if the mask has the value 255, then set the valid_mask to True which is x,y
                ys, xs = torch.nonzero(torch.tensor(combined_mask) == 255, as_tuple=True)
                for y, x in zip(ys.tolist(), xs.tolist()):
                    latent_x = x // 16
                    latent_y = y // 16
                    latent_t = (t + 3) // 4  
                    if 0 <= latent_t < 21 and 0 <= latent_x < 32 and 0 <= latent_y < 32:
                        token_idx = latent_t * 32 * 32 + latent_y * 32 + latent_x
                        valid_token[token_idx] = True
            frame_id = torch.arange(21).repeat_interleave(32*32)
            # frame_id.shape == [21*1024] == [21504]
            valid_mask={
                "valid_mask": valid_token,
                "frame_id": frame_id,
            }
            

        if self.is_i2v:
            video, first_frame = video_data, random_ref_frame_tmp
            data = {"text": text, "video": video,"dwpose_data": dwpose_data,
                    "dwpose_pose_data":dwpose_pose_data, "valid_token_ids":valid_mask,
                    "video_path": frames_path, "dwpose_path": dwpose_path, 
                    "first_frame": first_frame, "random_ref_dwpose_data": random_ref_dwpose_tmp}
        else:
            data = {"text": text, "video": video, "path": frames_path}
        return data
    
    def __len__(self):
        
        return len(self.video_list)
 



class TextVideoDataset_onestage(torch.utils.data.Dataset):
    def __init__(self, base_dir, max_num_frames=81, frame_interval=2, num_frames=81, height=480, width=832,attention_type="point", is_i2v=False,steps_per_epoch=1):
        # metadata = pd.read_csv(metadata_path)
        # self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        # self.text = metadata["text"].to_list()
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.steps_per_epoch = steps_per_epoch
        self.attention_type = attention_type
        # data_list = ['UBC_Fashion', 'self_collected_videos_pose', 'TikTok']

        self.sample_fps = frame_interval
        self.max_frames = max_num_frames
        self.misc_size = [height, width]
        self.video_list = []
        self.pose_dir=base_dir
        # Traverse each video in self.pose_dir and write into self.video_list
        for video_name in os.listdir(self.pose_dir):
            video_path = os.path.join(self.pose_dir, video_name)
            if os.path.isfile(video_path) or os.path.isdir(video_path):
                self.video_list.append(video_path)
        print("!!! dataset length: ", len(self.video_list))        


        random.shuffle(self.video_list)
            
        self.frame_process = v2.Compose([
            # v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def resize(self, image):
        width, height = image.size
        # 
        image = torchvision.transforms.functional.resize(
            image,
            (self.height, self.width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        # return torch.from_numpy(np.array(image))
        return image
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame

    def transform_keypoints(self, kpts, x1, y1, crop_w, crop_h, resize_w, resize_h):
        """
        Transform keypoints from normalized [0,1] to cropped+resized pixel coords.
        Points outside the crop region or specifically masked (e.g., idx 9,10,12,13 for body) are replaced with (0.0, 0.0).

        Args:
            kpts: input keypoints, formats:
                - {"candidate": ndarray (N, 2)} for body
                - ndarray (1, 68, 2) for face
                - ndarray (2, 21, 2) for hands
            x1, y1: crop top-left corner in original image
            crop_w, crop_h: crop size in original image
            resize_w, resize_h: size after resizing

        Returns:
            np.ndarray of shape (N, 2), resized keypoints (invalid points → [0.0, 0.0])
        """
        if kpts is None or len(kpts) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # ---- Load keypoints ----
        if isinstance(kpts, dict) and "candidate" in kpts:
            xy = kpts["candidate"] * [resize_w, resize_h]
            mask_indices = {8,11,9, 10, 12, 13}
        elif isinstance(kpts, np.ndarray) and kpts.ndim == 3 and kpts.shape[1:] == (68, 2):  # face
            xy = kpts[0] * [resize_w, resize_h]
            mask_indices = set()
        elif isinstance(kpts, np.ndarray) and kpts.ndim == 3 and kpts.shape[1:] == (21, 2):  # hands
            xy = np.concatenate([kpts[0], kpts[1]], axis=0) * [resize_w, resize_h]
            mask_indices = set()
        else:
            raise ValueError(f"Unsupported keypoint format: {type(kpts)}, shape: {getattr(kpts, 'shape', None)}")

        # ---- Crop + Resize ----
        transformed = []
        for idx, (x, y) in enumerate(xy):
            if idx in mask_indices:
                transformed.append([0.0, 0.0])
                continue

            # check if (x, y) falls into the crop box
            if not (x1 <= x < x1 + crop_w and y1 <= y < y1 + crop_h):
                transformed.append([0.0, 0.0])
                continue

            # shift to crop-local and resize
            x_crop = x - x1
            y_crop = y - y1

            x_resized = x_crop / crop_w * resize_w
            y_resized = y_crop / crop_h * resize_h

            transformed.append([x_resized, y_resized])

        return np.array(transformed, dtype=np.float32)
    def transform_pose_dict(self, pose, x1, y1, crop_w, crop_h, resize_w, resize_h):
        """
        Apply transform to all body/hands/face keypoints in a pose dict.
        """
        return {
            "bodies": self.transform_keypoints(pose.get("bodies", []), x1, y1, crop_w, crop_h, resize_w, resize_h),
            "hands": self.transform_keypoints(pose.get("hands", []), x1, y1, crop_w, crop_h, resize_w, resize_h),
            "faces": self.transform_keypoints(pose.get("faces", []), x1, y1, crop_w, crop_h, resize_w, resize_h),
            "hands_score": pose.get("hands_score", 0.0),
            "faces_score": pose.get("faces_score", 0.0)
        }
    def collect_keypoints(self, pose_dict, max_points=188):
        keypoints = []

        # 1. Bodies with interpolation
        kpts_body = pose_dict.get("bodies", [])
        if isinstance(kpts_body, (list, np.ndarray)):
            kpts_body = torch.tensor(kpts_body, dtype=torch.float32)
        if isinstance(kpts_body, torch.Tensor) and kpts_body.numel() > 0:
            connections = [(2, 3), (3, 4), (5, 6), (6, 7),(1, 2), (1, 5)]
            interpolated = []
            for i, j in connections:
                p1, p2 = kpts_body[i], kpts_body[j]
                if (p1 == 0).all() or (p2 == 0).all():
                    interpolated.append(torch.zeros(10, 2))
                else:
                    pts = [(1 - alpha) * p1 + alpha * p2 for alpha in torch.linspace(0, 1, 12)[1:-1]]
                    interpolated.append(torch.stack(pts))
            if interpolated:
                kpts_body = torch.cat([kpts_body] + interpolated, dim=0)
            keypoints.append(kpts_body)

        # 2. Hands
        kpts_hands = pose_dict.get("hands", [])
        if isinstance(kpts_hands, (list, np.ndarray)):
            kpts_hands = torch.tensor(kpts_hands, dtype=torch.float32)
        if isinstance(kpts_hands, torch.Tensor) and kpts_hands.numel() > 0:
            keypoints.append(kpts_hands)

        # 3. Faces
        kpts_face = pose_dict.get("faces", [])
        if isinstance(kpts_face, (list, np.ndarray)):
            kpts_face = torch.tensor(kpts_face, dtype=torch.float32)
        if isinstance(kpts_face, torch.Tensor) and kpts_face.numel() > 0:
            keypoints.append(kpts_face)

        # Concatenate all and clip to max length
        all_kpts = torch.cat(keypoints, dim=0)
        if all_kpts.shape[0] > max_points:
            all_kpts = all_kpts[:max_points]
        return all_kpts
    
    def extract_region_masks_from_all_kpts(self, all_kpts, image_size):
        """
        Args:
            all_kpts: Tensor of shape [188, 2], containing pixel coordinates (x, y)
            image_size: Tuple (H, W), height and width of the image

        Returns:
            face_mask, lhand_mask, rhand_mask, arm_shoulder_mask: 
            Each is a np.uint8 binary mask of shape (H, W) with values 0 or 255
        """
        H, W = image_size
        body_kpts = all_kpts[0:78]
        lhand_kpts = all_kpts[78:99]
        rhand_kpts = all_kpts[99:120]
        face_kpts = all_kpts[120:188]

        def poly_to_mask(poly):
            """Converts a polygon (in x,y) into a binary mask"""
            mask = np.zeros((H, W), dtype=np.uint8)
            if poly is not None and len(poly) >= 3:
                poly_int = np.array(poly).astype(np.int32)
                cv2.fillPoly(mask, [poly_int], 255)
            return mask

        def valid_poly(kpts):
            """
            Get a convex polygon from valid keypoints.
            Returns polygon in (x, y) format.
            """
            valid = (kpts > 0).all(dim=1)
            if valid.sum() < 3:
                return None
            kpts = kpts[valid]
            try:
                hull = ConvexHull(kpts)
                poly = kpts[hull.vertices]
                return poly.numpy()  # (x, y)
            except:
                return None

        def shoulder_arm_poly(kpts):
            """
            Construct a polygon from shoulder and arm keypoints.
            Returns polygon in (x, y) format.
            """
            indices = [5, 6, 7, 4, 3, 2]
            points = []
            for idx in indices:
                if (kpts[idx] > 0).all():
                    x, y = kpts[idx]
                    points.append([x.item(), y.item()])
            if len(points) >= 3:
                return np.array(points, dtype=np.float32)
            return None

        face_mask = poly_to_mask(valid_poly(face_kpts))
        lhand_mask = poly_to_mask(valid_poly(lhand_kpts))
        rhand_mask = poly_to_mask(valid_poly(rhand_kpts))
        arm_mask = poly_to_mask(shoulder_arm_poly(body_kpts))

        return face_mask, lhand_mask, rhand_mask, arm_mask

    def build_region_blocks_and_layout(
        self,valid_mask: torch.Tensor,  # [latent_t * latent_h * latent_w]
        latent_t: int, latent_h: int, latent_w: int, block_size: int
    ):
        """
        Build strictly separated region/nonregion blocks. Each region block contains only region tokens; each nonregion block contains only nonregion tokens.
        Args:
            valid_mask: Global token mask (True for region, False for nonregion)
            latent_t: Number of frames
            latent_h, latent_w: Spatial dimensions
            block_size: Number of tokens per block

        Returns:
            perm: [new_total_token,] Original token indices for valid tokens, -1 for padding locations
            pad_mask: [new_total_token,] True for valid tokens, False for padding
            block_layout: [n_block, n_block] Block sparse attention connectivity matrix
            region_block_ptrs, nonregion_block_ptrs: Block index ranges for each frame
        """
        n_token_per_frame = latent_h * latent_w
        region_indices_per_frame = []
        nonregion_indices_per_frame = []
        n_region_block = []
        n_nonregion_block = []

        # 1. Collect global indices of region and nonregion tokens for each frame
        for t in range(latent_t):
            start = t * n_token_per_frame
            end = (t + 1) * n_token_per_frame
            mask_flat = valid_mask[start:end]
            region_idx = torch.where(mask_flat)[0] + start
            nonregion_idx = torch.where(~mask_flat)[0] + start
            region_indices_per_frame.append(region_idx)
            nonregion_indices_per_frame.append(nonregion_idx)
            n_region_block.append((len(region_idx) + block_size - 1) // block_size)
            n_nonregion_block.append((len(nonregion_idx) + block_size - 1) // block_size)

        n_block = sum(n_region_block) + sum(n_nonregion_block)
        new_total_token = n_block * block_size

        # 2. Build perm and pad_mask, fill each block with -1 as padding if necessary
        perm = []
        pad_mask = []
        region_block_ptrs, nonregion_block_ptrs = [], []
        ptr = 0

        # First, fill region blocks for all frames
        for t in range(latent_t):
            idx = region_indices_per_frame[t].tolist()
            n_full = n_region_block[t]
            for b in range(n_full):
                block = idx[b*block_size : (b+1)*block_size]
                pad = block_size - len(block)
                perm += block + [-1] * pad
                pad_mask += [True] * len(block) + [False] * pad
            region_block_ptrs.append((ptr, ptr + n_full))
            ptr += n_full

        # Then, fill nonregion blocks for all frames
        for t in range(latent_t):
            idx = nonregion_indices_per_frame[t].tolist()
            n_full = n_nonregion_block[t]
            for b in range(n_full):
                block = idx[b*block_size : (b+1)*block_size]
                pad = block_size - len(block)
                perm += block + [-1] * pad
                pad_mask += [True] * len(block) + [False] * pad
            nonregion_block_ptrs.append((ptr, ptr + n_full))
            ptr += n_full

        perm = torch.tensor(perm, dtype=torch.long)
        pad_mask = torch.tensor(pad_mask, dtype=torch.bool)

        # 3. Build block_layout matrix
        block_layout = torch.zeros((n_block, n_block), dtype=torch.bool)
        # For each frame: fully connect all blocks within the same frame
        for t in range(latent_t):
            f_start = region_block_ptrs[t][0]
            f_end = nonregion_block_ptrs[t][1]
            block_layout[f_start:f_end, f_start:f_end] = 1
        # Connect all region blocks across frames (region-to-region cross-frame attention)
        for i in range(latent_t):
            for j in range(latent_t):
                if i == j: continue
                si, ei = region_block_ptrs[i]
                sj, ej = region_block_ptrs[j]
                block_layout[si:ei, sj:ej] = 1

        return perm, pad_mask, block_layout, region_block_ptrs, nonregion_block_ptrs

    # def __getitem__(self, data_id):
    def __getitem__(self, index):
        index = index % len(self.video_list)
        success=False
        for _try in range(5):
            try:
                if _try >0:
                    
                    index = random.randint(1,len(self.video_list))
                    index = index % len(self.video_list)
                
                clean = True
                frames_path = self.video_list[index]

                video_path = Path(frames_path)
                latent_dir = str(video_path.parent).replace("/videos", "/latent")
                latent_name = video_path.with_suffix(".tensors.pth").name
                latent_path = Path(latent_dir) / latent_name              
                latent_data = torch.load(latent_path, weights_only=True, map_location="cpu")
                latents = latent_data["latents"]
                prompt_emb = latent_data["prompt_emb"]
                image_emb = latent_data["image_emb"]
                dwpose_data = latent_data["dwpose_data"]
                random_ref_dwpose_data = latent_data["random_ref_dwpose_data"]
                valid_token_ids = latent_data["valid_token_ids"]
                text = "a person is dancing"
            except Exception as e:
                print(e)
        data = {"text": text, "dwpose_data": dwpose_data,
                "valid_token_ids":valid_token_ids,
                "latents": latents,"prompt_emb": prompt_emb, "image_emb": image_emb,
                "video_path": frames_path, 
                "random_ref_dwpose_data": random_ref_dwpose_data}

        return data
    

    def __len__(self):
        
        return len(self.video_list)
 



class LightningModelForTrain_onestage(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None,
        model_VAE=None,
        attention_type = None,
        lpips_loss = False,
        # 
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")

        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        self.attention_type = attention_type

        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe_VAE = model_VAE.pipe.eval()

        self.tiler_kwargs = model_VAE.tiler_kwargs

        concat_dim = 4


        self.dwpose_embedding = nn.Sequential(
                    nn.Conv3d(3, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,2,2), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2,2,2), padding=1),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2,2,2), padding=1),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, 5120, (1,2,2), stride=(1,2,2), padding=0))

        randomref_dim = 20
        self.randomref_embedding_pose = nn.Sequential(
                    nn.Conv2d(3, concat_dim * 4, 3, stride=1, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, randomref_dim, 3, stride=2, padding=1),
                    
                    )
        self.freeze_parameters()
        
        # self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)

        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.lpips_loss = lpips_loss
        if self.lpips_loss:
            self.lpips_loss_fn = lpips.LPIPS(net='vgg').to(self.device)
            self.lpips_loss_fn.eval() 
        
        
    def freeze_parameters(self):

        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        self.pipe_VAE.eval()
        self.pipe_VAE.requires_grad_(False)
        
        self.randomref_embedding_pose.eval()
        self.dwpose_embedding.eval()
        # for param in self.dwpose_embedding.parameters():
        #     param.requires_grad = False
        # for param in self.randomref_embedding_pose.parameters():
        #     param.requires_grad = False        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            # 
            try:
                state_dict = load_state_dict(pretrained_lora_path)
            except:
                state_dict = load_state_dict_from_folder(pretrained_lora_path)
            # 
            state_dict_new = {}
            state_dict_new_module = {}
            for key in state_dict.keys():
                
                if 'pipe.dit.' in key:
                    key_new = key.split("pipe.dit.")[1]
                    state_dict_new[key_new] = state_dict[key]
                if "dwpose_embedding" in key or "randomref_embedding_pose" in key:
                    state_dict_new_module[key] = state_dict[key]
            state_dict = state_dict_new
            state_dict_new = {}

            for key in state_dict_new_module:
                if "dwpose_embedding" in key:
                    state_dict_new[key.split("dwpose_embedding.")[1]] = state_dict_new_module[key]
            self.dwpose_embedding.load_state_dict(state_dict_new, strict=True)

            state_dict_new = {}
            for key in state_dict_new_module:
                if "randomref_embedding_pose" in key:
                    state_dict_new[key.split("randomref_embedding_pose.")[1]] = state_dict_new_module[key]
            self.randomref_embedding_pose.load_state_dict(state_dict_new,strict=True)

            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    
    
    

    def training_step(self, batch, batch_idx):
        # batch["dwpose_data"]/255.: [1, 3, 81,512, 512], batch["random_ref_dwpose_data"]/255.: [1, 512, 512, 3]
        text, video, path = batch["text"][0], batch["video"], batch["video_path"][0]
        dwpose_pose_data = batch["dwpose_pose_data"]#list
        valid_token_ids = batch["valid_token_ids"]

        #torch.Size([21504])
        ######### per frame heatmap#########
        # latent_xy = attention_region[0, 0, :, 1:3]
        # heatmap = torch.zeros(32, 32)
        # import matplotlib.pyplot as plt
        # for x, y in latent_xy:
        #     x = int(x.item())
        #     y = int(y.item())
        #     if 0 <= x < 32 and 0 <= y < 32:
        #         heatmap[y, x] += 1
        # plt.figure(figsize=(6, 6))
        # plt.imshow(heatmap.numpy(), cmap='hot', interpolation='nearest')
        # plt.title("Frame 0 Keypoints on 32x32 Latent Grid")
        # plt.colorbar(label="Hit Count")
        # plt.xlabel("Latent X")
        # plt.ylabel("Latent Y")
        # plt.grid(False)
        # plt.tight_layout()
        # save_path = "frame0_attention_heatmap.png"
        # plt.savefig(save_path)
        # import ipdb; ipdb.set_trace()
        ######### per frame heatmap#########

        # 'A person is dancing',  [1, 3, 81, 512, 512], 'data/example_dataset/train/[DLPanda.com][]7309800480371133711.mp4'
       
        self.pipe_VAE.device = self.device
        dwpose_data = self.dwpose_embedding((torch.cat([batch["dwpose_data"][:,:,:1].repeat(1,1,3,1,1), batch["dwpose_data"]], dim=2)/255.).to(self.device))

        random_ref_dwpose_data = self.randomref_embedding_pose((batch["random_ref_dwpose_data"]/255.).to(torch.bfloat16).to(self.device).permute(0,3,1,2)).unsqueeze(2) # [1, 20, 104, 60]
        with torch.no_grad():
            if video is not None:
                # prompt
                prompt_emb = self.pipe_VAE.encode_prompt(text)
                # video
                video = video.to(dtype=self.pipe_VAE.torch_dtype, device=self.pipe_VAE.device)
                latents = self.pipe_VAE.encode_video(video, **self.tiler_kwargs)[0]
                # image
                if "first_frame" in batch: # [1, 512, 512, 3]
                    first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                    _, _, num_frames, height, width = video.shape
                    image_emb = self.pipe_VAE.encode_image(first_frame, num_frames, height, width)
                else:
                    image_emb = {}
                
                batch = {"latents": latents.unsqueeze(0), "prompt_emb": prompt_emb, "image_emb": image_emb}
        
        # Data
        p1 = random.random()
        p = random.random()
        if p1 < 0.05:
            
            dwpose_data = torch.zeros_like(dwpose_data)
            random_ref_dwpose_data = torch.zeros_like(random_ref_dwpose_data)
        latents = batch["latents"].to(self.device)  # [1, 16, 21, 64, 64]
        prompt_emb = batch["prompt_emb"] # batch["prompt_emb"]["context"]:  [1, 1, 512, 4096]
        prompt_emb["context"] = prompt_emb["context"].to(self.device)
        image_emb = batch["image_emb"]
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"].to(self.device) # [1, 257, 1280]
            if p < 0.1:
                image_emb["clip_feature"] = torch.zeros_like(image_emb["clip_feature"]) # [1, 257, 1280]
        if "y" in image_emb:
            
            if p < 0.1:
                image_emb["y"] = torch.zeros_like(image_emb["y"])
            image_emb["y"] = image_emb["y"].to(self.device) + random_ref_dwpose_data  # [1, 20, 21, 64, 64]Add commentMore actions
            
        
        condition =  dwpose_data
        # 
        condition = rearrange(condition, 'b c f h w -> b (f h w) c').contiguous()
        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))

        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            add_condition = condition,
            valid_token_ids = valid_token_ids,
            attention_type = self.attention_type,
        )

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        
        ##############LPIPS Loss##################
        if self.lpips_loss:
            with torch.no_grad():       
                sigma=self.pipe.scheduler.sigmas[timestep_id].to(device=self.device)         
                pred_latents = noisy_latents-sigma*noise_pred
                pred_images = self.pipe_VAE.decode_video(pred_latents,**self.tiler_kwargs)
                gt_images = self.pipe_VAE.decode_video(latents,**self.tiler_kwargs)#[1, 3, 81, 512, 512]
                # # -------- Save Predicted Video --------
                # frames = pred_images[0]  # 取第一个 batch，[3, 81, 512, 512]
                # frames = frames.permute(1, 2, 3, 0).detach().cpu().numpy()  # [81, 512, 512, 3]
                # if frames.min() < 0:
                #     frames = (frames + 1) / 2.0  # [-1,1] -> [0,1]
                # frames = np.clip(frames, 0, 1)
                # frames = (frames * 255).round().astype(np.uint8)
                # imageio.mimsave("pred_video.mp4", frames, fps=10)

                # # -------- Save GT Video --------
                # frames_gt = gt_images[0]  # [3, 81, 512, 512]
                # frames_gt = frames_gt.permute(1, 2, 3, 0).float().detach().cpu().numpy()
                # if frames_gt.min() < 0:
                #     frames_gt = (frames_gt + 1) / 2.0
                # frames_gt = np.clip(frames_gt, 0, 1)
                # frames_gt = (frames_gt * 255).round().astype(np.uint8)
                # imageio.mimsave("gt_video.mp4", frames_gt, fps=10)  
                # import ipdb; ipdb.set_trace()           
                frame_indices = torch.linspace(0, 80, steps=21).long()  # shape [20]
                pred_imgs = pred_images[:, :, frame_indices, :, :]   # [B, 3, 21, H, W]
                gt_imgs = gt_images[:, :, frame_indices, :, :]       # [B, 3, 21, H, W]
                
                B = pred_imgs.size(0)
                pred_imgs = pred_imgs.permute(0,2,1,3,4).reshape(-1, 3, pred_imgs.size(-2), pred_imgs.size(-1)).to(self.device)
                gt_imgs   = gt_imgs.permute(0,2,1,3,4).reshape(-1, 3, gt_imgs.size(-2), gt_imgs.size(-1)).to(self.device)


                lpips_loss = self.lpips_loss_fn(pred_imgs.float(), gt_imgs.float()).mean()

                del pred_latents, pred_images, gt_images, pred_imgs, gt_imgs


            
        raw_loss = loss
        
        #plot to a png file
        if self.lpips_loss:
            loss = loss * self.pipe.scheduler.training_weight(timestep)+lpips_loss
            self.log("lpips_loss", lpips_loss, prog_bar=True,logger=True, on_step=True, on_epoch=False)
        else:
            loss = loss * self.pipe.scheduler.training_weight(timestep)
        
        # Record log
        self.log("train_loss", loss, prog_bar=True,logger=True, on_step=True, on_epoch=False)
        
        self.log("raw_loss", raw_loss, prog_bar=True,logger=True, on_step=True, on_epoch=False)
        return loss


    def configure_optimizers(self):
        # trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        # optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        # return optimizer
        trainable_modules = [
            {'params': filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())},
            {'params': self.dwpose_embedding.parameters()},
            {'params': self.randomref_embedding_pose.parameters()},
        ]
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        # trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters())) + \
        #                         list(filter(lambda named_param: named_param[1].requires_grad, self.dwpose_embedding.named_parameters())) + \
        #                         list(filter(lambda named_param: named_param[1].requires_grad, self.randomref_embedding_pose.named_parameters()))
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters())) 
        
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        # state_dict = self.pipe.denoising_model().state_dict()
        state_dict = self.state_dict()
        # state_dict.update()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    return args


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert config dict to object with attributes
    class Config:
        pass
    
    args = Config()
    for key, value in config.items():
        setattr(args, key, value)
    
    # Ensure boolean values are properly set
    boolean_attrs = [
        'tiled', 'use_gradient_checkpointing', 
        'use_gradient_checkpointing_offload', 'use_swanlab'
    ]
    
    for attr in boolean_attrs:
        if hasattr(args, attr) and getattr(args, attr) is not None:
            setattr(args, attr, bool(getattr(args, attr)))
    
    return args


def data_process(config):
    dataset = TextVideoDataset(
        config.dataset_path,
        max_num_frames=config.num_frames,
        frame_interval=config.frame_interval,
        num_frames=config.num_frames,
        height=config.height,
        width=config.width,
        is_i2v=config.image_encoder_path is not None,
        steps_per_epoch=config.steps_per_epoch,
        attention_type=config.attention_type,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=config.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=config.text_encoder_path,
        image_encoder_path=config.image_encoder_path,
        vae_path=config.vae_path,
        tiled=config.tiled,
        tile_size=(config.tile_size_height, config.tile_size_width),
        tile_stride=(config.tile_stride_height, config.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=config.output_path,
    )
    trainer.test(model, dataloader)



class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
    def test_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["video_path"][0]
        dwpose_data,random_ref_dwpose_data = batch["dwpose_data"],batch["random_ref_dwpose_data"]
        valid_token_ids = batch["valid_token_ids"]
        self.pipe.device = self.device
        if video is not None:
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            # image
            if "first_frame" in batch:
                first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe.encode_image(first_frame, num_frames, height, width)
            else:
                image_emb = {}
            data = {"dwpose_data": dwpose_data, "random_ref_dwpose_data": random_ref_dwpose_data,
                    "valid_token_ids":valid_token_ids,
                    "latents": latents, "prompt_emb": prompt_emb, "image_emb": image_emb}
            video_path = Path(path)
            latent_dir = str(video_path.parent).replace("/videos", "/latent")
            latent_name = video_path.with_suffix(".tensors.pth").name
            latent_path = Path(latent_dir) / latent_name
            latent_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, latent_path)



    
def train_onestage(config):
    if config.use_wandb:


        wandb.init(
            project="prj-warp-attention", 
            name=config.experiment_name,  
            config={
                "steps_per_epoch": config.steps_per_epoch,
                "every_n_train_steps": config.every_n_train_steps,
                "learning_rate": config.learning_rate,
                "train_architecture": config.train_architecture,
                "attention_type": config.attention_type,
            },
            dir=os.path.join(config.output_path, "wandb_logs"), 
            mode=config.wandb_mode if hasattr(config, 'wandb_mode') else "online", 
        )
        logger = [pl_loggers.WandbLogger()]

                                             
    else:
        logger = None


    dataset = TextVideoDataset_train(
        config.dataset_path,
        max_num_frames=config.num_frames,
        frame_interval=config.frame_interval,
        num_frames=config.num_frames,
        height=config.height,
        width=config.width,
        is_i2v=config.image_encoder_path is not None,
        steps_per_epoch=config.steps_per_epoch,
        attention_type=config.attention_type,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=config.dataloader_num_workers
    )
    model_VAE = LightningModelForDataProcess(
        text_encoder_path=config.text_encoder_path,
        image_encoder_path=config.image_encoder_path,
        vae_path=config.vae_path,
        tiled=config.tiled,
        tile_size=(config.tile_size_height, config.tile_size_width),
        tile_stride=(config.tile_stride_height, config.tile_stride_width),
    )
    model = LightningModelForTrain_onestage(
        attention_type=config.attention_type,
        dit_path=config.dit_path,
        learning_rate=config.learning_rate,
        train_architecture=config.train_architecture,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_target_modules=config.lora_target_modules,
        init_lora_weights=config.init_lora_weights,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=config.use_gradient_checkpointing_offload,
        pretrained_lora_path=config.pretrained_lora_path,
        model_VAE = model_VAE,
        lpips_loss = config.lpips_loss,
    )
    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.output_path, "checkpoints"),  # 显式设置目录
        filename="epoch{epoch}-step{step}",
        save_top_k=-1,
        every_n_train_steps=config.every_n_train_steps
    )
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=config.training_strategy,
        default_root_dir=config.output_path,
        accumulate_grad_batches=config.accumulate_grad_batches,
        log_every_n_steps=config.log_every_n_steps,
        callbacks=[checkpoint_callback], # save checkpoints every_n_train_steps 
        logger=logger,
    )
    trainer.fit(model, dataloader)
    if config.use_wandb:
        wandb.finish()
if __name__ == '__main__':
    cli_args = parse_args()
    config = load_config(cli_args.config)
    
    if config.task == "data_process":
        data_process(config)
    elif config.task == "train":
        # only support DiT 
        train_onestage(config)



# lora finetune
# CUDA_VISIBLE_DEVICES="0" python examples/unianimate_wan/train_unianimate_wan.py   --task train   --train_architecture lora --lora_rank 64 --lora_alpha 64  --dataset_path data/example_dataset   --output_path ./models_out_one_GPU   --dit_path "/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"    --max_epochs 10   --learning_rate 1e-4   --accumulate_grad_batches 1   --use_gradient_checkpointing --image_encoder_path "/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"  --use_gradient_checkpointing_offload 
# CUDA_VISIBLE_DEVICES="0,1" python examples/unianimate_wan/train_unianimate_wan.py  --task train   --train_architecture lora --lora_rank 128 --lora_alpha 128  --dataset_path data/example_dataset   --output_path ./models_out   --dit_path "/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"     --max_epochs 10   --learning_rate 1e-4   --accumulate_grad_batches 1   --use_gradient_checkpointing --image_encoder_path "/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" --use_gradient_checkpointing_offload --training_strategy "deepspeed_stage_2" 
# CUDA_VISIBLE_DEVICES="0,1,2,3" python examples/unianimate_wan/train_unianimate_wan.py  --task train   --train_architecture lora --lora_rank 128 --lora_alpha 128  --dataset_path data/example_dataset   --output_path ./models_out   --dit_path "/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"     --max_epochs 10   --learning_rate 1e-4   --accumulate_grad_batches 1   --use_gradient_checkpointing --image_encoder_path "/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" --use_gradient_checkpointing_offload --training_strategy "deepspeed_stage_2" 