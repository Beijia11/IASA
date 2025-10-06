import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData, WanUniAnimateVideoPipeline
from modelscope import snapshot_download, dataset_snapshot_download
from PIL import Image
import os
import pickle
from PIL import Image
import numpy as np
import random
import pickle
import torch
from io import BytesIO
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFilter
import  torch.nn  as nn
import cv2
import sys 
import time
from scipy.spatial import ConvexHull 
sys.path.append("../../")  

# define hight and width
height = 512
width = 512
seed = 0
max_frames = 81
use_teacache = False
attention_type = "Original"#[Original,point,bfground,orgionalregion,region]
test_list_path= [
    # Format: [frame_interval, reference image, driving pose sequence]
    [1, "/ocean/projects/cis240035p/blu3/Cospeech-distillation/dit/self2/images/RoyHvfJowZI_20.png", "/ocean/projects/cis240035p/blu3/Cospeech-distillation/dit/self2/dwpose/RoyHvfJowZI_20.mp4","/ocean/projects/cis240035p/blu3/Cospeech-distillation/dit/self2/dwpose_data/RoyHvfJowZI_20.pkl"],
]

misc_size = [height,width]
# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32, # Image Encoder is loaded with float32
)
model_manager.load_models(
    [
        [
            
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",

        ],
        "./Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
        "./Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
model_manager.load_lora_v2("./checkpoints/UniAnimate-Wan2.1-14B-Lora-12000.ckpt", lora_alpha=1.0)
#model_manager.load_lora_v2("/ocean/projects/cis240035p/blu3/Cospeech-distillation/dit/models_out/region_bg_causal/checkpoints/epochepoch=6-stepstep=560.ckpt", lora_alpha=1.0)

# if you use deepspeed to train UniAnimate-Wan2.1, multiple checkpoints may be need to load, use the following form:
# model_manager.load_lora_v2([
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00001-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00002-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00003-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00004-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00005-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00006-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00007-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00008-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00009-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00010-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00011-of-00011.safetensors",
#             ], lora_alpha=1.0)

pipe = WanUniAnimateVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.


def resize(image):
    
    image = torchvision.transforms.functional.resize(
        image,
        (height, width),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    )
    return torch.from_numpy(np.array(image))
def transform_keypoints( kpts, x1, y1, crop_w, crop_h, resize_w, resize_h):
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
def transform_pose_dict(pose, x1, y1, crop_w, crop_h, resize_w, resize_h):
    """
    Apply transform to all body/hands/face keypoints in a pose dict.
    """
    return {
        "bodies": transform_keypoints(pose.get("bodies", []), x1, y1, crop_w, crop_h, resize_w, resize_h),
        "hands": transform_keypoints(pose.get("hands", []), x1, y1, crop_w, crop_h, resize_w, resize_h),
        "faces": transform_keypoints(pose.get("faces", []), x1, y1, crop_w, crop_h, resize_w, resize_h),
        "hands_score": pose.get("hands_score", 0.0),
        "faces_score": pose.get("faces_score", 0.0)
    }    
def collect_keypoints(pose_dict, max_points=188):
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
    
def extract_region_masks_from_all_kpts( all_kpts, image_size):
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

for path_dir_per in test_list_path:
    sample_fps = path_dir_per[0]  # frame interval for sampling
    ref_image_path = path_dir_per[1]  # Assuming ref_image_path remains unchanged
    pose_file_path = path_dir_per[2]  # This is now the path to the .mp4 file
    pose_data_path = path_dir_per[3]
    
    dwpose_all = {}
    frames_all = {}

    # Open the video file
    cap = cv2.VideoCapture(pose_file_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Iterate through video frames and store them
    for ii_index in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no more frames

        # Convert to RGB and store the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)
        
        # Save the frame in frames_all dictionary with 4-digit index
        ii_index_str = f"{ii_index:04d}"
        dwpose_all[ii_index_str] = frame_image
        frames_all[ii_index_str]= Image.fromarray(cv2.cvtColor(cv2.imread(ref_image_path), cv2.COLOR_BGR2RGB))
        # Capture the first frame for frames_pose_ref
        if ii_index == 0:
            frames_pose_ref = frame_image  # Store the first frame directly

    cap.release()
    stride = sample_fps
    _total_frame_num = len(frames_all)
    cover_frame_num = (stride * max_frames)
    
    if _total_frame_num < cover_frame_num + 1:
        start_frame = 0
        end_frame = _total_frame_num-1
        stride = max((_total_frame_num//max_frames),1)
        end_frame = min(stride*max_frames, _total_frame_num)
    else:
        # start_frame = random.randint(0, _total_frame_num-cover_frame_num-1)
        start_frame = 0
        end_frame = start_frame + cover_frame_num
    
    with open(pose_data_path, "rb") as f:
        dwpose_data_dict = pickle.load(f)
    dwpose_poses = dwpose_data_dict["poses"]
    frame_list = []
    dwpose_list = []
    pose_data_list = []
    random_ref_frame = frames_all[list(frames_all.keys())[0]]
    if random_ref_frame.mode != 'RGB':
        random_ref_frame = random_ref_frame.convert('RGB')
    random_ref_dwpose = frames_pose_ref
    if random_ref_dwpose.mode != 'RGB':
        random_ref_dwpose = random_ref_dwpose.convert('RGB')
    
    # sample pose sequence
    for i_index in range(start_frame, end_frame, stride):
        if i_index < len(frames_all):  # Check index within bounds
            i_key = list(frames_all.keys())[i_index]
            i_frame = frames_all[i_key]
            if i_frame.mode != 'RGB':
                i_frame = i_frame.convert('RGB')
            
            i_dwpose = dwpose_all[i_key]
            if i_dwpose.mode != 'RGB':
                i_dwpose = i_dwpose.convert('RGB')
            i_dwpose_data = dwpose_poses[i_index]
            frame_list.append(i_frame)
            dwpose_list.append(i_dwpose)
            pose_data_list.append(i_dwpose_data)
    
    if (end_frame-start_frame) < max_frames:
        for _ in range(max_frames-(end_frame-start_frame)):
            i_key = list(frames_all.keys())[end_frame-1]
            
            i_frame = frames_all[i_key]
            if i_frame.mode != 'RGB':
                i_frame = i_frame.convert('RGB')
            i_dwpose = dwpose_all[i_key]
            
    
            frame_list.append(i_frame)
            dwpose_list.append(i_dwpose)

    dwpose_pose_data = []
    for dwpose_pose in pose_data_list:
        transformed_pose = transform_pose_dict(
            dwpose_pose, 0, 0,
            crop_w=512,
            crop_h=512,
            resize_w=width,  # width
            resize_h=height,  # height
        )     

        dwpose_pose_data.append(transformed_pose)
    have_frames = len(frame_list)>0
    middle_indix = 0

    if have_frames:

        l_hight = random_ref_frame.size[1]
        l_width = random_ref_frame.size[0]
        
        ref_frame = random_ref_frame 
        
        random_ref_frame_tmp = torchvision.transforms.functional.resize(
            random_ref_frame,
            (height, width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        random_ref_dwpose_tmp = resize(random_ref_dwpose) 
        
        dwpose_data_tmp = torch.stack([resize(ss).permute(2,0,1) for ss in dwpose_list], dim=0)

    
    dwpose_data = torch.zeros(max_frames, 3, misc_size[0], misc_size[1])

    if have_frames:
        
        dwpose_data[:len(frame_list), ...] = dwpose_data_tmp
        
    
    dwpose_data = dwpose_data.permute(1,0,2,3)

    def image_compose_width(imag, imag_1):
        # read the size of image1
        rom_image = imag
        width, height = imag.size
        # read the size of image2
        rom_image_1 = imag_1
        
        width1 = rom_image_1.size[0]
        # create a new image
        to_image = Image.new('RGB', (width+width1, height))
        # paste old images
        to_image.paste(rom_image, (0, 0))
        to_image.paste(rom_image_1, (width, 0))
        return to_image

    caption = "a person is dancing"
    video_out_condition = []
    for ii in range(dwpose_data_tmp.shape[0]):
        ss = dwpose_list[ii]
        video_out_condition.append(image_compose_width(random_ref_frame_tmp, torchvision.transforms.functional.resize(
            ss,
            (height, width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )))
    ##attention type
    if attention_type == "Original":
        valid_mask =torch.zeros(32*32*21, dtype=torch.bool)
    if attention_type == "point":
        attention_region = torch.zeros(max_frames,188, 3)
        T = 21 * 32 * 32 
        valid_mask = torch.zeros(T, dtype=torch.bool)

        for t in range(max_frames):
            pose_dict = dwpose_pose_data[t]
            all_kpts = collect_keypoints(pose_dict, 188)

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
        valid_mask=valid_mask.unsqueeze(0).to("cuda")
    if attention_type == "bfground" or attention_type == "orgionalregion":
        attention_region = torch.zeros(max_frames,188, 3)
        T = 21 * 32 * 32 
        valid_mask = torch.zeros(T, dtype=torch.bool)
        for t in range(max_frames):
            pose_dict = dwpose_pose_data[t]
            all_kpts = collect_keypoints(pose_dict, 188)#hand,arm,face,hand paralize
            face_mask, lhand_mask, rhand_mask, arm_mask = extract_region_masks_from_all_kpts(all_kpts, (height, width))
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
        valid_mask=valid_mask.unsqueeze(0).to("cuda")

    if attention_type == "region":
        attention_region = torch.zeros(max_frames,188, 3)
        T = 21 * 32 * 32 
        valid_token = torch.zeros(T, dtype=torch.bool)
        for t in range(max_frames):
            pose_dict = dwpose_pose_data[t]
            all_kpts = collect_keypoints(pose_dict, 188)#hand,arm,face,hand
            face_mask, lhand_mask, rhand_mask, arm_mask = extract_region_masks_from_all_kpts(all_kpts, (height, width))
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
        valid_token=valid_token.unsqueeze(0).to("cuda")
        frame_id=frame_id.unsqueeze(0).to("cuda")
        valid_mask={
            "valid_mask": valid_token,
            "frame_id": frame_id,
        }
    with torch.no_grad():
        _ = pipe(
            prompt="a person is dancing",
            negative_prompt="warmup",  
            input_image=ref_frame,
            num_inference_steps=1,  
            cfg_scale=1.0,
            seed=0,
            tiled=True,
            dwpose_data=dwpose_data,
            random_ref_dwpose=random_ref_dwpose_tmp,
            height=height,
            width=width,
            tea_cache_l1_thresh=None,
            tea_cache_model_id=None,
            valid_token_ids=valid_mask,
            attention_type=attention_type,
        )
    torch.cuda.synchronize()  
    print("[INFO] flexattention pipeline warmup done.")         
    # Image-to-video
    t_real_start = time.time()
    video = pipe(
        prompt="a person is dancing",
        negative_prompt="细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=ref_frame,
        num_inference_steps=15,
        # cfg_scale=1.5, # slow
        cfg_scale=1.0, # fast
        seed=seed, tiled=True,
        dwpose_data=dwpose_data,
        random_ref_dwpose=random_ref_dwpose_tmp,
        height=height,
        width=width,
        tea_cache_l1_thresh=0.3 if use_teacache else None,
        tea_cache_model_id="Wan2.1-I2V-14B-720P" if use_teacache else None,
        valid_token_ids=valid_mask,
        attention_type=attention_type,
    )
    t_real_end = time.time()
    print("[INFO] FlexAttention in pipe real generation done. Time taken: {:.2f}s".format(t_real_end - t_real_start))
    video_out = []
    for ii in range(len(video)):
        ss = video[ii]
        video_out.append(image_compose_width(video_out_condition[ii], ss))
    os.makedirs("./outputs", exist_ok=True)
    save_video(video_out, "outputs/video_{}_{}_{}.mp4".format(attention_type,ref_image_path.split('/')[-1], pose_file_path.split('/')[-1]), fps=fps, quality=5)


    # CUDA_VISIBLE_DEVICES="0" python examples/unianimate_wan/inference_unianimate_wan_480p.py
