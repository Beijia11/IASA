import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader
from train_unianimate_wan import load_config, parse_args
from train_unianimate_wan import TextVideoDataset_train
from diffsynth.pipelines.wan_video import DistillWanVideoPipeline,WanVideoPipeline
from diffsynth.models.wan_video_dit import DistillWanModel
from diffsynth import ModelManager, load_state_dict, load_state_dict_from_folder
from peft import LoraConfig, inject_adapter_in_model, get_peft_model
import os
import glob
import json
from safetensors.torch import load_file
import wandb
import lightning.pytorch.loggers as pl_loggers
import os
import random
from train_unianimate_wan import LightningModelForDataProcess, LightningModelForTrain_onestage
from einops import rearrange
from PIL import Image

# -----------------------
#  DMD Loss 示例（可自定义）
# -----------------------

class DMDLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, student_out, teacher_out):
        return F.mse_loss(student_out, teacher_out.detach())

# -----------------------
#  LightningModule
# -----------------------
class DistillationTrainer(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        lora_path,
        teacher_lora_path,
        student_lora_path,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        init_lora_weights="kaiming",
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        model_VAE=None,
        attention_type=None,
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
        
        self.load_embedding_weights(lora_path)
        self.freeze_parameters()
        teacher_config= LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=(init_lora_weights == "kaiming"),
            target_modules=lora_target_modules.split(","),
        )
        student_config= LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=(init_lora_weights == "kaiming"),
            target_modules=lora_target_modules.split(","),
        )
        self.pipe.dit = get_peft_model(self.pipe.dit, teacher_config, adapter_name='teacher')
        self.pipe.dit.add_adapter(adapter_name='student', peft_config=student_config)
        self.pipe.dit.load_adapter(teacher_lora_path, adapter_name="teacher", is_trainable=True)
        self.pipe.dit.load_adapter(student_lora_path, adapter_name="student", is_trainable=True)
        self.pipe.dit.set_adapter("student")
   
        self.pipe.device = self.device
        self.pipe.dit.to(self.device)
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        

        

     
    def load_embedding_weights(self, embedding_ckpt_path):
        try:
            state_dict = load_state_dict(embedding_ckpt_path)
        except:
            state_dict = load_state_dict_from_folder(embedding_ckpt_path )
        # 加载 dwpose_embedding
        emb_state = {k.replace("dwpose_embedding.", ""): v for k, v in state_dict.items() if "dwpose_embedding." in k}
        if emb_state:
            self.dwpose_embedding.load_state_dict(emb_state, strict=True)
        # 加载 randomref_embedding_pose
        ref_state = {k.replace("randomref_embedding_pose.", ""): v for k, v in state_dict.items() if "randomref_embedding_pose." in k}
        if ref_state:
            self.randomref_embedding_pose.load_state_dict(ref_state, strict=True)    
                
    def freeze_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.dit.train()
        # Freeze parameters
        self.pipe_VAE.eval()
        self.pipe_VAE.requires_grad_(False)
        
        self.randomref_embedding_pose.eval()
        self.dwpose_embedding.eval()
        for param in self.dwpose_embedding.parameters():
            param.requires_grad = False
        for param in self.randomref_embedding_pose.parameters():
            param.requires_grad = False
 
        

    def training_step(self, batch, batch_idx):
        # batch["dwpose_data"]/255.: [1, 3, 81,512, 512], batch["random_ref_dwpose_data"]/255.: [1, 512, 512, 3]
        text, video, path = batch["text"][0], batch["video"], batch["video_path"][0]
        dwpose_pose_data = batch["dwpose_pose_data"]#list
        valid_token_ids = batch["valid_token_ids"]

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

        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        # import ipdb; ipdb.set_trace()
        self.pipe.dit.set_adapter('student')
        student_pred = self.pipe.dit(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            add_condition=condition,
            valid_token_ids=valid_token_ids,
            attention_type=self.attention_type,
        )       
        sigma=self.pipe.scheduler.sigmas[timestep_id].to(device=self.device)         
        pred_latents = noisy_latents-sigma*student_pred
        pred_images = self.pipe_VAE.decode_video(pred_latents,**self.tiler_kwargs)
        import ipdb; ipdb.set_trace()
        
        with torch.no_grad():
            self.pipe.dit.set_adapter('teacher')
            teacher_pred = self.pipe.dit(
                noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
                add_condition=condition,
                valid_token_ids=valid_token_ids,
                attention_type=self.attention_type,
            )
        import ipdb; ipdb.set_trace()
        teacher_pred = teacher_pred.detach().cpu()
        del teacher_pred  # 显式释放
        torch.cuda.empty_cache()
        
        # 推理 student LoRA adapter


        teacher_loss = torch.nn.functional.mse_loss(teacher_pred.float(), training_target.float())
        student_loss = torch.nn.functional.mse_loss(student_pred.float(), training_target.float())
        ################################

        raw_loss = teacher_loss
        #plot to a png file
        teacher_loss = teacher_loss * self.pipe.scheduler.training_weight(timestep)
        
        # Record log
        self.log("train_loss", teacher_loss, prog_bar=True,logger=True, on_step=True, on_epoch=False)
        self.log("raw_loss", raw_loss, prog_bar=True,logger=True, on_step=True, on_epoch=False)
        return teacher_loss, student_loss


    def configure_optimizers(self):
        # 只优化 student adapter 的参数
        # import ipdb; ipdb.set_trace()
        student_params = [
            param
            for name, param in self.pipe.dit.named_parameters()
            if "lora" in name and ".student." in name and param.requires_grad
        ]
        optimizer = torch.optim.AdamW(student_params, lr=self.learning_rate)
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
    
    

# -----------------------
#  主训练流程
# -----------------------
def main_train(config):
    if config.use_wandb:
        wandb.init(
            project="prj-wan-distillation", 
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


    # 2. Dataset & Loader
    dataset = TextVideoDataset_train(
        base_dir=config.dataset_path,
        max_num_frames=config.num_frames,
        frame_interval=config.frame_interval,
        num_frames=config.num_frames,
        height=config.height,
        width=config.width,
        attention_type=config.attention_type,
        is_i2v=True
    )
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=config.dataloader_num_workers,

    )
    model_VAE = LightningModelForDataProcess(
        text_encoder_path=config.text_encoder_path,
        image_encoder_path=config.image_encoder_path,
        vae_path=config.vae_path,
        tiled=config.tiled,
        tile_size=(config.tile_size_height, config.tile_size_width),
        tile_stride=(config.tile_stride_height, config.tile_stride_width),
    )
    # 3. Lightning Model
    model = DistillationTrainer(
        attention_type=config.attention_type,
        dit_path=config.dit_path,
        lora_path=config.lora_path,
        teacher_lora_path=config.teacher_lora_path,
        student_lora_path=config.student_lora_path,
        learning_rate=config.learning_rate,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_target_modules=config.lora_target_modules,
        init_lora_weights=config.init_lora_weights,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=config.use_gradient_checkpointing_offload,
        model_VAE = model_VAE,
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

# -----------------------
#  启动配置
# -----------------------
if __name__ == "__main__":
    cli_args = parse_args()
    config = load_config(cli_args.config)
    main_train(config)
