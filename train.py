import math 
import torch.nn.functional as F
import os 
import random 
import torch
import numpy as np

from tqdm.auto import tqdm 
from einops import rearrange
from pipelines import VideoGenerationPipeline
from util import save_videos_grid, generate_pos_coords, generate_frame_mask
from modelscope_t2v.ms_pipeline import MSVideoGenerationPipeline
from text_grounding_input import GroundingNetInput

class Trainer:

    def __init__(
        self,
        logger,
        accelerator,
        train_dataset,
        tokenizer=None,
        shift_init_state=None,
        mask_gt_frames=False,
        add_pos_emb=False,
        shift_channel_index: int = 2,
        train_task: list = ["infilling"],
        num_process: int = 1,
        train_batch_size: int = 1,
        max_train_steps: int = 500,
        gradient_accumulation_steps: int = 1,
        num_frames: int= 8,
        causal_tmp_attention: bool = False,
        use_clip_visual_encoder: bool = False, 
    ):

        num_update_steps_per_epoch = math.ceil((len(train_dataset)//train_batch_size)/gradient_accumulation_steps)
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
        total_batch_size = train_batch_size * num_process * gradient_accumulation_steps
        self.logger = logger 
        if accelerator.is_local_main_process:
            self.logger.info("***** Building Tranier with below parameters *****")
            self.logger.info(f"  Training tasks = {' '.join(train_task)}")
            self.logger.info(f"  Num examples = {len(train_dataset)}")
            self.logger.info(f"  Num Epochs = {num_train_epochs}")
            self.logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
            self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            self.logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
            self.logger.info(f"  Total optimization steps = {max_train_steps}")
        
        self.tokenizer = tokenizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.train_batch_size = train_batch_size
        self.max_train_steps = max_train_steps
        self.train_task = train_task
        self.num_train_epochs = num_train_epochs
        self.cur_best_val_loss = 9999
        self.use_clip_visual_encoder = use_clip_visual_encoder

        self.shift_init_state = shift_init_state
        self.shift_channel_index = shift_channel_index
        self.add_pos_emb = add_pos_emb
        if add_pos_emb:
            self.pos_coords = generate_pos_coords(w=16, h=16, f=num_frames)
        
        self.mask_gt_frames = mask_gt_frames
        self.causal_tmp_attention = causal_tmp_attention
        self.grounding_input = GroundingNetInput()
              
    def train(self,
              unet,
              vae,
              text_encoder,
              noise_scheduler,
              optimizer,
              lr_scheduler,
              weight_dtype,
              train_dataloader,
              accelerator,
              first_epoch,
              global_step,
              max_grad_norm,
              output_dir,
              checkpointing_steps,
              cat_cond,
              scheduler=None,
              validataion_steps=None,
              val_dataloader=None,
              seed=None,
              validation_data=None,
              ):


        progress_bar = tqdm(range(global_step, self.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        unet.set_task(self.train_task)
        if val_dataloader is not None:
            video_pipeline = VideoGenerationPipeline(
                add_pos_emb=self.add_pos_emb,
                mask_gt_frame=self.mask_gt_frames,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=self.tokenizer,
                unet=unet,
                scheduler=scheduler,
                shift_init_state=self.shift_init_state,
                shift_channel_index =self.shift_channel_index 
            )

        for epoch in range(first_epoch, self.num_train_epochs):
            unet.train()
            train_loss = 0.0 
            for step, batch in enumerate(train_dataloader):

                with accelerator.accumulate(unet):
                    # Convert videos to latent space
                    loss, avg_loss = self.shared_step(
                        unet=unet,
                        vae=vae,
                        text_encoder=text_encoder,
                        accelerator=accelerator,
                        batch=batch,
                        noise_scheduler=noise_scheduler,
                        weight_dtype=weight_dtype,
                        cat_cond = cat_cond,
                    )
                    train_loss += avg_loss.item() / self.gradient_accumulation_steps
                    
                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log(values={"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if global_step % checkpointing_steps == 0 and global_step != 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            self.logger.info(f"Saved state to {save_path}")
                    
                    if global_step % validataion_steps == 0 and global_step != 0:
                        if accelerator.is_main_process:
                            generator = torch.Generator(device=unet.device)
                            generator.manual_seed(seed)
                            with torch.no_grad():
                                val_loss = self.validataion(unet=unet, vae=vae, text_encoder=text_encoder,
                                                            noise_scheduler=noise_scheduler, accelerator=accelerator,
                                                            weight_dtype=weight_dtype, val_dataloader=val_dataloader,
                                                            cat_cond=cat_cond)

                                accelerator.log(values={"validation_loss": val_loss},step=global_step)

                            if val_loss < self.cur_best_val_loss:
                                self.cur_best_val_loss = val_loss
                                save_path = os.path.join(output_dir, f"best-checkpoint-{global_step}")
                                accelerator.save_state(save_path)
                                self.logger.info(f"Saved best-checkpoitn to {save_path}")

                            # randomly generate a video in validation set
                            print("Sampling videos")
                            index = random.randint(0, len(val_dataloader.dataset))
                            data = val_dataloader.dataset[index]
                            prompt = data["prompt_text"]
                            gt_sample = data["pixel_values"]
                            pixel_values = gt_sample.unsqueeze(0).to(weight_dtype)
                            video_length = pixel_values.shape[1]
                            gt_pixels = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                            gt_latents = vae.encode(gt_pixels.to(vae.device)).latent_dist.sample()
                            gt_latents = rearrange(gt_latents, "(b f) c h w -> b c f h w", f=video_length)
                            gt_latents = gt_latents * 0.18215

                            sample = video_pipeline(prompt=prompt,
                                                    generator=generator,
                                                    gt_latents=gt_latents,
                                                    task=self.train_task[0],
                                                    cat_cond=cat_cond,
                                                    causal_tmp_attention=self.causal_tmp_attention,
                                                    **validation_data).video
                            torch.cuda.empty_cache()
                            # rescale gt
                            gt_sample = (gt_sample + 1.0) / 2.0
                            gt_sample = rearrange(gt_sample, "f c h w -> c f h w")
                            samples = torch.concat([gt_sample.unsqueeze(0), sample])
                            # accelerator.log(values={"video":samples.cpu().detach().numpy()},step=global_step)
                            save_videos_grid(samples, f"{output_dir}/{prompt}-{global_step}.gif")
                            

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= self.max_train_steps:
                    break
            
            accelerator.end_training()
        
    def shift_noisy_state(self, noisy_latent, shift_channel_index=2):
        # noisy latent shape: batch_size, num_channels, num_frames, height, width
        new_latent = torch.randn_like(noisy_latent)
        new_latent[:, :shift_channel_index, :, :, :] = noisy_latent[:, :shift_channel_index, :,:,:]
        new_latent[:, shift_channel_index:, 1: , :, :] = noisy_latent[:, :shift_channel_index,:-1,:,:]
        return new_latent
    
    def shared_step(self,
                    unet,
                    vae,
                    text_encoder, 
                    accelerator,
                    batch,
                    noise_scheduler, 
                    weight_dtype,
                    cat_cond):
        
        task = random.choice(self.train_task)
        # Convert videos to latent space
        pixel_values = batch["pixel_values"].to(weight_dtype)
        video_length = pixel_values.shape[1]
        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
        latents = latents * 0.18215

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # Sample noise 
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        if self.shift_init_state:
            noise = self.shift_noisy_state(noisy_latent=noise, shift_channel_index=self.shift_channel_index)
        
        noisy_latent = noise_scheduler.add_noise(latents,noise,timesteps)
        noisy_latent, loss_mask = self.get_task_noisy_latent(latents, noisy_latent, task ,cat_cond)

        if self.add_pos_emb:
            coords = torch.from_numpy(self.pos_coords).cuda(noisy_latent.device)
            pos_embedding = unet.spatial_conv(coords).transpose(0, 1).unsqueeze(0)
            noisy_latent += pos_embedding
        
        if self.mask_gt_frames:
            attention_mask = generate_frame_mask(video_length, task)
            attention_mask = attention_mask.to(latents.device)
        elif self.causal_tmp_attention:
            attention_mask = (torch.triu(torch.ones(video_length,video_length))==1).transpose(0,1) 
            attention_mask = attention_mask.to(noisy_latent.device)
        else:
            attention_mask = None 
        
        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]
   
        # Get the target for loss depending on the prediction type
        if noise_scheduler.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")
        
        # Predict the noise residual and compute loss
        model_pred = unet(noisy_latent, timesteps, encoder_hidden_states, attention_mask=attention_mask).sample
        mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = torch.masked_select(mse_loss, loss_mask).mean()
        #Gather the losses across all processes for logging (if we use distributed training).
        avg_loss = accelerator.gather(loss.repeat(self.train_batch_size,)).mean()
        return loss, avg_loss
        

    def validataion(self, 
                    unet,
                    vae,
                    text_encoder,
                    noise_scheduler,
                    accelerator,
                    weight_dtype,
                    val_dataloader,
                    cat_cond,
                    ):
        
        unet.eval()
        val_loss = 0.0 
        self.logger.info("Start validation...")
        progress_bar = tqdm(range(len(val_dataloader)),disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Validation Samples ")
        for step, batch in enumerate(val_dataloader):
            with torch.no_grad():
                _, avg_loss = self.shared_step(
                    unet=unet,
                    vae=vae,
                    text_encoder=text_encoder,
                    accelerator=accelerator,
                    batch=batch,
                    noise_scheduler=noise_scheduler,
                    weight_dtype=weight_dtype,
                    cat_cond=cat_cond
                )
                val_loss += avg_loss.item()
                progress_bar.update(1)
            torch.cuda.empty_cache()

        val_avg_loss = val_loss / len(val_dataloader)
        self.logger.info(f"Validataion Loss is {val_avg_loss}") 
        return val_avg_loss


    def get_task_noisy_latent(self, latent, noisy_latent, task, cat_cond=False):
        """
        latent: b, c, f, w, h 
        noisy_laent:  b, c, f, w, h
        """
        
        if task == "prediction":
            if not cat_cond:
                # replace the first one to gt
                noisy_latent[:,:,0,:,:] = latent[:,:,0,:,:]
                loss_mask = torch.ones_like(noisy_latent)
                loss_mask[:,:,0,:,:] = torch.zeros_like(latent[:,:,0,:,:])
            else:
                first_frame=latent[:,:,0,:,:]
                first_frame_all = first_frame.unsqueeze(2).expand_as(noisy_latent)
                noisy_latent = torch.cat((noisy_latent,first_frame_all),dim=1)
                loss_mask = torch.ones_like(latent)

        elif task == "infilling":
            # replace the first and last one to gt
            noisy_latent[:,:,0,:,:] = latent[:,:,0,:,:] 
            noisy_latent[:,:,-1,:,:] = latent[:,:,-1,:,:]
            loss_mask = torch.ones_like(noisy_latent)
            loss_mask[:,:,0,:,:] = torch.zeros_like(latent[:,:,0,:,:])
            loss_mask[:,:,-1,:,:] = torch.zeros_like(latent[:,:,0,:,:])
        elif task == "rewind":
            # replace the last one to gt 
            noisy_latent[:,:,-1,:,:] = latent[:,:,-1,:,:]
            loss_mask = torch.ones_like(noisy_latent)
            loss_mask[:,:,-1,:,:] = torch.zeros_like(latent[:,:,0,:,:])
        elif task == "vanilla":
            loss_mask =torch.ones_like(noisy_latent)
        else:
            raise NotImplementedError
        return noisy_latent, loss_mask == 1
    
    def get_modelscope_inputs(
        self, 
        vae, 
        text_encoder, 
        noise_scheduler, 
        batch, 
        weight_dtype, 
        use_img_free_guidance,
        position_net=None, 
        cat_init=False,
        only_cat_first_frame=False,
        visual_encoder=None,
        controlnet_encoder=None,
    ):
        
        # process input image
        pixel_values = batch["pixel_values"].to(weight_dtype)
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)
        video_length = pixel_values.shape[1]
        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
        latents = vae.encode(pixel_values).sample()
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
        latents = latents * 0.18215

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # Sample noise
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        init_noisy_latent = noise_scheduler.add_noise(latents, noise, timesteps)
        noisy_latent = init_noisy_latent
        encoder_hidden_states = text_encoder.encode_with_transformer(batch["prompt_ids"])
        
        grounding_context = None
        if position_net is not None and "boxes" in batch.keys():
            # process position info
            if len(batch["boxes"].shape) == 4:
                # merge batch_size and num_frame
                batch_size, num_frame, num_obj, _ = batch["boxes"].shape
                batch["boxes"] = batch["boxes"].view(batch_size*num_frame, num_obj, -1)
                batch["text_embeddings"] = batch["text_embeddings"].view(batch_size*num_frame, num_obj, -1)
                batch["masks"] = batch["masks"].view(batch_size*num_frame, -1)
           
            grounding_input = self.grounding_input.prepare(batch)
            if random.random() < 0.1:  # random drop for guidance
                grounding_input = self.grounding_input.get_null_input()
            
            grounding_context = position_net(
                boxes=grounding_input["boxes"],
                masks=grounding_input["masks"],
                positive_embeddings=grounding_input["positive_embeddings"])
            grounding_context = {
                "context": grounding_context,
                "masks": batch["masks"]
            }

        visual_feat = None
        control_feat = None
        if cat_init:
            if position_net is not None and "cond_pixel" in batch.keys():
                cond_pixel_values = batch["cond_pixel"].unsqueeze(1).to(weight_dtype)
                cond_pixel_values = rearrange(cond_pixel_values, "b f c h w -> (b f) c h w")
                cond_latents = vae.encode(cond_pixel_values).sample()
                cond_latents = cond_latents * 0.18215
                cond_latents = rearrange(cond_latents, "(b f) c h w -> b c f h w", f=video_length)
                first_frame = cond_latents[:,:,0,:,:]
            else:
                first_frame = latents[:, :, 0, :, :]
            
            if use_img_free_guidance:
                if random.random() < 0.25:
                    first_frame = torch.zeros_like(first_frame).to(first_frame.device)

            if not only_cat_first_frame:
                first_frame_all = first_frame.unsqueeze(2).expand_as(noisy_latent)
            else:
                first_frame_all = first_frame.unsqueeze(2).expand_as(noisy_latent)
                zeros = torch.zeros_like(first_frame_all)
                first_frame_all = first_frame_all.clone()
                first_frame_all[:,:,1:,:,:] = zeros[:,:,1:,:,:]
            noisy_latent = torch.cat((noisy_latent, first_frame_all), dim=1)
            visual_feat = None  # for cat this is always None
       # else:
        # if not self.use_clip_visual_encoder:
        #     assert visual_encoder is not None, "Visual encoder should not be NONE"
        #     first_latent = rearrange(latents,"b c f h w -> b f c h w")[:,0,:,:]
        if visual_encoder is not None:
            assert controlnet_encoder is None, "ControlNot encoder should be None"
            first_latent = rearrange(latents, "b c f h w -> b f c h w")[:, 0, :, :]
            if use_img_free_guidance:
                if random.random() < 0.1:
                    first_latent = torch.zeros_like(first_latent).to(first_latent.device)
            visual_feat = visual_encoder(first_latent)
            visual_feat = rearrange(visual_feat, "b c w h -> b (w h) c")
            visual_feat = visual_feat.unsqueeze(1).repeat(1, video_length, 1, 1)
            visual_feat = rearrange(visual_feat, "b f l d -> (b f) l d")
            #pixel_values =rearrange(pixel_values, " (b f) c h w -> b f c h w")
            #first_frame_pixel_value = pixel_values[:,0,:,:,:].squeeze(1)
        elif self.use_clip_visual_encoder:
            import ipdb; ipdb.set_trace()
        if controlnet_encoder is not None:
            assert controlnet_encoder is not None, "We should use ControlNot encoder here"
            first_latent = rearrange(latents, "b c f h w -> b f c h w")[:, 0, :, :]
            if use_img_free_guidance:
                if random.random() < 0.1:
                    first_latent = torch.zeros_like(first_latent).to(first_latent.device)
            control_feat = controlnet_encoder(init_noisy_latent, first_latent, timesteps, encoder_hidden_states)
        else:
            pass

        return latents, noise, noisy_latent, encoder_hidden_states, timesteps, grounding_context, visual_feat, control_feat

    def modelscope_shared_step(
            self,
            unet,
            vae,
            text_encoder,
            accelerator,
            batch,
            noise_scheduler,
            weight_dtype,
            use_img_free_guidance,
            only_cat_first_frame,
            position_net,
            cat_init,
            visual_encoder,
            controlnet_encoder,
    ):
       
        latents, noise, noisy_latent, encoder_hidden_states, timesteps, grounding_context, visual_feat, control_feat = self.get_modelscope_inputs(
            vae=vae,
            text_encoder=text_encoder,
            noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
            batch=batch,
            position_net=position_net,
            cat_init=cat_init,
            use_img_free_guidance=use_img_free_guidance,
            only_cat_first_frame=only_cat_first_frame,
            visual_encoder=visual_encoder,
            controlnet_encoder=controlnet_encoder,
        )
      
        # Get the target for loss depending on the prediction type
        if noise_scheduler.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")
        
        # predict  the noise residual and compute loss 
        model_pred = unet(x=noisy_latent, t=timesteps, y=encoder_hidden_states, grounding_input=grounding_context,
                          visual_feat=visual_feat, control_feat=control_feat)
        mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        # TODO: remove the debugging code below before sync
        # has_inf_or_nan = (torch.isinf(mse_loss) | (mse_loss != mse_loss)).any().item()
        # if has_inf_or_nan:
        #     import ipdb; ipdb.set_trace()
        loss = mse_loss.mean()

        # Gather the losses across all processes for logging (if we use distributed training).
        avg_loss = accelerator.gather(loss.repeat(self.train_batch_size,)).mean()

        return loss, avg_loss
    
    def validation_model_scope(
                   self,
                   unet,
                   vae,
                   text_encoder,
                   accelerator,
                   noise_scheduler,
                   validation_dataloader,
                   weight_dtype,
                   ):
        unet.eval()
        val_loss = 0.0
        self.logger.info("Start validation ...")
        progress_bar = tqdm(range(len(validation_dataloader)), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Validation Samples")
        for step, batch in enumerate(validation_dataloader):
            with torch.no_grad():
                _, avg_loss = self.model_scope_shared_step(
                          unet=unet,
                          vae=vae,
                          text_encoder=text_encoder,
                          batch=batch,
                          accelerator=accelerator,
                          noise_scheduler=noise_scheduler,
                          weight_dtype=weight_dtype
                )
                val_loss += avg_loss.item()
                progress_bar.update(1)
            torch.cuda.empty_cache()
        
        val_avg_loss = val_loss / len(validation_dataloader)
        self.logger.info(f"Validation Loss is {val_avg_loss}")
        return val_avg_loss

    def train_model_scope(self,
                          unet,
                          vae,
                          text_encoder,
                          position_net,
                          visual_encoder,
                          controlnet_encoder,
                          noise_scheduler,
                          optimizer,
                          lr_scheduler,
                          weight_dtype,
                          train_dataloader,
                          accelerator,
                          first_epoch,
                          global_step,
                          max_grad_norm,
                          device,
                          output_dir,
                          checkpointing_steps,
                          use_img_free_guidance=False,
                          scheduler=None,
                          validation_steps=None,
                          seed=None,
                          diffusion=None,
                          cat_init=False,
                          validation_dataloader=None,
                          only_cat_first_frame=False,
                          ):
        
        progress_bar = tqdm(range(global_step, self.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        
        if validation_dataloader is not None:
            video_pipeline = MSVideoGenerationPipeline(
                vae=vae,
                text_encoder=text_encoder,
                unet=unet,
                diffusion=diffusion,
                position_net=position_net,
                visual_encoder=visual_encoder,
            )
      
        for epoch in range(first_epoch, self.num_train_epochs):
            unet.train()
            train_loss = 0.0

            for step, batch in enumerate(train_dataloader):
                # if step % 20 == 0:
                #     print("control zero conv weights", controlnet_encoder.input_hint_block.weight.sum())
                #     print("control zero conv weights", controlnet_encoder.zero_convs[0].weight.sum())
                #     print("unet zero conv weights", unet.output_blocks[0][0].in_layers[2].weight[:,-1000:].sum())
                with accelerator.accumulate(unet):
                    loss, avg_loss = self.modelscope_shared_step(
                        unet=unet,
                        vae=vae,
                        text_encoder=text_encoder,
                        accelerator=accelerator,
                        batch=batch,
                        noise_scheduler=noise_scheduler,
                        weight_dtype=weight_dtype,
                        use_img_free_guidance=use_img_free_guidance,
                        only_cat_first_frame=only_cat_first_frame,
                        position_net=position_net,
                        cat_init=cat_init,
                        visual_encoder=visual_encoder,
                        controlnet_encoder=controlnet_encoder,
                    )
                    train_loss += avg_loss.item() / self.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        def combine_para(gens):
                            for gen in gens:
                                if gen is not None:
                                    for ele in gen.parameters():
                                        yield ele
                        accelerator.clip_grad_norm_(combine_para([unet, position_net, visual_encoder,controlnet_encoder]), max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if accelerator.is_local_main_process:
                        accelerator.log(values={"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if global_step % checkpointing_steps == 0 and global_step != 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            if accelerator.is_local_main_process:
                                self.logger.info(f"Saved state to {save_path}")

                    if global_step % validation_steps == 0 and global_step != 0:
                        # if accelerator.is_main_process:
                        #     generator = torch.Generator(device=unet.device)
                        #     generator.manual_seed(seed)
                        #     with torch.no_grad():
                        #         val_loss = self.validation_model_scope(
                        #                                     unet=unet,
                        #                                     vae=vae,
                        #                                     text_encoder=text_encoder,
                        #                                     noise_scheduler=noise_scheduler,
                        #                                     accelerator=accelerator,
                        #                                     weight_dtype=weight_dtype,
                        #                                     validation_dataloader=validation_dataloader,
                        #                                     )

                        #         accelerator.log(values={"validation_loss": val_loss},step=global_step)

                            # if val_loss < self.cur_best_val_loss:
                                # self.cur_best_val_loss = val_loss
                            save_path = os.path.join(output_dir, f"best-checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            self.logger.info(f"Saved best-checkpoitn to {save_path}")

                            # randomly generate a video in validation set
                            print("Sampling videos")
                            if position_net is None:
                                index = random.randint(0, len(validation_dataloader.dataset))
                                data = validation_dataloader.dataset[index]
                                prompt = data["prompt_text"]
                                gt_sample = data["pixel_values"]
                                pixel_values = gt_sample.unsqueeze(0).to(weight_dtype)

                                first_frame_pixel = None
                                if cat_init:
                                    first_frame_pixel = pixel_values[:, 0, :, :]

                                if position_net is not None:
                                    grounding_input = self.grounding_input.prepare(data)
                                    grounding_context = position_net(**grounding_input)


                                video_length = pixel_values.shape[1]
                                gt_pixels = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                                gt_latents = vae.encode(gt_pixels.to(vae.device)).sample()
                                gt_latents = rearrange(gt_latents, "(b f) c h w -> b c f h w", f=video_length)
                                gt_latents = gt_latents * 0.18215
                                height, width = gt_latents.shape[-2], gt_latents.shape[-1]


                                sample = video_pipeline(
                                    prompt=prompt,
                                    video_length=video_length,
                                    height=height,
                                    width=width,
                                    num_inference_steps=50,
                                    guidance_scale=9,
                                    first_frame_pixel=first_frame_pixel,
                                )

                                torch.cuda.empty_cache()
                                gt_sample = (gt_sample + 1.0 ) / 2.0
                                gt_sample = rearrange(gt_sample, "f c h w -> c f h w")
                                samples = torch.concat([gt_sample.unsqueeze(0), sample])
                                save_videos_grid(samples, f"{output_dir}/{prompt}-{global_step}.gif")

                if accelerator.is_local_main_process:
                    logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)

                if global_step >= self.max_train_steps:
                    break

            accelerator.end_training()
