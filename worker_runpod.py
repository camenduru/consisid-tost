import os, shutil, json, requests, random, time, runpod
from urllib.parse import urlsplit

import torch
import numpy as np
from PIL import Image, ImageOps
from datetime import datetime

import insightface
from insightface.app import FaceAnalysis
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from diffusers import CogVideoXDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.training_utils import free_memory

from util.utils import *
from util.rife_model import load_rife_model
from models.utils import process_face_embeddings
from models.transformer_consisid import ConsisIDTransformer3DModel
from models.pipeline_consisid import ConsisIDPipeline
from models.eva_clip import create_model_and_transforms
from models.eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from models.eva_clip.utils_qformer import resize_numpy_image_long

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/content/ConsisID-preview"
lora_path = None
lora_rank = 128
dtype = torch.bfloat16
if os.path.exists(os.path.join(model_path, "transformer_ema")):
    subfolder = "transformer_ema"
else:
    subfolder = "transformer"        
transformer = ConsisIDTransformer3DModel.from_pretrained_cus(model_path, subfolder=subfolder)
scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
try:
    is_kps = transformer.config.is_kps
except:
    is_kps = False
face_helper = FaceRestoreHelper(
    upscale_factor=1,
    face_size=512,
    crop_ratio=(1, 1),
    det_model='retinaface_resnet50',
    save_ext='png',
    device=device,
    model_rootpath=os.path.join(model_path, "face_encoder")
)
face_helper.face_parse = None
face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device, model_rootpath=os.path.join(model_path, "face_encoder"))
face_helper.face_det.eval()
face_helper.face_parse.eval()
model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', os.path.join(model_path, "face_encoder", "EVA02_CLIP_L_336_psz14_s6B.pt"), force_custom_clip=True)
face_clip_model = model.visual
face_clip_model.eval()
eva_transform_mean = getattr(face_clip_model, 'image_mean', OPENAI_DATASET_MEAN)
eva_transform_std = getattr(face_clip_model, 'image_std', OPENAI_DATASET_STD)
if not isinstance(eva_transform_mean, (list, tuple)):
    eva_transform_mean = (eva_transform_mean,) * 3
if not isinstance(eva_transform_std, (list, tuple)):
    eva_transform_std = (eva_transform_std,) * 3
eva_transform_mean = eva_transform_mean
eva_transform_std = eva_transform_std
face_main_model = FaceAnalysis(name='antelopev2', root=os.path.join(model_path, "face_encoder"), providers=['CUDAExecutionProvider'])
handler_ante = insightface.model_zoo.get_model(f'{model_path}/face_encoder/models/antelopev2/glintr100.onnx', providers=['CUDAExecutionProvider'])
face_main_model.prepare(ctx_id=0, det_size=(640, 640))
handler_ante.prepare(ctx_id=0)
face_clip_model.to(device, dtype=dtype)
face_helper.face_det.to(device)
face_helper.face_parse.to(device)
transformer.to(device, dtype=dtype)
free_memory()
pipe = ConsisIDPipeline.from_pretrained(model_path, transformer=transformer, scheduler=scheduler, torch_dtype=dtype)
if lora_path:
    pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
    pipe.fuse_lora(lora_scale=1 / lora_rank)
scheduler_args = {}
if "variance_type" in pipe.scheduler.config:
    variance_type = pipe.scheduler.config.variance_type
    if variance_type in ["learned", "learned_range"]:
        variance_type = "fixed_small"
    scheduler_args["variance_type"] = variance_type
pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
pipe.to(device)
upscale_model = load_sd_upscale("/content/model_real_esran/RealESRGAN_x4.pth", device)
frame_interpolation_model = load_rife_model("/content/model_rife")

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate_local(input):
    values = input["input"]

    input_image = values['input_image']
    input_image = download_file(url=input_image, save_dir='/content', file_name='input_image')
    prompt = values['prompt']
    num_inference_steps = values['num_inference_steps']
    guidance_scale = values['guidance_scale']
    seed = values['seed']

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)
    print(seed)

    image_input = Image.open(input_image).convert("RGB")
    id_image = np.array(ImageOps.exif_transpose(image_input))
    id_image = resize_numpy_image_long(id_image, 1024)
    id_cond, id_vit_hidden, align_crop_face_image, face_kps = process_face_embeddings(face_helper, face_clip_model, handler_ante, 
                                                                            eva_transform_mean, eva_transform_std, 
                                                                            face_main_model, device, dtype, id_image, 
                                                                            original_id_image=id_image, is_align_face=True, 
                                                                            cal_uncond=False)
    if is_kps:
        kps_cond = face_kps
    else:
        kps_cond = None
    tensor = align_crop_face_image.cpu().detach()
    tensor = tensor.squeeze()
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.numpy() * 255
    tensor = tensor.astype(np.uint8)
    image  = ImageOps.exif_transpose(Image.fromarray(tensor))
    prompt = prompt.strip('"')
    generator = torch.Generator(device).manual_seed(seed) if seed else None
    latents = pipe(
        prompt=prompt,
        image=image,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=49,
        use_dynamic_cfg=False,
        guidance_scale=guidance_scale,
        generator=generator,
        id_vit_hidden=id_vit_hidden,
        id_cond=id_cond,
        kps_cond=kps_cond,
        output_type="pt",
    ).frames
    batch_size = latents.shape[0]
    batch_video_frames = []
    for batch_idx in range(batch_size):
        pt_image = latents[batch_idx]
        pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])
        image_np = VaeImageProcessor.pt_to_numpy(pt_image)
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)
        batch_video_frames.append(image_pil)
    all_frames = [frame for sublist in batch_video_frames for frame in sublist]
    total_frames = len(all_frames)
    desired_duration_seconds = 6
    fps = math.ceil(total_frames / desired_duration_seconds)
    video_path = save_video(all_frames, fps)
    free_memory()
    source = video_path
    destination = '/content/consisid-tost.mp4'
    shutil.move(source, destination)

    result = f"/content/consisid-tost.mp4"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate_local})