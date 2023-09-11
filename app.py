import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
repo_id_embeds = "sd-concepts-library/latest-bagan"
pipe = StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path,
    torch_dtype=torch.float16
).to("cuda")
pipe.load_textual_inversion(repo_id_embeds)


g_cuda = torch.Generator(device='cuda')

def inference(prompt, num_samples, height=512, width=512, num_inference_steps=50, guidance_scale=7.5):
    with torch.autocast("cuda"), torch.inference_mode():
        return pipe(
                prompt, height=int(height), width=int(width),
                num_images_per_prompt=int(num_samples),
                num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                generator=g_cuda
            ).images
st.text(prompt)
st.write("hi")
st.write(inference)
