def facebook_dino_vits8(image_path):
    from transformers import ViTFeatureExtractor, ViTModel
    from PIL import Image
    import requests

    image = Image.open(image_path)
    feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits8')
    model = ViTModel.from_pretrained('facebook/dino-vits8')
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

    return last_hidden_states



def google_vit_base_patch16_224_in21k(image_path):
    from transformers import ViTImageProcessor, ViTModel
    from PIL import Image
    import requests

    image = Image.open(image_path)
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

    return last_hidden_states



def microsoft_xclip_base_patch16_zero_shot(video_path, labels):
    import av
    import torch
    import numpy as np

    from transformers import AutoProcessor, AutoModel

    def read_video_pyav(container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        '''
        Sample a given number of frame indices from the video.
        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.
        Returns:
            indices (`List[int]`): List of sampled frame indices
        '''
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices

    container = av.open(video_path)

    # sample 8 frames
    frame_sample_rate = container.streams.video[0].frames // 8
    indices = sample_frame_indices(clip_len=8, frame_sample_rate=frame_sample_rate,
                                   seg_len=container.streams.video[0].frames)
    video = read_video_pyav(container, indices)

    processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16-zero-shot")
    model = AutoModel.from_pretrained("microsoft/xclip-base-patch16-zero-shot").to("cuda")

    inputs = processor(
        text=labels,
        videos=list(video),
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_video = outputs.logits_per_video  # this is the video-text similarity score
    probs = logits_per_video.softmax(dim=1)

    return probs



def CompVis_stable_diffusion_v1_4(prompt, output_path):
    import torch
    import os
    from diffusers import StableDiffusionPipeline

    model_id = 'CompVis/stable-diffusion-v1-4'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to('cuda')

    image = pipe(prompt).images[0]
    image.save(output_path)

    return os.path.abspath(output_path)



def stabilityai_sd_vae_ft_mse(prompt, output_path):
    import torch
    import os
    from diffusers import StableDiffusionPipeline

    model_id = 'stabilityai/sd-vae-ft-mse-original'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to('cuda')

    image = pipe(prompt).images[0]
    image.save(output_path)

    return os.path.abspath(output_path)



def Realistic_Vision_V1_4(prompt, negative_prompt):
    from transformers import pipeline

    model = pipeline('text-to-image', model='SG161222/Realistic_Vision_V1.4')

    result = model(prompt, negative_prompt=negative_prompt)

    return result



def stabilityai_sd_vae_ft_ema():
    from diffusers.models import AutoencoderKL
    from diffusers import StableDiffusionPipeline
    model = "CompVis/stable-diffusion-v1-4"
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)

    return pipe



def EimisAnimeDiffusion_1_0v(prompt):
    from huggingface_hub import hf_hub_download
    hf_hub_download('eimiss/EimisAnimeDiffusion_1.0v', prompt)



def Linaqruf_anything_v3_0(prompt, max_length=512):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_name_or_path = "Linaqruf/anything-v3.0"
    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Tokenize the input text
    inputs = tokenizer(prompt, truncation=True, padding=True, max_length=max_length, return_tensors="pt")

    # Set the device
    model.to("cuda")
    inputs.to("cuda")

    # Make the prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Return the logits
    return logits



def text_to_image(prompt, output_path):
    from transformers import CLIPProcessor, CLIPModel
    import torch
    import os
    from PIL import Image

    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("wavymulder/Analog-Diffusion")
    processor = CLIPProcessor.from_pretrained("wavymulder/Analog-Diffusion")

    # Tokenize the text
    inputs = processor(prompt, return_tensors="pt", padding=True)

    # Generate the image
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"])

    # Convert the image tensor to PIL image
    image = Image.fromarray(outputs[0].numpy().astype("uint8"))
    image.save(output_path)

    return os.path.abspath(output_path)



def Lykon_DreamShaper(prompt):
    import requests

    url = "https://api-inference.huggingface.co/models/Lykon/DreamShaper"

    headers = {
        "Authorization": "Bearer api_key",
        "Content-Type": "application/json",
    }

    data = {
        "inputs": prompt
    }

    response = requests.post(url, headers=headers, json=data)
    output = response.json()

    return output



def darkstorm2150_Protogen_v2_2_Official_Release(prompt, output_path):
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    import torch
    import os

    model_id = "darkstorm2150/Protogen_v2.2_Official_Release"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir="D:\python\data")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, cache_dir="D:\python\data")
    pipe = pipe.to("cuda")

    image = pipe(prompt, num_inference_steps=25).images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def gsdf_Counterfeit_V2_5(prompt):
    from transformers import pipeline

    # Extracting arguments
    model = 'gsdf/Counterfeit-V2.5'
    tokenizer = 'gsdf/Counterfeit-V2.5'
    device = 'cuda'

    # Initializing the pipeline
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)

    # Generating the text
    text = generator(prompt, max_length=100)[0]['generated_text']

    return text



def vintedois_diffusion_v0_1(prompt, output_path):
    import torch
    import os
    from diffusers import StableDiffusionPipeline

    model_id = "22h/vintedois-diffusion-v0-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    image = pipe(prompt).images[0]

    image.save(output_path)

    return os.path.abspath(output_path)



def kha_white_manga_ocr_base(image_path):
    import torch
    from PIL import Image
    from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

    model = VisionEncoderDecoderModel.from_pretrained("kha-white/manga-ocr-base")
    processor = GPT2TokenizerFast.from_pretrained("kha-white/manga-ocr-base")
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    image = Image.open(image_path)
    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted tokens and labels
    predicted_tokens = processor.tokenizer.convert_ids_to_tokens(outputs.logits.argmax(dim=2)[0])
    predicted_labels = processor.tokenizer.convert_ids_to_tokens(outputs.logits.argmax(dim=2)[0])

    return predicted_tokens, predicted_labels



def blip_image_captioning_base(image_path, prompt):
    from PIL import Image
    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    raw_image = Image.open(image_path)
    inputs = processor(raw_image, prompt, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption



def blip_image_captioning_large(image_path, prompt):
    from PIL import Image
    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    raw_image = Image.open(image_path)
    inputs = processor(raw_image, prompt, return_tensors="pt")
    out = model.generate(**inputs)

    return processor.decode(out[0], skip_special_tokens=True)



def microsoft_trocr_base_printed(image_path):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image

    image = Image.open(image_path)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text



def blip2_opt_2_7b(image_path, prompt):
    from PIL import Image
    import torch
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", device_map={"": 0}, torch_dtype=torch.float16)

    image = Image.open(image_path)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text



def microsoft_trocr_small_handwritten(image_path):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    import requests

    image = Image.open(image_path).convert('RGB')
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten')
    pixel_values = processor(images=image, return_tensors='pt').pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text



def naver_clova_ix_donut_base(image_path, prompt):
    import torch
    import re
    from PIL import Image
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    # Load the model and tokenizer
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    image = Image.open(image_path)

    # prepare decoder inputs
    task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
    question = "When is the coffee break?"
    prompt = task_prompt.replace("{user_input}", prompt)
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token

    return processor.token2json(sequence)



def microsoft_git_base_coco(image_path):
    from transformers import AutoProcessor, AutoModelForCausalLM
    from PIL import Image

    processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

    image = Image.open(image_path)

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_caption



def microsoft_trocr_large_handwritten(image_path):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text



def ydshieh_vit_gpt2_coco_en(image_path):
    import torch
    from PIL import Image
    from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel

    loc = "ydshieh/vit-gpt2-coco-en"
    feature_extractor = ViTFeatureExtractor.from_pretrained(loc)
    tokenizer = AutoTokenizer.from_pretrained(loc)
    model = VisionEncoderDecoderModel.from_pretrained(loc)
    model.eval()

    def predict(image):
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
        with torch.no_grad():
            output_ids = model.generate(pixel_values, max_length=16, num_beams=4,
                                        return_dict_in_generate=True).sequences
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds

    with Image.open(image_path) as image:
        preds = predict(image)

    return preds



def microsoft_trocr_base_handwritten(image_path):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image

    image = Image.open(image_path).convert('RGB')
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    pixel_values = processor(images=image, return_tensors='pt').pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text



def donut_base_finetuned_cord_v2(image_path):
    import torch
    import re
    from PIL import Image
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    # Load the model and tokenizer
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    image = Image.open(image_path)

    # prepare decoder inputs
    task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
    question = "When is the coffee break?"
    prompt = task_prompt.replace("{user_input}", prompt)
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token

    return processor.token2json(sequence)



def git_large_coco(image_path, question):
    from transformers import AutoProcessor, AutoModelForCausalLM
    from PIL import Image
    import torch
    processor = AutoProcessor.from_pretrained("microsoft/git-base-textvqa")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textvqa")

    image = Image.open(image_path).convert("RGB")

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    input_ids = processor(text=question, add_special_tokens=False).input_ids
    input_ids = [processor.tokenizer.cls_token_id] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)

    generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)

    return processor.batch_decode(generated_ids, skip_special_tokens=True)



def google_pix2struct_base(PATH_TO_SAVE, USERNAME, MODEL_NAME):
    from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

    model = Pix2StructForConditionalGeneration.from_pretrained(PATH_TO_SAVE)
    processor = Pix2StructProcessor.from_pretrained(PATH_TO_SAVE)

    model.push_to_hub(f"{USERNAME}/{MODEL_NAME}")
    processor.push_to_hub(f"{USERNAME}/{MODEL_NAME}")



def google_pix2struct_textcaps_base(image_path):
    from PIL import Image
    from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

    image = Image.open(image_path)
    model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base")
    processor = Pix2StructProcessor.from_pretrained("google/pix2struct-textcaps-base")
    inputs = processor(images=image, return_tensors="pt")
    predictions = model.generate(**inputs)

    return processor.decode(predictions[0], skip_special_tokens=True)



def git_base(image_path):
    from transformers import AutoProcessor, AutoModelForCausalLM
    from PIL import Image

    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

    image = Image.open(image_path)

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_caption



def microsoft_trocr_large_printed(image_path):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text



def git_large_textcaps(image_path):
    from transformers import AutoProcessor, AutoModelForCausalLM
    import requests
    from PIL import Image
    model_name = "microsoft/git-large-textcaps"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    image = Image.open(image_path)

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_caption



def git_large_r_textcaps(image_path):
    from transformers import AutoProcessor, AutoModelForCausalLM
    import requests
    from PIL import Image
    model_name = "microsoft/git-large-r-textcaps"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    image = Image.open(image_path)

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_caption)



def microsoft_trocr_small_stage1(image_path):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    import torch

    image = Image.open(image_path).convert('RGB')
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-stage1')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-stage1')

    pixel_values = processor(image, return_tensors='pt').pixel_values
    decoder_input_ids = torch.tensor([[model.config.decoder.decoder_start_token_id]])
    outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)

    return outputs



def microsoft_trocr_small_printed(image_path):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image

    image = Image.open(image_path).convert('RGB')
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed')
    pixel_values = processor(images=image, return_tensors='pt').pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text



def dpt_large_redesign(prompt):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained('nielsr/dpt-large-redesign')
    tokenizer = AutoTokenizer.from_pretrained('nielsr/dpt-large-redesign')

    # Tokenize the input text
    inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors='pt')
    inputs = {k: v.to('cpu') for k, v in inputs.items()}

    # Forward pass through the model
    outputs = model(**inputs)

    # Get the predicted label
    predicted_label = torch.argmax(outputs.logits).item()

    return predicted_label



def glpn_kitti(image_path, output_path):
    from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
    import torch
    import os
    import numpy as np
    from PIL import Image

    image = Image.open(image_path)
    feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-kitti")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti")
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype(np.uint8)
    depth = Image.fromarray(formatted)
    depth.save(output_path)
    return os.path.abspath(output_path)



def Intel_dpt_large(image_path, output_path):
    from transformers import DPTImageProcessor, DPTForDepthEstimation
    import torch
    import os
    import numpy as np
    from PIL import Image

    image = Image.open(image_path)
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype(np.uint8)
    depth = Image.fromarray(formatted)

    depth.save(output_path)
    return os.path.abspath(output_path)



def glpn_nyu(image_path, output_path):
    from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
    import torch
    import os
    import numpy as np
    from PIL import Image

    image = Image.open(image_path, output_path)
    feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(predicted_depth.unsqueeze(1), size=image.size[::-1], mode="bicubic",
                                                 align_corners=False)
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype(np.uint8)
    depth = Image.fromarray(formatted)
    depth.save(output_path)

    return os.path.abspath(output_path)



def glpn_nyu_finetuned_diode(image_path, output_path):
    from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
    import torch
    import os
    import numpy as np
    from PIL import Image

    image = Image.open(image_path)
    feature_extractor = GLPNFeatureExtractor.from_pretrained("sayakpaul/glpn-nyu-finetuned-diode")
    model = GLPNForDepthEstimation.from_pretrained("sayakpaul/glpn-nyu-finetuned-diode")
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(predicted_depth.unsqueeze(1), size=image.size[::-1], mode="bicubic",
                                                 align_corners=False)
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype(np.uint8)
    depth = Image.fromarray(formatted)
    depth.save(output_path)

    return os.path.abspath(output_path)



def intel_dpt_hybrid_midas(image_path, output_path):
    from PIL import Image
    import numpy as np
    import os
    import torch
    from transformers import DPTForDepthEstimation, DPTFeatureExtractor

    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype(np.uint8)
    depth = Image.fromarray(formatted)
    depth.save(output_path)

    return os.path.abspath(output_path)



def glpn_nyu_finetuned_diode_221122_030603(image_path, output_path):
    from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
    import torch
    import os
    import numpy as np
    from PIL import Image

    image = Image.open(image_path)
    feature_extractor = GLPNFeatureExtractor.from_pretrained("sayakpaul/glpn-nyu-finetuned-diode-221122-030603")
    model = GLPNForDepthEstimation.from_pretrained("sayakpaul/glpn-nyu-finetuned-diode-221122-030603")
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype(np.uint8)
    depth = Image.fromarray(formatted)
    depth.save(output_path)

    return os.path.abspath(output_path)



def glpn_kitti_finetuned_diode(image_path, output_path):
    from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
    import torch
    import os
    import numpy as np
    from PIL import Image

    image = Image.open(image_path)
    feature_extractor = GLPNFeatureExtractor.from_pretrained("sayakpaul/glpn-kitti-finetuned-diode")
    model = GLPNForDepthEstimation.from_pretrained("sayakpaul/glpn-kitti-finetuned-diode")
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype(np.uint8)
    depth = Image.fromarray(formatted)
    depth.save(output_path)

    return os.path.abspath(output_path)



def glpn_nyu_finetuned_diode_221122_044810(image_path, output_path):
    from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
    import torch
    import os
    import numpy as np
    from PIL import Image

    image = Image.open(image_path)
    feature_extractor = GLPNFeatureExtractor.from_pretrained("sayakpaul/glpn-nyu-finetuned-diode-221122-044810")
    model = GLPNForDepthEstimation.from_pretrained("sayakpaul/glpn-nyu-finetuned-diode-221122-044810")
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype(np.uint8)
    depth = Image.fromarray(formatted)
    depth.save(output_path)

    return os.path.abspath(output_path)



def glpn_kitti_finetuned_diode_221214_123047(image_path, output_path):
    from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
    import torch
    import os
    import numpy as np
    from PIL import Image

    image = Image.open(image_path)
    feature_extractor = GLPNFeatureExtractor.from_pretrained("sayakpaul/glpn-kitti-finetuned-diode-221214-123047")
    model = GLPNForDepthEstimation.from_pretrained("sayakpaul/glpn-kitti-finetuned-diode-221214-123047")
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype(np.uint8)
    depth = Image.fromarray(formatted)
    depth.save(output_path)

    return os.path.abspath(output_path)



def microsoft_resnet_50(image_path):
    from transformers import AutoImageProcessor, ResNetForImageClassification
    import torch
    from PIL import Image

    image = Image.open(image_path)

    processor = AutoImageProcessor.from_pretrained('microsoft/resnet-50')
    model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50')

    inputs = processor(image, return_tensors='pt')

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()

    return model.config.id2label[predicted_label]



def facebook_convnext_large_224(image_path):
    from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
    import torch
    from PIL import Image

    image = Image.open(image_path)
    feature_extractor = ConvNextFeatureExtractor.from_pretrained('facebook/convnext-large-224')
    model = ConvNextForImageClassification.from_pretrained('facebook/convnext-large-224')

    inputs = feature_extractor(image, return_tensors='pt')

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()

    return model.config.id2label[predicted_label]



def microsoft_beit_base_patch16_224_pt22k_ft22k(image_path):
    from transformers import BeitImageProcessor, BeitForImageClassification
    from PIL import Image

    image = Image.open(image_path)
    processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    return model.config.id2label[predicted_class_idx]



def google_vit_base_patch16_224(image_path):
    from transformers import ViTImageProcessor, ViTForImageClassification
    from PIL import Image
    import requests

    image = Image.open(image_path)
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]



def martinezomg_vit_base_patch16_224_diabetic_retinopathy(image_path):
    from transformers import pipeline

    image_classifier = pipeline('image-classification', 'martinezomg/vit-base-patch16-224-diabetic-retinopathy')
    result = image_classifier(image_path)

    return result



def nateraw_vit_age_classifier(image_path):
    from PIL import Image
    from transformers import ViTFeatureExtractor, ViTForImageClassification

    im = Image.open(image_path)

    model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
    transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')

    inputs = transforms(im, return_tensors='pt')
    output = model(**inputs)

    proba = output.logits.softmax(1)
    preds = proba.argmax(1)

    return preds



def google_vit_base_patch16_384(image_path):
    from transformers import ViTFeatureExtractor, ViTForImageClassification
    from PIL import Image

    image = Image.open(image_path)
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384')
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    return model.config.id2label[predicted_class_idx]






def microsoft_beit_base_patch16_224(image_path):
    from transformers import BeitImageProcessor, BeitForImageClassification
    from PIL import Image
    import requests

    image = Image.open(image_path)
    processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')
    model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224')
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    return predicted_class



def lysandre_tiny_vit_random(image_path):
    import torch
    from PIL import Image
    from torchvision.transforms import functional as F
    from transformers import ViTFeatureExtractor, ViTForImageClassification

    model_name_or_path = "lysandre/tiny-vit-random"
    # Load the image
    image = Image.open(image_path)

    # Preprocess the image
    image = F.to_tensor(image)

    # Load the model and tokenizer
    feature_extractor = ViTFeatureExtractor.from_pretrained("lysandre/tiny-vit-random")
    model = ViTForImageClassification.from_pretrained(model_name_or_path).to("cuda")

    # Tokenize the image
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Make predictions
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    outputs = model(**inputs)

    # Get the predicted class label
    predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()

    return predicted_class_idx



def fxmarty_resnet_tiny_beans(image_path):
    from transformers import pipeline

    classifier = pipeline('image-classification', model='fxmarty/resnet-tiny-beans')
    results = classifier(image_path)

    return results



def google_mobilenet_v1_0_75_192(image_path):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from PIL import Image
    import requests

    image = Image.open(image_path)
    preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v1_0.75_192")
    model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v1_0.75_192")
    inputs = preprocessor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    return model.config.id2label[predicted_class_idx]



def nvidia_mit_b0(image_path):
    from transformers import SegformerFeatureExtractor, SegformerForImageClassification
    from PIL import Image
    import requests

    image = Image.open(image_path)
    feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/mit-b0')
    model = SegformerForImageClassification.from_pretrained('nvidia/mit-b0')
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    return predicted_class



def vit_base_patch16_224_augreg2_in21k_ft_in1k(image_path):
    import torch
    from PIL import Image
    from torchvision.transforms import functional as F
    from transformers import ViTFeatureExtractor, ViTForImageClassification

    model_path = "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k"
    # Load the pre-trained model
    model = ViTForImageClassification.from_pretrained(model_path)

    # Load and preprocess the image
    image = Image.open(image_path)
    image = F.resize(image, (224, 224))
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Generate features from the image
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs.pop("pixel_values")
    inputs = {k: v.to(torch.device("cuda")) for k, v in inputs.items()}

    # Perform inference
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted class label
    predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class



def google_mobilenet_v2_1_0_224(image_path):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from PIL import Image
    import requests

    image = Image.open(image_path)
    preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
    model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
    inputs = preprocessor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    return model.config.id2label[predicted_class_idx]



def microsoft_swin_tiny_patch4_window7_224(image_path):
    from transformers import AutoFeatureExtractor, SwinForImageClassification
    from PIL import Image

    image = Image.open(image_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    return model.config.id2label[predicted_class_idx]



def microsoft_swinv2_tiny_patch4_window8_256(image_path):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from PIL import Image
    import requests

    image = Image.open(image_path)
    processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    return model.config.id2label[predicted_class_idx]



def anime_ai_detect(prompt):
    from transformers import pipeline
    model_name = "saltacc/anime-ai-detect"
    classifier = pipeline("text-classification", model=model_name)
    result = classifier(prompt)

    return result



def swin_tiny_patch4_window7_224_bottom_cleaned_data(image_path):
    from transformers import AutoFeatureExtractor, SwinForImageClassification
    from PIL import Image

    image = Image.open(image_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "Soulaimen/swin-tiny-patch4-window7-224-bottom_cleaned_data")
    model = SwinForImageClassification.from_pretrained("Soulaimen/swin-tiny-patch4-window7-224-bottom_cleaned_data")
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    return model.config.id2label[predicted_class_idx]



def microsoft_table_transformer_structure_recognition(image_path):
    from transformers import AutoImageProcessor, TableTransformerModel
    from huggingface_hub import hf_hub_download
    from PIL import Image

    file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")
    image = Image.open(image_path).convert("RGB")

    image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
    model = TableTransformerModel.from_pretrained("microsoft/table-transformer-structure-recognition")

    # prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")

    # forward pass
    outputs = model(**inputs)

    # the last hidden states are the final query embeddings of the Transformer decoder
    # these are of shape (batch_size, num_queries, hidden_size)
    last_hidden_states = outputs.last_hidden_state

    return last_hidden_states



def facebook_regnet_y_008(image_path):
    from transformers import AutoFeatureExtractor, RegNetForImageClassification
    import torch
    from PIL import Image

    image = Image.open(image_path)

    feature_extractor = AutoFeatureExtractor.from_pretrained('zuppif/regnet-y-040')
    model = RegNetForImageClassification.from_pretrained('zuppif/regnet-y-040')

    inputs = feature_extractor(image, return_tensors='pt')

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()

    return model.config.id2label[predicted_label]



def microsoft_table_transformer_detection(image_path):
    from transformers import AutoImageProcessor, TableTransformerModel
    from PIL import Image
    image = Image.open(image_path)
    image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    model = TableTransformerModel.from_pretrained("microsoft/table-transformer-detection")

    # prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")

    # forward pass
    outputs = model(**inputs)

    # the last hidden states are the final query embeddings of the Transformer decoder
    # these are of shape (batch_size, num_queries, hidden_size)
    last_hidden_states = outputs.last_hidden_state

    return last_hidden_states



def facebook_detr_resnet_50(image_path):
    from transformers import DetrImageProcessor, DetrForObjectDetection
    import torch
    from PIL import Image
    import requests

    image = Image.open(image_path)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    return outputs



def hustvl_yolos_tiny(image_path):
    from transformers import YolosFeatureExtractor, YolosForObjectDetection
    from PIL import Image

    image = Image.open(image_path)
    feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    return logits, bboxes



def facebook_detr_resnet_101(image_path):
    from transformers import DetrImageProcessor, DetrForObjectDetection
    from PIL import Image

    image = Image.open(image_path)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    return outputs



def google_owlvit_base_patch32(image_path, prompt):
    from PIL import Image
    import torch
    from transformers import OwlViTProcessor, OwlViTForObjectDetection

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    image = Image.open(image_path)

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    return results



def keremberke_yolov8m_table_extraction(image_path, output_path):
    from ultralyticsplus import YOLO, render_result
    import os

    model = YOLO('keremberke/yolov8m-table-extraction')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    results = model.predict(image_path)
    print(results[0].boxes)

    render = render_result(model=model, image=image_path, result=results[0])

    render.save(output_path)

    return os.path.abspath(output_path)



def detr_doc_table_detection(image_path):
    from transformers import DetrImageProcessor, DetrForObjectDetection
    import torch
    from PIL import Image

    image = Image.open(image_path)
    processor = DetrImageProcessor.from_pretrained("TahaDouaji/detr-doc-table-detection")
    model = DetrForObjectDetection.from_pretrained("TahaDouaji/detr-doc-table-detection")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box}")



def hustvl_yolos_small(image_path):
    from transformers import YolosFeatureExtractor, YolosForObjectDetection
    from PIL import Image
    import requests

    image = Image.open(image_path)
    feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')

    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)

    logits = outputs.logits
    bboxes = outputs.pred_boxes

    return logits, bboxes



def facebook_detr_resnet_101_dc5(image_path):
    from transformers import DetrFeatureExtractor, DetrForObjectDetection
    from PIL import Image
    import requests

    image = Image.open(image_path)
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-101-dc5')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101-dc5')
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    return logits, bboxes



def deformable_detr(image_path):
    from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
    import torch
    from PIL import Image
    import requests

    image = Image.open(image_path)
    processor = AutoImageProcessor.from_pretrained('SenseTime/deformable-detr')
    model = DeformableDetrForObjectDetection.from_pretrained('SenseTime/deformable-detr')
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)

    return outputs



def keremberke_yolov8m_hard_hat_detection(image_path, output_path, conf=0.25, iou=0.45, agnostic_nms=False,
                                          max_det=1000):
    from ultralyticsplus import YOLO, render_result
    import os
    model = YOLO('keremberke/yolov8m-hard-hat-detection')

    model.overrides['conf'] = conf
    model.overrides['iou'] = iou
    model.overrides['agnostic_nms'] = agnostic_nms
    model.overrides['max_det'] = max_det

    results = model.predict(image_path)

    print(results[0].boxes)
    render = render_result(model=model, image=image_path, result=results[0])
    render.save(output_path)

    return os.path.abspath(output_path)



def keremberke_yolov5m_license_plate(image_path, output_path, size=640, augment=False, conf=0.25, iou=0.45,
                                     agnostic=False,
                                     multi_label=False, max_det=1000):
    import yolov5
    import os

    model = yolov5.load('keremberke/yolov5m-license-plate')
    model.conf = conf
    model.iou = iou
    model.agnostic = agnostic
    model.multi_label = multi_label
    model.max_det = max_det

    results = model(image_path, size=size, augment=augment)
    predictions = results.pred[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    results.save(output_path)

    return boxes, scores, categories, os.path.abspath(output_path)



def keremberke_yolov8m_valorant_detection(image_path, output_path):
    from ultralyticsplus import YOLO, render_result
    import os

    model = YOLO('keremberke/yolov8m-valorant-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    results = model.predict(image_path)
    print(results[0].boxes)

    render = render_result(model=model, image=image_path, result=results[0])
    render.save(output_path)
    return os.path.abspath(output_path)



def keremberke_yolov8m_csgo_player_detection(image_path, output_path):
    from ultralyticsplus import YOLO, render_result
    import os
    model = YOLO('keremberke/yolov8m-csgo-player-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    results = model.predict(image_path)
    print(results[0].boxes)

    render = render_result(model=model, image=image_path, result=results[0])

    render.save(output_path)
    return os.path.abspath(output_path)



def keremberke_yolov8s_table_extraction(image_path, output_path):
    from ultralyticsplus import YOLO, render_result
    import os
    model = YOLO('keremberke/yolov8s-table-extraction')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    results = model.predict(image_path)
    print(results[0].boxes)

    render = render_result(model=model, image=image_path, result=results[0])
    render.save(output_path)
    return os.path.abspath(output_path)



def google_owlvit_large_patch14(image_path, prompt):
    import requests
    from PIL import Image
    import torch
    from transformers import OwlViTProcessor, OwlViTForObjectDetection

    processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")

    image = Image.open(image_path)

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    i = 0
    text = prompt[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    score_threshold = 0.1
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if score >= score_threshold:
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")



def keremberke_yolov8m_nlf_head_detection(image_path, output_path, conf=0.25, iou=0.45, agnostic_nms=False,
                                          max_det=1000):
    from ultralyticsplus import YOLO, render_result
    import os
    model = YOLO('keremberke/yolov8m-nlf-head-detection')
    model.overrides['conf'] = conf
    model.overrides['iou'] = iou
    model.overrides['agnostic_nms'] = agnostic_nms
    model.overrides['max_det'] = max_det

    results = model.predict(image_path)
    print(results[0].boxes)

    render = render_result(model=model, image=image_path, result=results[0])
    render.save(output_path)

    return os.path.abspath(output_path)



def keremberke_yolov8m_forklift_detection(image_path, output_path, conf=0.25, iou=0.45, agnostic_nms=False,
                                          max_det=1000):
    from ultralyticsplus import YOLO, render_result
    import os
    model = YOLO('keremberke/yolov8m-forklift-detection')
    model.overrides['conf'] = conf
    model.overrides['iou'] = iou
    model.overrides['agnostic_nms'] = agnostic_nms
    model.overrides['max_det'] = max_det

    results = model.predict(image_path)
    print(results[0].boxes)

    render = render_result(model=model, image=image_path, result=results[0])
    render.save(output_path)

    return os.path.abspath(output_path)



def google_owlvit_base_patch16(image_path, prompt):
    import torch
    import requests
    from PIL import Image
    from transformers import OwlViTProcessor, OwlViTForObjectDetection

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")

    image = Image.open(image_path)

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    return results



def keremberke_yolov8m_plane_detection(image_path, output_path, conf=0.25, iou=0.45, agnostic_nms=False, max_det=1000):
    from ultralyticsplus import YOLO, render_result
    import os
    model = YOLO('keremberke/yolov8m-plane-detection')
    model.overrides['conf'] = conf
    model.overrides['iou'] = iou
    model.overrides['agnostic_nms'] = agnostic_nms
    model.overrides['max_det'] = max_det

    results = model.predict(image_path)
    print(results[0].boxes)

    render = render_result(model=model, image=image_path, result=results[0])
    render.save(output_path)

    return os.path.abspath(output_path)



def keremberke_yolov8s_csgo_player_detection(image_path, output_path):
    from PIL import Image
    from io import BytesIO
    from urllib.request import urlopen
    from ultralyticsplus import YOLO, render_result
    import os
    model = YOLO('keremberke/yolov8s-csgo-player-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    image = Image.open(image_path)
    results = model.predict(image)

    print(results[0].boxes)

    render = render_result(model=model, image=image, result=results[0])
    render.save(output_path)

    return os.path.abspath(output_path)



def keremberke_yolov8m_blood_cell_detection(image_path, output_path, conf=0.25, iou=0.45, agnostic_nms=False,
                                            max_det=1000):
    from ultralyticsplus import YOLO, render_result
    import os

    model = YOLO('keremberke/yolov8m-blood-cell-detection')
    model.overrides['conf'] = conf
    model.overrides['iou'] = iou
    model.overrides['agnostic_nms'] = agnostic_nms
    model.overrides['max_det'] = max_det

    results = model.predict(image_path)
    print(results[0].boxes)

    render = render_result(model=model, image=image_path, result=results[0])
    render.save(output_path)

    return os.path.abspath(output_path)



def keremberke_yolov8s_hard_hat_detection(image_path, output_path):
    from ultralyticsplus import YOLO, render_result
    import os
    model = YOLO('keremberke/yolov8s-hard-hat-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    results = model.predict(image_path)
    print(results[0].boxes)

    render = render_result(model=model, image=image_path, result=results[0])
    render.save(output_path)

    return os.path.abspath(output_path)



def fcakyon_yolov5s_v7_0(image_path, output_path, conf=0.25, iou=0.45, agnostic=False, multi_label=False, max_det=1000):
    import yolov5
    import os
    model = yolov5.load('fcakyon/yolov5s-v7.0')
    model.conf = conf
    model.iou = iou
    model.agnostic = agnostic
    model.multi_label = multi_label
    model.max_det = max_det

    results = model(image_path)

    predictions = results.pred[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    results.show()

    results.save(save_dir=output_path)

    return os.path.abspath(output_path)



def keremberke_yolov8n_table_extraction(image_path, output_path, conf=0.25, iou=0.45, agnostic_nms=False, max_det=1000):
    from ultralyticsplus import YOLO, render_result
    import os
    model = YOLO('keremberke/yolov8n-table-extraction')
    model.overrides['conf'] = conf
    model.overrides['iou'] = iou
    model.overrides['agnostic_nms'] = agnostic_nms
    model.overrides['max_det'] = max_det

    results = model.predict(image_path)
    print(results[0].boxes)

    render = render_result(model=model, image=image_path, result=results[0])
    render.save(output_path)

    return os.path.abspath(output_path)



def clipseg_rd64_refined(image_path, prompt):
    from PIL import Image
    from transformers import AutoProcessor, CLIPSegModel

    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

    image = Image.open(image_path)

    inputs = processor(
        text=[prompt], images=image, return_tensors="pt", padding=True
    )

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)

    return probs



def keremberke_yolov8n_csgo_player_detection(image_path, output_path, conf=0.25, iou=0.45, agnostic_nms=False,
                                             max_det=1000):
    from ultralyticsplus import YOLO, render_result
    import os
    model = YOLO('keremberke/yolov8n-csgo-player-detection')

    model.overrides['conf'] = conf
    model.overrides['iou'] = iou
    model.overrides['agnostic_nms'] = agnostic_nms
    model.overrides['max_det'] = max_det

    results = model.predict(image_path)

    print(results[0].boxes)
    render = render_result(model=model, image=image_path, result=results[0])
    render.save(output_path)

    return os.path.abspath(output_path)



def keremberke_yolov5s_license_plate(image_path, output_path, size=640, augment=False, conf=0.25, iou=0.45,
                                     agnostic=False,
                                     multi_label=False, max_det=1000):
    import yolov5
    import os

    model = yolov5.load('keremberke/yolov5s-license-plate')
    model.conf = conf
    model.iou = iou
    model.agnostic = agnostic
    model.multi_label = multi_label
    model.max_det = max_det

    results = model(image_path, size=size, augment=augment)
    predictions = results.pred[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    results.show()

    results.save(output_path)

    return boxes, scores, categories, os.path.abspath(output_path)



def openmmlab_upernet_convnext_small(image_path):
    from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
    from PIL import Image

    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

    image = Image.open(image_path).convert("RGB")

    inputs = image_processor(images=image, return_tensors="pt")

    outputs = model(**inputs)

    logits = outputs.logits

    return logits



def keremberke_yolov8n_blood_cell_detection(image_path, output_path, conf=0.25, iou=0.45, agnostic_nms=False,
                                            max_det=1000):
    from ultralyticsplus import YOLO, render_result
    import os
    model = YOLO('keremberke/yolov8n-blood-cell-detection')
    model.overrides['conf'] = conf
    model.overrides['iou'] = iou
    model.overrides['agnostic_nms'] = agnostic_nms
    model.overrides['max_det'] = max_det

    results = model.predict(image_path)
    print(results[0].boxes)

    render = render_result(model=model, image=image_path, result=results[0])
    render.save(output_path)

    return os.path.abspath(output_path)



def nvidia_segformer_b0_finetuned_ade_512_512(image_path):
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    from PIL import Image

    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    return logits



def nvidia_segformer_b5_finetuned_ade_640_640(image_path):
    from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
    from PIL import Image

    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-512-512")

    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    return logits



def nvidia_segformer_b2_finetuned_cityscapes_1024_1024(image_path):
    from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
    from PIL import Image

    feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b2-finetuned-cityscapes-1024-1024')
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b2-finetuned-cityscapes-1024-1024')

    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits

    return logits



def nvidia_segformer_b0_finetuned_cityscapes_1024_1024(image_path):
    from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
    from PIL import Image
    import requests

    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")

    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    return logits



def facebook_detr_resnet_50_panoptic(image_path):
    import io
    import requests
    from PIL import Image
    import torch
    import numpy as np
    from transformers import DetrFeatureExtractor, DetrForSegmentation
    from transformers.models.detr.feature_extraction_detr import rgb_to_id

    image = Image.open(image_path)
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50-panoptic')
    model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-50-panoptic')
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    processed_sizes = torch.as_tensor(inputs['pixel_values'].shape[-2:]).unsqueeze(0)
    result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]
    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
    panoptic_seg_id = rgb_to_id(panoptic_seg)

    return panoptic_seg_id



def facebook_maskformer_swin_base_coco(image_path):
    from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
    from PIL import Image
    import requests

    feature_extractor = MaskFormerFeatureExtractor.from_pretrained('facebook/maskformer-swin-base-coco')
    model = MaskFormerForInstanceSegmentation.from_pretrained('facebook/maskformer-swin-base-coco')

    image = Image.open(image_path)

    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)

    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    predicted_panoptic_map = result['segmentation']

    return predicted_panoptic_map



def mattmdjaga_segformer_b2_clothes(image_path):
    from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
    from PIL import Image
    import matplotlib.pyplot as plt
    import torch.nn as nn
    import os
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]
    plt.imshow(pred_seg)



def facebook_mask2former_swin_base_coco_panoptic(image_path):
    import requests
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    processor = AutoImageProcessor.from_pretrained('facebook/mask2former-swin-base-coco-panoptic')
    model = Mask2FormerForUniversalSegmentation.from_pretrained('facebook/mask2former-swin-base-coco-panoptic')

    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    predicted_panoptic_map = result['segmentation']

    return predicted_panoptic_map



def facebook_mask2former_swin_large_cityscapes_semantic(image_path):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    processor = AutoImageProcessor.from_pretrained('facebook/mask2former-swin-large-cityscapes-semantic')
    model = Mask2FormerForUniversalSegmentation.from_pretrained('facebook/mask2former-swin-large-cityscapes-semantic')

    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits
    predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    return class_queries_logits, masks_queries_logits, predicted_semantic_map



def facebook_maskformer_swin_large_ade(image_path):
    from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
    from PIL import Image
    import requests

    image = Image.open(image_path)
    processor = MaskFormerImageProcessor.from_pretrained('facebook/maskformer-swin-large-ade')
    inputs = processor(images=image, return_tensors='pt')
    model = MaskFormerForInstanceSegmentation.from_pretrained('facebook/maskformer-swin-large-ade')
    outputs = model(**inputs)
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits
    predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    return class_queries_logits, masks_queries_logits, predicted_semantic_map



def shi_labs_oneformer_ade20k_swin_large(image_path):
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    from PIL import Image

    image = Image.open(image_path)

    processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large")

    semantic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
    semantic_outputs = model(**semantic_inputs)

    predicted_semantic_map = \
    processor.post_process_semantic_segmentation(semantic_outputs, target_sizes=[image.size[::-1]])[0]

    return predicted_semantic_map



def facebook_mask2former_swin_large_coco_panoptic(image_path):
    import torch
    import requests
    from PIL import Image
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")

    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    predicted_panoptic_map = result["segmentation"]

    return predicted_panoptic_map



def facebook_mask2former_swin_small_coco_instance(image_path):
    import torch
    import requests
    from PIL import Image
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    processor = AutoImageProcessor.from_pretrained('facebook/mask2former-swin-small-coco-instance')
    model = Mask2FormerForUniversalSegmentation.from_pretrained('facebook/mask2former-swin-small-coco-instance')

    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    result = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    predicted_instance_map = result['segmentation']

    return predicted_instance_map



def shi_labs_oneformer_ade20k_swin_tiny(image_path, semantic=True, instance=True, panoptic=True):
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    from PIL import Image
    import requests

    image = Image.open(image_path)

    processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
    model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")

    pt = "pt"

    semantic_inputs = processor(images=image, task_inputs=[semantic], return_tensors=pt)
    semantic_outputs = model(**semantic_inputs)

    predicted_semantic_map = \
    processor.post_process_semantic_segmentation(semantic_outputs, target_sizes=[image.size[::-1]])[0]

    instance_inputs = processor(images=image, task_inputs=[instance], return_tensors=pt)
    instance_outputs = model(**instance_inputs)

    predicted_instance_map = \
    processor.post_process_instance_segmentation(instance_outputs, target_sizes=[image.size[::-1]])[0]["segmentation"]

    panoptic_inputs = processor(images=image, task_inputs=[panoptic], return_tensors=pt)
    panoptic_outputs = model(**panoptic_inputs)

    predicted_panoptic_map = \
    processor.post_process_panoptic_segmentation(panoptic_outputs, target_sizes=[image.size[::-1]])[0]["segmentation"]

    return predicted_semantic_map, predicted_instance_map, predicted_panoptic_map



def keremberke_yolov8m_building_segmentation(image_path):
    from ultralyticsplus import YOLO, render_result

    model = YOLO('keremberke/yolov8m-building-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    results = model.predict(image_path)

    return results



def nvidia_segformer_b5_finetuned_cityscapes_1024_1024(image_path):
    from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
    from PIL import Image
    import requests

    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to("cuda")

    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    logits = outputs.logits

    return logits



def facebook_mask2former_swin_tiny_coco_instance(image_path):
    from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
    from PIL import Image
    import requests
    from diffusers.utils import load_image

    feature_extractor = MaskFormerFeatureExtractor.from_pretrained('facebook/mask2former-swin-tiny-coco-instance')
    model = MaskFormerForInstanceSegmentation.from_pretrained('facebook/mask2former-swin-tiny-coco-instance').to('cuda')

    image = load_image(image_path)
    inputs = feature_extractor(images=image, return_tensors='pt').to('cuda')
    outputs = model(**inputs)

    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    predicted_panoptic_map = result['segmentation']

    return class_queries_logits, masks_queries_logits, predicted_panoptic_map



def facebook_maskformer_swin_base_ade(image_path):
    from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
    from PIL import Image
    import requests
    from diffusers.utils import load_image

    feature_extractor = MaskFormerFeatureExtractor.from_pretrained('facebook/maskformer-swin-base-ade')
    model = MaskFormerForInstanceSegmentation.from_pretrained('facebook/maskformer-swin-base-ade', ignore_mismatched_sizes=True).to('cuda')

    image = load_image(image_path)
    inputs = feature_extractor(images=image, return_tensors='pt').to('cuda')
    outputs = model(**inputs)

    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    predicted_panoptic_map = result['segmentation']

    return class_queries_logits, masks_queries_logits, predicted_panoptic_map



def keremberke_yolov8m_pcb_defect_segmentation(image_path):
    from ultralyticsplus import YOLO, render_result

    model = YOLO('keremberke/yolov8m-pcb-defect-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    results = model.predict(image_path)

    return results



def facebook_maskformer_swin_tiny_coco(image_path):
    from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
    from PIL import Image
    import requests
    from diffusers.utils import load_image

    feature_extractor = MaskFormerFeatureExtractor.from_pretrained('facebook/maskformer-swin-tiny-coco')
    model = MaskFormerForInstanceSegmentation.from_pretrained('facebook/maskformer-swin-tiny-coco').to('cuda')

    image = load_image(image_path)
    inputs = feature_extractor(images=image, return_tensors='pt').to('cuda')
    outputs = model(**inputs)

    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    predicted_panoptic_map = result['segmentation']

    return class_queries_logits, masks_queries_logits, predicted_panoptic_map



def yolov8m_pothole_segmentation(image_path):
    from ultralyticsplus import YOLO, render_result

    model = YOLO('keremberke/yolov8m-pothole-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    results = model.predict(image_path)

    return results



def keremberke_yolov8s_building_segmentation(image_path):
    from ultralyticsplus import YOLO, render_result

    model = YOLO('keremberke/yolov8s-building-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    results = model.predict(image_path)

    return results



def yolov8s_pothole_segmentation(image_path):
    from ultralyticsplus import YOLO, render_result

    model = YOLO('keremberke/yolov8s-pothole-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    results = model.predict(image_path)

    return results



def yolov8n_pothole_segmentation(image_path):
    from ultralyticsplus import YOLO, render_result

    model = YOLO('keremberke/yolov8n-pothole-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    results = model.predict(image_path)

    return results



def keremberke_yolov8n_pcb_defect_segmentation(image_path):
    from ultralyticsplus import YOLO, render_result

    model = YOLO('keremberke/yolov8n-pcb-defect-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    results = model.predict(image_path)

    return results



def lambdalabs_sd_image_variations_diffusers(original_image_path, output_path, guidance_scale=3):
    from diffusers import StableDiffusionImageVariationPipeline
    from PIL import Image
    import torchvision.transforms as transforms
    import os

    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        'lambdalabs/sd-image-variations-diffusers',
        revision='v2.0',
    ).to('cuda')

    im = Image.open(original_image_path)
    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
        ),
        transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711]
        ),
    ])
    inp = tform(im).to("cuda").unsqueeze(0)
    out = sd_pipe(inp, guidance_scale=guidance_scale)
    out['images'][0].save(output_path)

    return os.path.abspath(output_path)



def lllyasviel_sd_controlnet_openpose(control_image_path, prompt, output_image_path) -> None:
    import torch
    import os
    from huggingface_hub import HfApi
    from pathlib import Path
    from diffusers.utils import load_image
    from PIL import Image
    import numpy as np
    from controlnet_aux import PidiNetDetector, HEDdetector, OpenposeDetector, LineartDetector
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    checkpoint = "lllyasviel/sd-controlnet-openpose"

    image = load_image(control_image_path)
    # image = image.resize((512, 512))

    processor =  OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    # processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

    control_image = processor(image)

    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)

    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    image.save(output_image_path)

    return os.path.abspath(output_image_path)



def lllyasviel_sd_controlnet_hed(control_image_path, prompt, output_image_path) -> None:
    import torch
    import os
    from huggingface_hub import HfApi
    from pathlib import Path
    from diffusers.utils import load_image
    from PIL import Image
    import numpy as np
    from controlnet_aux import PidiNetDetector, HEDdetector, OpenposeDetector, LineartDetector
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    checkpoint = "lllyasviel/sd-controlnet-hed"

    image = load_image(control_image_path)
    # image = image.resize((512, 512))

    processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
    # processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

    control_image = processor(image)

    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)

    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    image.save(output_image_path)

    return os.path.abspath(output_image_path)



def lllyasviel_sd_controlnet_seg(control_image_path, prompt, output_image_path) -> None:
    import torch
    import os
    from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
    from huggingface_hub import HfApi
    from pathlib import Path
    from diffusers.utils import load_image
    from PIL import Image
    import numpy as np
    from controlnet_aux import PidiNetDetector, HEDdetector, OpenposeDetector, LineartDetector
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )
    palette = np.asarray([
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ])

    checkpoint = "lllyasviel/sd-controlnet-seg"

    image = load_image(control_image_path)
    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

    pixel_values = image_processor(image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = image_segmentor(pixel_values)

    seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    color_seg = color_seg.astype(np.uint8)

    control_image = Image.fromarray(color_seg)


    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)

    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    image.save(output_image_path)

    return os.path.abspath(output_image_path)



def lllyasviel_sd_controlnet_depth(image_path, prompt,  output_path):
    from transformers import pipeline
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    from PIL import Image
    import numpy as np
    import torch
    from diffusers.utils import load_image
    import os

    depth_estimator = pipeline('depth-estimation')

    image = load_image(image_path)
    image = depth_estimator(image)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-depth', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet,
                                                             safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    image = pipe(prompt, image, num_inference_steps=20).images[0]
    image.save(output_path)

    return os.path.abspath(output_path)



def lllyasviel_sd_controlnet_scribble(control_image_path, prompt, output_image_path) -> None:
    import torch
    import os
    from huggingface_hub import HfApi
    from pathlib import Path
    from diffusers.utils import load_image
    from PIL import Image
    import numpy as np
    from controlnet_aux import PidiNetDetector, HEDdetector, OpenposeDetector, LineartDetector
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    checkpoint = "lllyasviel/sd-controlnet-scribble"

    image = load_image(control_image_path)
    # image = image.resize((512, 512))

    processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
    # processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

    control_image = processor(image)

    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)

    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    image.save(output_image_path)

    return os.path.abspath(output_image_path)



def lllyasviel_control_v11p_sd15_canny(control_image_path, prompt, output_image_path ,low_threshold=100, high_threshold=200):
    import torch
    import os
    from huggingface_hub import HfApi
    from pathlib import Path
    from diffusers.utils import load_image
    import numpy as np
    import cv2
    from PIL import Image
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    checkpoint = "lllyasviel/control_v11p_sd15_canny"
    image = load_image(control_image_path)
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)

    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    generator = torch.manual_seed(33)
    image = pipe(
        prompt,
        num_inference_steps=20,
        generator=generator,
        image=control_image,
    ).images[0]
    image.save(output_image_path)
    return os.path.abspath(output_image_path)


def lllyasviel_control_v11p_sd15_lineart(control_image_path, prompt, output_image_path) -> None:
    import torch
    import os
    from huggingface_hub import HfApi
    from pathlib import Path
    from diffusers.utils import load_image
    from PIL import Image
    import numpy as np
    from controlnet_aux import PidiNetDetector, HEDdetector, OpenposeDetector, LineartDetector
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    checkpoint = "lllyasviel/control_v11p_sd15_lineart"

    image = load_image(control_image_path)
    image = image.resize((512, 512))

    # processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
    processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

    control_image = processor(image)

    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)

    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    image.save(output_image_path)

    return os.path.abspath(output_image_path)



def lllyasviel_sd_controlnet_normal(image_path, prompt,  output_path):
    from PIL import Image
    from transformers import pipeline
    import numpy as np
    import os
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    import torch
    import cv2
    from diffusers.utils import load_image

    # Load image
    image = load_image(image_path).convert("RGB")

    # Depth estimation
    depth_estimator = pipeline("depth-estimation", model='Intel/dpt-hybrid-midas')
    image = depth_estimator(image)["predicted_depth"][0]
    image = image.numpy()

    # Normalize depth image
    image_depth = image.copy()
    image_depth -= np.min(image_depth)
    image_depth /= np.max(image_depth)

    # Threshold for background
    bg_threshold = 0.4

    # Apply Sobel filter
    x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    x[image_depth < bg_threshold] = 0
    y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    y[image_depth < bg_threshold] = 0
    z = np.ones_like(x) * np.pi * 2.0
    image = np.stack([x, y, z], axis=2)
    image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
    image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image)

    # Load ControlNet model
    controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-normal",
                                                 torch_dtype=torch.float16)

    # Load StableDiffusionControlNetPipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet,
                                                             safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    # Apply controlnet to image
    image = pipe(prompt, image, num_inference_steps=20).images[0]

    # Save output image
    image.save(output_path)

    return os.path.abspath(output_path)



def llllyasviel_control_v11p_sd15_scribble(control_image_path, prompt, output_image_path) -> None:
    import torch
    import os
    from huggingface_hub import HfApi
    from pathlib import Path
    from diffusers.utils import load_image
    from PIL import Image
    import numpy as np
    from controlnet_aux import PidiNetDetector, HEDdetector, OpenposeDetector
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    checkpoint = "lllyasviel/control_v11p_sd15_scribble"

    image = load_image(control_image_path)

    # processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
    processor = HEDdetector.from_pretrained('lllyasviel/Annotators')

    control_image = processor(image, scribble=True)

    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)

    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    image.save(output_image_path)

    return os.path.abspath(output_image_path)



def lllyasviel_control_v11p_sd15_openpose(control_image_path, prompt, output_image_path) -> None:
    import torch
    import os
    from huggingface_hub import HfApi
    from pathlib import Path
    from diffusers.utils import load_image
    from PIL import Image
    import numpy as np
    from controlnet_aux import PidiNetDetector, HEDdetector, OpenposeDetector
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    checkpoint = "lllyasviel/control_v11p_sd15_openpose"

    image = load_image(control_image_path)

    # processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
    processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

    control_image = processor(image, safe=True)

    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)

    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    image.save(output_image_path)

    return os.path.abspath(output_image_path)



def lllyasviel_control_v11e_sd15_ip2p(control_image_path, prompt, output_image_path) -> None:
    import torch
    import os
    from huggingface_hub import HfApi
    from pathlib import Path
    from diffusers.utils import load_image
    from PIL import Image
    import numpy as np
    from controlnet_aux import PidiNetDetector, HEDdetector
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    checkpoint = "lllyasviel/control_v11e_sd15_ip2p"

    image = load_image(control_image_path)

    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)

    image = pipe(prompt, num_inference_steps=30, generator=generator, image=image).images[0]
    image.save(output_image_path)

    return os.path.abspath(output_image_path)


def lllyasviel_control_v11p_sd15_softedge(control_image_path: str, prompt: str, output_image_path: str) -> None:
    import torch
    import os
    from huggingface_hub import HfApi
    from pathlib import Path
    from diffusers.utils import load_image
    from PIL import Image
    import numpy as np
    from controlnet_aux import PidiNetDetector, HEDdetector
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    checkpoint = "lllyasviel/control_v11p_sd15_softedge"

    image = load_image(control_image_path)

    # processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
    processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')

    control_image = processor(image, safe=True)
    control_image.save(control_image_path)

    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)

    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    image.save(output_image_path)

    return os.path.abspath(output_image_path)



def swin2SR_lightweight_x2_64(input_image, output_path):
    import torch
    from PIL import Image
    from torchvision.transforms import ToTensor
    from transformers import Swin2SRForImageSuperResolution, AutoImageProcessor
    import os
    import numpy as np

    # Load the pre-trained model and processor
    model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-lightweight-x2-64").to("cuda")
    processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-lightweight-x2-64")

    # Preprocess the input image
    image = Image.open(input_image)
    image_tensor = ToTensor()(image).unsqueeze(0)

    # Run the model to generate the super-resolution image
    inputs = processor(image_tensor, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)

    # Postprocess the output image
    output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.moveaxis(output, source=0, destination=-1)
    output = (output * 255.0).round().astype(np.uint8)
    sr_image = Image.fromarray(output)
    sr_image.save(output_path)

    return os.path.abspath(output_path)



def lllyasviel_control_v11p_sd15_mlsd(image_path: str, prompt: str, output_path: str) -> None:
    import torch
    import os
    from huggingface_hub import HfApi
    from pathlib import Path
    from diffusers.utils import load_image
    from PIL import Image
    import numpy as np
    from controlnet_aux import MLSDdetector
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    checkpoint = "lllyasviel/control_v11p_sd15_mlsd"
    image = load_image(image_path)
    prompt = prompt
    processor = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    control_image = processor(image)
    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16
    ).to('cuda')
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    generator = torch.manual_seed(0)
    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def lllyasviel_control_v11p_sd15_normalbae(control_image_path: str, prompt: str,
                                           output_image_path: str) -> None:
    import torch
    import os
    from huggingface_hub import HfApi
    from pathlib import Path
    from diffusers.utils import load_image
    from PIL import Image
    import numpy as np
    from controlnet_aux import NormalBaeDetector
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    image = load_image(control_image_path)
    processor = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
    control_image = processor(image)
    control_image.save(control_image_path)
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_normalbae', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    generator = torch.manual_seed(33)
    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    image.save(output_image_path)

    return os.path.abspath(output_image_path)



def GreeneryScenery_SheepsControlV3(image_path, text_guidance=None):
    from transformers import pipeline

    model = pipeline('image-to-image', model='GreeneryScenery/SheepsControlV3')
    result = model({'image': image_path, 'text_guidance': text_guidance})

    return result



def GreeneryScenery_SheepsControlV5(input_text, model_name, tokenizer_name):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Tokenize the input text
    inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")

    # Make predictions
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=1)

    return predictions



def google_maxim_s3_deblurring_gopro(image_path):
    from huggingface_hub import from_pretrained_keras
    from PIL import Image
    from diffusers.utils import load_image
    import tensorflow as tf
    import numpy as np
    import requests

    image = load_image(image_path)
    image = np.array(image)
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (256, 256))

    model = from_pretrained_keras("google/maxim-s3-deblurring-gopro").to("cuda")

    # put image on GPU
    image = tf.expand_dims(image, 0).to("cuda")
    predictions = model.predict(image)

    return predictions



def lllyasviel_control_v11p_sd15s2_lineart_anime(control_image_path, prompt, output_path):
    import torch
    import os
    from huggingface_hub import HfApi
    from pathlib import Path
    from diffusers.utils import load_image
    from PIL import Image
    import numpy as np
    from controlnet_aux import LineartAnimeDetector
    from transformers import CLIPTextModel
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    checkpoint = "lllyasviel/control_v11p_sd15s2_lineart_anime"
    image = load_image(control_image_path)
    image = image.resize((512, 512))
    processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    control_image = processor(image)
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder",
                                                 num_hidden_layers=11, torch_dtype=torch.float16)
    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                             text_encoder=text_encoder, controlnet=controlnet,
                                                             torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    generator = torch.manual_seed(0)
    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    image.save(output_path)

    return os.path.abspath(output_path)



def lllyasviel_control_v11p_sd15_inpaint(original_image_path, mask_image_path, prompt, output_path):
    from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, DDIMScheduler
    from diffusers.utils import load_image
    import numpy as np
    import torch
    import os

    '''
    description: This function uses the pretrained model from diffusers to inpaint the image

    original_image_path: path to the original image
    mask_image_path: path to the mask image
    prompt: prompt for the image
    output_path: path to the output image
    '''

    init_image = load_image(original_image_path)
    init_image = init_image.resize((512, 512))

    generator = torch.Generator(device="cuda").manual_seed(1)

    mask_image = load_image(mask_image_path)
    mask_image = mask_image.resize((512, 512))

    def make_inpaint_condition(image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image

    control_image = make_inpaint_condition(init_image, mask_image).to("cuda")

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # generate image
    image = pipe(
        prompt,
        num_inference_steps=20,
        generator=generator,
        eta=1.0,
        image=init_image,
        mask_image=mask_image,
        control_image=control_image,
    ).images[0]

    image.save(output_path)

    return os.path.abspath(output_path)



def google_ddpm_celebahq_256(output_path):
    import torch
    import os
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('google/ddpm-celebahq-256').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def google_ddpm_ema_celebahq_256(output_path):
    import torch
    import os
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('google/ddpm-ema-celebahq-256').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def google_ddpm_ema_church_256(output_path):
    import torch
    import os
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('google/ddpm-ema-church-256').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def CompVis_ldm_celebahq_256(output_path):
    from diffusers import DiffusionPipeline
    import os
    sde_ve = DiffusionPipeline.from_pretrained("CompVis/ldm-celebahq-256").to("cuda")
    image = sde_ve()[0]
    image[0].save(output_path)
    return os.path.abspath(output_path)



def google_ddpm_church_256(output_path):
    import torch
    import os
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('google/ddpm-church-256').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def google_ncsnpp_celebahq_256(output_path):
    from diffusers import DiffusionPipeline
    import os
    sde_ve = DiffusionPipeline.from_pretrained("google/ncsnpp-celebahq-256").to("cuda")
    image = sde_ve()[0]
    image[0].save(output_path)
    return os.path.abspath(output_path)



def johnowhitaker_sd_class_wikiart_from_bedrooms(output_path):
    import torch
    import os
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('johnowhitaker/sd-class-wikiart-from-bedrooms').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def ddpm_cifar10_32(output_path):
    import torch
    import os
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('google/ddpm-cifar10-32').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def google_ddpm_ema_bedroom_256(output_path):
    import torch
    import os
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('google/ddpm-ema-bedroom-256').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def google_ncsnpp_ffhq_1024(output_path):
    from diffusers import DiffusionPipeline
    import os
    sde_ve = DiffusionPipeline.from_pretrained("google/ncsnpp-ffhq-1024").to("cuda")
    image = sde_ve()[0]
    image[0].save(output_path)
    return os.path.abspath(output_path)



def ocariz_universe_1400(output_path):
    import torch
    import os
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('ocariz/universe_1400').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def minecraft_skin_diffusion_v2(output_path):
    import torch
    import os
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('WiNE-iNEFF/Minecraft-Skin-Diffusion-V2').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def minecraft_skin_diffusion(output_path):
    import torch
    import os
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('WiNE-iNEFF/Minecraft-Skin-Diffusion').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def sd_class_butterflies_32(output_path):
    import torch
    import os
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('clp/sd-class-butterflies-32').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def MFawad_sd_class_butterflies_32(output_path):
    import torch
    import os
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('MFawad/sd-class-butterflies-32').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)
    return os.path.abspath(output_path)


def google_ncsnpp_ffhq_256(output_path):
    from diffusers import DiffusionPipeline
    import os
    sde_ve = DiffusionPipeline.from_pretrained("google/ncsnpp-ffhq-256").to("cuda")
    image = sde_ve()[0]
    image[0].save(output_path)
    return os.path.abspath(output_path)



def google_ddpm_ema_cat_256(output_path):
    import torch
    import os
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('google/ddpm-ema-cat-256').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def ocariz_butterfly_200(output_path):
    import torch
    import os
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('ocariz/butterfly_200').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def ntrant7_sd_class_butterflies_32(output_path):
    import torch
    import os
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('ntrant7/sd-class-butterflies-32').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def apocalypse_19_shoe_generator(output_path):
    import torch
    import os
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('Apocalypse-19/shoe-generator').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def pravsels_ddpm_ffhq_vintage_finetuned_vintage_3epochs(output_path):
    import torch
    import os
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('pravsels/ddpm-ffhq-vintage-finetuned-vintage-3epochs').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def microsoft_xclip_base_patch32(video_path, labels):
    import av
    import torch
    import numpy as np

    from transformers import AutoProcessor, AutoModel
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    def read_video_pyav(container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        '''
        Sample a given number of frame indices from the video.
        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.
        Returns:
            indices (`List[int]`): List of sampled frame indices
        '''
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices

    container = av.open(video_path)

    # sample 8 frames
    frame_sample_rate = container.streams.video[0].frames // 8
    indices = sample_frame_indices(clip_len=8, frame_sample_rate=frame_sample_rate, seg_len=container.streams.video[0].frames)
    video = read_video_pyav(container, indices)

    processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
    model = AutoModel.from_pretrained("microsoft/xclip-base-patch32").to("cuda")

    inputs = processor(
        text=labels,
        videos=list(video),
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_video = outputs.logits_per_video  # this is the video-text similarity score
    probs = logits_per_video.softmax(dim=1)

    return probs



def myunus1_diffmodels_galaxies_scratchbook(output_path):
    import torch
    import os
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('myunus1/diffmodels_galaxies_scratchbook').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def utyug1_sd_class_butterflies_32(output_path):
    import torch
    import os
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('utyug1/sd-class-butterflies-32').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def sd_class_pandas_32():
    import torch
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('schdoel/sd-class-AFHQ-32')
    image = pipeline().images[0]
    return image



def facebook_timesformer_base_finetuned_k400(video):
    from transformers import AutoImageProcessor, TimesformerForVideoClassification
    import numpy as np
    import torch

    video = list(np.random.randn(8, 3, 224, 224))
    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
    inputs = processor(video, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])



def MCG_NJU_videomae_base(video_path):
    from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
    from transformers import VideoMAEFeatureExtractor, VideoMAEForPreTraining
    import numpy as np
    import torch
    from decord import VideoReader, cpu

    # video = list(np.random.randn(num_frames, 3, 224, 224))

    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)

    video = videoreader.get_batch([i for i in range(len(videoreader))]).asnumpy()

    video = list(video.transpose(0, 3, 1, 2))

    preprocessing_pipeline = Compose([
        Resize((224, 224)),
        CenterCrop((224, 224)),
    ])

    frames = [preprocessing_pipeline(torch.tensor(frame, dtype=torch.float16)) for frame in video]
    frames = np.stack([frame / 255 for frame in frames])

    # sample 16 frames from the video
    frames = frames[np.linspace(0, len(frames) - 1, 16, dtype=int)]

    feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-ssv2")
    model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base-ssv2")

    pixel_values = feature_extractor(list(frames), return_tensors="pt").pixel_values

    # feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-short-ssv2")
    # model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base-short-ssv2")
    #
    # pixel_values = feature_extractor(video, return_tensors="pt").pixel_values

    num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
    seq_length = (16 // model.config.tubelet_size) * num_patches_per_frame

    bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()

    outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
    loss = outputs.loss
    return loss



def facebook_timesformer_base_finetuned_k600(video_path):
    from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose

    from transformers import AutoImageProcessor, TimesformerForVideoClassification
    import numpy as np
    import torch
    from decord import VideoReader, cpu

    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)

    video = videoreader.get_batch([i for i in range(len(videoreader))]).asnumpy()

    video = list(video.transpose(0, 3, 1, 2))

    preprocessing_pipeline = Compose([
        Resize((448, 448)),
        CenterCrop((448, 448)),
    ])

    frames = [preprocessing_pipeline(torch.tensor(frame, dtype=torch.float16)) for frame in video]
    frames = np.stack([frame / 255 for frame in frames])

    # sample 16 frames from the video
    frames = frames[np.linspace(0, len(frames) - 1, 16, dtype=int)]

    processor = AutoImageProcessor.from_pretrained('facebook/timesformer-base-finetuned-k600')
    model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-base-finetuned-k600').to('cuda')
    inputs = processor(list(frames), return_tensors='pt').to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]



def MCG_NJU_videomae_base_finetuned_kinetics(video_path):
    from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    import numpy as np
    import torch
    from decord import VideoReader, cpu

    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)

    video = videoreader.get_batch([i for i in range(len(videoreader))]).asnumpy()

    video = list(video.transpose(0, 3, 1, 2))

    preprocessing_pipeline = Compose([
        Resize((224, 224)),
        CenterCrop((224, 224)),
    ])

    frames = [preprocessing_pipeline(torch.tensor(frame, dtype=torch.float16)) for frame in video]
    frames = np.stack([frame / 255 for frame in frames])

    # sample 16 frames from the video
    frames = frames[np.linspace(0, len(frames) - 1, 16, dtype=int)]

    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics')
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics').to('cuda')
    inputs = processor(list(frames), return_tensors='pt').to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]



def facebook_timesformer_hr_finetuned_k400(video_path):
    from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose

    from transformers import AutoImageProcessor, TimesformerForVideoClassification
    import numpy as np
    import torch
    from decord import VideoReader, cpu

    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)

    video = videoreader.get_batch([i for i in range(len(videoreader))]).asnumpy()

    video = list(video.transpose(0, 3, 1, 2))

    preprocessing_pipeline = Compose([
        Resize((448, 448)),
        CenterCrop((448, 448)),
    ])

    frames = [preprocessing_pipeline(torch.tensor(frame, dtype=torch.float16)) for frame in video]
    frames = np.stack([frame / 255 for frame in frames])

    # sample 16 frames from the video
    frames = frames[np.linspace(0, len(frames) - 1, 16, dtype=int)]

    processor = AutoImageProcessor.from_pretrained('facebook/timesformer-hr-finetuned-k400')
    model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-hr-finetuned-k400').to('cuda')
    inputs = processor(list(frames), return_tensors='pt').to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]



def facebook_timesformer_base_finetuned_ssv2(video_path):
    from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose

    from transformers import AutoImageProcessor, TimesformerForVideoClassification
    import numpy as np
    import torch
    from decord import VideoReader, cpu

    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)

    video = videoreader.get_batch([i for i in range(len(videoreader))]).asnumpy()

    video = list(video.transpose(0, 3, 1, 2))

    preprocessing_pipeline = Compose([
        Resize((448, 448)),
        CenterCrop((448, 448)),
    ])

    frames = [preprocessing_pipeline(torch.tensor(frame, dtype=torch.float16)) for frame in video]
    frames = np.stack([frame / 255 for frame in frames])

    # sample 16 frames from the video
    frames = frames[np.linspace(0, len(frames) - 1, 16, dtype=int)]

    processor = AutoImageProcessor.from_pretrained('facebook/timesformer-base-finetuned-ssv2')
    model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-base-finetuned-ssv2').to('cuda')
    inputs = processor(list(frames), return_tensors='pt').to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]




def facebook_timesformer_hr_finetuned_ssv2(video_path):
    from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose

    from transformers import AutoImageProcessor, TimesformerForVideoClassification
    import numpy as np
    import torch
    from decord import VideoReader, cpu

    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)

    video = videoreader.get_batch([i for i in range(len(videoreader))]).asnumpy()

    video = list(video.transpose(0, 3, 1, 2))

    preprocessing_pipeline = Compose([
        Resize((448, 448)),
        CenterCrop((448, 448)),
    ])

    frames = [preprocessing_pipeline(torch.tensor(frame, dtype=torch.float16)) for frame in video]
    frames = np.stack([frame / 255 for frame in frames])

    # sample 16 frames from the video
    frames = frames[np.linspace(0, len(frames) - 1, 16, dtype=int)]

    processor = AutoImageProcessor.from_pretrained('facebook/timesformer-hr-finetuned-ssv2')
    model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-hr-finetuned-ssv2').to('cuda')
    inputs = processor(list(frames), return_tensors='pt').to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]



def videomae_large(video_path):
    from transformers import VideoMAEFeatureExtractor, VideoMAEForPreTraining
    import numpy as np
    import torch
    from decord import VideoReader, cpu
    from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose

    # video = list(np.random.randn(num_frames, 3, 224, 224))

    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)

    video = videoreader.get_batch([i for i in range(len(videoreader))]).asnumpy()

    video = list(video.transpose(0, 3, 1, 2))

    preprocessing_pipeline = Compose([
        Resize((224, 224)),
        CenterCrop((224, 224)),
    ])

    frames = [preprocessing_pipeline(torch.tensor(frame, dtype=torch.float16)) for frame in video]
    frames = np.stack([frame / 255 for frame in frames])

    # sample 16 frames from the video
    frames = frames[np.linspace(0, len(frames) - 1, 16, dtype=int)]

    feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-large")
    model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-large").to('cuda')

    pixel_values = feature_extractor(list(frames), return_tensors="pt").pixel_values.to('cuda')

    # feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-short-ssv2")
    # model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base-short-ssv2")
    #
    # pixel_values = feature_extractor(video, return_tensors="pt").pixel_values

    num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
    seq_length = (16 // model.config.tubelet_size) * num_patches_per_frame

    bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()

    outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
    loss = outputs.loss

    return loss



def MCG_NJU_videomae_base_finetuned_ssv2(video_path):
    from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose

    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    import numpy as np
    import torch
    from decord import VideoReader, cpu

    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)

    video = videoreader.get_batch([i for i in range(len(videoreader))]).asnumpy()

    video = list(video.transpose(0, 3, 1, 2))

    preprocessing_pipeline = Compose([
        Resize((224, 224)),
        CenterCrop((224, 224)),
    ])

    frames = [preprocessing_pipeline(torch.tensor(frame, dtype=torch.float16)) for frame in video]
    frames = np.stack([frame / 255 for frame in frames])

    # sample 16 frames from the video
    frames = frames[np.linspace(0, len(frames) - 1, 16, dtype=int)]

    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base-finetuned-ssv2')
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-ssv2').to('cuda')
    inputs = processor(list(frames), return_tensors='pt').to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]



def MCG_NJU_videomae_base_short(video_path):
    from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    import numpy as np
    import torch
    from decord import VideoReader, cpu

    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)

    video = videoreader.get_batch([i for i in range(len(videoreader))]).asnumpy()

    video = list(video.transpose(0, 3, 1, 2))

    preprocessing_pipeline = Compose([
        Resize((224, 224)),
        CenterCrop((224, 224)),
    ])

    frames = [preprocessing_pipeline(torch.tensor(frame, dtype=torch.float16)) for frame in video]
    frames = np.stack([frame / 255 for frame in frames])

    # sample 16 frames from the video
    frames = frames[np.linspace(0, len(frames) - 1, 16, dtype=int)]

    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base-short')
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-short').to('cuda')
    inputs = processor(list(frames), return_tensors='pt').to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]



def MCG_NJU_videomae_large_finetuned_kinetics(video_path):
    from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    import numpy as np
    import torch
    from decord import VideoReader, cpu

    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)

    video = videoreader.get_batch([i for i in range(len(videoreader))]).asnumpy()

    video = list(video.transpose(0, 3, 1, 2))

    preprocessing_pipeline = Compose([
        Resize((224, 224)),
        CenterCrop((224, 224)),
    ])

    frames = [preprocessing_pipeline(torch.tensor(frame, dtype=torch.float16)) for frame in video]
    frames = np.stack([frame / 255 for frame in frames])

    # sample 16 frames from the video
    frames = frames[np.linspace(0, len(frames) - 1, 16, dtype=int)]

    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-large-finetuned-kinetics')
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-large-finetuned-kinetics').to('cuda')
    inputs = processor(list(frames), return_tensors='pt').to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]



def MCG_NJU_videomae_base_short_finetuned_kinetics(video_path):
    from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    import numpy as np
    import torch
    from decord import VideoReader, cpu

    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)

    video = videoreader.get_batch([i for i in range(len(videoreader))]).asnumpy()

    video = list(video.transpose(0, 3, 1, 2))

    preprocessing_pipeline = Compose([
        Resize((224, 224)),
        CenterCrop((224, 224)),
    ])

    frames = [preprocessing_pipeline(torch.tensor(frame, dtype=torch.float16)) for frame in video]
    frames = np.stack([frame / 255 for frame in frames])

    # sample 16 frames from the video
    frames = frames[np.linspace(0, len(frames) - 1, 16, dtype=int)]

    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base-short-finetuned-kinetics')
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-short-finetuned-kinetics').to('cuda')
    inputs = processor(list(frames), return_tensors='pt').to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]



def videomae_base_finetuned_RealLifeViolenceSituations_subset(video_path):
    from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    import numpy as np
    import torch
    from decord import VideoReader, cpu

    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)

    video = videoreader.get_batch([i for i in range(len(videoreader))]).asnumpy()

    video = list(video.transpose(0, 3, 1, 2))

    preprocessing_pipeline = Compose([
        Resize((224, 224)),
        CenterCrop((224, 224)),
    ])

    frames = [preprocessing_pipeline(torch.tensor(frame, dtype=torch.float16)) for frame in video]
    frames = np.stack([frame / 255 for frame in frames])

    # sample 16 frames from the video
    frames = frames[np.linspace(0, len(frames) - 1, 16, dtype=int)]

    processor = VideoMAEImageProcessor.from_pretrained('videomae/base-finetuned-RealLifeViolenceSituations-subset')
    model = VideoMAEForVideoClassification.from_pretrained('videomae/base-finetuned-RealLifeViolenceSituations-subset').to('cuda')
    inputs = processor(list(frames), return_tensors='pt').to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]



def fcakyon_timesformer_large_finetuned_k400(video_path):
    from transformers import AutoImageProcessor, TimesformerForVideoClassification
    import numpy as np
    import torch

    video = list(np.random.randn(96, 3, 224, 224))
    processor = AutoImageProcessor.from_pretrained("fcakyon/timesformer-large-finetuned-k400")
    model = TimesformerForVideoClassification.from_pretrained("fcakyon/timesformer-large-finetuned-k400")
    inputs = processor(video, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])



def fcakyon_timesformer_hr_finetuned_k400(video_path):
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    import numpy as np
    import torch
    from decord import VideoReader, cpu

    video = list(np.ones((16, 3, 224, 224)))

    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)

    video = videoreader.get_batch([i for i in range(len(videoreader))]).asnumpy()

    video = list(video.transpose(0, 3, 1, 2))

    preprocessing_pipeline = Compose([
        Resize((448, 448)),
        CenterCrop((448, 448)),
    ])

    frames = [preprocessing_pipeline(torch.tensor(frame, dtype=torch.float16)) for frame in video]
    frames = np.stack([frame / 255 for frame in frames])

    # sample 16 frames from the video
    frames = frames[np.linspace(0, len(frames) - 1, 16, dtype=int)]

    processor = VideoMAEImageProcessor.from_pretrained('fcakyon/timesformer-hr-finetuned-k400')
    model = VideoMAEForVideoClassification.from_pretrained('fcakyon/timesformer-hr-finetuned-k400').to('cuda')
    inputs = processor(list(frames), return_tensors='pt').to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]



def videomae_small_finetuned_kinetics(video_path):
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    import numpy as np
    import torch
    from decord import VideoReader, cpu

    video = list(np.ones((16, 3, 224, 224)))

    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)

    video = videoreader.get_batch([i for i in range(len(videoreader))]).asnumpy()

    video = list(video.transpose(0, 3, 1, 2))

    preprocessing_pipeline = Compose([
        Resize((224, 224)),
        CenterCrop((224, 224)),
    ])

    frames = [preprocessing_pipeline(torch.tensor(frame, dtype=torch.float16)) for frame in video]
    frames = np.stack([frame / 255 for frame in frames])

    # sample 16 frames from the video
    frames = frames[np.linspace(0, len(frames) - 1, 16, dtype=int)]

    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-small-finetuned-kinetics')
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-small-finetuned-kinetics').to('cuda')
    inputs = processor(list(frames), return_tensors='pt').to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]



def videomae_base_finetuned_ucf101_subset(video_path: str):
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    import numpy as np
    import torch
    from decord import VideoReader, cpu

    video = list(np.ones((16, 3, 224, 224)))

    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)

    video = videoreader.get_batch([i for i in range(len(videoreader))]).asnumpy()

    video = list(video.transpose(0, 3, 1, 2))

    preprocessing_pipeline = Compose([
        Resize((224, 224)),
        CenterCrop((224, 224)),
    ])

    frames = [preprocessing_pipeline(torch.tensor(frame, dtype=torch.float16)) for frame in video]
    frames = np.stack([frame / 255 for frame in frames])

    # sample 16 frames from the video
    frames = frames[np.linspace(0, len(frames) - 1, 16, dtype=int)]

    processor = VideoMAEImageProcessor.from_pretrained('zahrav/videomae-base-finetuned-ucf101-subset')
    model = VideoMAEForVideoClassification.from_pretrained('zahrav/videomae-base-finetuned-ucf101-subset').to('cuda')
    inputs = processor(list(frames), return_tensors='pt').to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]



def sayakpaul_videomae_base_finetuned_ucf101_subset(video_file_path: str):
    """
       Classifies the input video using the specified model.

       :param video_file_path: str, path to the input video file.
       :param model_name: str, name of the model to be used for classification. Default is "sayakpaul/videomae-base-finetuned-ucf101-subset".
       :return: str, name of the predicted class.
       """
    import torch
    import imageio
    from torchvision.transforms import Compose, Lambda
    from pytorchvideo.transforms import Normalize, ShortSideScale, UniformTemporalSubsample
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    # Load the fine-tuned model
    model = VideoMAEForVideoClassification.from_pretrained('sayakpaul/videomae-base-finetuned-ucf101-subse')

    # Load the processor
    image_processor = VideoMAEImageProcessor.from_pretrained('sayakpaul/videomae-base-finetuned-ucf101-subse')

    # Load video
    video_reader = imageio.get_reader(video_file_path, "ffmpeg")
    video_frames = [frame for frame in video_reader]
    video_tensor = torch.tensor(video_frames).permute(3, 0, 1, 2)  # Change to (C, T, H, W)

    # Preprocess the video
    mean = image_processor.image_mean
    std = image_processor.image_std
    resize_to = (image_processor.size["height"], image_processor.size["width"])

    transform = Compose([
        UniformTemporalSubsample(model.config.num_frames),
        Lambda(lambda x: x / 255.0),
        Normalize(mean, std),
        ShortSideScale(resize_to),
    ])

    preprocessed_video = transform(video_tensor)

    # Make predictions
    with torch.no_grad():
        inputs = {"pixel_values": preprocessed_video.unsqueeze(0)}
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    # Map the predicted index to the corresponding label
    predicted_class_label = model.config.id2label[predicted_class_idx]

    return predicted_class_label



def openai_clip_vit_base_patch32(image_path, labels):
    from PIL import Image
    import requests
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(image_path)

    inputs = processor(text=labels, images=[image], return_tensors="pt", padding=True).to("cuda")
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    return probs



def openai_clip_vit_large_patch14(image_path, labels):
    from PIL import Image
    import requests
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    image = Image.open(image_path)

    inputs = processor(text=labels, images=[image], return_tensors="pt", padding=True).to("cuda")
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    return probs



def laion_clip_vit_bigG_14_laion2B_39B_b160k(image, possible_class_names):
    from transformers import pipeline

    classifier = pipeline('image-classification', model='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    result = classifier(image, possible_class_names=possible_class_names)

    return result



def openai_clip_vit_base_patch16(image_path, labels):
    from PIL import Image
    import requests
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    image = Image.open(image_path)

    inputs = processor(text=labels, images=[image], return_tensors="pt", padding=True).to("cuda")
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    return probs



def laion_clip_vit_b_16_laion2b_s34b_b88k(image_path, labels):
    from transformers import pipeline

    classify = pipeline('image-classification', model='laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
    result = classify(image_path, labels)

    return result



def patrickjohncyh_fashion_clip(image_path, labels):
    from transformers import CLIPProcessor, CLIPModel
    import PIL

    model = CLIPModel.from_pretrained('patrickjohncyh/fashion-clip').to('cuda')
    processor = CLIPProcessor.from_pretrained('patrickjohncyh/fashion-clip')

    image = PIL.Image.open(image_path)

    inputs = processor(text=labels, images=[image], return_tensors='pt', padding=True).to('cuda')
    logits_per_image = model(**inputs).logits_per_image
    probs = logits_per_image.softmax(dim=-1).tolist()[0]

    return probs



def laion_clip_convnext_large_d_320_laion2B_s29B_b131K_ft_soup(image_path, labels):
    from transformers import CLIPProcessor, CLIPModel

    processor = CLIPProcessor.from_pretrained("laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup")
    model = CLIPModel.from_pretrained("laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup")

    inputs = processor(text=labels, images=image_path, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    return outputs



def laion_clip_convnext_base_w_laion_aesthetic_s13B_b82K(image_path, labels):
    from transformers import pipeline

    model = pipeline('image-classification', model='laion/CLIP-convnext_base_w-laion_aesthetic-s13B-b82K')
    result = model(image_path, labels)

    return result



def laion_clip_convnext_xxlarge_laion2B_s34B_b82K_augreg_soup(image_path, labels):
    from transformers import CLIPProcessor, CLIPModel
    import PIL

    # Load the CLIP model
    model_name = "laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup"
    model = CLIPModel.from_pretrained(model_name)

    # Load the CLIP processor
    processor = CLIPProcessor.from_pretrained(model_name)

    image = PIL.Image.open(image_path)
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Forward pass through the model
    outputs = model(**inputs)

    # Get the predicted class probabilities
    logits = outputs.logits
    probabilities = logits.softmax(dim=-1)

    # Get the predicted class labels
    if labels is not None:
        predicted_labels = [labels[i] for i in probabilities.argmax(dim=-1).tolist()]
    else:
        predicted_labels = probabilities.argmax(dim=-1).tolist()

    return predicted_labels



def laion_clip_image_classification(image_path, class_names):
    from transformers import pipeline

    image_classification = pipeline('image-classification',
                                    model='laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K-augreg')
    result = image_classification(image_path, class_names)
    return result



def clip_rsicd_v2(image_path, labels):
    from PIL import Image
    import requests
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained("flax-community/clip-rsicd-v2")
    processor = CLIPProcessor.from_pretrained("flax-community/clip-rsicd-v2")

    image = Image.open(image_path)

    inputs = processor(text=[f"a photo of a {l}" for l in labels], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    return probs



def tiny_random_CLIPSegModel(image_path, labels):

    from transformers import  AutoProcessor, CLIPSegModel
    import PIL

    # Load the model and tokenizer
    model = CLIPSegModel.from_pretrained('hf-tiny-model-private/tiny-random-CLIPSegModel').to("cuda")
    processor = AutoProcessor.from_pretrained('hf-tiny-model-private/tiny-random-CLIPSegModel')

    image = PIL.Image.open(image_path)

    # Tokenize the texts
    inputs = processor(text=labels, images=[image], return_tensors="pt", padding=True).to("cuda")

    # Forward pass through the model
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    return probs



def timm_eva02_enormous_patch14_plus_clip_224_laion2b_s9b_b144k(image_path, labels):

    from transformers import AutoProcessor, CLIPModel
    import PIL

    # Load the model and tokenizer
    model = CLIPModel.from_pretrained('timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k').to("cuda")
    processor = AutoProcessor.from_pretrained('timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k')

    image = PIL.Image.open(image_path)

    # Tokenize the texts
    inputs = processor(text=labels, images=[image], return_tensors="pt", padding=True).to("cuda")

    # Forward pass through the model
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    return probs



def laion_clip_convnext_large_d_laion2B_s26B_b102K_augreg(image_path, classes):
    from transformers import pipeline

    clip = pipeline('image-classification', model='laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg')
    clip(image_path, classes)



def laion_clip_convnext_large_d_320_laion2B_s29B_b131K_ft(image_path, labels):
    from transformers import AutoProcessor, CLIPModel
    import PIL

    # Load the model and tokenizer
    model = CLIPModel.from_pretrained('laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft').to("cuda")
    processor = AutoProcessor.from_pretrained('laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft')

    image = PIL.Image.open(image_path)

    # Tokenize the texts
    inputs = processor(text=labels, images=[image], return_tensors="pt", padding=True).to("cuda")

    # Forward pass through the model
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    return probs



def OFA_Sys_chinese_clip_vit_base_patch16(image_path, labels):
    from PIL import Image
    import requests
    from transformers import ChineseCLIPProcessor, ChineseCLIPModel

    model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").to("cuda")
    processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

    image = Image.open(image_path)

    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to("cuda")
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    return probs



def clip_vit_base_patch32_ko(image_path, labels):
    import requests
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    repo = "Bingsu/clip-vit-base-patch32-ko"
    model = AutoModel.from_pretrained(repo).to("cuda")
    processor = AutoProcessor.from_pretrained(repo)

    image = Image.open(image_path)
    inputs = processor(text=labels, images=[image], return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs



def chinese_clip_vit_large_patch14(image_path, labels):
    from PIL import Image
    import requests
    from transformers import ChineseCLIPProcessor, ChineseCLIPModel

    model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14").to("cuda")
    processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14")

    image = Image.open(image_path)

    inputs = processor(text=labels, images=[image], return_tensors="pt", padding=True).to("cuda")
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # probs: [[0.0066, 0.0211, 0.0031, 0.9692]]

    return probs





def clipseg_rd64_refined(labels, image_path):
    from transformers import  AutoProcessor, CLIPSegModel
    import PIL

    # Load the model and tokenizer
    model = CLIPSegModel.from_pretrained('CIDAS/clipseg-rd64-refined').to("cuda")
    processor = AutoProcessor.from_pretrained('CIDAS/clipseg-rd64-refined')

    image = PIL.Image.open(image_path)

    # Tokenize the texts
    inputs = processor(text=labels, images=[image], return_tensors="pt", padding=True).to("cuda")

    # Forward pass through the model
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    return probs



def git_base(image):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load the model and tokenizer
    model_name = "microsoft/git-base"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Perform the desired operations using the model and tokenizer
    # ...

    # Return the result
    return result



def microsoft_table_transformer_structure_recognition(api_key, table_image_path):
    import requests
    from PIL import Image
    import io

    # Read the image file
    with open(table_image_path, 'rb') as image_file:
        image_data = image_file.read()

    # Send the image data to the API
    response = requests.post(
        url="https://api.cognitive.microsoft.com/vision/v3.2/read/analyze",
        headers={
            "Content-Type": "application/octet-stream",
            "Ocp-Apim-Subscription-Key": api_key
        },
        data=image_data
    )

    # Get the response JSON
    response_json = response.json()

    # Extract the table structure from the response
    table_structure = response_json["analyzeResult"]["readResults"][0]["tables"][0]["rows"]

    return table_structure



def CompVis_ldm_celebahq_256(output_path, num_inference_steps=200):
    import torch
    from PIL import Image
    from diffusers import DiffusionPipeline
    import os

    model_id = "CompVis/ldm-celebahq-256"
    pipeline = DiffusionPipeline.from_pretrained(model_id)
    image = pipeline(num_inference_steps=num_inference_steps)[0]
    image[0].save(output_path)

    return os.path.abspath(output_path)




def ocariz_universe_1400(output_path):
    from diffusers import DDPMPipeline
    import os

    pipeline = DDPMPipeline.from_pretrained('ocariz/universe_1400').to('cuda')
    image = pipeline().images[0]
    image.save(output_path)

    return os.path.abspath(output_path)



def google_ddpm_celebahq_256(api_key, output_path):
    import diffusers
    from diffusers import DDPMPipeline
    import os

    model_id = "google/ddpm-celebahq-256"
    sample = 0
    ddpm = DDPMPipeline.from_pretrained(model_id, api_key=api_key)
    image = ddpm()[sample]
    image[0].save(output_path)
    return os.path.abspath(output_path)



def google_ncsnpp_ffhq_256(output_path):
    import os
    from diffusers import DiffusionPipeline
    model_id = "google/ncsnpp-ffhq-256"
    sde_ve = DiffusionPipeline.from_pretrained(model_id)
    image = sde_ve()[0]
    image[0].save(output_path)
    return os.path.abspath(output_path)



def martinezomg_vit_base_patch16_224_diabetic_retinopathy(image_path):
    from transformers import pipeline

    image_classifier = pipeline('image-classification', 'martinezomg/vit-base-patch16-224-diabetic-retinopathy')
    result = image_classifier(image_path)

    return result



def fxmarty_resnet_tiny_beans(image_path):
    from transformers import pipeline

    classifier = pipeline('image-classification', model='fxmarty/resnet-tiny-beans')
    results = classifier(image_path)

    return results



def nvidia_mit_b0(image_path):
    from transformers import SegformerFeatureExtractor, SegformerForImageClassification
    from PIL import Image
    import requests

    image = Image.open(image_path)
    feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/mit-b0')
    model = SegformerForImageClassification.from_pretrained('nvidia/mit-b0').to('cuda')
    inputs = feature_extractor(images=image, return_tensors='pt').to('cuda')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    return predicted_class



def facebook_regnet_y_008(image_path):
    from transformers import AutoFeatureExtractor, RegNetForImageClassification
    import torch
    from datasets import load_dataset
    from PIL import Image

    image = Image.open(image_path)

    feature_extractor = AutoFeatureExtractor.from_pretrained('zuppif/regnet-y-040')
    model = RegNetForImageClassification.from_pretrained('zuppif/regnet-y-040').to('cuda')

    inputs = feature_extractor(image, return_tensors='pt').to('cuda')

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()
    print(model.config.id2label[predicted_label])



def nlpconnect_vit_gpt2_image_captioning(image_paths):
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    import torch
    from PIL import Image

    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    def predict_step(image_paths):
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")
            images.append(i_image)

        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(pixel_values, **gen_kwargs)
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds

    return predict_step(image_paths)



def promptcap_coco_vqa(question, image_path):
    import torch
    from promptcap import PromptCap
    import PIL

    model = PromptCap("vqascore/promptcap-coco-vqa")

    if torch.cuda.is_available():
        model.cuda()

    # image = PIL.Image.open(image_path)

    return model.caption(question, image_path)



def openai_clip_vit_base_patch32(image_path, labels):
    import torch
    from PIL import Image
    from torchvision.transforms import functional as F
    from transformers import CLIPProcessor, CLIPModel

    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(torch.device("cuda"))
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Prepare the images
    image = Image.open(image_path)
    image = F.resize(image, (224, 224))
    image = F.to_tensor(image)

    # Encode the images
    inputs = processor(text=labels, images=[image], return_tensors="pt", padding=True).to(torch.device("cuda"))
    # image_features = model.get_image_features(**inputs)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    return probs



def blip2_flan_t5_xl(image_path, question):
    import requests
    from PIL import Image
    from transformers import BlipProcessor, Blip2ForConditionalGeneration
    import torch

    processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to(torch.device('cuda'))

    raw_image = Image.open(image_path).convert('RGB')

    inputs = processor(raw_image, question, return_tensors="pt").to(torch.device('cuda'))
    out = model.generate(**inputs)

    return processor.decode(out[0], skip_special_tokens=True)



def blip2_flan_t5_xxl(image_path, question):
    import requests
    from PIL import Image
    from transformers import BlipProcessor, Blip2ForConditionalGeneration
    import torch

    processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl").to(torch.device('cuda'))

    raw_image = Image.open(image_path).convert('RGB')

    inputs = processor(raw_image, question, return_tensors="pt").to(torch.device('cuda'))
    out = model.generate(**inputs)

    return processor.decode(out[0], skip_special_tokens=True)



def google_deplot(image_path: str, text: str) -> str:
    from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
    import requests
    from PIL import Image
    import torch

    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot').to(torch.device('cuda:0'))
    processor = Pix2StructProcessor.from_pretrained('google/deplot')

    image = Image.open(image_path)

    inputs = processor(images=image, text=text, return_tensors='pt').to(torch.device('cuda:0'))

    predictions = model.generate(**inputs, max_new_tokens=512)

    return processor.decode(predictions[0], skip_special_tokens=True)



def microsoft_resnet_18(image_path):
    from transformers import AutoFeatureExtractor, ResNetForImageClassification
    import torch
    import PIL

    feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/resnet-18')
    model = ResNetForImageClassification.from_pretrained('microsoft/resnet-18')
    image = PIL.Image.open(image_path)
    inputs = feature_extractor(image, return_tensors='pt')

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]



def facebook_convnext_base_224(image_path):
    from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
    import torch
    from datasets import load_dataset
    import PIL

    dataset = load_dataset('huggingface/cats-image')
    feature_extractor = ConvNextFeatureExtractor.from_pretrained('facebook/convnext-base-224')
    model = ConvNextForImageClassification.from_pretrained('facebook/convnext-base-224')
    image = PIL.Image.open(image_path)
    inputs = feature_extractor(image, return_tensors='pt')

    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_label = logits.argmax(-1).item()

    return model.config.id2label[predicted_label]



def facebook_convnext_tiny_224(image_Path):
    from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
    import torch
    from datasets import load_dataset
    import PIL

    dataset = load_dataset('huggingface/cats-image')

    feature_extractor = ConvNextFeatureExtractor.from_pretrained('facebook/convnext-tiny-224')
    model = ConvNextForImageClassification.from_pretrained('facebook/convnext-tiny-224').to(torch.device('cuda'))

    image = PIL.Image.open(image_Path)
    inputs = feature_extractor(image, return_tensors='pt').to(torch.device('cuda'))

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()

    return model.config.id2label[predicted_label]



def keremberke_yolov8s_pcb_defect_segmentation(image_path):
    from ultralyticsplus import YOLO, render_result
    import PIL

    model = YOLO('keremberke/yolov8s-pcb-defect-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    image = PIL.Image.open(image_path)

    results = model.predict(image)
    boxes = results[0].boxes
    masks = results[0].masks

    render = render_result(model=model, image=image, result=results[0])
    render.show()

    return boxes, masks



def lllyasviel_sd_controlnet_canny(image_path: str, output_path: str, low_threshold: int=100, high_threshold: int=200) -> None:
    import cv2
    from PIL import Image
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    import torch
    import numpy as np
    from diffusers.utils import load_image
    import os

    image = Image.open(image_path)
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet,
                                                             safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    image = pipe("bird", image, num_inference_steps=5).images[0]
    image.save(output_path)
    return os.path.abspath(output_path)



def lllyasviel_control_v11p_sd15_seg(prompt, control_image_path, output_path):
    import torch
    import os
    from huggingface_hub import HfApi
    from pathlib import Path
    from diffusers.utils import load_image
    from PIL import Image
    import numpy as np
    from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    ada_palette = np.asarray([
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ])

    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

    checkpoint = "lllyasviel/control_v11p_sd15_seg"

    image = Image.open(control_image_path)
    # prompt = "old house in stormy weather with rain and wind"

    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)
    seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
    for label, color in enumerate(ada_palette):
        color_seg[seg == label, :] = color
    color_seg = color_seg.astype(np.uint8)
    control_image = Image.fromarray(color_seg)

    control_image.save("./data/control.png")

    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16).to("cuda")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)
    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]

    image.save(output_path)
    return os.path.abspath(output_path)



def google_ddpm_cifar10_32(output_path):
    import os
    import torch
    from PIL import Image
    from diffusers import DDPMPipeline

    # Install diffusers if not already installed
    try:
        import diffusers
    except ImportError:
        #        !pip install diffusers
        import diffusers

    # Load the DDPMPipeline model
    ddpm = DDPMPipeline.from_pretrained('google/ddpm-cifar10-32').to(torch.device('cuda'))

    # Generate an image using the DDPMPipeline model
    image = ddpm().images[0]

    # Save the generated image to the specified output path
    image.save(output_path)

    # Return the absolute path of the saved image
    return os.path.abspath(output_path)


def ceyda_butterfly_cropped_uniq1K_512(output_path):
    import torch
    import numpy as np
    from PIL import Image
    from huggan.pytorch.lightweight_gan.lightweight_gan import LightweightGAN
    import os

    gan = LightweightGAN.from_pretrained("ceyda/butterfly_cropped_uniq1K_512")
    gan.eval()

    batch_size = 1
    with torch.no_grad():
        ims = gan.G(torch.randn(batch_size, gan.latent_dim)).clamp_(0., 1.) * 255
        ims = ims.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)

    Image.fromarray(ims[0]).save(output_path)
    return os.path.abspath(output_path)



def google_ddpm_bedroom_256(output_path):
    import os
    import torch
    from PIL import Image
    from diffusers import DDPMPipeline

    # Install diffusers if not already installed
    try:
        import diffusers
    except ImportError:
        #        !pip install diffusers
        import diffusers

    model_id = "google/ddpm-bedroom-256"

    # Load the DDPMPipeline model
    ddpm = DDPMPipeline.from_pretrained(model_id).to(torch.device("cuda"))

    # Generate an image using the DDPMPipeline model
    image = ddpm().images[0]

    # Save the generated image to the specified output path
    image.save(output_path)

    # Return the path to the saved image
    return os.path.abspath(output_path)



def google_ncsnpp_church_256(output_path):
    import diffusers
    from diffusers import DiffusionPipeline
    import torch
    import os

    sde_ve = DiffusionPipeline.from_pretrained('google/ncsnpp-church-256').to(torch.device('cuda:0'))
    image = sde_ve()[0]
    image[0].save(output_path)
    return os.path.abspath(output_path)



def facebook_timesformer_hr_finetuned_k600(video_path):
    from transformers import AutoImageProcessor, TimesformerForVideoClassification
    import numpy as np
    from decord import VideoReader, cpu
    import torch

    # load the video frames from path

    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)
    video = videoreader.get_batch([i for i in range(len(videoreader))]).asnumpy()

    # transform video to (time, channel, height, width) format and set height and width to 224
    video = np.transpose(video, (0, 3, 1, 2))
    # normalize the video frames
    min_values = video.min(axis=(1, 2, 3), keepdims=True)
    max_values = video.max(axis=(1, 2, 3), keepdims=True)

    # Normalize each frame to the range [0, 1]
    video = (video - min_values) / (max_values - min_values)

    # video = video[:, :, 0:224, 0:224]
    print(video.shape)

    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-hr-finetuned-k600", do_rescale=False)
    model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-hr-finetuned-k600").to(torch.device("cuda"))
    inputs = [processor(images=frame, return_tensors="pt", image_size=(448, 448)).pixel_values for frame in video]
    inputs = torch.stack(inputs).to(torch.device("cuda")).squeeze()
    print(inputs.shape)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]



def videomae_small_finetuned_ssv2(video_path):
    import numpy as np
    import torch
    from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
    from decord import VideoReader, cpu

    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)
    video = videoreader.get_batch([i for i in range(len(videoreader))]).asnumpy()
    print(video.shape)

    # transform video to (time, channel, height, width) format and set height and width to 224
    video = np.transpose(video, (0, 3, 1, 2))
    # video = video[:, :, 0:224, 0:224]
    print(video.shape)
    video = list(np.random.randn(16, 3, 224, 224))


    feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-small-finetuned-ssv2")
    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-small-finetuned-ssv2")
    inputs = feature_extractor(video, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]





def videomae_base_finetuned_ucf101(file_path: str, clip_len: int=16) -> str:
    from decord import VideoReader, cpu
    import torch
    import numpy as np
    from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
    from huggingface_hub import hf_hub_download

    np.random.seed(0)

    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices


    videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)

    frame_sample_rate = int(len(videoreader) // clip_len)

    indices = sample_frame_indices(clip_len=clip_len, frame_sample_rate=frame_sample_rate, seg_len=len(videoreader))
    video = videoreader.get_batch(indices).asnumpy()

    feature_extractor = VideoMAEFeatureExtractor.from_pretrained("nateraw/videomae-base-finetuned-ucf101")
    model = VideoMAEForVideoClassification.from_pretrained("nateraw/videomae-base-finetuned-ucf101")

    inputs = feature_extractor(list(video), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_label = logits.argmax(-1).item()

    return model.config.id2label[predicted_label]



def laion_CLIP_convnext_base_w_laion2B_s13B_b82K(image_path, labels):
    from transformers import pipeline
    clip = pipeline('image-classification', model='laion/CLIP-convnext_base_w-laion2B-s13B-b82K')
    return clip(image_path, labels)


def google_ddpm_cat_256(output_path):
    import os
    import torch
    from PIL import Image
    from diffusers import DDPMPipeline

    # Install diffusers if not already installed
    try:
        import diffusers
    except ImportError:
        import diffusers

    model_id = "google/ddpm-cat-256"

    # Load the DDPMPipeline model
    ddpm = DDPMPipeline.from_pretrained(model_id).to(torch.device("cuda"))

    # Generate an image using the DDPMPipeline model
    image = ddpm().images[0]

    # Save the generated image to the specified output path
    image.save(output_path)

    # Return the path of the saved image
    return os.path.abspath(output_path)



def hotdog_not_hotdog(image_path):
    import torch
    from PIL import Image
    from torchvision.transforms import functional as F
    from transformers import ViTFeatureExtractor, ViTForImageClassification

    # Load the pre-trained model and feature extractor
    model_name = "julien-c/hotdog-not-hotdog"
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)

    # Load and preprocess the image
    image = Image.open(image_path)
    image = F.to_tensor(image)
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Make predictions
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)

    # Return the predicted label
    return predictions.item()

