import torch
import gc
import scipy.io.wavfile as wavfile
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, VitsModel
from diffusers import AutoPipelineForText2Image
from datasets import load_dataset
import os
import time

from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv(override=True)
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def process_meeting(audio_path, language="English", progress_callback=None):
    results = {}
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # ==========================================
    # Phase 1: Adaptive Speech-to-Text (Whisper)
    # ==========================================
    if progress_callback: progress_callback(f"🎙️ Phase 1: Transcribing audio ({language})...")
    asr_model = "openai/whisper-medium" if language == "Persian" else "openai/whisper-small.en"

    pipe_asr = pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        torch_dtype=compute_dtype,
        device=device_str
    )

    if language == "Persian":
        transcription_result = pipe_asr(
            audio_path,
            chunk_length_s=30,
            return_timestamps=True,
            generate_kwargs={"task": "transcribe", "language": "persian"}
        )
    else:
        transcription_result = pipe_asr(
            audio_path,
            chunk_length_s=30,
            return_timestamps=True
        )

    transcription = transcription_result["text"].strip()
    if not transcription:
        transcription = "هیچ صدایی برای تحلیل پیاده‌سازی نشد." if language == "Persian" else "No speech detected in audio."
        
    results["transcription"] = transcription
    del pipe_asr
    free_memory()

    # ==========================================
    # Phase 2: Domain-Agnostic LLM Understanding
    # ==========================================
    if progress_callback: progress_callback("🧠 Phase 2: Analyzing content & extracting insights...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    
    model_id = "Qwen/Qwen2.5-3B-Instruct" if language == "Persian" else "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        token=hf_token
    )

    # دستورالعمل‌ها کاملاً به زبان انگلیسی ولی با تعیین دقیق زبان خروجی
    if language == "Persian":
        sys_msg = (
            "You are a versatile, universal audio intelligence assistant. Analyze the transcription by:\n"
            "1. Identifying the content genre/type (e.g., Story/Narrative, News, Business Meeting, Interview, Resume/Profile, Scientific Lecture, Casual Conversation).\n"
            "2. Providing a comprehensive, structured summary tailored to its specific genre.\n"
            "3. Extracting the core highlights, takeaways, or key facts.\n"
            "Format the output in clean Markdown.\n"
            "CRITICAL REQUIREMENT: You MUST generate your entire response strictly in Persian (Farsi) language."
        )
        short_sys = (
            "Summarize the essence of this content into exactly one captivating, complete sentence suitable for a spoken audio teaser. "
            "CRITICAL REQUIREMENT: Output ONLY that single sentence strictly in Persian (Farsi) language without any extra introductory text."
        )
    else:
        sys_msg = (
            "You are a versatile, universal audio intelligence assistant. Analyze the transcription by:\n"
            "1. Identifying the content genre/type (e.g., Story/Narrative, News, Business Meeting, Interview, Resume/Profile, Scientific Lecture, Casual Conversation).\n"
            "2. Providing a comprehensive, structured summary tailored to its specific genre.\n"
            "3. Extracting the core highlights, takeaways, or key facts.\n"
            "Format the output in clean Markdown."
        )
        short_sys = (
            "Summarize the essence of this content into exactly one captivating, complete sentence suitable for a spoken audio teaser. "
            "Output ONLY that single sentence."
        )

    # ۱. تولید گزارش ساخت‌یافته متناسب با محتوا
    messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": f"Transcription:\n{transcription}"}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
    outputs = model.generate(inputs, max_new_tokens=1000, temperature=0.3)
    dossier_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    results["minutes"] = dossier_text

    # ۲. تولید جمله صوتی برای TTS
    messages_short = [{"role": "system", "content": short_sys}, {"role": "user", "content": dossier_text}]
    inputs_short = tokenizer.apply_chat_template(messages_short, return_tensors="pt", add_generation_prompt=True).to("cuda")
    outputs_short = model.generate(inputs_short, max_new_tokens=120, temperature=0.5)
    audio_brief_text = tokenizer.decode(outputs_short[0][len(inputs_short[0]):], skip_special_tokens=True).strip()
    results["brief_text"] = audio_brief_text

    # ۳. پرامپت انگلیسی تصویرساز (همیشه انگلیسی باقی می‌ماند)
    img_sys = (
        "Based on the analysis, design an artistic cover illustration prompt in English. "
        "Adapt the visual style to match the genre (e.g., cinematic fantasy for stories, editorial graphic for news, modern technical for science, professional studio portrait for resumes, minimalist corporate for meetings). "
        "Output ONLY the detailed visual prompt in one sentence, no quotes or intro."
    )
    messages_img = [{"role": "system", "content": img_sys}, {"role": "user", "content": dossier_text}]
    inputs_img = tokenizer.apply_chat_template(messages_img, return_tensors="pt", add_generation_prompt=True).to("cuda")
    outputs_img = model.generate(inputs_img, max_new_tokens=80, temperature=0.7)
    dynamic_visual_prompt = tokenizer.decode(outputs_img[0][len(inputs_img[0]):], skip_special_tokens=True).strip()

    del model, tokenizer, inputs, outputs, inputs_short, outputs_short, inputs_img, outputs_img
    free_memory()

    # ==========================================
    # Phase 3: Text-to-Speech (Dynamic Filename)
    # ==========================================
    if progress_callback: progress_callback("🔊 Phase 3: Synthesizing Audio Teaser...")
    timestamp = int(time.time())
    audio_out_path = f"audio_briefing_{timestamp}.wav"

    if language == "Persian":
        tts_model_id = "facebook/mms-tts-fas"
        tts_tokenizer = AutoTokenizer.from_pretrained(tts_model_id)
        tts_model = VitsModel.from_pretrained(tts_model_id).to(device_str)
        tts_inputs = tts_tokenizer(audio_brief_text, return_tensors="pt").to(device_str)
        with torch.no_grad():
            speech_output = tts_model(**tts_inputs).waveform
        speech_audio = speech_output.squeeze().cpu().numpy()
        wavfile.write(audio_out_path, tts_model.config.sampling_rate, speech_audio)
        del tts_model, tts_tokenizer
    else:
        synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device=device_str)
        embeddings_dataset = load_dataset("matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True)
        speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        speech = synthesiser(audio_brief_text, forward_params={"speaker_embeddings": speaker_embedding})
        wavfile.write(audio_out_path, speech["sampling_rate"], speech["audio"])
        del synthesiser, embeddings_dataset

    results["audio_brief_path"] = audio_out_path
    free_memory()

    # ==========================================
    # Phase 4: Adaptive Cover Art Generation
    # ==========================================
    if progress_callback: progress_callback("🎨 Phase 4: Generating Adaptive Cover Art...")
    pipe_img = AutoPipelineForText2Image.from_pretrained(
        "segmind/SSD-1B", 
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        variant="fp16"
    )
    pipe_img.to(device_str)
    
    # پرامپت داینامیک با رندومایز شدن برای جلوگیری از تکرار حالت هنری
    final_image_prompt = f"Masterpiece artwork, {dynamic_visual_prompt}, high atmospheric lighting, concept design, 8k resolution"
    image = pipe_img(prompt=final_image_prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
        
    img_out_path = f"dossier_cover_{timestamp}.png"
    image.save(img_out_path)
    results["cover_image"] = img_out_path

    del pipe_img
    free_memory()

    if progress_callback: progress_callback("✅ All Processing Complete!")
    return results