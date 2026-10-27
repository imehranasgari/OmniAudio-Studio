import gradio as gr
from core import process_meeting

def process_ui(audio_file, language, progress=gr.Progress()):
    if not audio_file:
        return (
            "⚠️ No audio input provided. Please record or upload an audio file first.",
            None,
            None,
            "Error: Missing audio source."
        )
    
    def log_progress(msg):
        progress(0, desc=msg)
        
    results = process_meeting(audio_file, language=language, progress_callback=log_progress)
    
    return (
        results["minutes"],
        results["audio_brief_path"],
        results["cover_image"],
        "✅ Pipeline finished successfully!"
    )

custom_css = """
footer {visibility: hidden}
.gradio-container {max-width: 1200px !important; margin: auto;}
"""

with gr.Blocks(title="OmniAudio Studio", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown(
        """
        # 🎙️ OmniAudio Studio
        ### Universal Multimodal Audio Intelligence Pipeline
        Transform any audio stream—interviews, spoken resumes, podcasts, news, meetings, or narratives—into **structured intelligence**, **adaptive cover art**, and a **spoken audio teaser**.
        """
    )
    
    with gr.Row():
        # Input Section
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload", "microphone"], 
                type="filepath", 
                label="Input Audio Stream (Microphone or File Upload)"
            )
            language_input = gr.Radio(
                choices=["English", "Persian"], 
                value="English", 
                label="Audio Language"
            )
            submit_btn = gr.Button("🚀 Run Multimodal Analysis", variant="primary", size="lg")
            status_text = gr.Textbox(label="Execution Status", interactive=False, placeholder="Ready")
            
        # Output Section
        with gr.Column(scale=1):
            out_image = gr.Image(type="filepath", label="Adaptive Generative Cover Art")
            out_audio = gr.Audio(type="filepath", label="Spoken Audio Brief / Executive Teaser")
            out_text = gr.Markdown(label="Extracted Intelligence & Structured Dossier")
            
    submit_btn.click(
        fn=process_ui,
        inputs=[audio_input, language_input],
        outputs=[out_text, out_audio, out_image, status_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7862)