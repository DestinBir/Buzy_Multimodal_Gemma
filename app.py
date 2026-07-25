import os

import gradio as gr

from src.inference import Buzy_inference


def _sample_path(*parts):
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_assets")
    path = os.path.join(sample_dir, *parts)
    return path if os.path.exists(path) else None


def load_invoice_example():
    path = _sample_path("invoice.pdf")
    files = [path] if path else None
    return files, None, None, "Summarize this invoice and flag any anomalies."


def load_contract_example():
    path = _sample_path("contract.pdf")
    files = [path] if path else None
    return files, None, None, "What are the key obligations and risks in this contract?"


def load_meeting_example():
    path = _sample_path("meeting.mp3")
    files = [path] if path else None
    return None, None, files, "Summarize the key decisions and action items from this meeting."


def build_demo():
    with gr.Blocks(title="Buzy AI") as demo:
        gr.Markdown("# Buzy AI")
        gr.Markdown("### Autonomous Operating System for Organizations")
        gr.Markdown(
            "Upload documents, images, and meeting audio, ask a business question, "
            "and get a structured, source-attributed analysis."
        )

        with gr.Row():
            image_input = gr.File(
                label="Upload Images",
                file_count="multiple",
                file_types=["image"],
            )
            document_input = gr.File(
                label="Upload Documents",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".txt"],
            )
            audio_input = gr.File(
                label="Upload Audio",
                file_count="multiple",
                file_types=["audio"],
            )

        prompt = gr.Textbox(
            label="Business Question",
            lines=4,
            placeholder="e.g. What are the biggest risks in this contract, and what should we do about them?",
        )

        with gr.Row():
            analyze_btn = gr.Button("Analyze", variant="primary")
            clear_btn = gr.ClearButton(
                [image_input, document_input, audio_input, prompt, gallery, output],
                value="Clear",
            )

        gr.Markdown("#### One-click examples")
        with gr.Row():
            invoice_btn = gr.Button("Analyze Invoice")
            contract_btn = gr.Button("Review Contract")
            meeting_btn = gr.Button("Summarize Meeting")

        gr.Markdown("---")

        gallery = gr.Gallery(label="Uploaded Images", columns=4, height=200, visible=True)
        output = gr.Markdown(label="Business Analysis Report")

        analyze_btn.click(
            fn=Buzy_inference,
            inputs=[image_input, document_input, audio_input, prompt],
            outputs=[output, gallery],
        )

        invoice_btn.click(
            fn=load_invoice_example,
            inputs=[],
            outputs=[document_input, image_input, audio_input, prompt],
        )
        contract_btn.click(
            fn=load_contract_example,
            inputs=[],
            outputs=[document_input, image_input, audio_input, prompt],
        )
        meeting_btn.click(
            fn=load_meeting_example,
            inputs=[],
            outputs=[image_input, document_input, audio_input, prompt],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue().launch(theme=gr.themes.Soft())
