import gradio as gr
import os


def video_identity(video):
    return video


demo = gr.Interface(video_identity, 
                    gr.Video(), 
                    "playable_video", 
                    examples=[
                        os.path.join(os.path.dirname(__file__), 
                                     "input/output.mp4")], 
                    cache_examples=True)

if __name__ == "__main__":
    demo.launch()
