import gradio as gr
from models.stylegene.api import synthesize_descendant

description = """<p style="text-align: center; font-weight: bold;">
        <span style="font-size: 28px">StyleGene: Crossover and Mutation of Region-Level Facial Genes for Kinship Face Synthesis</span>
        <br>
        <span style="font-size: 18px" id="paper-info">
            [<a href="https://wmpscc.github.io/stylegene/" target="_blank">Project Page</a>]
            [<a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Li_StyleGene_Crossover_and_Mutation_of_Region-Level_Facial_Genes_for_Kinship_CVPR_2023_paper.pdf" target="_blank">Paper</a>]
            [<a href="https://github.com/CVI-SZU/StyleGene" target="_blank">GitHub</a>]
        </span>
        <br> 
        <a> Tips: One picture should have only one face.</a>
    </p>"""

block = gr.Blocks()
with block:
    gr.HTML(description)
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Upload photos of father and mother")
            with gr.Row():
                img1 = gr.Image(label="Father")
                img2 = gr.Image(label="Mother")
            gr.Markdown("### Select the child's age and gender")
            with gr.Row():
                age = gr.Dropdown(label="Age",
                                  choices=["0-2", "3-9", "10-19", "20-29", "30-39",
                                           "40-49", "50-59", "60-69", "70+"], value="3-9")
                gender = gr.Dropdown(label="Gender", choices=["male", "female"], value="female")
            gr.Markdown("### Adjust your child's resemblance to parents")
            bar1 = gr.Slider(label="gamma", minimum=0, maximum=1, value=0.47)
            bar2 = gr.Slider(label="eta", minimum=0, maximum=1, value=0.4)
            bt_run = gr.Button("Run")
            gr.Markdown("""## Disclaimer
                            This method is intended for academic research purposes only and is strictly prohibited for commercial use.
                            Users are required to comply with all local laws and regulations when using this method.""")

        with gr.Column():
            gr.Markdown("### Results")
            img3 = gr.Image(label="Generated child")
            with gr.Row():
                img1_align = gr.Image(label="Father")
                img2_align = gr.Image(label="Mother")


    def run(father, mother, gamma, eta, age, gender):
        attributes = {'age': age, 'gender': gender, 'gamma': float(gamma), 'eta': float(eta)}
        img_F, img_M, img_C = synthesize_descendant(father, mother, attributes)
        return img_F, img_M, img_C


    bt_run.click(run, [img1, img2, bar1, bar2, age, gender], [img1_align, img2_align, img3])

block.launch(share=True, server_name="0.0.0.0", server_port=7860, show_error=True, debug=True)
