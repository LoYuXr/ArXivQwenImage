"""
QwenImage Parquet Dataset Config - Testing New Model Factory Code
Usage: accelerate launch --config_file accelerate_cfg/1m4g_bf16.yaml train_OpenSciDraw_loop.py configs/260119/test_newcode_qwenimage_parquet_dataset_config_local.py
"""

_base_ = '../base_config.py'

# ====== Model Type (NEW: for model factory) ======
model_type = 'QwenImage'

# ====== 输入路径 ======
base_filesys_path = "/home/v-yuxluo/data/"

# ====== 输出路径 ======
model_output_dir = '/home/v-yuxluo/data/experiments/260119_qwenimage_test/models'
image_output_dir = '/home/v-yuxluo/data/experiments/260119_qwenimage_test/samples'
logging_dir = 'logs'

# ====== 日志与监控 ======
report_to = 'wandb'
tracker_name = 'QwenImage-NewCode-Test'
run_name = "test_newcode_qwenimage_260119_local"
seed = 42
cache_dir = "/home/v-yuxluo/data/huggingface_cache"

# ====== 模型权重与结构 ======
transformer_cfg = dict(
    type='QwenImageTransformer2DModel',
)

# ====== LoRA ======
use_lora = True
lora_layers = "to_k,to_q,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1"
rank = 128
lora_alpha = 128
lora_dropout = 0.0
layer_weighting = 5.0

# ====== 训练设置 ======
train_batch_size = 2
num_train_epochs = 2
gradient_accumulation_steps = 1
optimizer = "prodigy"
learning_rate = 1.0
lr_warmup_steps = 0
use_8bit_adam = False
max_train_steps = 50000
gradient_checkpointing = True
checkpointing_steps = 1000
resume_from_checkpoint = "latest"

# ====== 数据 ======
train_gpus_num = 4
dataloader_num_workers = 8
train_iteration_func_name = 'Qwen_Image_train_iteration_func_parquet'

use_parquet_dataset = True
dataset = dict(
    type='ArXiVParquetDatasetV2',
    base_dir=base_filesys_path,
    parquet_base_path='ArXiV_parquet/qwenimage_latents/',
    num_train_examples=602000,
    num_workers=dataloader_num_workers,
    stat_data=True,
)

data_sampler = dict(
    type='ArXiVMixScaleBatchSampler',
    batch_size=train_batch_size,
)

max_sequence_length = 1024

# ====== 精度 / 性能 ======
mixed_precision = "bf16"

# ====== 验证与推理 ======
validation_func_name = 'QwenImage_validation_func'
validation_steps = 500
num_inference_steps = 50
true_cfg_scale = 4.0
negative_prompt = " "

validation_prompts = [
    "The figure illustrates a process hooking mechanism using the LD_PRELOAD environment variable to inject a custom data collection library, siren.so, into target ELF binary executables during runtime. The global layout is a top-down flowchart depicting the sequence of interactions from environment setup to data analysis. At the top, a light blue rectangular box labeled 'Environment Variable: LD_PRELOAD=siren.so' initiates the process. This points downward to a green rectangle labeled 'Dynamic Linker: ld.so', which branches into two paths: one to a light blue box 'Injected Library: siren.so' and another to a green box 'Shared Libraries: DT_NEEDED'. Both converge into a large green rectangular container labeled 'ELF Binary Executable', which contains three internal components arranged vertically. The first is a light blue hexagon labeled 'Constructor: Data Collection and UDP Sender', followed by a green rectangle 'Application Code: main()', and then another light blue hexagon 'Destructor: Data Collection and UDP Sender'. These indicate that the injected library's data collection routines are triggered at both process startup (via constructor) and shutdown (via destructor). An arrow from the destructor leads to a light blue rectangle 'Message Receiver: UDP Server', which in turn connects to a light blue cylinder labeled 'Database: SQLite'. From the database, a downward arrow leads to a light blue rectangle 'Post-processing and Consolidation: Python', which then connects leftward to another light blue rectangle 'Statistics and Similarity Analysis: Python'. All elements shaded in light blue represent components of the SIREN architecture, while green elements denote standard system or application components. The arrows indicate the direction of control flow and data transmission, showing how injected data is sent via UDP, received, stored, processed, and finally analyzed. The diagram emphasizes the non-intrusive nature of the hooking mechanism, leveraging dynamic linking to collect runtime data without modifying the target application’s source code.", #2508.18950_0_img_SINGLE.png 
    "The figure presents an overview of four distinct end-to-end Task-Oriented Dialogue (TOD) approaches, arranged vertically as subfigures (a) through (d), each illustrating a different methodology for integrating language models into dialogue systems.\n\n[1] Global Layout and Structure:\nThe figure is divided into four horizontal sections, each representing a different approach. Each section contains a central model component at the top, with input/output modules below or connected via arrows. The layout follows a top-down flow, where user inputs lead to model processing and then to outputs such as actions or responses. Subfigure labels (a), (b), (c), and (d) are placed beneath each section, along with descriptive captions explaining the approach.\n\n[2] Visual Modules and Attributes:\nIn subfigure (a), labeled 'Full-shot approach with fine-tuning LM', a large light green rounded rectangle at the top represents a 'Pre-trained Language Model (e.g., GPT2, T5)', marked with a red flame icon. Below it, five light blue rectangular boxes labeled 'User', 'Belief State', 'DB', 'Action', and 'Resp' are aligned horizontally. Arrows connect these boxes to the model, indicating bidirectional interaction between the model and all components except 'Resp', which receives output from the model.\n\nSubfigure (b), titled 'Zero-shot approach via schema-guided prompting LLM', features a similar light green rounded rectangle labeled 'Large Language Model (e.g., GPT 3.5, GPT-4)', marked with a blue snowflake icon. Below, two yellow rounded rectangles labeled 'DST Prompter' and 'Policy Prompter' receive input from 'User' and 'DB' respectively, and feed into the LLM. The LLM outputs to 'Action' and 'Resp', both light blue boxes.\n\nSubfigure (c), 'Zero-shot approach via autonomous Agent LLM', shows a light green rounded rectangle containing a robot icon and a pink rounded rectangle labeled 'Instruction following LLM'. This module is labeled 'Large Language Model' and marked with a blue snowflake. A bidirectional arrow connects the 'User' box to the LLM, with 'Resp' labeled on the return path. To the right, a set of yellow boxes labeled 'API tool-1' through 'API tool-n' are connected to the LLM via a blue circular arrow, indicating iterative interaction.\n\nSubfigure (d), 'Spec-TOD (ours): Few-shot approach with specialized instruction-tuned LLM', displays a light green rounded rectangle labeled 'Specialized Task-Oriented LLM', marked with a red flame icon. Inside, a robot icon with a gear symbol is adjacent to a pink rounded rectangle labeled 'Specified-Task Instruct.'. A bidirectional arrow connects the 'User' box to this module, with 'Resp' labeled on the return path. To the right, a vertical stack of yellow boxes labeled 'Task-1 Spec. Rep.', 'Task-2 Spec. Rep.', ..., 'Task-m Spec. Rep.' is connected to the 'Specified-Task Instruct.' box via a blue circular arrow, indicating iterative refinement using task-specific representations.\n\n[3] Connections and Arrows:\nIn (a), arrows show bidirectional communication between the pre-trained LM and 'User', 'Belief State', and 'DB', while unidirectional arrows point from the LM to 'Action' and 'Resp'.\n\nIn (b), arrows go from 'User' to 'DST Prompter', from 'DB' to 'Policy Prompter', and from both prompters to the LLM. The LLM sends outputs to 'Action' and 'Resp'.\n\nIn (c), a bidirectional arrow links 'User' and the LLM, with 'Resp' labeled on the response path. A blue circular arrow connects the LLM to the API tools, indicating iterative tool calling.\n\nIn (d), a bidirectional arrow connects 'User' and the LLM, with 'Resp' on the return path. A blue circular arrow links the 'Specified-Task Instruct.' box to the stack of task-specific representations, suggesting iterative refinement using these representations.", #2507.04841/2507.04841_0_img_SINGLE.png
    "The figure illustrates a network architecture for a single-step diffusion model with an enhanced decoder. The global layout is horizontal, progressing from left to right, with multiple parallel input streams converging into a central processing unit before diverging again toward the output. On the far left, three distinct input conditioning vectors, labeled c₁, c₂, and c₃, are represented as gray rounded rectangles. Each of these inputs is processed by a separate blue parallelogram-shaped module labeled ε, indicating an encoder or feature extraction component. These encoders are marked as 'Frozen' according to the legend at the bottom right, which uses a blue snowflake icon to denote frozen modules. The outputs of these encoders are combined via two circular summation nodes (⊕), where the first summation node receives the output of ε(c₁) and ε(c₂), and the second summation node combines the result with ε(c₃). Additionally, a noise latent vector z_T, shown as a gray rounded rectangle, is fed directly into the first summation node. The combined feature representation from both summation nodes is then passed into a large, centrally located orange bowtie-shaped module labeled 'UNet'. This UNet is marked as 'Trained' in the legend, indicated by an orange flame icon, signifying it is the primary trainable component of the architecture. The UNet outputs a denoised latent representation, denoted as -ẑ₀, shown as a gray rounded rectangle. This output is then fed into a blue parallelogram-shaped decoder module labeled D, also marked as 'Frozen'. Prior to entering the decoder, an additional orange parallelogram-shaped module labeled ε_f, which is trained, provides auxiliary features that are concatenated or fused with the main latent stream before decoding. The final output emerges from the decoder D. The connections between all components are depicted using gray arrows, indicating the flow of data. The overall structure emphasizes a multi-scale feature fusion strategy, where conditioned features from multiple encoders are aggregated and combined with noise to guide the UNet’s denoising process, followed by reconstruction through a frozen decoder enhanced by an additional trained feature extractor ε_f.", #2507.20331
    ]

resolution_list = [ #w, h
    [576, 960],
    [576, 960],
    [1008, 576],
    
]