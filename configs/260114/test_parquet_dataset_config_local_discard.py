_base_ = '../base_config.py'

# ====== 输入路径 ======
base_filesys_path = "/home/v-yuxluo/yuxuanluo"
# ====== 输出路径 ======
model_output_dir  = '/home/v-yuxluo/data/experiments/260115/models'
image_output_dir  = '/home/v-yuxluo/data/experiments/260115/samples'
logging_dir       = 'logs'

# ====== 日志与监控 ======
report_to   = 'wandb'
tracker_name = 'test'
run_name = "yuxuan test openscidraw 260115 local A100"
seed = 42
cache_dir = "/home/v-yuxluo/data/huggingface_cache"

# Transformer 架构

# ====== 模型权重与结构 ======
transformer_cfg = dict(
    type='QwenImageTransformer2DModel',
)

# ====== LoRA ======
use_lora = True
lora_layers = "to_k,to_q,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1"
rank = 128
lora_alpha = 128 #是否需要保持一致
layer_weighting = 5.0


# ====== 训练设置 ======
train_batch_size = 2
gradient_accumulation_steps = 1
# resolution = 512
optimizer = "prodigy" #Adamw, muon, etc.
learning_rate = 1.0
lr_warmup_steps = 0
use_8bit_adam = False
max_train_steps = 50000
gradient_checkpointing = True
checkpointing_steps = 1000
resume_from_checkpoint = "latest"


# ====== 数据 ======
train_gpus_num = 4
dataloader_num_workers = 8 # --dataloader_num_workers
train_iteration_func_name = 'Qwen_Image_train_iteration_func_parquet' #'Qwen_Image_train_iteration_func' #Qwen_Image_train_iteration_func_parquet

use_parquet_dataset = True
dataset = dict(
        type='ArXiVParquetDataset',
        base_dir=base_filesys_path,
        parquet_base_path='ArXiV_parquet/QwenImageParquet_260115/',
        num_train_examples=602000,
        num_workers=dataloader_num_workers,
        debug_mode = True
)


data_sampler = dict(
    type='DistributedBucketSampler',
    batch_size=train_batch_size,
    #num_replicas=accelerator.num_processes,  # <--- 总卡数
    #rank=accelerator.process_index,      # <--- 当前卡是第几号  #在后面再传
    drop_last=True,
    shuffle=True
)

max_sequence_length = 1024 # 2048

# ====== 精度 / 性能 ======
mixed_precision = "bf16"


# ====== 验证与推理 ======
validation_func_name = 'QwenImage_validation_func'
validation_steps = 500
num_inference_steps = 50
true_cfg_scale = 4.0
negative_prompt = " "

validation_prompts = [
    "The figure presents three distinct machine learning architectures for single-shot inverse design, labeled A, B, and C, each illustrating a different approach to mapping from a response to a design. The overall layout is horizontal, with each panel arranged side-by-side and clearly labeled with a capital letter and title above it. Panel A, titled 'Direct Inverse Mapping', shows a simple feedforward neural network with three layers. On the left, two blue circular nodes represent the 'Response' input layer. These connect via black lines to three gray circular nodes in the hidden layer. From there, connections extend to four red circular nodes on the right, labeled 'Design', representing the output layer. All connections are fully connected between adjacent layers. Panel B, titled 'Tandem Neural Network', features a more complex structure. It begins with two blue circular nodes labeled 'Response' on the left. These connect to three gray hidden nodes, which then connect to four red nodes labeled 'Design'. This first part mirrors Panel A. However, this 'Design' output is fed into a second network enclosed within a dashed rectangular box labeled 'Pretrained Forward Network'. Inside this box, the four red 'Design' nodes connect to three gray hidden nodes, which then connect to two blue circular nodes labeled 'Predicted Response'. This creates a feedback loop where the generated design is passed through a forward model to predict the response, enabling iterative refinement or validation. Panel C, titled 'Conditional Generative Model (only showing generation network)', illustrates a generative architecture. On the left, two blue circular nodes labeled 'Response' are shown above two gray circular nodes labeled 'Noise or Latent Variables'. Both sets of inputs converge into a shared hidden layer of three gray nodes. From there, connections lead to four red circular nodes on the right, labeled 'Design'. This setup reflects a conditional generative model where both the desired response and random latent variables are used to generate a design. The caption clarifies that only the generation network is shown, omitting components like discriminators or encoders required for training. All panels use consistent visual attributes: blue circles for input (Response), gray circles for hidden layers, and red circles for output (Design). Connections are represented by thin black lines with arrowheads indicating direction. Text labels are placed near or directly below the corresponding nodes. The figure emphasizes conceptual clarity over architectural complexity, noting that real implementations may involve deeper or more intricate networks.",
    "The figure illustrates the architecture of LLaVA, a multimodal model designed to process both visual and linguistic inputs to generate language responses. The global layout is structured horizontally into three main layers: the bottom layer contains input modules, the middle layer consists of feature representations and projection components, and the top layer represents the language model generating the final output. The diagram flows from left to right and bottom to top, indicating the data processing pipeline.\n\nIn the bottom layer, two distinct inputs are shown: an 'Image' denoted as Xv on the left, and a 'Language Instruction' denoted as Xq on the right. The image input Xv is processed by a green rectangular module labeled 'Vision Encoder', which extracts visual features. These features are then passed through an orange rectangular module labeled 'Projection', which transforms them into a compatible representation Zv. This projected visual feature Zv is connected via a curved black arrow to a sequence of yellow diamond-shaped nodes labeled Hv, representing the visual feature embeddings. Similarly, the language instruction Xq is fed into a white rectangular box (implied to be a language encoder or tokenizer), whose output is connected via a straight black arrow to a sequence of yellow diamond-shaped nodes labeled Hq, representing the language instruction embeddings.\n\nThe middle layer consists of these two sequences of diamond-shaped nodes: Hv (visual features) and Hq (language instruction features). Both sequences are connected via multiple gray lines to the top layer, which is a large light-blue rectangular block labeled 'Language Model'. These connections indicate that both visual and linguistic features are fused and fed into the language model for joint processing.\n\nThe top layer, the 'Language Model', generates the final output: a 'Language Response' denoted as Xa. This output is represented by a sequence of red diamond-shaped nodes, each connected via gray lines from the language model, indicating the generation of tokens or words in the response.\n\nThe visual attributes include color-coded modules: green for the Vision Encoder, orange for the Projection, and light blue for the Language Model. The feature representations Hv and Hq are depicted as yellow diamonds, while the output Xa is shown as red diamonds. All connections are drawn as arrows or lines, with curved arrows for the visual pathway and straight arrows for the language pathway, emphasizing the flow of information from inputs to the language model and finally to the generated response. The diagram includes explicit labels for all major components and data streams, ensuring clarity in the architectural design.",
    "The figure illustrates the architecture of the ChatEd chatbot system, designed for interactive question-answering based on uploaded documents. The global layout is divided into two main sections: a left-side data ingestion pipeline and a central-right interaction and response generation workflow. The entire system is enclosed within a large rectangular boundary, emphasizing the scope of the chatbot’s operational environment.\n\nOn the left, the process begins with an 'Admin/Instructor' role, represented by a blue rounded rectangle. This entity uploads documents, indicated by a downward arrow labeled 'Upload documents', to a 'Text embedder' module, shown as a light blue rounded rectangle. The text embedder processes the documents and saves their embeddings into a 'Database', depicted as a dark gray rounded rectangle, via an arrow labeled 'Save into'.\n\nThe central-right portion of the diagram details the user interaction flow. A 'User', also represented by a blue rounded rectangle, initiates a query by asking a 'Question', which is shown as a light blue rounded rectangle. The question triggers a 'Search in database' operation, leading to the Database. From the Database, a dashed arrow labeled 'Query with document' points to an 'LLM' (Large Language Model), shown as a purple rounded rectangle. The LLM generates a response, indicated by a downward arrow labeled 'Generate', resulting in an 'Answer', displayed as a red rounded rectangle.\n\nThe Answer is then sent back to the User, completing the query-response cycle. Additionally, the Answer is fed into a 'Chat history' module, shown as a dark gray rounded rectangle, via an upward arrow labeled 'Update'. The Chat history is also connected to the User with a rightward arrow labeled 'See', indicating that the user can view the conversation history. Furthermore, a feedback loop exists from the Chat history back to the User, suggesting ongoing interaction and context retention.\n\nAll modules are represented as rounded rectangles with distinct colors: blue for human roles (User, Admin/Instructor), light blue for intermediate data or input (Question, Text embedder), dark gray for storage components (Database, Chat history), purple for the core AI model (LLM), and red for the final output (Answer). Arrows indicate the direction of data or control flow, with labels specifying the nature of the action or data transfer. The diagram uses solid lines for direct flows and a dashed line for the retrieval of contextual information from the database to the LLM, highlighting the retrieval-augmented generation mechanism.",
    "The figure illustrates the architecture of CSAGNN, a graph neural network model designed for job-member matching. The global layout is a left-to-right flow, starting with input nodes on the far left and progressing through multiple processing stages to a final prediction and loss computation on the right. The structure is modular, with distinct blocks for embedding generation, attention mechanisms, neighbor sampling, aggregation, and prediction. The entire process is iterative, with a loop indicated by the '×(l−1)' notation, suggesting multiple layers or iterations of the neighbor sampling and aggregation steps. On the left, the input consists of a job node j₀ (blue diamond) and multiple member nodes m₀, m₁, m₂ (yellow circles), each connected to skill nodes s₀, s₁, s₂ (green triangles). These skills are associated with the job j₀ and are used to guide the attention mechanism. Each member node feeds into a module labeled 'Pre-trained Embeddings', which includes a BERT component (gray rectangle) for processing text information. The output of this module is concatenated with the pre-trained WHIN embedding, forming an initial representation for each member. This is visually represented by a horizontal bar with two color segments: pink (pre-trained embedding) and yellow (BERT output). These initial member representations h_m₀^(0), h_m₁^(0), h_m₂^(0) are then passed to a 'Skill-based cross-attention layer' (gray vertical rectangle). This layer re-weights the member representations based on the skills required by the job j₀, producing updated representations h_m₀, h_m₁, h_m₂. The job node j₀ also has its own initial representation h_j₀, shown as a blue bar, which is derived from its own text and pre-trained embeddings. The next stage involves a 'Neighbor Sampler' (gray box with dashed border) that selects relevant neighbors for each member based on similarity to the job's required skills. The sampled neighbors' representations are then aggregated using an 'Aggregation' module (gray rectangle), producing updated representations h_m₀^(1), h_m₁^(1), h_m₂^(1). This process is repeated for l−1 layers, as indicated by the '×(l−1)' label above the subsequent identical blocks, meaning the neighbor sampling and aggregation steps are applied iteratively. After the final iteration, the updated member representations h_m₀^(l), h_m₁^(l), h_m₂^(l) are combined with the job representation h_j₀ via an 'Average' operation (arrow pointing to a box labeled 'Average'). The result is fed into a set of 'MLP layers' (triangle-shaped block) to produce a 'Predict value' y_m₀,j₀. This predicted value is compared to the 'Ground-truth value' ŷ_m₀,j₀ using a 'loss' function, completing the training loop. Visual attributes include: job nodes as blue diamonds, member nodes as yellow circles, skill nodes as green triangles, BERT and embedding modules as gray rectangles, attention and aggregation layers as gray vertical bars, and the MLP as a triangle. Arrows indicate data flow, with solid lines for direct connections and dashed lines for optional or conditional paths. The figure uses color coding to distinguish between different types of embeddings (pink for pre-trained, yellow for BERT) and to highlight the iterative nature of the neighbor sampling and aggregation process.",
]

resolution_list = [ #w, h
    [1056,  528],
    [1056,  528],
    [1056,  528],
    [1056,  528],
]