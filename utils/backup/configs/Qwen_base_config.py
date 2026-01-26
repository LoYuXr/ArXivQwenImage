# ====== 通用运行设置（通常不变/默认） ======
seed = 42
device = 'cuda'
dtype = 'float32'

# ====== 模型/资源通用 ======
revision = None
variant = None

bnb_quantization_config_path = None

# Transformer 架构

# ====== 模型权重与结构 ======
pretrained_model_name_or_path = (
    # "/mnt/ephemeral/model_checkpoints/models--Qwen--Qwen-Image/"
    # "snapshots/9e31a1d489546029df1adfc9e12a08edafc692e7"
    "Qwen/Qwen-Image-2512"
)
huggingface_token = "***REMOVED***"

# LoRA 通用默认
use_lora = True
lora_layers = "to_k,to_q,to_v"
rank = 64
lora_alpha = 4
lora_dropout = 0.0
layer_weighting = 5.0

# VAE 架构默认
pos_embedding = 'rope'
layer_embedding = 'rope'
single_layer_decoder = 'vit'
decoder_arch = 'vit'
max_layers = 48


# 分辨率与增强
resolution = 512
center_crop = False
random_flip = False

# 训练规模与优化器默认
train_batch_size = 1
num_train_epochs = 1
max_train_steps = 80000
gradient_accumulation_steps = 1
gradient_checkpointing = True
cache_latents = False      # --cache_latents

optimizer = "AdamW"
use_8bit_adam = False
learning_rate = 2e-4
lr_scheduler = "constant"
lr_warmup_steps = 500
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 1e-4
adam_epsilon = 1e-08
max_grad_norm = 1.0

prodigy_beta3 = None
prodigy_decouple = True
prodigy_use_bias_correction = True
prodigy_safeguard_warmup = True


checkpointing_steps = 500
resume_from_checkpoint = "latest"
checkpoints_total_limit = None

# 混合精度
mixed_precision = "bf16"
allow_tf32 = False
upcast_before_saving = False
offload = False

#可视化
report_to = "wandb"
push_to_hub = False
hub_token = None
hub_model_id = None
cache_dir = None

# 验证/推理默认
scale_lr = False           # 等价 CLI: --scale_lr
lr_num_cycles = 1          # --lr_num_cycles
lr_power = 1.0             # --lr_power

weighting_scheme = "none"  #choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"], help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
logit_mean = 0.0                      # --logit_mean
logit_std = 1.0                       # --logit_std
mode_scale = 1.29                     # --mode_scale

# validation_guidance_scale = 1.0
# validation_prompts = [
#   "The image is a promotional graphic for a new collection. It features a woman in a red dress with a red headscarf. The woman is looking directly at the camera with a slight smile. The background is a solid red color, which matches the color of her dress and headscarf. The text \"NEW COLLECTION.\" is prominently displayed at the top of the image, and \"Must Haves\" is written at the bottom. The overall style of the image is sleek and modern, with a focus on the woman and the color red.",
#   "The image is a promotional graphic for a free online course. It features a young woman with blonde hair, wearing a black long-sleeved top, sitting at a desk and typing on a laptop. The laptop is open and appears to be in use. The woman is looking down at the laptop screen, suggesting she is engaged in the activity. The graphic includes text elements that provide information about the course. There is a circular logo with the text \"YOUR SCHOOL\" in white letters on a blue background. Below the logo, the words \"FREE ONLINE COURSE\" are written in bold, capital letters. To the right of the logo, there is a date \"JUNE 20\" in a smaller font size. The background of the image is a light blue with a subtle grid pattern, giving it a clean and organized look. The overall style of the image is modern and professional, designed to attract potential students to the online course.",
#   "The image is a festive New Year greeting card. It features a warm, golden-yellow background with a subtle bokeh effect that gives the impression of a starry night or a glittering celebration. In the center, there is a large, shiny gold number \"2022\" that stands out prominently. To the left of the number, there is a smaller gold number \"2\" and to the right, a smaller gold number \"2\". These numbers are also shiny and gold, and they are positioned in such a way that they appear to be part of the larger \"2022\" number. In front of the numbers, there is a golden gift box with a bow on top, suggesting a sense of giving or receiving. The gift box is also shiny and gold, matching the overall color scheme of the image. At the top of the image, there is a text that reads \"Happy New Year!\" in a large, elegant font that is also gold in color. Below this, there is a motivational phrase that says \"Give wings to your dreams and let them come true in 2022.\" This text is smaller than the New Year greeting but still prominent and clear. The overall style of the image is celebratory and inspirational, designed to inspire hope and positivity for the new year. The use of gold and the bokeh effect contribute to a luxurious and dreamy atmosphere.",
#   "The image displays a menu from a restaurant named \"GREAT FOOD RESTAURANT.\" The menu is divided into two sections: \"MAIN COURSE\" and \"DRINKS.\" Under the main course, there are four items listed with their respective prices: \"Tuna Tartar\" for $20.00, \"Red Onions\" for $20.00, \"Butter Chicken\" for $20.00, and \"Washabi Sushi\" for $20.00. In the drinks section, there are five items listed: \"Orange Juice\" for $20.00, \"Apple Juice\" for $20.00, \"Chocolate Milk\" for $20.00, \"Cheesecake\" for $20.00, and \"Ice Cream\" for $20.00. The background of the menu features a black and orange color scheme with splashes of orange paint, giving it a vibrant and artistic feel. There are two photographs of food items on the menu: one at the top left corner showing a plate of food that appears to be a salad, and another at the bottom right corner showing a dish that looks like tacos. The text on the menu is white, which stands out against the black background, making it easy to read. The overall design of the menu is modern and visually appealing"
# ]

# validation_boxes = [
#   [[0, 0, 512, 512], [0, 0, 512, 512], [400, 128, 512, 384], [0, 360, 232, 512], [232, 48, 464, 120], [32, 320, 152, 440], [0, 408, 184, 512], [40, 464, 136, 488], [176, 136, 472, 440], [368, 112, 512, 248], [32, 192, 152, 312], [32, 72, 152, 192]],
#   [[0, 0, 512, 512], [0, 0, 512, 512], [8, 8, 504, 504], [32, 368, 360, 440], [368, 376, 480, 440], [48, 392, 344, 424], [384, 392, 456, 424], [32, 24, 344, 336], [48, 32, 352, 344], [56, 40, 368, 352], [32, 448, 200, 464], [56, 48, 360, 352], [352, 464, 376, 488], [448, 56, 480, 88], [320, 128, 456, 264], [336, 160, 432, 240]],
#   [[0, 0, 512, 512], [0, 0, 512, 512], [32, 200, 480, 456], [128, 80, 392, 128], [152, 144, 360, 184], [192, 304, 280, 408], [104, 304, 288, 408], [104, 304, 416, 408], [88, 288, 440, 408]],
#   [[0, 0, 512, 512], [0, 0, 512, 512], [368, 0, 512, 216], [248, 72, 344, 104], [352, 176, 400, 256], [248, 152, 392, 168], [248, 48, 400, 72], [8, 128, 240, 384], [64, 0, 192, 208], [48, 408, 304, 512], [208, 440, 392, 512], [248, 280, 328, 296], [248, 176, 336, 256], [248, 312, 336, 400], [352, 304, 400, 408]]
# ]
