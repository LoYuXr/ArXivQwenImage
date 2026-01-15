# Compute FID between two S3 prefixes
# python test_fid_vc5k.py \
#   --src /mnt/ephemeral/experiments/vc5k_fid_base_512 \
#   --dst s3://core-cn-model-trainer.canva.com/usr/yuxuanluo/experiments/828-qwen-image-lora-H200-ablation+512res_Prodigy_64Rank_fullLoRA_weight5_10w/merged_rgba/ \
#   --aws-region us-east-1 \
#   --model-trainer-arn arn:aws:iam::051687089423:role/service.core-cn-model-trainer \
#   --device cuda \
#   --cleanup

# python test_fid_vc5k.py \
#   --src /mnt/ephemeral/experiments/vc5k_fid_base_512 \
#   --dst s3://core-cn-model-trainer.canva.com/usr/yuxuanluo/experiments/inference_fid_vc5k/910_H200_Prodigy_Rank64_weigh2_5/910_inference_test_512res_Prodigy_64Rank_fullLoRA_weight2_5_5w/merged_rgba \
#   --aws-region us-east-1 \
#   --model-trainer-arn arn:aws:iam::051687089423:role/service.core-cn-model-trainer \
#   --device cuda \
#   --cleanup

# python test_fid_vc5k.py \
#   --src /mnt/ephemeral/experiments/vc5k_fid_base_512 \
#   --dst s3://core-cn-model-trainer.canva.com/usr/yuxuanluo/experiments/inference_fid_vc5k/911_H200_Adamw_2e_4_Rank64_weight5/905-lora-H200-512res_Adamw2e_4_64Rank_weight5_5w/merged_rgba \
#   --aws-region us-east-1 \
#   --model-trainer-arn arn:aws:iam::051687089423:role/service.core-cn-model-trainer \
#   --device cuda \
#   --cleanup;

python test_fid_vc5k.py \
  --src /mnt/ephemeral/experiments/vc5k_fid_base_512 \
  --dst s3://core-cn-model-trainer.canva.com/usr/yuxuanluo/experiments/inference_fid_vc5k/913_H200_Prodigy_Rank128_weigh2_5/913_inference_test_512res_Prodigy_128Rank_fullLoRA_weight2_5_10w/merged_rgba \
  --aws-region us-east-1 \
  --model-trainer-arn arn:aws:iam::051687089423:role/service.core-cn-model-trainer \
  --device cuda \
  --cleanup;

python test_fid_vc5k.py \
  --src /mnt/ephemeral/experiments/vc5k_fid_base_512 \
  --dst s3://core-cn-model-trainer.canva.com/usr/yuxuanluo/experiments/inference_fid_vc5k/913_H200_Prodigy_Rank128_weigh2_5/913_inference_test_512res_Prodigy_128Rank_fullLoRA_weight2_5_9w/merged_rgba \
  --aws-region us-east-1 \
  --model-trainer-arn arn:aws:iam::051687089423:role/service.core-cn-model-trainer \
  --device cuda \
  --cleanup;

python test_fid_vc5k.py \
  --src /mnt/ephemeral/experiments/vc5k_fid_base_512 \
  --dst s3://core-cn-model-trainer.canva.com/usr/yuxuanluo/experiments/inference_fid_vc5k/913_H200_Prodigy_Rank128_weigh2_5/913_inference_test_512res_Prodigy_128Rank_fullLoRA_weight2_5_8w/merged_rgba \
  --aws-region us-east-1 \
  --model-trainer-arn arn:aws:iam::051687089423:role/service.core-cn-model-trainer \
  --device cuda \
  --cleanup;
