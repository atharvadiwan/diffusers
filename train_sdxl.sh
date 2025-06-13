export MODEL_NAME="/workspace/data/stable_diffusion_weights/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="./exp/sdxl_asv_exp1"
# export HUB_MODEL_ID="naruto-lora"
# export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  examples/text_to_image/train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=150 \
  --lr_scheduler="cosine" \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=500 \
  --validation_epochs=10 \
  --validation_prompt="The image depicts a narrow road that appears to be a bridge or overpass, likely leading to an industrial area. The road is bordered by a metal fence on the left side, with a mix of red and black stripes. The road is lined with various types of vehicles, including cars and trucks. The time of day seems to be during sunset" \
  --seed=1337