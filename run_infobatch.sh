pip install -r requirements.txt
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
cd ..
CUDA_VISIBLE_DEVICES=0 python3 finetune_infobatch.py \
    --base_model '<path-to-llama-7b-hf>' \
    --data_path alpaca_data_dq_k5_1k.json \
    --output_dir './lora-alpaca' \
    --num_epochs 15 \
    --val_set_size=1000 \
    --learning_rate 1.5e-4 \
    --batch_size 64 \
    --cutoff_len=512 \
    --output_dir='./lora-alpaca' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=8
