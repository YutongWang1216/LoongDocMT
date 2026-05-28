

echo "[$(date)] SFT Training Start"
FORCE_TORCHRUN=1 llamafactory-cli train full_sft.yaml > logs/train_sft.log 2>&1 && \
echo "[$(date)] SFT Training End"

echo "[$(date)] DPO Training Start"
FORCE_TORCHRUN=1 llamafactory-cli train lora_dpo.yaml > logs/train_dpo.log 2>&1 && \
echo "[$(date)] DPO Training End"
