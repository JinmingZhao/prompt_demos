source activate transformers
export PYTHONPATH=/data7/MEmoBert

data_dir=/data7/emobert/exp/promote_pretrain/data/iemocap
result_dir=/data7/emobert/exp/promote_pretrain/results/iemocap
for cvNo in `seq 1 1`;
do
    CUDA_VISIBLE_DEVICES=7 python run_mask.py \
        --model_name_or_path  /data2/zjm/tools/LMs/bert_base_en \
        --do_train --do_eval \
        --train_file ${data_dir}/${cvNo}/trn_val_mask_itwas.csv \
        --validation_file ${data_dir}/${cvNo}/tst_mask_itwas.csv \
        --max_seq_length 50 \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 64 \
        --gradient_accumulation_steps 2 \
        --learning_rate 2e-5 \
        --max_grad_norm 5.0 \
        --num_train_epochs 10 \
        --evaluation_strategy 'epoch' \
        --lr_scheduler_type 'linear' \
        --output_dir ${result_dir}/${cvNo}/bert_prompt_base_uncased_2e5_epoch10_bs64_trnval
done