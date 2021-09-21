source activate transformers
export PYTHONPATH=/data7/MEmoBert

gpuid=$1
data_dir=/data7/emobert/exp/promote_pretrain/data/iemocap
result_dir=/data7/emobert/exp/promote_pretrain/results/iemocap
for cvNo in `seq 1 10`;
do
    CUDA_VISIBLE_DEVICES=${gpuid} python run_nsp.py \
        --model_name_or_path  /data2/zjm/tools/LMs/bert_base_en \
        --do_train --do_eval \
        --train_file ${data_dir}/${cvNo}/trn_val_nsp_iam.csv \
        --validation_file ${data_dir}/${cvNo}/tst_nsp_iam.csv \
        --max_seq_length 50 \
        --per_device_batch_size 32 \
        --learning_rate 2e-5 \
        --max_grad_norm 5.0 \
        --num_train_epochs 10 \
        --evaluation_strategy 'epoch' \
        --lr_scheduler_type 'linear' \
        --output_dir ${result_dir}/bert_prompt_nsp_iam_base_uncased_2e5_epoch10_bs32_trnval/${cvNo}
done