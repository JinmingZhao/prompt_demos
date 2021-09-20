# 在文本情感分类上进行验证
包含两种prompt方法一种是基于 mask langauge model的，一种是基于 next sentence predition 任务的

# IEMOCAP 
10-folder Cross-Validation Speaker-Independent(average perfomance) 
happy [3407] anger [4963] sad [6517] neutral [8699]

| Model | UAR    | F1 |
| :----| :----: | :----:  |
| Bert-base-uncase + Finetune |  |  |
| Bert-base-uncase + prompt(It was )  |  |  |
| Bert-base-uncase + prompt(I am )  |  |  |
| Bert-base-uncase + prompt + NSP  |  |  |

# MSP-IMPROV 
12-folder Cross-Validation Speaker-Independent(average perfomance)

| Model | UAR    | F1 |
| :----| :----: | :----:  |
| Bert-base-uncase + Finetune |  |  |
| Bert-base-uncase + prompt(It was )  |  |  |
| Bert-base-uncase + prompt(I am )  |  |  |
| Bert-base-uncase + prompt + NSP  |  |  |

# reference
NSP-BERT: A Prompt-based Zero-Shot Learner Through an Original Pre-training Task--Next Sentence Prediction