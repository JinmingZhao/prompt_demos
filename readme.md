# 在文本情感分类上进行验证
包含两种prompt方法一种是基于 mask langauge model的，一种是基于 next sentence predition 任务的

# IEMOCAP 
10-folder Cross-Validation Speaker-Independent(average perfomance) 
happy [3407] anger [4963] sad [6517] neutral [8699]

| Model | UAR    | F1 |
| :----| :----: | :----:  |
| Bert-base-uncase + Finetune | 0.72164 | 0.70953 |
| Bert-base-uncase + prompt(it was )  | 0.72145 | 0.7039 |
| Bert-base-uncase + prompt(i am )  | 0.72337 | 0.70505 |
| Bert-base-uncase + prompt+NSP(it was)  | 0.73989 | 0.72735 |
| Bert-base-uncase + prompt+NSP(i am)  |  0.74204 | 0.72568 |

# MSP-IMPROV 
12-folder Cross-Validation Speaker-Independent(average perfomance)

| Model | UAR    | F1 |
| :----| :----: | :----:  |
| Bert-base-uncase + Finetune | 0.6226  | 0.6379  |
| Bert-base-uncase + prompt(it was)  |  |  |
| Bert-base-uncase + prompt(i am)  |  |  |
| Bert-base-uncase + prompt+NSP(it was)  |  |  |
| Bert-base-uncase + prompt+NSP(i am)  |  |  |

# reference
Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing. Penfeng Liu et al. CMU.
NSP-BERT: A Prompt-based Zero-Shot Learner Through an Original Pre-training Task--Next Sentence Prediction. Yi Sun et al.  中国人民解放军陆军工程大学.