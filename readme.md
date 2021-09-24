# 在文本情感分类上进行验证
包含两种prompt方法一种是基于 mask langauge model的，一种是基于 next sentence predition 任务的
多尝试几种不同的prompt template.
比如
i am happy.
it was happy.
i'm happy.
overall, it expresses happy emotion.
it it happy.
happy.
happy emotion.


# IEMOCAP 
10-folder Cross-Validation Speaker-Independent(average perfomance) 
happy [3407] anger [4963] sad [6517] neutral [8699]

| Model | UAR    | F1 |
| :----| :----: | :----:  |
| Bert-base-uncase + Finetune | 0.72164 | 0.70953 |
| Bert-base-uncase + prompt(it was )  | 0.72145 | 0.7039 |
| Bert-base-uncase + prompt(i am )  | 0.72337 | 0.70505 |
| Bert-base-uncase + prompt(i‘m )  | 0.71742 | 0.69939 |
| Bert-base-uncase + prompt(only mask )  | 0.70493 | 0.7039 |
| Bert-base-uncase + prompt+NSP(it was)  | 0.73989 | 0.72735 |
| Bert-base-uncase + prompt+NSP(i am)  |  0.74204 | 0.72568 |
| Bert-base-uncase + prompt+NSP(i'm)  | 0.74238 | 0.72547 |
| Bert-base-uncase + prompt+NSP(only mask)  | 0.73911 | 0.72252 |



# MSP-IMPROV 
12-folder Cross-Validation Speaker-Independent(average perfomance)
| Model | UAR    | F1 |
| :----| :----: | :----:  |
| Bert-base-uncase + Finetune | 0.6226  | 0.6379  |
| Bert-base-uncase + prompt(it was)  | 0.6202 |  0.6274 |
| Bert-base-uncase + prompt(i am)  | 0.6201 | 0.6239 |
| Bert-base-uncase + prompt(i'm)  | 0.6198 | 0.6164 |
| Bert-base-uncase + prompt(only mask)  | 0.6168 | 0.6213 |
| Bert-base-uncase + prompt+NSP(it was)  | 0.5966 | 0.6254 |
| Bert-base-uncase + prompt+NSP(i am)  | 0.5997 | 0.6249 |
| Bert-base-uncase + prompt+NSP(i'm)  | 0.6078 | 0.6261 |
| Bert-base-uncase + prompt+NSP(only mask)  | 0.6127 | 0.6378 |

# reference
Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing. Penfeng Liu et al. CMU.
NSP-BERT: A Prompt-based Zero-Shot Learner Through an Original Pre-training Task--Next Sentence Prediction. Yi Sun et al.  中国人民解放军陆军工程大学.