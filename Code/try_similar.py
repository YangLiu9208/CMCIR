import numpy as np
# import paddle as P
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


from scipy import spatial
from scipy.special import softmax
import numpy as np

def cosine_similarity(a, b):
    return 1 - spatial.distance.cosine(a, b)

def euclidean_metric(a, b):
    return np.linalg.norm(a - b)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained("bert-base-cased")

# 0错1半对2全对
ans_text = ["购买门票一共需要250元.","购买门票一共需要250元."]
student_texts = [
    ("一共需要250元。", 1),
    ("购买门票一共要250元。", 2),
    ("要200元。", 0),
    ("要250元。", 1),
    ("购买门票一共需要205元。", 0),
    ("购买门票", 0),
    ("一共需要256元", 0),
    ("购买门票一共要250元。", 2),
    ("购买门票一共需要2400元。", 0),
    ("一共需要20506元。", 0),
    ("一共需要250 $ \\tugai $元。", 1),
    ("一共需要 $ 250 ^ { \\circ } $", 1),
    ("购买门票一共需要250元。 $ \\tugai $", 2),
    ("购买门票一共需要240元。", 0),
    ("购买门票一共需要240元.", 0),
    ("一共需要250元。", 1),
    ("购买门两票一共248元。", 0),
    ("购买门票一共需要250元。", 2),
    ("购买门票共要250元。", 2),
    ("需要125元。", 0),
    ("购买门要一共需要75元。", 0),
    ("$ \\tugai $ 购买门票一共需要230元。", 0),
    ("购买门票一共需要 $ \\tugai \\tugai 50 + 43 = 93 $(元)", 0),
    ("一共要500元", 0),
    ("一共需要230元钱。", 0),
]

# ans_text = "购买门票一共需要NUM元."
# student_texts = [
#     ("一共需要NUM元。", 1),
#     ("购买门票一共要NUM元。", 2),
#     ("要NUM元。", 0),
#     ("要NUM元。", 1),
#     ("购买门票一共需要NUM元。", 0),
#     ("购买门票", 0),
#     ("一共需要NUM元", 0),
#     ("购买门票一共要NUM元。", 2),
#     ("购买门票一共需要NUM元。", 0),
#     ("一共需要NUM元。", 0),
#     ("一共需要NUM元。", 1),
#     ("一共需要NUM", 1),
#     ("购买门票一共需要NUM元。", 2),
#     ("购买门票一共需要NUM元。", 0),
#     ("购买门票一共需要NUM元.", 0),
#     ("一共需要NUM元。", 1),
#     ("购买门两票一共NUM元。", 0),
#     ("购买门票一共需要NUM元。", 2),
#     ("购买门票共要NUM元。", 2),
#     ("需要NUM元。", 0),
#     ("购买门要一共需要NUM元。", 0),
#     ("购买门票一共需要NUM元。", 0),
#     ("购买门票一共需要NUM(元)", 0),
#     ("一共要NUM元", 0),
#     ("一共需要NUM元钱。", 0),
#     ("买票要NUM元钱。", 0),
#

# text = '3. (易错题)某市电影院推出两种买票方案.方案一 成人: $40$元/人 儿童: $20$元/人 方案二 团体: $6$人以上(包括$6$人) $25$元/人 （1）杨老师和李老师带领实验小学四年级$38$名学生去看电影，选那种方案划算？'
# ck_words = ['成人', '儿童', '老师', '学生']
# ck_pair1 = ['成人', '儿童']
# ck_pair2 = ['老师', '学生']

text = '3. (易错题)某市电影院推出两种买票方案.方案一 成人:$40$元/人 儿童:$20$元/人 方案二 团体:$6$人以上(包括$6$人) $25$元/人 （2）如果是$38$个大人和$2$个儿童去看电影，选那种方案划算？'
ck_words = ['成人', '儿童', '大人']
ck_pair1 = ['成人', '儿童']
ck_pair2 = ['大人', '儿童']

ans_tokens_dict = tokenizer(ans_text, padding=True)
ans_input_batch = ans_tokens_dict["input_ids"]
ans_attention_mask_batch = ans_tokens_dict["attention_mask"]
ans_token_type_ids_batch = ans_tokens_dict["token_type_ids"]

# ids, ret = tokenizer.encode(text)
ans_input_var = torch.LongTensor(ans_input_batch)
ans_attention_mask_var = torch.LongTensor(ans_attention_mask_batch)
ans_token_type_ids_var = torch.LongTensor(ans_token_type_ids_batch)
pooled_output  = model(input_ids=ans_input_var, attention_mask=ans_attention_mask_var,
                token_type_ids=ans_token_type_ids_var)                 # eager execution

# print(ans_outputs[1][0])

for student_data in student_texts:
    student_text = student_data[0]
    student_flag = student_data[1]
    student_tokens_dict = tokenizer([student_text], padding=True)
    student_input_batch = student_tokens_dict["input_ids"]
    student_attention_mask_batch = student_tokens_dict["attention_mask"]
    student_token_type_ids_batch = student_tokens_dict["token_type_ids"]

    student_input_var = torch.LongTensor(student_input_batch)
    student_attention_mask_var = torch.LongTensor(student_attention_mask_batch)
    student_token_type_ids_var = torch.LongTensor(student_token_type_ids_batch)
    student_outputs = model(input_ids=student_input_var, attention_mask=student_attention_mask_var,
                    token_type_ids=student_token_type_ids_var)                 # eager execution

    ans_embedding = ans_outputs[1][0].detach().numpy()
    student_embedding = student_outputs[1][0].detach().numpy()

    # print("欧式距离度量：", ans_text, student_text, student_flag, euclidean_metric(ans_embedding, student_embedding))
    # print("Cosine相似度度量：", ans_text, student_text, student_flag, cosine_similarity(ans_embedding, student_embedding))
    # print()
    ans_tokens_list = [[s for s in ans_text]]
    student_tokens_list = [s for s in student_text]
    # print(ans_tokens_list)
    print("BLEU度量：", ans_text, student_text, sentence_bleu(ans_tokens_list, student_tokens_list, weights=(0.5, 0.5), smoothing_function=SmoothingFunction().method1))