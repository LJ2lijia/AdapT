import json

adapt_file = open("/home/zhuyq/CodeGeeX/output-230613/humaneval/adapt/0.8_0.3.jsonl_results.jsonl")
sample_file = open("/home/zhuyq/CodeGeeX/output-230613/humaneval/baseline/he_15_0.2.jsonl_results.jsonl")

adapt_lines = adapt_file.readlines()
sample_lines = sample_file.readlines()

adapt_dict = {}
sample_dict = {}

for line in adapt_lines:
    id = json.loads(line)['task_id']
    if id not in adapt_dict.keys():
        adapt_dict[id] = 0
    
    passed = json.loads(line)['passed']
    if passed == True:
        if id in adapt_dict.keys():
            adapt_dict[id] += 1
        else:
            adapt_dict[id] = 1

print("adapt_result", adapt_dict)
for line in sample_lines:
    id = json.loads(line)['task_id']
    if id not in sample_dict.keys():
        sample_dict[id] = 0
    
    passed = json.loads(line)['passed']
    if passed == True:
        if id in sample_dict.keys():
            sample_dict[id] += 1
        else:
            sample_dict[id] = 1
print("sample_result", sample_dict)
key_list = []
for k in adapt_dict.keys():
    if adapt_dict[k] == 0:
        if sample_dict[k] == 0:
            key_list.append(k)

for k in key_list:
    adapt_dict.pop(k, None)
    sample_dict.pop(k, None)


import csv
# 打开文件 
file = open('/home/zhuyq/CodeGeeX/dict_csv.csv','w',encoding='utf-8',newline='')
#先设置列名，并写入csv文件
csv_writer= csv.DictWriter(file, fieldnames=adapt_dict.keys())
csv_writer.writeheader()  

csv_writer.writerow(adapt_dict)   #数据写入csv文件
file.close()

file_1 = open('/home/zhuyq/CodeGeeX/dict_csv_sample.csv','w',encoding='utf-8',newline='')
#关闭文件夹
csv_writer_1= csv.DictWriter(file_1, fieldnames=sample_dict.keys())
csv_writer_1.writeheader()
csv_writer_1.writerow(sample_dict)
file_1.close()
