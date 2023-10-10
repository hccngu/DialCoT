import json

input_path = 'data/GSM8K/DialCoT-A.json'
input_path_2 = 'data/GSM8K/CoT-FT.json'
output_path = 'data/GSM8K/selfask-A.json'
output_path_2 = 'data/GSM8K/selfask-S.json'

res = {}

for i, line in enumerate(open(input_path, 'r').readlines()):
    if i >= 7473: break
    line_dic = json.loads(line)
    res[line_dic['source'][8:].strip().split('\n')[0]] = line_dic['target'].strip()

# with open(output_path, 'w', encoding='utf-8') as f_w:
#     for i, line in enumerate(open(input_path_2, 'r').readlines()):
#         line_dic = json.loads(line)
#         questions = res[line_dic['source'].strip()].split('\n')
#         answers = line_dic['target'].split('\n')
#         assert len(questions) == len(answers)
#         temp = {}
#         temp['source'] = line_dic['source']
#         temp['target'] = ''
#         for j in range(len(questions)):
#             temp['target'] = temp['target'] + 'Follow up: ' + questions[j] + '\nIntermediate answer: ' + answers[j] + '\n'
#         temp['target'] = temp['target'].strip()
#         f_w.write(json.dumps(temp, ensure_ascii=False) + '\n')
#         f_w.flush()


with open(output_path_2, 'w', encoding='utf-8') as f_w:
    for i, line in enumerate(open(input_path_2, 'r').readlines()):
        line_dic = json.loads(line)
        questions = res[line_dic['source'].strip()].split('\n')
        answers = line_dic['target'].split('\n')
        assert len(questions) == len(answers)
        temp = {}
        temp['source'] = line_dic['source'] + '\nFollow up: '
        for j in range(len(questions)):
            if j == len(questions) - 1:
                questions[j] = 'Final question:' + questions[j][14:]
            temp['target'] = questions[j]
            f_w.write(json.dumps(temp, ensure_ascii=False) + '\n')
            f_w.flush()
            temp['source'] = temp['source'] + temp['target'] + '\nIntermediate answer: '
            temp['target'] = answers[j]
            f_w.write(json.dumps(temp, ensure_ascii=False) + '\n')
            f_w.flush()
            temp['source'] = temp['source'] + temp['target'] + '\nFollow up: '
            #  + '\nIntermediate answer: ' + answers[j] + '\n'

print('a')