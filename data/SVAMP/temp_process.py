import json
import random

# with open('data/zero-shot-cot-test-data/MultiArith/MultiArith.json', 'r') as f:
#     data = json.load(f)


# random_numbers = random.sample(range(600), 400)

# res_test = []
# res_dev = []
# for i, line in enumerate(data):
#     if i in random_numbers:
#         res_test.append(line)
#     else:
#         res_dev.append(line)

# print(len(data), len(res_test), len(res_dev))

# with open("data/zero-shot-cot-test-data/MultiArith/MultiArith_test.json", "w") as file:
#     json.dump(res_test, file)

# with open("data/zero-shot-cot-test-data/MultiArith/MultiArith_dev.json", "w") as file:
#     json.dump(res_dev, file)


with open('data/zero-shot-cot-test-data/MultiArith/MultiArith_test.json', 'r') as f:
    data = json.load(f)

f_w = open('data/zero-shot-cot-test-data/MultiArith/MultiArith_test.txt', 'w')

for line in data:
    temp = {}
    question = line['sQuestion']
    answer = line['lEquations'][0] + ' #### ' + str(line['lSolutions'][0])
    temp['question'] = question
    temp['answer'] = answer
    f_w.write(str(temp) + '\n')
    f_w.flush()
f_w.close()