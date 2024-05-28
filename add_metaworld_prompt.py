with open("metaworld_data/folder/train/metadata.csv", "r") as f:
    origin_prompts = [s.strip().split(",") for s in f.readlines()]

with open("./prompts_exp.txt", "r") as f:
    prompts_exp = [s.strip().split(",") for s in f.readlines()]

for task in range(len(prompts_exp)):
    for item in range(len(prompts_exp[task])):
        prompts_exp[task][item] = " ".join(prompts_exp[task][item].split("-")[:-1])

# print(origin_prompts)
# print(prompts_exp)

for p in origin_prompts:
    for p_exp in prompts_exp:
        if p[1] in p_exp:
            p[1] = p_exp[0]
            p_exp.append(p_exp.pop(0))

# print(origin_prompts)

origin_prompts = "\n".join(map(lambda x: ",".join(x), origin_prompts))

# for i in origin_prompts:
#     print(i)

with open("metaworld_data/folder/train/metadata.csv", "w") as f:
    f.write(origin_prompts)
