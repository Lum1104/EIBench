import json

score_files = ["score_7866.jsonl", "score_7862.jsonl", "score_7861.jsonl"]

overall_stds = []
overall_avgs = []
overall_min_scores = []
overall_max_scores = []

happy_avgs = []
happy_stds = []
happy_min_scores = []
happy_max_scores = []

sad_avgs = []
sad_stds = []
sad_min_scores = []
sad_max_scores = []

angry_avgs = []
angry_stds = []
angry_min_scores = []
angry_max_scores = []

excited_avgs = []
excited_stds = []
excited_min_scores = []
excited_max_scores = []

# open the files and read the content
for file in score_files:
    scores_json = []
    happy_scores = []
    sad_scores = []
    angry_scores = []
    excited_scores = []
    with open(file, "r") as f:
        for line in f:
            scores_json.append(json.loads(line))
    for score_json in scores_json:
        for key, scores in score_json.items():
            if "happy" in key.lower():
                happy_scores.append(int(scores))
            elif "sad" in key.lower():
                sad_scores.append(int(scores))
            elif "angry" in key.lower():
                angry_scores.append(int(scores))
            elif "excitement" in key.lower():
                excited_scores.append(int(scores))

    # calculate the average scores
    happy_avg = sum(happy_scores) / len(happy_scores)
    sad_avg = sum(sad_scores) / len(sad_scores)
    angry_avg = sum(angry_scores) / len(angry_scores)
    excited_avg = sum(excited_scores) / len(excited_scores)

    # calculate the standard deviation
    happy_std = (
        sum([(score - happy_avg) ** 2 for score in happy_scores]) / len(happy_scores)
    ) ** 0.5
    sad_std = (
        sum([(score - sad_avg) ** 2 for score in sad_scores]) / len(sad_scores)
    ) ** 0.5
    angry_std = (
        sum([(score - angry_avg) ** 2 for score in angry_scores]) / len(angry_scores)
    ) ** 0.5
    excited_std = (
        sum([(score - excited_avg) ** 2 for score in excited_scores])
        / len(excited_scores)
    ) ** 0.5
    # calculate the min and max scores
    happy_min = min(happy_scores)
    sad_min = min(sad_scores)
    angry_min = min(angry_scores)
    excited_min = min(excited_scores)

    happy_max = max(happy_scores)
    sad_max = max(sad_scores)
    angry_max = max(angry_scores)
    excited_max = max(excited_scores)

    # calculate overall average
    overall_avg = (happy_avg + sad_avg + angry_avg + excited_avg) / 4
    # calculate overall standard deviation
    overall_std = (
        (happy_std**2 + sad_std**2 + angry_std**2 + excited_std**2) / 4
    ) ** 0.5
    # calculate overall min and max
    overall_min = min(happy_min, sad_min, angry_min, excited_min)

    # print the results
    print(f"File: {file}")
    print(
        f"Happy: avage: {happy_avg} std: ({happy_std}) [min, max] [{happy_min}, {happy_max}]"
    )
    print(f"({happy_avg}, {happy_std}, [{happy_min}, {happy_max}])")
    print(
        f"Angry: average: {angry_avg} std: ({angry_std}) [min, max] : [{angry_min}, {angry_max}]"
    )
    print(f"({angry_avg}, {angry_std}, [{angry_min}, {angry_max}])")
    print(
        f"Sad: average: {sad_avg} std: ({sad_std}) [min, max] : [{sad_min}, {sad_max}]"
    )
    print(f"({sad_avg}, {sad_std}, [{sad_min}, {sad_max}])")
    print(f"Excited")
    print(f"({excited_avg}, {excited_std}, [{excited_min}, {excited_max}])")
    print(f"Overall:")
    print(
        f"({overall_avg}, {overall_std}, [{overall_min}, {max(happy_max, sad_max, angry_max, excited_max)}])"
    )
    print("\n")
    # append the results
    overall_avgs.append(overall_avg)
    overall_stds.append(overall_std)
    overall_min_scores.append(overall_min)
    overall_max_scores.append(max(happy_max, sad_max, angry_max, excited_max))

    happy_avgs.append(happy_avg)
    happy_stds.append(happy_std)
    happy_min_scores.append(happy_min)
    happy_max_scores.append(happy_max)

    sad_avgs.append(sad_avg)
    sad_stds.append(sad_std)
    sad_min_scores.append(sad_min)
    sad_max_scores.append(sad_max)

    angry_avgs.append(angry_avg)
    angry_stds.append(angry_std)
    angry_min_scores.append(angry_min)
    angry_max_scores.append(angry_max)

    excited_avgs.append(excited_avg)
    excited_stds.append(excited_std)
    excited_min_scores.append(excited_min)
    excited_max_scores.append(excited_max)


# calculate the metric across three file

# calculate the average scores
happy_avg = sum(happy_avgs) / len(happy_avgs)
sad_avg = sum(sad_avgs) / len(sad_avgs)
angry_avg = sum(angry_avgs) / len(angry_avgs)
excited_avg = sum(excited_avgs) / len(excited_avgs)

# calculate the standard deviation
happy_std_temp = 0
sad_std_temp = 0
angry_std_temp = 0
excited_std_temp = 0
for happy_std in happy_stds:
    happy_std_temp += happy_std**2
happy_std = (happy_std_temp / len(happy_stds)) ** 0.5

for sad_std in sad_stds:
    sad_std_temp += sad_std**2
sad_std = (sad_std_temp / len(sad_stds)) ** 0.5

for angry_std in angry_stds:
    angry_std_temp += angry_std**2
angry_std = (angry_std_temp / len(angry_stds)) ** 0.5

for excited_std in excited_stds:
    excited_std_temp += excited_std**2
excited_std = (excited_std_temp / len(excited_stds)) ** 0.5

# calculate the min and max scores
happy_min = min(happy_min_scores)
sad_min = min(sad_min_scores)
angry_min = min(angry_min_scores)
excited_min = min(excited_min_scores)

happy_max = max(happy_max_scores)
sad_max = max(sad_max_scores)
angry_max = max(angry_max_scores)
excited_max = max(excited_max_scores)

# calculate overall average
overall_avg = (happy_avg + sad_avg + angry_avg + excited_avg) / 4
# calculate overall standard deviation
overall_std = ((happy_std**2 + sad_std**2 + angry_std**2 + excited_std**2) / 4) ** 0.5
# calculate overall min and max
overall_min = min(happy_min, sad_min, angry_min, excited_min)
overall_max = max(happy_max, sad_max, angry_max, excited_max)

# print the results
print("Overall")
print(f"({happy_avg}, {happy_std}, [{happy_min}, {happy_max}])")
print(f"({angry_avg}, {angry_std}, [{angry_min}, {angry_max}])")
print(f"({sad_avg}, {sad_std}, [{sad_min}, {sad_max}])")
print(f"({excited_avg}, {excited_std}, [{excited_min}, {excited_max}])")
print(f"({overall_avg}, {overall_std}, [{overall_min}, {overall_max}])")
