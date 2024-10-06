import json
import re
import argparse

def extract_scores_from_jsonl(file_path):
    scores = []
    happy = []
    sad = []
    angry = []
    excit = []
    score_pattern = re.compile(r"\{score: (\d+)/\d+\}")
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            for path, score_str in data.items():
                match = score_pattern.search(score_str)
                if match:
                    correct, total = int(match[0].split(" ")[1].split("/")[0]), int(match[0].split(" ")[1].split("/")[1].replace("}", ""))
                    score_value = correct / total
                    scores.append(score_value)
                else:
                    print(path)
                if "Happy" in path:
                    happy.append(score_value)
                elif "sad" in path:
                    sad.append(score_value)
                elif "Angry" in path:
                    angry.append(score_value)
                elif "excit" in path:
                    excit.append(score_value)
    return scores, happy, angry, sad, excit

def calculate_average_score(scores):
    if not scores:
        return 0
    return sum(scores) / len(scores)

def main(file_path):
    scores, happy, angry, sad, excit = extract_scores_from_jsonl(file_path)
    average_score = calculate_average_score(scores)
    happy_score = calculate_average_score(happy)
    angry_score = calculate_average_score(angry)
    sad_score = calculate_average_score(sad)
    excit_score = calculate_average_score(excit)

    # print(f"Scores: {scores}")
    print(f"happy Score: {happy_score:.5f}")
    print(f"angry Score: {angry_score:.5f}")
    print(f"sad Score: {sad_score:.5f}")
    print(f"excit Score: {excit_score:.5f}")
    print(f"average Score: {average_score:.5f}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and calculate scores from JSONL file.")
    parser.add_argument("--file-path", type=str, help="Path to the JSONL file.")
    args = parser.parse_args()
    main(args.file_path)
