import json
import re
import argparse

def extract_scores_from_jsonl(file_path):
    scores = []

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
    return scores

def calculate_average_score(scores):
    if not scores:
        return 0
    return sum(scores) / len(scores)

def main(file_path):
    scores = extract_scores_from_jsonl(file_path)
    average_score = calculate_average_score(scores)
    print(f"average Score: {average_score:.5f}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and calculate scores from JSONL file.")
    parser.add_argument("--file-path", type=str, help="Path to the JSONL file.")
    args = parser.parse_args()
    main(args.file_path)
