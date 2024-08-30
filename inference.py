import json
import argparse
from transformers import AutoTokenizer
from utils import CustomDatasetForDev
from vllm_inference import CausalLMWithvLLM
import os

def main(model_path, test_json_path, test_json_output_path):
    # 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ds = CustomDatasetForDev(test_json_path, tokenizer)

    # 모델 설정 및 추론
    llm = CausalLMWithvLLM(
        model_path=model_path, 
        use_chat_template=False, 
        verbose=False,
        model_kwargs={'max_model_len': 4096, 'tensor_parallel_size': 1, 'gpu_memory_utilization': 0.9},
        generation_config={'temperature': 0, 'max_tokens': 4096, 'repetition_penalty': 1.0}
    )
    preds = llm(ds.inp)

    # 파일 읽기
    try:
        with open(test_json_path, "r", encoding="utf-8") as f:
            result = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{test_json_path} 파일을 찾을 수 없습니다.")
    except json.JSONDecodeError:
        raise ValueError(f"{test_json_path} 파일을 JSON 형식으로 읽을 수 없습니다.")

    # 추론 결과 추가 및 저장
    if len(preds) != len(result):
        raise ValueError("추론 결과의 수와 데이터셋의 수가 일치하지 않습니다.")

    for idx, r in enumerate(preds):
        result[idx]["output"] = r

    try:
        with open(test_json_output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False, indent=4))
        print(f"추론 결과가 {test_json_output_path}에 저장되었습니다.")
    except IOError as e:
        raise IOError(f"{test_json_output_path} 파일을 저장하는 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for summarization model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file.")
    
    args = parser.parse_args()
    
    main(args.model_path, args.input_file, args.output_file)
