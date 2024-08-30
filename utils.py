import yaml
import json
import torch
from torch.utils.data import Dataset
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 모든 GPU에 대해 시드 고정
    # Deterministic 모드 활성화 (GPU 성능이 약간 저하될 수 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_yaml(data, file_path):
    with open(file_path, 'w', encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)

def load_config(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

class CustomDatasetForDev(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX=-100
        self.inp = []
        self.label = []

        # PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''
        PROMPT = 'You are EXAONE model from LG AI Research, a helpful assistant.'

        with open(fname, 'r', encoding="utf-8") as f:
            data = json.load(f)

        def make_chat(inp):
            chat = ["[Conversation]"]
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']
                chat.append(f"화자{speaker}: {utterance}")
            chat = "\n".join(chat)

            question = f"[Question]\n위 {', '.join(inp['subject_keyword'])} 주제에 대한 대화를 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
        for example in data:
            chat = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False
            )
            target = example["output"]

            self.inp.append(source)
            self.label.append(target)
        

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx]
