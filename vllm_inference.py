import logging
from typing import List, Union, Optional
from torch import nn
from transformers.modeling_utils import ModuleUtilsMixin
from vllm import LLM, SamplingParams, ModelRegistry
from vllm import ModelRegistry

def default(val, d):
    if val is not None:
        return val
    return d


class CausalLMWithvLLM(nn.Module, ModuleUtilsMixin):
    default_generation_config = {
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1.0,
    }

    def __init__(
        self,
        model_path: str = None,
        use_chat_template: bool = False,
        verbose: bool = False,
        model_kwargs: Optional[dict] = None,
        generation_config: Optional[dict] = None,
    ):
        super().__init__()
        self.model = None
        self.tokenizer = None  # 초기화되지 않은 tokenizer
        self.processor = None
        self.verbose = verbose
        self.use_chat_template = use_chat_template
        self.model_kwargs = default(model_kwargs, {})
        self.generation_config = default(
            generation_config, self.default_generation_config
        )
        self.model = LLM(model=model_path, trust_remote_code=True, **self.model_kwargs)
        self.post_init()

        if self.verbose:
            logging.basicConfig(level=logging.DEBUG)

    def post_init(self):
        stop_tokens = ["Instruction:", "Instruction", "Response:", "Response", "<|eot_id|>"]
        self.generation_config = SamplingParams(
            **self.generation_config, stop=stop_tokens
        )

    def forward(self, text: Union[str, List[str]]) -> List[str]:
        if isinstance(text, str):
            text = [text]

        if self.use_chat_template:
            if not self.tokenizer:
                raise ValueError("Tokenizer must be initialized when use_chat_template is True.")
            text = [
                self.tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': t}], tokenize=False, add_generation_prompt=True
                ) for t in text
            ]

        outputs = self.model.generate(
            prompts=text, sampling_params=self.generation_config
        )
        generated_text = [output.outputs[0].text for output in outputs]
        return generated_text
