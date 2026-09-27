from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override


class Qwen25Handler(OSSHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, dtype, **kwargs)
        
    
    @override
    def _format_prompt(self, messages, function):
        # OSSHandler already inserts BFCL's prompt-mode tool instructions.
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
