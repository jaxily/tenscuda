"""The one prompt used by the matrix scripts, so CPU and TT runs see identical tokens."""
import torch

PROMPT = "Explain in two sentences why the sky is blue."


def prompt_ids(tok) -> torch.Tensor:
    msgs = [{"role": "user", "content": PROMPT}]
    if tok.chat_template:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                       return_dict=True)["input_ids"]
    return tok(PROMPT, return_tensors="pt").input_ids
