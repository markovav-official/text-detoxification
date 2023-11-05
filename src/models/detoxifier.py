import os
import sys
import warnings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


warnings.filterwarnings("ignore")


def detoxify(text: str, model_path='../models/t5-small-finetuned-toxic-en-to-neutral/best'):
    prefix = "paraphrase: "
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    input_ids = tokenizer(prefix + text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True,temperature=0)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python <path>/detoxifier.py "text" "model_path"')
        print('Automatically using default model path: "./models/t5-small-finetuned-toxic-en-to-neutral-en/best" if run from project root directory.')
        exit(1)
    text = sys.argv[1]
    if len(text) < 1:
        print('Text cannot be empty.')
        exit(1)
    model_path = sys.argv[2] if len(sys.argv) > 2 else './models/t5-small-finetuned-toxic-en-to-neutral-en/best'
    if not os.path.exists(model_path) or not os.path.isdir(model_path) or not os.path.exists(model_path + '/pytorch_model.bin'):
        print('Model not found in "' + model_path + '". Please specify a valid model path.')
        exit(1)
    print(detoxify(text, model_path))
