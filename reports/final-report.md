# Introduction

The task of text detoxification involves the identification and modification of toxic content within text data to make it neutral and non-offensive. This process is vital in maintaining healthy interactions on social platforms, forums, and in digital communications. The goal is to create an automated system capable of detecting and altering toxic text while preserving the original intent and information content of the messages.

Text detoxification has wide-ranging applications including online content moderation, improving the quality of discourse in social media, and preventing the spread of harmful language. It is a step toward creating more inclusive and respectful online communities.

# Data Analysis

The dataset provided for the detoxification task contains 577,777 entries with columns indicating reference text, translations, similarity, length difference, and toxicity flags for both reference and translation. To refine the dataset for the detoxification task, it was transformed into a format with two columns: "toxic-en" and "neutral-en".

Additional data was sourced from the following datasets:

- [s-nlp/paranmt_for_detox](https://huggingface.co/datasets/s-nlp/paranmt_for_detox)
- [s-nlp/en_paradetox_content](https://huggingface.co/datasets/s-nlp/en_paradetox_content)
- [s-nlp/paradetox](https://huggingface.co/datasets/s-nlp/paradetox)

These datasets were also converted to the "toxic-en" and "neutral-en" format and combined with the original data, resulting in a comprehensive dataset stored in `combined.tsv` with a total of 631,231 entries. For the toxicity classifier, a `badwords.txt` file was downloaded from [Bad Words file (English) for Censor List](https://www.phorum.org/phorum5/read.php?63,127769) to aid in identifying toxic content.

# Model Specification

The model selected for the detoxification task was `t5-small`, a transformer-based model known for its effectiveness in natural language understanding and generation tasks. The specifications of the `t5-small` model include:

* A smaller footprint than its larger counterparts, making it more manageable for training with limited computational resources.
* The ability to understand the context of sentences, which is crucial for the paraphrasing required in detoxification.
* Pre-trained on a diverse corpus of text, allowing it to perform a wide range of language tasks with the addition of a task-specific prefix.

# Training Process

The model was trained using the `transformers` library with a tailored training and validation dataset. A prefix of "paraphrase:" was used to instruct the model on the task at hand. Training was performed using the `Seq2SeqTrainer` with the following parameters:

* Number of epochs: 3
* Learning Rate: $1 \cdot 10^{-4}$
* Training metric: sacrebleu

The training aimed to optimize the model's ability to paraphrase toxic sentences into non-toxic equivalents while maintaining the original meaning. The best-performing model was saved under `models\t5-small-finetuned-toxic-en-to-neutral-en\best`.

# Evaluation

The model's performance was evaluated using the reduction rate of toxic comments as the primary metric:

$\text{Reduction Rate}= 1 − \frac{\text{Number of Toxic Comments After Detoxification}}{\text{Number of Toxic Comments Before Detoxification}}$

This metric assesses the model's effectiveness in reducing toxicity by comparing the number of toxic comments before and after the detoxification process.

# Results

The `t5-small` model achieved a 62.313354% reduction rate score, indicating a substantial decrease in toxic comments after detoxification. This performance demonstrates the capability of context-aware language models to effectively reduce toxicity in text. The results suggest that the `t5-small` model can be a valuable tool for automating text detoxification, with potential improvements available through further fine-tuning of larger, more powerful models.

# References

[Text Detoxification using Large Pre-trained Neural Models by Dale et al., page 14](https://arxiv.org/abs/2109.08914)

[Text Detoxification using Large Pre-trained Neural Models](https://semion.io/doc/text-detoxification-using-large-pre-trained-neural-models)

[Text Detoxification with Parallel Data](https://dardem.github.io/text_detoxification/)
