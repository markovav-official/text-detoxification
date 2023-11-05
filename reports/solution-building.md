
# Evaluation (metric):

In the context of text detoxification, the evaluation metric chosen is the reduction rate of toxic comments. It is quantified as:

$\text{Reduction Rate}= 1 − \frac{\text{Number of Toxic Comments After Detoxification}}{\text{Number of Toxic Comments Before Detoxification}}$

This metric reflects the effectiveness of the detoxification process; the closer the value is to 1, the more successful the detoxification method is in reducing the prevalence of toxic comments.

# Baseline:

In this study, my baseline approach for text detoxification was to remove offensive words identified from a [publicly available list](https://www.phorum.org/phorum5/read.php?63,127769). This method simply scans and eliminates any matches from the text.

**Pros:**

* **Speed:** This method is computationally fast because it relies on straightforward string matching and removal.
* **Simplicity:** It's easy to implement as it doesn't require complex algorithms or predictive modeling.

**Cons:**

* **Context Ignorance:** It doesn't account for the context in which words are used, which can lead to inaccuracies.
* **Fixed Vocabulary:** The method's effectiveness is limited to a static list of words and fails to adapt to evolving language use.
* **Word Forms:** It is unable to recognize variations of the same word, missing potentially toxic content.

The baseline detoxification metric was **23.491062%**, which represents the proportion of toxic comments that were left even after the application of this method.

# Hypothesis 1: T5-Small Model

For my first hypothesis, I utilized the `t5-small` transformer model for detoxification tasks. This model is designed to grasp the subtleties of language, which is a significant leap from the baseline approach.

**Pros:**

* **Context Awareness:** The model has the ability to understand context, which is crucial for accurate content moderation.
* **Adaptive:** It can learn from new data, potentially improving its performance over time as it is exposed to more examples.

**Cons:**

* **Computational Cost:** The `t5-small` model requires more computational resources, making it slower in comparison to the baseline method.

This machine learning-based approach resulted in a detoxification metric of **62.313354%**, showing a marked decrease in the number of toxic comments post-detoxification.

# Hypothesis 2: The T5-Large Model

My second hypothesis was that a larger model, like `t5-large`, would yield even better results due to its increased capacity to understand and process language. However, I did not test this hypothesis due to a lack of sufficient computational resources. Nonetheless, I remain confident that such a model would enhance the detoxification performance.

# Results:

The findings from the approaches I examined suggest that machine learning models are more effective for text detoxification compared to simpler methods. The `t5-small` model particularly showed a significant improvement over the baseline method, indicating that models which account for language context are better suited for this task.

While I did not test the `t5-large` model, it stands to reason based on the model's design and capabilities that it could potentially provide an even higher detoxification metric.

In future work, I aim to explore a balance between performance and computational cost. Moreover, the idea of integrating fast, simple filtering techniques with more sophisticated machine learning models might offer a more efficient and effective approach to text detoxification.
