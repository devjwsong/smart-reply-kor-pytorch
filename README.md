# smart-reply-kor-pytorch
**Smart Reply** is an automated response recommendation system usually used in messenger, mail platforms, etc.

It recommends several candidate responses based on the sender's input for the receiver to answer the message more conveniently with one touch/click.

<br/>

This is the Korean Smart Reply system via Intent Classification with Intent Capsule Network from [*Xia, C., Zhang, C., Yan, X., Chang, Y., & Yu, P. S. (2018). Zero-shot user intent detection via capsule neural networks. arXiv preprint arXiv:1809.00385*](https://arxiv.org/abs/1809.00385).

The IntentCapsNet used is the implementation of the original repository [intent-capsnet-kor-pytorch](https://github.com/devJWSong/intent-capsnet-kor-pytorch) with DistilKoBERT encoder.

<br/>

---

### Details

This Smart Reply system is based on multinomial classification, consisting of 3 main parts.

1. Intent Classification

   The IntentCapsNet detects the sender's intent by processing the input message.

   The sender's intent is called "question" intent for convenience.

2. Intent Mapping

   This defines the relation between each question intent and according "response" intent sets.

   In other words, there are fixed links between question intent and its corresponding response intents.

   You can see the details from `data/intent_map.json`.

3. Intent Group

   This file has candidate responses grouped by each question intent group.

   The system samples several responses from each group for recommendation.

   You can also check and add/modify the answers in `data/intent_text.json`.

<br/>

The overall architecture of Korean Smart Reply is as follows.

<img src="https://user-images.githubusercontent.com/16731987/86425622-a00c8680-bd20-11ea-8b31-2caf31ae15db.png" alt="The overall architecture of Korean Smart Reply system.">

<br/>

<hr style="background: transparent; border: 0.5px dashed;"/>

### How to run

1. Install all required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

2. Run below codes to train & test a model.

   ```shell
   python src/main.py --model_type=MODEL_TYPE --mode=MODE --bert_embedding_frozen=TRUE_OR_FALSE
   ```

   - `--model_type`: You should select one model type among three, `bert_capsnet`, `basic_capsnet`, `w2v_capsnet`.
   - `--mode`: You should choose one of two tasks, `seen_class` or `zero_shot`.
   - `--bert_embedding_frozen`: This matters when you use `bert_capsnet`, which specify whether the embedding layer of DistilKoBERT should be frozen or not. This parameter is `True` or `False` and if you omit this, it is fixed in `False`.

<br/>

