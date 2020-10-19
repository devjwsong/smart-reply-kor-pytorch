# smart-reply-kor-pytorch
**Smart Reply** is an automated response recommendation system usually used in messenger, mail platforms, etc.

It recommends several candidate responses based on the sender's input for the receiver to answer the message more conveniently with one touch/click.

<br/>

This is the Korean Smart Reply system via Intent Classification with Intent Capsule Network from *Zero-shot user intent detection via capsule neural networks*[[1]](#1).

The IntentCapsNet used is the implementation of the original repository *intent-capsnet-kor-pytorch*[[2]](#2) with DistilKoBERT encoder.

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

---

### How to run

1. Install all required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

2. For intent classification, first you need to train the intent classifier.

   ```shell
   python src/main.py --mode='train_model' --bert_embedding_frozen=TRUE_OR_FALSE
   ```

   - `--mode`: You should choose one of two tasks, `train_model` or `test_smart_reply`. The former is to train the intent classifier with given train data before using Smart Reply. The latter is for the actual usage of Smart Reply with the trained intent classifier.
   - `bert_embedding_frozen`: This specifies whether the embedding layer of DistilKoBERT should be frozen or not. This parameter is `True` or `False` and if you omit this, it is fixed to `False`.

   <br/>

3. Finally, run below command to test Smart Reply system.

   ```shell
   python src/main.py --mode='test_smart_reply' --input=INPUT_MESSAGE
   ```

   - `--input`: This is the input message you want to send to the Smart Reply system. This will be considered as the actual input message from the sender and Smart Reply will give you corresponding answers.
   
   <br/>

---

### References

<a id="1">[1]</a> 
*Xia, C., Zhang, C., Yan, X., Chang, Y., & Yu, P. S. (2018). Zero-shot user intent detection via capsule neural networks. arXiv preprint arXiv:1809.00385*. ([https://arxiv.org/abs/1809.00385](https://arxiv.org/abs/1809.00385))

<a id="2">[2]</a> 
*intent-capsnet-kor-pytorch*. ([https://github.com/devJWSong/intent-capsnet-kor-pytorch](https://github.com/devJWSong/intent-capsnet-kor-pytorch))

<br/>

---

