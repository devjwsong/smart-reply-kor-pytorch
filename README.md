# smart-reply-kor-pytorch

This is the Korean **Smart Reply** system based on Intent Classification via Intent Capsule Network from [*Xia, C., Zhang, C., Yan, X., Chang, Y., & Yu, P. S. (2018). Zero-shot user intent detection via capsule neural networks. arXiv preprint arXiv:1809.00385*](https://arxiv.org/abs/1809.00385).

The intent classifier used is the same implementation from [intent-capsnet-kor-pytorch](https://github.com/devJWSong/intent-capsnet-kor-pytorch) with DistilKoBERT encoder.

<br/>

---

### Details

This Smart Reply system consists of 3 main parts.

1. Intent Classification

   The IntentCapsNet with DistilKoBERT encoder detects the intent of the sender's utterance. This intent is called "question intent" for convenience.

2. Intent Mapping

   The intent map defines the relation between each question intent and its according "response intents". In other words, there is a fixed response intent set of the receiver depending on the sender's question intent.

   You can check the details from `data/intent_map.json`.

3. Intent Group

   This file has various candidate responses grouped by each response intent.

   You can also see or modify/add responses in `data/intent_text.json`.

The overall architecture of this system is as follows.

<img src="https://user-images.githubusercontent.com/16731987/86424587-ec09fc00-bd1d-11ea-8a50-e09ec51650f4.png" alt="The overall architecture of Korean Smart Reply system via Intent Classification.">

