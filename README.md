# smart-reply-kor-pytorch
**Smart Reply** is an automated response recommendation system usually used in messenger, mail platforms, etc.

It recommends several candidate responses based on the sender's input for the receiver to answer the message more conveniently with one touch/click.

<br/>

This is the **Korean Smart Reply** system via Intent Classification with Intent Capsule Network from *Zero-shot user intent detection via capsule neural networks*[[1]](#1).

The IntentCapsNet used is the implementation of the repository *intent-capsnet-kor-pytorch*[[2]](#2) with DistilKoBERT[[3]](#3) encoder. (An LSTM encoder was used in the original paper.)

<br/>

---

### Details

This Smart Reply system is based on multinomial classification, consisting of 3 main parts.

1. **Intent Classification**

   The IntentCapsNet detects the sender's intent by processing the input message.

   The sender's intent is called "question" intent for convenience.

2. **Intent Mapping**

   This defines the relation between each question intent and according "response" intent sets.

   In other words, there are fixed links between question intent and its corresponding response intents.

3. **Intent Group**

   This file has candidate responses grouped by each response intent group.

   The system samples several responses from each group for recommendation.


<br/>

The overall architecture of Korean Smart Reply is as follows.

<img src="https://user-images.githubusercontent.com/16731987/86425622-a00c8680-bd20-11ea-8b31-2caf31ae15db.png" alt="The overall architecture of Korean Smart Reply system.">

<br/>

---

### Configurations

You can set various arguments by modifying `config.json` in the top directory.

The description of each variable is as follows. (Those not introduced in below table are set automatically and should not be changed.)

| Argument                | Type              | Description                                                  | Default          |
| ----------------------- | ----------------- | ------------------------------------------------------------ | ---------------- |
| `data_dir`              | `String`          | The name of the parent directory where data files are stored. | `"data"`         |
| `ckpt_dir`              | `String`          | The path for saved checkpoints.                              | `"saved_models"` |
| `train_name`            | `String`          | The prefix of the train data file's name.                    | `"train"`        |
| `test_name`             | `String`          | The prefix of the test data file's name.                     | `"test"`         |
| `intent_map_name`       | `String`          | The prefix of the intent map file's name.                    | `"intent_map"`   |
| `intent_text_name`      | `String`          | The prefix of the response text file's name.                 | `"intent_text"`  |
| `bert_embedding_frozen` | `Boolean`         | This determines whether the embedding layer of DistilKoBERT is frozen or not during fine-tuning. | `false`          |
| `dropout`               | `Number(float)`   | The dropout rate.                                            | `0.5`            |
| `batch_size`            | `Number`(`int`)   | The batch size.                                              | `16`             |
| `num_epochs`            | `Number`(`int`)   | The total number of epochs.                                  | `20`             |
| `max_len`               | `Number`(`int`)   | The maximum length of a sentence.                            | `128`            |
| `d_a`                   | `Number(int)`     | The dimension size of internal vector during the self-attention. | `20`             |
| `caps_prop`             | `Number(int)`     | The number of properties in each capsule.                    | `10`             |
| `r`                     | `Number(int)`     | The number of semantic features.                             | `3`              |
| `num_iters`             | `Number(int)`     | The number of iterations for the dynamic routing algorithm.  | `3`              |
| `alpha`                 | `Number(float)`   | The coefficient value for encouraging the discrepancies among different attention heads in the loss function. | `1e-4`           |
| `learning_rate`         | `Number`(`float`) | The learning rate.                                           | `1e-4`           |
| `device`                | `String`          | The device type. (`"cuda"` or `"cpu"`) If this is set to `"cuda"`, then the device configuration is set to `torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')`. If this variable is `"cpu"`, then the setting becomes just `torch.devcie('cpu')`. | `"cuda"`         |
| `len_limit`             | `Number(int)`     | This defines the minimum length(number of tokens) which is passed for the smart reply. If the input is too short, then it is regarded as non-informative query which does not require the response recommandation. | `5`              |
| `end_command`           | `String`          | The command to stop the inference.                           | `"Abort!"`       |
| `num_candidates`        | `Number(int)`     | The number of responses to be recommended.                   | `3`              |

<br/>

---

### Data

For this project, I made a small dataset myself which have sentences likely used in business circumstances and corresponding intent labels.

There is the train set for training and the test set for evaluation/validation.

The number of total intents is 14, which are defined as follows.

| Index | Intent label | Meaning                                   | Example                                       |
| ----- | ------------ | ----------------------------------------- | --------------------------------------------- |
| 0     | request      | I want to request / assign a task to you. | 이 일을 맡아주실 수 있나요?                   |
| 1     | complaint    | I want to complain about something.       | 보내주신 자료가 좀 잘못된 것 같습니다.        |
| 2     | info         | This is an objective information.         | 마무리하고 퇴근하라는 팀장님의 지시입니다.    |
| 3     | question     | I want to ask a question.                 | 제가 잘 몰라서 그런데 설명해주실 수 있습니까? |
| 4     | ok           | I accept this.                            | 알겠습니다.                                   |
| 5     | no           | I reject this.                            | 그건 안 될 것 같습니다.                       |
| 6     | thank        | I want to thank you.                      | 정말 감사합니다.                              |
| 7     | sorry        | I want to apologize.                      | 정말 죄송합니다.                              |
| 8     | laugh        | I feel so happy!                          | 좋은 일이네요!                                |
| 9     | sad          | I feel so sad...                          | 정말 안타깝네요.                              |
| 10    | welcome      | You're welcome.                           | 천만에요.                                     |
| 11    | dontknow     | I don't know about that.                  | 잘 모르겠습니다.                              |
| 12    | hello        | Hi / Hello.                               | 만나서 반갑습니다.                            |
| 13    | bye          | Good bye.                                 | 좋은 하루 되세요.                             |

<br/>

You can add additional sentences for improving model performances.

As you can see in the data file, `{data_dir}/{train_name}.txt` or `{data_dir}/{test_name}.txt`, the intent index, intent label and the sentence is split by one tab in one line.

<br/>

In `{data_dir}/{intent_map_name}.json`, the relations between question intents and response intents are defined.

One question intent has a list of corresponding response intents.

And in `{data_dir}/{intent_text_name}.json`, you can check the pre-defined responses to be recommended.

One response intent has a list of example responses.

You can add more responses for each intent if you want.

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
   python src/main.py --config_path=PATH_TO_CONFIGURATION_FILE --mode='train' --ckpt_name=CHECKPOINT_NAME
   ```

   - `--config_path`: This indicates the path to the configuration file. (default: `"config.json"`)
   - `--mode`: You should choose one of two tasks, `train` or `inference`. The former is to train the intent classifier with given train data before using the smart Reply. The latter is for the actual usage with the trained intent classifier.
   - `--ckpt_name`: This specify the checkpoint file name. This would be the name of trained intent classifier and you can continue your training with this model in the case of resuming training. If you want to conduct training from the beginning, this parameter should be omitted. When inferencing, this would be the name of the checkpoint you want to test. (default: `None`)

   <br/>

3. Finally, run below command to use the smart reply system.

   ```shell
   python src/main.py --config_path=PATH_TO_CONFIGURATION_FILE --mode='inference' --ckpt_name=CHECKPOINT_NAME
   ```

   <br/>

---

### References

<a id="1">[1]</a> 
*Xia, C., Zhang, C., Yan, X., Chang, Y., & Yu, P. S. (2018). Zero-shot user intent detection via capsule neural networks. arXiv preprint arXiv:1809.00385*. ([https://arxiv.org/abs/1809.00385](https://arxiv.org/abs/1809.00385))

<a id="2">[2]</a> 
*intent-capsnet-kor-pytorch*. ([https://github.com/devJWSong/intent-capsnet-kor-pytorch](https://github.com/devJWSong/intent-capsnet-kor-pytorch))

<a id="3">[3]</a> 
*DistilKoBERT*. ([https://github.com/monologg/DistilKoBERT](https://github.com/monologg/DistilKoBERT))

<br/>

---

