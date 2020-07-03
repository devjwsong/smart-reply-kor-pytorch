from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

import torch.optim as optim
import torch
import os
import time
import argparse
import json
import random
import data_process, capsule_nn


def setting_for_training(ckpt_dir, bert_embedding_frozen):
    # Load data dict.
    print("Reading dataset...")
    data_path = "../data"

    data = data_process.read_datasets(data_path)

    train_class_num = len(data['train_class_dict'])
    test_class_num = len(data['test_class_dict'])

    # Set basic configs for training.
    config = {'keep_prob': 0.8,
              'hidden_size': data['word_emb_size'],
              'batch_size': 16,
              'vocab_size': data['vocab_size'],
              'epoch_num': 20,
              'seq_len': data['max_len'],
              'pad_id': data['pad_id'],
              'train_class_num': train_class_num,
              'test_class_num': test_class_num,
              'word_emb_size': data['word_emb_size'],
              'd_a': 20,
              'd_m': 256,
              'caps_prop': 10,
              'r': 3,
              'iter_num': 3,
              'alpha': 0.0001,
              'learning_rate': 0.0001,
              'sim_scale': 4,
              'num_layers': 2,
              'ckpt_dir': ckpt_dir,
              'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
              'bert_embedding_frozen': bert_embedding_frozen
    }

    # Save model config for later usage
    config_memo = config
    if torch.cuda.is_available():
        config_memo['device'] = 'cuda'
    else:
        config_memo['device'] = 'cpu'

    # If the saving directory does not exist, make one.
    if not os.path.exists(config['ckpt_dir']):
        print("Making check point directory...")
        os.mkdir(config['ckpt_dir'])
    with open(f'{ckpt_dir}/config.json', 'w') as f:
        json.dump(config, f)

    # Initialize dataloaders.
    train_sampler = RandomSampler(data['train_set'], replacement=True, num_samples=data['train_set'].__len__())
    train_loader = DataLoader(data['train_set'], batch_size=config['batch_size'], sampler=train_sampler)
    test_loader = DataLoader(data['test_set'], batch_size=config['batch_size'], shuffle=True)

    return config, train_loader, test_loader


def setting_for_smart_reply(ckpt_dir, input_sentence):
    assert os.path.isfile(f"{ckpt_dir}/best_model.pth"), "There is no trained IntentCapsNet model."
    assert os.path.isfile(f"{ckpt_dir}/config.json"), "There is no configuration of IntentCapsnet model."

    # Load model config
    print("Loading the configuration of trained model...")
    with open(f'{ckpt_dir}/config.json', 'r') as f:
        config = json.load(f)
    if torch.cuda.is_available():
        config['device'] = torch.device('cuda')
    else:
        config['device'] = torch.device('cpu')

    # Specify the filtering length.
    len_limit = 20
    is_pass = True

    # Tokenize the sentence.
    print("Preprocessing the input...")
    from kobert_transformers import get_tokenizer
    tokenizer = get_tokenizer()
    not_padded_x = tokenizer.tokenize('[CLS] ' + input_sentence + ' [SEP]')
    not_padded_x = tokenizer.convert_tokens_to_ids(not_padded_x)

    if len(not_padded_x) > len_limit:
        is_pass = False

    # Add paddings.
    if config['seq_len'] < len(not_padded_x):
        x = not_padded_x[0:config['seq_len']]
    else:
        x = not_padded_x + [config['pad_id']] * (config['seq_len'] - len(not_padded_x))

    x = torch.LongTensor(x).unsqueeze(0) # (1, L)

    return config, x, is_pass


def train(config, train_loader, test_loader):
    # Total training/testing time.
    total_train_time = 0.0
    total_test_time = 0.0

    # Initialize the model and optimizer.
    model = capsule_nn.CapsuleNetwork(config, is_train=True).to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training starts.
    print("Training starts.")
    best_test_acc = 0.0
    for epoch in range(1, config['epoch_num']+1):
        model.train()
        y_list = []
        pred_list = []
        loss_val = None

        # One batch.
        start_time = time.time()
        for batch in tqdm(train_loader):
            batch_x, batch_y, batch_one_hot_label = batch

            attentions, output_logits, prediction_vecs, _ = model(batch_x, is_train=True)
            loss_val = model.get_loss(batch_one_hot_label, output_logits, attentions)

            optimizer.zero_grad()
            loss_val.backward()

            optimizer.step()

            batch_pred = torch.argmax(output_logits, 1)
            y_list += batch_y.tolist()
            pred_list += batch_pred.tolist()

        train_time = time.time() - start_time
        total_train_time += train_time

        # Calculate accuracy and f1 score of one epoch.
        acc = accuracy_score(y_list, pred_list)
        f1 = f1_score(y_list, pred_list, average='weighted')

        print(f"################### Epoch: {epoch} ###################")
        print(f"Train loss: {loss_val.item()}")
        print(f"Train Acc: {round(acc, 4)} || Train F1: {round(f1, 4)}")
        print(f"Train time: {round(train_time, 4)}")

        # Execute evaluation depending on each task.
        cur_test_acc, cur_test_f1, test_time = evaluate(test_loader, model)

        # If f1 score has increased, save the model.
        if cur_test_acc > best_test_acc:
            best_test_acc = cur_test_acc
            torch.save(model.state_dict(), f"{config['ckpt_dir']}/best_model.pth")
            print("************ Best model saved! ************")

        print("------------------------------------------------------")
        print(f"Best Test Acc: {round(best_test_acc, 4)}")
        print(f"Test Acc: {round(cur_test_acc, 4)} || Current Test F1: {round(cur_test_f1, 4)}")
        total_test_time += test_time
        print("Testing time", round(test_time, 4))

    print(f"Overall training time: {total_train_time}")
    print(f"Overall testing time: {total_test_time}")
    
    
def evaluate(test_loader, model):
    model.eval()
    y_list = []
    pred_list = []

    # Evaluation starts.
    with torch.no_grad():
        start_time = time.time()

        # One batch.
        for batch in tqdm(test_loader):
            batch_x, batch_y, batch_one_hot_label = batch

            attentions, output_logits, prediction_vecs, _ = model(batch_x, is_train=False)

            y_list += batch_y.tolist()
            pred_list += torch.argmax(output_logits, 1).tolist()

        test_time = time.time() - start_time
        acc = accuracy_score(y_list, pred_list)
        f1 = f1_score(y_list, pred_list, average='weighted')
        
    return acc, f1, test_time


def test_smart_reply(config, x):
    # Load model.
    print("Loading the model...")
    model = capsule_nn.CapsuleNetwork(config, is_train=False).to(config['device'])
    model.load_state_dict(torch.load(f"{config['ckpt_dir']}/best_model.pth"))
    model.eval()

    # Extract output intent.
    print("Working on intent classification...")
    attentions, output_logits, prediction_vecs, _ = model(x, is_train=False)
    intent = torch.argmax(output_logits, 1).item()

    # Get response list.
    print("Getting candidate smart replies...")
    response_list = get_candidates(intent, num_candidates=3)

    print(response_list)


def get_candidates(intent, num_candidates=3):
    intent_map_path = '../data/intent_map.json'
    intent_text_path = '../data/intent_text.json'

    with open(intent_map_path) as f:
        intent_map = json.load(f)

    with open(intent_text_path) as f:
        intent_text = json.load(f)

    response_intent_list = intent_map[str(intent)]
    response_dict = {}
    for intent in response_intent_list:
        response_dict[intent] = 1

    if len(response_intent_list) >= num_candidates:
        response_dict = {k: v for i, (k, v) in enumerate(response_dict.items()) if i < num_candidates}
    else:
        left = num_candidates - len(response_intent_list)
        for i in range(left):
            intent = response_intent_list[i % len(response_intent_list)]
            response_dict[intent] += 1

    response_list = []
    for intent, num in response_dict.items():
        text_list = intent_text[str(intent)]
        if num > len(text_list):
            num = len(text_list)
        responses = random.sample(text_list, num)

        response_list += responses

    return response_list


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Argument parser for various parameters.")
    parser.add_argument('--mode', type=str, required=True, help="Training model or testing smart reply?")
    parser.add_argument('--bert_embedding_frozen', type=bool, default=False, help="Do you want to freeze BERT's embedding layer or not?")
    parser.add_argument('--input', type=str, default=False, help="Actual input sentence when using smart reply system.")

    args = parser.parse_args()

    assert args.mode == 'train_model' or args.mode == 'test_smart_reply', "Please specify correct mode."

    ckpt_dir = "../saved_model"

    if args.mode == 'train_model':
        config, train_loader, test_loader = setting_for_training(ckpt_dir, args.bert_embedding_frozen)
        train(config, train_loader, test_loader)
    elif args.mode == 'test_smart_reply':
        assert args.input is not None, "Please type the input sentence for smart replying system."

        config, x, is_pass = setting_for_smart_reply(ckpt_dir, args.input)

        if is_pass:
            test_smart_reply(config, x)
        else:
            print("No need to conduct Smart Reply.")

