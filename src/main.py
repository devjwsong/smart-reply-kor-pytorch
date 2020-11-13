from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from custom_data import *
from capsule_nn import *

import torch.optim as optim
import torch
import os
import time
import argparse
import json
import random
import numpy as np


class Manager():
    def __init__(self, config_path, mode, ckpt_name=None):
        # Setting config
        print("Setting the configurations...")
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        if self.config['device'] == "cuda":
            self.config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        elif self.config['device'] == "cpu":
            self.config['device'] = torch.device('cpu')
        
        # Tokenizer setting
        print("Loading the tokenizer & vocab...")
        self.tokenizer = get_tokenizer()
        vocab = self.tokenizer.token2idx
        self.config['vocab_size'] = len(vocab)
        self.config['pad_id'] = vocab['[PAD]']
        self.config['cls_id'] = vocab['[CLS]']
        self.config['sep_id'] = vocab['[SEP]']
        self.config['unk_id'] = vocab['[UNK]']
            
        if mode == 'train':
            # Making the ckpt directory
            if not os.path.exists(self.config['ckpt_dir']):
                print("Making checkpoint directory...")
                os.mkdir(self.config['ckpt_dir']) 
                
            # Loading data
            train_set = CustomDataset(f"{self.config['data_dir']}/{self.config['train_name']}.txt", self.tokenizer, self.config)
            test_set = CustomDataset(f"{self.config['data_dir']}/{self.config['test_name']}.txt", self.tokenizer, self.config)
        
            train_class_dict = train_set.class_dict
            test_class_dict = test_set.class_dict
            
            # Checking class labels
            if len(train_class_dict) >= len(test_class_dict):
                for class_name, class_idx in test_class_dict.items():
                    assert class_name in train_class_dict and train_class_dict[class_name] == test_class_dict[class_name], \
                        print("There is unseen class or false index in the test set. Please correct it.")
            
            for class_name, class_idx in train_class_dict.items():
                assert class_idx in range(self.config['num_classes']), \
                    print("There is a class index out of range. Please check.")
            
            train_sampler = RandomSampler(train_set, replacement=True, num_samples=train_set.__len__())
            self.train_loader = DataLoader(train_set, batch_size=self.config['batch_size'], sampler=train_sampler)
            self.test_loader = DataLoader(test_set, batch_size=self.config['batch_size'], shuffle=True)
            
            # Initialize the model and optimizer.
            print("Initializing the model...")
            self.model = CapsuleNetwork(self.config, is_train=True).to(self.config['device'])
            self.optim = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
            self.best_f1 = 0.0
            
            if ckpt_name is not None:
                assert os.path.exists(f"{self.config['ckpt_dir']}/{ckpt_name}"), f"There is no checkpoint named {ckpt_name}."

                print("Loading checkpoint...")
                checkpoint = torch.load(f"{self.config['ckpt_dir']}/{ckpt_name}")
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optim.load_state_dict(checkpoint['optim_state_dict'])
                self.best_f1 = checkpoint['f1']
                
        elif mode == 'inference':
            assert os.path.exists(f"{self.config['ckpt_dir']}/{ckpt_name}"), f"There is no checkpoint named {ckpt_name}."
            
            # Initialized the model
            print("Initializing the model...")
            self.model = CapsuleNetwork(self.config, is_train=True).to(self.config['device'])
            self.model.eval()
            
            print("Loading checkpoint...")
            checkpoint = torch.load(f"{self.config['ckpt_dir']}/{ckpt_name}")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Get intent map & intent text json files.
            print(f"Loading {self.config['intent_map_name']}.json & {self.config['intent_text_name']}.json...")
            with open(f"{self.config['data_dir']}/{self.config['intent_map_name']}.json") as f:
                self.intent_map = json.load(f)

            with open(f"{self.config['data_dir']}/{self.config['intent_text_name']}.json") as f:
                self.intent_text = json.load(f)
            
    def train(self):
        # Total training/testing time.
        total_train_time = 0.0
        total_test_time = 0.0

        # Training starts.
        print("Training starts.")
        best_test_acc = 0.0
        for epoch in range(1, self.config['num_epochs']+1):
            print(f"################### Epoch: {epoch} ###################")

            self.model.train()
            train_losses = []
            y_list = []
            pred_list = []

            # One batch.
            start_time = time.time()
            for batch in tqdm(self.train_loader):
                batch_x, batch_y, batch_one_hot_label = batch

                attentions, output_logits, prediction_vecs, _ = self.model(batch_x, is_train=True)
                loss_val = self.model.get_loss(batch_one_hot_label, output_logits, attentions)

                self.optim.zero_grad()
                loss_val.backward()

                self.optim.step()

                batch_pred = torch.argmax(output_logits, 1)
                y_list += batch_y.tolist()
                pred_list += batch_pred.tolist()

                train_losses.append(loss_val.item())

            train_time = time.time() - start_time
            total_train_time += train_time

            # Calculate accuracy and f1 score of one epoch.
            acc = accuracy_score(y_list, pred_list)
            f1 = f1_score(y_list, pred_list, average='weighted')

            print(f"Train loss: {np.mean(train_losses)}")
            print(f"Train Acc: {round(acc, 4)} || Train F1: {round(f1, 4)}")
            print(f"Train time: {round(train_time, 4)}")

            # Execute evaluation depending on each task.
            cur_test_acc, cur_test_f1, test_time = self.evaluate()

            # If f1 score has increased, save the model.
            if cur_test_f1 > self.best_f1:
                self.best_f1 = cur_test_f1
                state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'f1': self.best_f1,
                }
                torch.save(state_dict, f"{self.config['ckpt_dir']}/best_ckpt.tar")
                print("************ Best model saved! ************")

            print("------------------------------------------------------")
            print(f"Best Test F1: {round(self.best_f1, 4)}")
            print(f"Test Acc: {round(cur_test_acc, 4)} || Test F1: {round(cur_test_f1, 4)}")
            total_test_time += test_time
            print("Testing time", round(test_time, 4))

        print(f"Overall training time: {total_train_time}")
        print(f"Overall testing time: {total_test_time}")
    
    def evaluate(self):
        print("Evaluation starts.")
        self.model.eval()
        y_list = []
        pred_list = []

        with torch.no_grad():
            start_time = time.time()
            for batch in tqdm(self.test_loader):
                batch_x, batch_y, batch_one_hot_label = batch

                attentions, output_logits, prediction_vecs, _ = self.model(batch_x, is_train=False)

                y_list += batch_y.tolist()
                pred_list += torch.argmax(output_logits, 1).tolist()

            test_time = time.time() - start_time
            acc = accuracy_score(y_list, pred_list)
            f1 = f1_score(y_list, pred_list, average='weighted')

        return acc, f1, test_time
    
    def inference(self):
        print(f"If you end the system, please type '{self.config['end_command']}'.")
        while True:
            input_sent = input("Input: ")
            
            if input_sent == self.config["end_command"]:
                print("Good Bye!")
                break
            
            tokens = self.tokenizer.tokenize(input_sent)

            if len(tokens) < self.config['len_limit']:
                print("The input is too short for smart reply.")
            else:
                tokens = self.tokenizer.convert_tokens_to_ids(tokens)
                tokens  = [self.config['cls_id']] + tokens + [self.config['sep_id']]

                if len(tokens) <= self.config['max_len']:
                    tokens += [self.config['pad_id']] * (self.config['max_len'] - len(tokens))
                else:
                    tokens = tokens[:self.config['max_len']]
                    tokens[-1] = self.config['sep_id']

                x = torch.LongTensor(tokens).unsqueeze(0)  # (1, L)

                # Extract output intent.
                print("Working on intent classification...")
                attentions, output_logits, prediction_vecs, _ = self.model(x, is_train=False)
                intent = torch.argmax(output_logits, 1).item()

                # Get response list.
                print("Getting candidate smart replies...")
                response_list = self.get_candidates(intent, num_candidates=self.config['num_candidates'])

                print(response_list)

    def get_candidates(self, intent, num_candidates=3):
        response_intent_list = self.intent_map[str(intent)]
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
            text_list = self.intent_text[str(intent)]
            if num > len(text_list):
                num = len(text_list)
            responses = random.sample(text_list, num)

            response_list += responses

        return response_list


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Argument parser for various parameters.")
    parser.add_argument('--config_path', required=True, help="The path to configuration file.")
    parser.add_argument('--mode', type=str, required=True, help="Training model or testing smart reply?")
    parser.add_argument('--ckpt_name', required=False, help="Best checkpoint file.")

    args = parser.parse_args()

    assert args.mode == 'train' or args.mode == 'inference', "Please specify correct mode."
    
    if args.mode == 'train':
        if args.ckpt_name is not None:
            manager = Manager(args.config_path, args.mode, ckpt_name=args.ckpt_name)
        else:
            manager = Manager(args.config_path, args.mode)
              
        manager.train()
        
    elif args.mode == 'inference':
        assert args.ckpt_name is not None, "Please specify the trained model checkpoint."
        
        manager = Manager(args.config_path, args.mode, ckpt_name=args.ckpt_name)
        
        manager.inference()
