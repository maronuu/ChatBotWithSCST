import os
import sys
import argparse
import logging
import random

import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn.functional as F
import ptan

from . import utils, model, data

SAVE_DIR = 'saves'
IMAGE_DIR = 'images'


class TrainerCrossEntropy:
    def __init__(self, genre, use_cuda, name_of_run, log, model, batch_size, lr, n_epochs):   
        is_cuda_available = torch.cuda.is_available()
        if (not is_cuda_available) and use_cuda:
            raise RuntimeError("GPU is not available, but you tried to use it.")
        self.use_cuda = use_cuda
        self.genre = genre
        self.name_of_run = name_of_run
        self.log = log
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.random_state = np.random.RandomState(data.SEED)

        # device
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        # save directory
        self.path_to_save_dir = os.path.join(SAVE_DIR, name_of_run)
        os.mkdir(self.path_to_save_dir, exist_ok=True)

        # load training data and embedding dict
        dialogues, self.emb_dict = data.load_dialogue_and_dict(
            genre_filter=self.genre)
        data.save_emb_dict(self.path_to_save_dir, self.emb_dict)
        self.train_data = data.encode_phrase_pairs(dialogues, self.emb_dict)
        self.random_state.shuffle(self.train_data)
        self.log.info(f"Training data is loaded. ({len(self.train_data)} records)")

        # make test data
        self.train_data, self.test_data = data.train_test_split(self.train_data, train_size=0.95)
        self.log.info(f"Training data: {len(self.train_data)} | Test data: {len(self.test_data)}")

        # define model
        self.net = model
        log.info(f"Model: {self.net.__repr__}")

        # Summary Writer
        self.writer = None

        # Optimizer
        self.lr = lr
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def run_test(self):
        eof = self.emb_dict[data.END]
        total_bleu = 0.
        num_bleu = 0
        for phrase1, phrase2 in self.test_data:
            seq = data.input_mat_to_packed(phrase1, self.net.embedding, device=self.device)
            hidden = self.net.encode(seq)
            logits, actions = self.net.decode_sequences(
                hidden=hidden,
                length=data.MAX_LENGTH,
                init_emb=seq[0:1],
                end_of_decoding=eof,
            )
            total_bleu += utils.calc_bleu_score_for_seq(actions, phrase2[1:])  # omit first token of reference sequence
            num_bleu += 1
        
        return total_bleu / num_bleu
    
    def _write(self, epoch, loss, bleu_score_train, bleu_score_test):
        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('bleu_train', bleu_score_train, epoch)
        self.writer.add_scalar('bleu_test', bleu_score_test, epoch)

    def _plot_and_save_fig(self, yy, ylabel_name, fig_name, save_dir=IMAGE_DIR):
        if not os.path.isdir('./images'):
            os.mkdir('./images')
        xx = np.arange(0, len(yy), 1)
        plt.figure()
        plt.plot(xx, yy)
        plt.xlabel('Epochs')
        plt.ylabel(ylabel_name)
        plt.title(fig_name)
        plt.savefig(os.path.join(save_dir, fig_name + '.png'))

    def run_train(self):
        # log format
        fmt = "%(asctime)-15s %(levelname)s %(message)s"
        logging.basicConfig(format=fmt, level=logging.INFO)
        # temporary best score
        best_score = None
        # writer
        self.writer = SummaryWriter(comment='-' + self.name_of_run)

        # record
        loss_record = []
        bleu_train_record = []
        bleu_test_record = []

        for epoch in range(self.n_epochs):
            total_bleu = 0.
            num_bleu = 0
            loss_list = []
            for batch in data.batch_iterator(self.train_data, self.batch_size):
                packed_question_seq, packed_responce_seqs, questions, responces = data.pairs_to_packed_seq(batch, embedding=self.net.embedding, device=self.device)

                # question_seq -> [Encoder] -> encoded
                encoded = self.net.encode(packed_question_seq)

                output_list = list()  # output list
                ref_list = list()  # reference(target) list
                for i, responce_seq in enumerate(packed_responce_seqs):
                    hidden = (encoded[j][:, i:i+1].contiguous() for j in [0, 1])
                    ref_seq = responces[i][1:]  # omit first token.

                    if random.random() < 0.5:
                        # decode training data directly.
                        out_g = self.net.decode_train_data(hidden, responce_seq)
                        out = torch.max(out_g.data, dim=1)[1]
                        out = out.cpu().numpy()
                        total_bleu += utils.calc_bleu_score_for_seq(out, ref_seq)
                    else:
                        # decode and generate phrase.
                        out_g, actions = self.net.decode_sequences(
                            hidden=hidden,
                            length=len(ref_seq),
                            init_emb=responce_seq[0:1],
                            mode='argmax'
                        )
                        total_bleu += utils.calc_bleu_score_for_seq(actions, ref_seq)
                    
                    output_list.append(out_g)
                    ref_list += ref_seq
                    num_bleu += 1
                
                # concatenate all outputs from each batch.
                out_tensor = torch.cat(output_list)
                ref_tensor = torch.Tensor(ref_list).to(torch.long).to(self.device)
                # update
                self.optimizer.zero_grad()
                loss_g = F.cross_entropy(out_tensor, ref_tensor)
                loss_g.backward()
                self.optimizer.step()
                loss_list.append(loss_g.item())

            # scores
            loss = np.mean(loss_list)
            bleu_score_train = total_bleu / num_bleu
            bleu_score_test = self.run_test()
            loss_record.append(loss)
            bleu_train_record.append(bleu_score_train)
            bleu_test_record.append(bleu_score_test)
            best_score = max(best_score, bleu_score_test)
            self.log.info(f"Epoch {epoch} || mean loss: {loss:.5f} || mean BLEU score: {bleu_score_train:.5f} || test BLEU score: {bleu_score_test:.5f} || current best test BLEU score: {best_score:.5f}")

            # tensorboard
            self._write(epoch, loss, bleu_score_train, bleu_score_test)

            # save
            if epoch % 5 == 0:
                data_path = os.path.join(
                    self.path_to_save_dir,
                    f"epoch_{epoch:.03d}_{bleu_score_train:.5f}_{bleu_score_test:.5f}.dat"
                    torch.save(self.net.state_dict(), data_path)
                )

        self.writer.close()
        # save record as figures.
        self._plot_and_save_fig(loss_record, "Loss", "loss-train-cross-entropy")
        self._plot_and_save_fig(bleu_train_record, "BLEU score on training", "bleu-train-cross-entropy")
        self._plot_and_save_fig(bleu_test_record, "BLEU score on test", "bleu-test-cross-entropy")




class TrainerReinforce:
    def __init__(self, genre, use_cuda, name_of_run, log, model, model_path, batch_size, lr, n_epochs):
        is_cuda_available = torch.cuda.is_available()
        if (not is_cuda_available) and use_cuda:
            raise RuntimeError("GPU is not available, but you tried to use it.")
        self.use_cuda = use_cuda
        self.log = log
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.random_state = np.random.RandomState(data.SEED)

        # device
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        # save directory
        self.path_to_save_dir = os.path.join(SAVE_DIR, name_of_run)
        os.mkdir(self.path_to_save_dir, exist_ok=True)

        # load training data and embedding dict
        dialogues, self.emb_dict = data.load_dialogue_and_dict(
            genre_filter=genre)
        # embedding dict
        data.save_emb_dict(self.path_to_save_dir, self.emb_dict)
        self.rev_emb_dict = {idx: word for word, idx in self.emb_dict.items()}
        self.begin_token = torch.Tensor([self.emb_dict[data.BEGIN]]).to(torch.long).to(self.device)
        # train data
        self.train_data = data.encode_phrase_pairs(dialogues, self.emb_dict)
        self.random_state.shuffle(self.train_data)
        self.log.info(f"Training data is loaded. ({len(self.train_data)} records)")

        # make test data
        self.train_data, self.test_data = data.train_test_split(self.train_data, train_size=0.95)
        self.train_data = data.ph_ph_pairs_to_ph_phs_pairs(self.train_data)
        self.test_data = data.ph_ph_pairs_to_ph_phs_pairs(self.test_data)
        self.log.info(f"Training data: {len(self.train_data)} | Test data: {len(self.test_data)}")

        # define model
        self.net = model
        self.log.info(f"Model: {self.net.__repr__}")

        # load model
        self.net.load_state_dict(torch.load(model_path))
        self.log.info("Model is loaded successfully.")
        self.log.info("Continuing...")

        # Summary Writer
        self.writer = None

        # Optimizer
        self.lr = lr
        self.optimizer = None

    def run_test(self):
        eof = self.emb_dict[data.END]
        total_bleu = 0.
        num_bleu = 0
        for phrase1, phrase2 in self.test_data:
            seq = data.input_mat_to_packed(phrase1, self.net.embedding, device=self.device)
            hidden = self.net.encode(seq)
            logits, actions = self.net.decode_sequences(
                hidden=hidden,
                length=data.MAX_LENGTH,
                init_emb=seq[0:1],
                end_of_decoding=eof,
            )
            ref_seq = [t[1:] for t in phrase2]
            total_bleu += utils.calc_bleu_score_for_seq(actions, ref_seq)  # omit first token of reference sequence
            num_bleu += 1
        
        return total_bleu / num_bleu
    
    def _write(self, epoch, loss, bleu_score_train, bleu_score_test):
        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('bleu_train', bleu_score_train, epoch)
        self.writer.add_scalar('bleu_test', bleu_score_test, epoch)

    def _plot_and_save_fig(self, yy, ylabel_name, fig_name, save_dir=IMAGE_DIR):
        if not os.path.isdir('./images'):
            os.mkdir('./images')
        xx = np.arange(0, len(yy), 1)
        plt.figure()
        plt.plot(xx, yy)
        plt.xlabel('Epochs')
        plt.ylabel(ylabel_name)
        plt.title(fig_name)
        plt.savefig(os.path.join(save_dir, fig_name + '.png'))

    def run_train(self):
        # log format
        fmt = "%(asctime)-15s %(levelname)s %(message)s"
        logging.basicConfig(format=fmt, level=logging.INFO)
        # temporary best score
        best_score = None
        # writer
        self.writer = SummaryWriter(comment='-' + self.name_of_run)
        # optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        # record
        loss_record = []
        bleu_train_record = []
        bleu_test_record = []

        with

        for epoch in range(self.n_epochs):
            total_bleu = 0.
            num_bleu = 0
            loss_list = []
            for batch in data.batch_iterator(self.train_data, self.batch_size):
                packed_question_seq, packed_responce_seqs, questions, responces = data.pairs_to_packed_seq(batch, embedding=self.net.embedding, device=self.device)

                # question_seq -> [Encoder] -> encoded
                encoded = self.net.encode(packed_question_seq)

                output_list = list()  # output list
                ref_list = list()  # reference(target) list
                for i, responce_seq in enumerate(packed_responce_seqs):
                    hidden = (encoded[j][:, i:i+1].contiguous() for j in [0, 1])
                    ref_seq = responces[i][1:]  # omit first token.

                    if random.random() < 0.5:
                        # decode training data directly.
                        out_g = self.net.decode_train_data(hidden, responce_seq)
                        out = torch.max(out_g.data, dim=1)[1]
                        out = out.cpu().numpy()
                        total_bleu += utils.calc_bleu_score_for_seq(out, ref_seq)
                    else:
                        # decode and generate phrase.
                        out_g, actions = self.net.decode_sequences(
                            hidden=hidden,
                            length=len(ref_seq),
                            init_emb=responce_seq[0:1],
                            mode='argmax'
                        )
                        total_bleu += utils.calc_bleu_score_for_seq(actions, ref_seq)
                    
                    output_list.append(out_g)
                    ref_list += ref_seq
                    num_bleu += 1
                
                # concatenate all outputs from each batch.
                out_tensor = torch.cat(output_list)
                ref_tensor = torch.Tensor(ref_list).to(torch.long).to(self.device)
                # update
                self.optimizer.zero_grad()
                loss_g = F.cross_entropy(out_tensor, ref_tensor)
                loss_g.backward()
                self.optimizer.step()
                loss_list.append(loss_g.item())

            # scores
            loss = np.mean(loss_list)
            bleu_score_train = total_bleu / num_bleu
            bleu_score_test = self.run_test()
            loss_record.append(loss)
            bleu_train_record.append(bleu_score_train)
            bleu_test_record.append(bleu_score_test)
            best_score = max(best_score, bleu_score_test)
            self.log.info(f"Epoch {epoch} || mean loss: {loss:.5f} || mean BLEU score: {bleu_score_train:.5f} || test BLEU score: {bleu_score_test:.5f} || current best test BLEU score: {best_score:.5f}")

            # tensorboard
            self._write(epoch, loss, bleu_score_train, bleu_score_test)

            # save
            if epoch % 5 == 0:
                data_path = os.path.join(
                    self.path_to_save_dir,
                    f"epoch_{epoch:.03d}_{bleu_score_train:.5f}_{bleu_score_test:.5f}.dat"
                    torch.save(self.net.state_dict(), data_path)
                )

        self.writer.close()
        # save record as figures.
        self._plot_and_save_fig(loss_record, "Loss", "loss-train-cross-entropy")
        self._plot_and_save_fig(bleu_train_record, "BLEU score on training", "bleu-train-cross-entropy")
        self._plot_and_save_fig(bleu_test_record, "BLEU score on test", "bleu-test-cross-entropy")