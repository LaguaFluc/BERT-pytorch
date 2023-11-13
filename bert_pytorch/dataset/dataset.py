from torch.utils.data import Dataset
import tqdm
import torch
import random


class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        """
        Initializes the object with the given parameters.
        
        Args:
            corpus_path (str): The path to the corpus file.
            vocab (Vocab): The vocabulary object.
            seq_len (int): The maximum sequence length.
            encoding (str, optional): The encoding of the corpus file. Defaults to "utf-8".
            corpus_lines (int, optional): The number of lines in the corpus file. Defaults to None.
            on_memory (bool, optional): Whether to load the entire corpus into memory. Defaults to True.
        """
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        # 构造NSP的正负样本
        t1, t2, is_next_label = self.random_sent(item)
        # 构造MLM的正样本
        # t1_random: List[int], t1_label: List[int]
        t1_random, t1_label = self.random_word(t1)
        # 构造MLM的负样本
        t2_random, t2_label = self.random_word(t2)

        # 连接t1和t2，作为一个句子输入到BERT中
        # sentence_input: [CLS] t1 [SEP] t2 [SEP]
        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        # bert_input: List[int], bert_label: List[int], segment_label: List[int]
        # MLM: 进行MLM训练的输入
        # 注：这里的bert_input和bert_label的长度都可能小于seq_len
        # 因为a = [1, 2, 3], 使用a[:10]得到的数组还是[1, 2, 3]长度为3
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        # 这样做的话，padding是放在bert_input的后面，也就是句子的后面
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, index):
        # 以0.5的均匀分布的概率来选择正样本、负样本
        # 以选择的t1为例，首先选择一个对应于t1的正样本，
        # 接着选择一个对应于t1的负样本
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        # 获取corpus一行中的前一句话、后一句话，构成正样本
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

    def get_random_line(self):
        # 为了选择负样本
        # 这里是已经固定了t1, 接着我们想要找到下一句话
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        # 这个line可能已经读到了corpus的最后一行，遇到了StopIteration, 所以我们需要从头再来
        line = self.file.__next__()
        # 读到了最后一行，从头再来
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]
