# import pytorch_lightning as pl
# import torch
# import torch.nn as nn
# import numpy as np
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from transformers import AutoModel,AutoTokenizer, BertTokenizer, BertModel
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from scikitplot.metrics import plot_confusion_matrix
# from neptune.new.types import Filea

# # resolving import problem
# import sys 
# sys.path.append('..')
# from utils.utils import set_bert_training, get_dataset_path


# class HerbertSentiment(pl.LightningModule):
#     """Polish Herbert for sentiment analysis"""
#     def __init__(self, train_dataloader=None,val_dataloader=None, test_dataloader=None, output_size=3, hidden_dim=1024, n_layers=2,
#                 gru=False, bidirectional=True, dropout=0.5, herbert_training=True, lr = 1e-4,
#                 device = "cuda", training_step_size = 2, gamma=0.9, logger=None, 
#                 dataset='default', lang='pl', explaining=False, option="train"):
#         super().__init__()

#         # caching whether gru layer should be present
#         self.gru = gru

#         if lang=='pl':
#             self.herbert = AutoModel.from_pretrained("allegro/herbert-base-cased").to(device)
#             print("Loaded herbert")
#         elif lang=='en':
#             self.herbert = BertModel.from_pretrained("bert-base-uncased").to(device)
#             print("Loaded bert")
#         elif lang=='ru':
#             self.herbert = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-conversational").to(device)
#         else:
#             print('Language not supported.')
#             return

#         if lang=='pl':
#             self.tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
#         elif lang=='en':
#             self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         elif lang=='ru':
#             self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-conversational")
#         else:
#             print("Language not supported.")
#             return

#         # new_tokens = ['$URL$', '$NUMBER$','COVID','covid','coronavirus','ковид','коронавирус']#,'plandemie'
#         # self.tokenizer.add_tokens(new_tokens)
#         # self.herbert.resize_token_embeddings(len(self.tokenizer))

#         self.herbert.train()
#         self.herbert_output_size = 768
        
#         # architecture
#         if not self.gru:
#             self.fc = nn.Linear(self.herbert_output_size, output_size)
#         else:
#             self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_size)
#         self.softmax = nn.Softmax(1)
#         self.dropout = nn.Dropout(dropout)
#         if self.gru:
#             self.rnn = nn.GRU(
#                 self.herbert_output_size,
#                 hidden_dim,
#                 num_layers = n_layers,
#                 bidirectional = bidirectional,
#                 batch_first = True,
#                 dropout = 0 if n_layers < 2 else dropout
#             )
            
#         self.cross_entropy_loss = nn.CrossEntropyLoss(weight=torch.Tensor([0.34, 0.36, 0.32]))
        
#         self.metrics = nn.ModuleDict({
#             'accuracy': pl.metrics.Accuracy(),
#             'recall': pl.metrics.classification.Recall(num_classes=output_size, average='macro'),
#             'f1': pl.metrics.classification.F1(num_classes=output_size, average='macro'),
#             'precision': pl.metrics.classification.Precision(num_classes=output_size, average='macro')
#         })
#         self.test_metrics = []

#         self.neptune_logger = logger
#         self.lr = lr
#         self.training_step_size = training_step_size
#         self.gamma = gamma
        
#         # Setting bert training
#         set_bert_training(herbert_training, self.herbert)
        
#         self.test_dataloader = test_dataloader
#         self.train_dataloader = train_dataloader
#         self.val_dataloader = val_dataloader

#         self.dataset = dataset
#         self.explaining = explaining
#         self.gpu = device
#         self.mode = option
        
#         self.target_names = ['negative', 'neutral', 'positive']
        

#     def forward(self, words, **kwargs):
#         if self.explaining: #xai
#             tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
#             tokens = tokenizer(list(words), padding = True, truncation = True, return_tensors='pt')
#             input_ids = tokens['input_ids'].to(self.gpu)
#             attention_mask = tokens['attention_mask'].to(self.gpu)
#             if not self.gru:
#                 embedded = self.herbert(input_ids, attention_mask = attention_mask).pooler_output
#             else:
#                 embedded = self.herbert(input_ids, attention_mask = attention_mask)[0]
#         else:
#             input_ids = words[1].to(self.gpu)
#             attention_mask = words[0].to(self.gpu)
#             if not self.gru:
#                 embedded = self.herbert(input_ids, attention_mask = attention_mask).pooler_output
#             else:
#                 embedded = self.herbert(input_ids, attention_mask = attention_mask)[0]

#         if self.gru:
#             _, hidden = self.rnn(embedded)

#             if self.rnn.bidirectional:
#                 hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
#             else:
#                 hidden = self.dropout(hidden[-1,:,:])

#         if self.gru:
#             output = self.fc(hidden)
#         else:
#             output = self.fc(embedded)

#         soft = self.softmax(output)

#         if self.explaining:
#             return soft.detach().numpy()
#         return soft


#     def compute_metrics(self, preds, labels):
#         new_metrics = {}
#         for name, metric in self.metrics.items():
#             new_metrics[name] = metric(torch.from_numpy(preds).cuda(), torch.from_numpy(labels).cuda())
#         return new_metrics


#     def training_step(self, training_batch, batch_idx):
#         x, y = training_batch
#         logits = self.forward(x)
#         loss = self.cross_entropy_loss(logits, y)

#         self.neptune_logger["train/batch loss"].log(loss)
#         return dict(
#             loss=loss,
#             log=dict(
#                 loss=loss
#             ),
#             labels=y.detach().cpu().numpy(),
#             predictions=logits.detach().cpu().numpy()
#         )


#     def validation_step(self, val_batch, batch_idx):
#         x, y = val_batch
#         logits = self.forward(x)
#         loss = self.cross_entropy_loss(logits, y)
#         self.neptune_logger["val/batch loss"].log(loss)
#         return dict(
#             valid_loss=loss,
#             log=dict(
#                 valid_loss=loss
#             ),
#             labels=y.detach().cpu().numpy(),
#             predictions=logits.detach().cpu().numpy()
#         )


#     def test_step(self, test_batch, batch_idx):
#         x, y = test_batch
#         logits = self.forward(x)
#         loss = self.cross_entropy_loss(logits, y)
#         if self.neptune_logger is not None:
#             self.neptune_logger["test/batch loss"].log(loss)
#         return dict(
#             test_loss=loss,
#             log=dict(
#                 test_loss=loss
#             ),
#             labels=y.detach().cpu().numpy(),
#             inputs=x[2],
#             predictions=logits.detach().cpu().numpy(),
#         )


#     def training_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
#         labels = np.concatenate([x['labels'] for x in outputs])
#         predictions = np.concatenate([x['predictions'] for x in outputs])
#         metrics = self.compute_metrics(predictions, labels)
#         self.log_everything(metrics,avg_loss,mode='train')

#         self.optimizer.step()
         

#     def validation_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
#         labels = np.concatenate([x['labels'] for x in outputs])
#         predictions = np.concatenate([x['predictions'] for x in outputs])
#         print("Validation report")
#         print(classification_report(labels,np.argmax(predictions,axis=1)))
#         metrics = self.compute_metrics(predictions, labels)
#         self.log_everything(metrics,avg_loss,mode='val')
#         self.log('val_accuracy', metrics['accuracy'])

#         # stopping bert training at some point
#         if self.current_epoch >= 7:
#             print('Stopped bert training')
#             self.bert_training = False
#             set_bert_training(False, self)


#     def test_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
#         labels = np.concatenate([x['labels'] for x in outputs])
#         predictions = np.concatenate([x['predictions'] for x in outputs])

#         # cache test metrics
#         self.test_metrics = self.compute_metrics(predictions, labels)

#         if self.mode == 'test':
#             # Show incorrect predictions
#             inputs = np.concatenate([x['inputs'] for x in outputs])
#             print(inputs.shape)
#             print(labels.shape)
#             print(predictions.shape)

#             try:
#                 os.mkdir('errors')
#             except OSError:
#                 pass

#             try:
#                 os.remove('errors/very_incorrect_preds.txt')
#                 os.remove('errors/slightly_incorrect_preds.txt')
#                 os.remove('errors/very_correct_preds.txt')
#                 os.remove('errors/slightly_correct_preds.txt')
#             except OSError:
#                 pass

#             very_incorrect = open('errors/very_incorrect_preds.txt', 'a')
#             slightly_incorrect = open('errors/slightly_incorrect_preds.txt', 'a')
#             very_correct = open('errors/very_correct_preds.txt', 'a')
#             slightly_correct = open('errors/slightly_correct_preds.txt', 'a')

#             for i, x in enumerate(inputs):
#                 if labels[i] != np.argmax(predictions[i]):
#                     #weekly incorrect predictions
#                     if np.max(predictions[i])<0.6:
#                         slightly_incorrect.write("Text: " + x + "\n True label: " + str(labels[i]) + "\n Predicted label: " + str(np.argmax(predictions[i])) +"\n Detailed predictions: " + str(predictions[i]) +"\n"+"\n")

#                     #strong incorrect predictions
#                     if np.max(predictions[i])>0.9:
#                         very_incorrect.write("Text: " + x + "\n True label: " + str(labels[i]) + "\n Predicted label: " + str(np.argmax(predictions[i])) +"\n Detailed predictions: " + str(predictions[i]) +"\n"+"\n")
#                 else:
#                     #week good predictions
#                     if np.max(predictions[i])<0.6:
#                         slightly_correct.write("Text: " + x + "\n True label: " + str(labels[i]) + "\n Predicted label: " + str(np.argmax(predictions[i])) +"\n Detailed predictions: " + str(predictions[i]) +"\n"+"\n")
#                     #strong good predictions
#                     if np.max(predictions[i])>0.9:
#                         very_correct.write("Text: " + x + "\n True label: " + str(labels[i]) + "\n Predicted label: " + str(np.argmax(predictions[i])) + "\n Detailed predictions: " + str(predictions[i]) +"\n"+"\n")

#             very_incorrect.close()
#             slightly_incorrect.close()
#             very_correct.close()
#             slightly_correct.close()

#         elif self.mode == 'labeling':
#             inputs = np.concatenate([x['inputs'] for x in outputs])

#             text = []
#             predicted_labels = []
#             for i, x in enumerate(inputs):
#                 text.append(x)
#                 predicted_labels.append(str(np.argmax(predictions[i])))
            
#             df = pd.DataFrame({'text': text, 'label': predicted_labels})

#             filepath = get_dataset_path(self.dataset).replace('.json', '').replace('.csv', '') + "_labeled.csv"
#             try:
#                 os.remove(filepath)
#             except OSError:
#                 pass
#             df.to_csv(filepath, index = False, header = False)

#         # classes
#         print("Test report")
#         print(classification_report(labels,np.argmax(predictions,axis=1), labels=[0,1,2], target_names=['negative','neutral','positive']))
#         print("Confusion matrix")
#         print(confusion_matrix(labels,np.argmax(predictions,axis=1), normalize='true'))
#         print(np.unique(labels, return_counts=True))
#         fig, ax = plt.subplots(figsize=(16, 12))
#         plot_confusion_matrix(labels,np.argmax(predictions,axis=1), normalize='true', ax=ax)
#         fig, ax = plt.subplots(figsize=(16, 12))
#         plot_confusion_matrix(labels,np.argmax(predictions,axis=1), ax=ax)
#         if self.neptune_logger:
#             self.neptune_logger["test/confusion_matrix_normalized"].upload(File.as_image(fig))
#             self.neptune_logger["test/confusion_matrix"].upload(File.as_image(fig))
#         metrics = self.compute_metrics(predictions, labels)
#         self.log_everything(metrics,avg_loss,mode='test')


#     def configure_optimizers(self):
#         # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)#Adam - basic one 
#         self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)#imitation of BertAdam

#         self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=self.training_step_size,gamma=self.gamma)
#         #self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=2000, num_training_steps=140000)

#         return [self.optimizer], [self.scheduler]


#     def train_dataloader(self):
#         return self.train_dataloader

#     def val_dataloader(self):
#         return self.val_dataloader

#     def test_dataloader(self):
#         return self.test_dataloader
    

#     def predictor(self, tokens):
#         results = []
#         logits=self.forward(tokens)
#         results.append(logits.cpu().detach().numpy()[0])
#         results_array = np.array(results)
#         return results_array


#     def log_everything(self,metrics,loss,mode='train'):
#         if self.neptune_logger:
#             self.neptune_logger[mode+"/epoch loss"].log(loss)
#             self.neptune_logger[mode+"/epoch accuracy"].log(metrics['accuracy'])
#             self.neptune_logger[mode+"/epoch f1"].log(metrics['f1'])
#             self.neptune_logger[mode+"/epoch precision"].log(metrics['precision'])
#             self.neptune_logger[mode+"/epoch recall"].log(metrics['recall'])

#from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
#from transformers import XLMTokenizer, RobertaModel, get_constant_schedule_with_warmup
# from utils.utils import set_bert_training, get_pkl_filename
from scikitplot.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchmetrics as tm
import torch.nn as nn
import numpy as np
import torch
import os


class HerbertSentiment(pl.LightningModule):
    
    """Polish Herbert for sentiment analysis"""
    def __init__(
        self,
        gru=False,
        lang="pl",
        dropout=0.5,
        hidden_dim=1024,
        bidirectional=True,
        output_dim=3,
        n_layers=2,
        lr=1e-5,
        lr_decay_step_size=2,
        lr_decay=0.9,
        model_path=None,
        pretrain_path=None,
        train_herbert=True,
        dataset=None,
        train_dataloader=None,
        val_dataloader=None, 
        test_dataloader=None,
        device="cuda",
        predicting=True, #TODO try to remove, maybe use predict mode
        explaining=False,
        train_adapter=False,
    ):
        super().__init__()

        self.gru = gru
        self.herbert_output_size = 768
        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(dropout)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        if self.gru:
            self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
            self.rnn = nn.GRU(self.herbert_output_size,
                            hidden_dim,
                            num_layers = n_layers,
                            bidirectional = bidirectional,
                            batch_first = True,
                            dropout = 0 if n_layers < 2 else dropout)
        else:
            self.fc = nn.Linear(self.herbert_output_size, output_dim)
        
        self.lr = lr
        self.training_step_size = lr_decay_step_size
        self.gamma = lr_decay
        self.bert_training = train_herbert
        #set_bert_training(self.bert_training, self.herbert)
        
        self.test_dataloader = test_dataloader
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.dataset = dataset

        self.model_path = model_path
        self.predicting = predicting
        self.explaining = explaining
        self.hardware = device
        self.train_adapter = train_adapter
        
        self.metrics = nn.ModuleDict({
            'accuracy': tm.Accuracy()
            # 'recall': tm.Recall(num_classes=output_dim, average='macro'),
            # 'f1': tm.F1Score(num_classes=output_dim, average='macro'),
            # 'precision': tm.Precision(num_classes=output_dim, average='macro')
        })
        self.previous_best_val_accuracy = None

    def forward(self, words):
        if self.predicting or self.explaining:
            tokens = self.tokenizer(list(words), padding = True, truncation = True, return_tensors='pt')
            input_ids = tokens['input_ids'].to(self.hardware)
            attention_mask = tokens['attention_mask'].to(self.hardware)
            if self.gru:
                embedded = self.herbert(input_ids, attention_mask = attention_mask)[0]
            else:
                embedded = self.herbert(input_ids, attention_mask = attention_mask).pooler_output
        else:
            input_ids = words[1].to(self.hardware)
            attention_mask = words[0].to(self.hardware)
            if self.gru:
                embedded = self.herbert(input_ids, attention_mask = attention_mask)[0]
            else:
                embedded = self.herbert(input_ids, attention_mask = attention_mask).pooler_output
                

        if self.gru:
            _, hidden = self.rnn(embedded)

            if self.rnn.bidirectional:
                hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
            else:
                hidden = self.dropout(hidden[-1,:,:])

        if self.gru:
            output = self.fc(hidden)
        else:
            output = self.fc(embedded)

        soft = self.softmax(output)

        if self.explaining:#TODO Check
            return soft.detach().numpy()
        return soft

    def embed(self, words):
        tokens = self.tokenizer(list(words), padding = True, truncation = True, return_tensors='pt')
        input_ids = tokens['input_ids'].to(self.hardware)
        attention_mask = tokens['attention_mask'].to(self.hardware)
        embedded = self.herbert(input_ids, attention_mask = attention_mask).pooler_output
        return embedded

    def compute_metrics(self, preds, labels):
        new_metrics = {}
        for name, metric in self.metrics.items():
            new_metrics[name] = metric(
                torch.from_numpy(preds).to(self.hardware), 
                torch.from_numpy(labels).to(self.hardware)
            )
        return new_metrics

    def training_step(self, training_batch, batch_idx):
        x, y = training_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        return dict(
            loss=loss,
            log=dict(
                loss=loss.detach()
            ),
            labels=y.detach().cpu().numpy(),
            predictions=logits.detach().cpu().numpy()
        )

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return dict(
            valid_loss=loss,
            log=dict(
                valid_loss=loss.detach()
            ),
            labels=y.detach().cpu().numpy(),
            predictions=logits.detach().cpu().numpy()
        )

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return dict(
            test_loss=loss,
            log=dict(
                test_loss=loss.detach()
            ),
            labels=y.detach().cpu().numpy(),
            inputs=x[2],
            predictions=logits.detach().cpu().numpy(),
        )

    def training_epoch_end(self, outputs):
        #avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        labels = np.concatenate([x['labels'] for x in outputs])
        predictions = np.concatenate([x['predictions'] for x in outputs])
        #metrics = self.compute_metrics(predictions, labels)
        self.optimizer.step()
         
    def validation_epoch_end(self, outputs):
        labels = np.concatenate([x['labels'] for x in outputs])
        predictions = np.concatenate([x['predictions'] for x in outputs])
        print("Validation report")
        print(classification_report(labels,np.argmax(predictions,axis=1)))
        metrics = self.compute_metrics(predictions, labels)
        acc = metrics['accuracy']
        self.log('val_accuracy', acc)
        
        # caching model if some improvement
        # if not self.previous_best_val_accuracy or acc > self.previous_best_val_accuracy:
        #     # save model
        #     filename = get_pkl_filename(self.model_path, self.dataset)
        #     if not self.train_adapter:
        #         torch.save(self.state_dict(), filename)
        #         print(f"Accuracy improved from {self.previous_best_val_accuracy} to {acc}. Caching model to {filename}\n")
        #     # save adapters
        #     if self.train_adapter:
        #         adapters_dir = filename.split(".")[0] + "_adapters"
        #         if not os.path.exists(adapters_dir):
        #             os.makedirs(adapters_dir)
        #         self.herbert.save_all_adapters(adapters_dir)
        #         print(f"Accuracy improved from {self.previous_best_val_accuracy} to {acc}. Caching adapters to {adapters_dir}\n")
        #     #
        #     self.previous_best_val_accuracy = acc

        # # stopping bert training at some point to only finetune other layers TODO refactor hardcode
        # if self.current_epoch >= 7:
        #     print('Stopped bert training')
        #     self.bert_training = False
        #     set_bert_training(False, self)

    def test_epoch_end(self, outputs):
        #avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        labels = np.concatenate([x['labels'] for x in outputs])
        predictions = np.concatenate([x['predictions'] for x in outputs])
        #inputs = np.concatenate([x['inputs'] for x in outputs])
        #self.test_metrics = self.compute_metrics(predictions, labels)

        print("Test report")
        print(classification_report(labels, np.argmax(predictions,axis=1), labels=[0,1,2]))#, target_names=self.target_names))
        print("Confusion matrix")
        print(confusion_matrix(labels, np.argmax(predictions,axis=1), normalize='true'))
        print(np.unique(labels, return_counts=True))
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(labels, np.argmax(predictions,axis=1), normalize='true', ax=ax)
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(labels, np.argmax(predictions,axis=1), ax=ax)
        #metrics = self.compute_metrics(predictions, labels)

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)#imitation of BertAdam
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.training_step_size, gamma=self.gamma)
        return [self.optimizer], [self.scheduler]

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader

    def test_dataloader(self):
        return self.test_dataloader
    
    # TODO check what is it used for
    def predictor(self, tokens):
        print(tokens)
        results = []
        logits=self.forward(tokens)
        results.append(logits.cpu().detach().numpy()[0])
        results_array = np.array(results)
        return results_array
