from transformers import AutoModel, BertModel, HerbertTokenizerFast, BertTokenizer, AutoTokenizer
from lime.lime_text import LimeTextExplainer
from typing import List
import argparse
import torch
import time
import shap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
    parser.add_argument("--lr_decay", default=0.9, type=float, help="learning rate decay")
    parser.add_argument("--hidden_dim", default=1024, type=int, help="hidden dimension of the layer after HerBert")
    parser.add_argument("--epochs", default=15, type=int, help="number of epochs")
    parser.add_argument("--gru", default=False, type=bool, help="whether or not to include gru layer")
    parser.add_argument("--bidirectional", default=True, type=bool, help="should the model be bidirectional")
    parser.add_argument("--n_layers", default=2, type=int, help="number of layers of GRU (after HerBert)")
    parser.add_argument("--dropout", default=0.5, type=float, help="specfiy dropout coefficient")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size for training")
    parser.add_argument("--seed", default=1234, type=int, help="seed for randomness")
    parser.add_argument("--mode", default='train', const='train', nargs='?', choices=['train','explain','test','labeling', 'pseudolabeling'], help="set the mode")
    parser.add_argument("--output_dim", default=3, type=int,  help="number of classes")
    parser.add_argument("--train_herbert", default=True, type=bool, help="should herbert be trained")
    parser.add_argument("--lr_decay_step_size", default=2, type=int,  help="how often decrease lr")
    parser.add_argument("--dataset", default='', type=str, help="dataset to train and test on")
    parser.add_argument("--cross_validation", default=False, type=bool, help="should cross validation be performed")
    parser.add_argument("--test_cv", default=False, type=bool, help="should testing of cross validation be performed")
    parser.add_argument("--crossval_folds", default=6, type=int, help="number of folds for cross validation to divide dataset and define number of iterations")
    parser.add_argument("--model_path", default='', type=str,  help="path to the pretrained model that will be loaded")
    parser.add_argument("--gpu", default=0, type=int,  help="id of the gpu to train on")#nargs='+',
    parser.add_argument("--workers", default=16, type=int,  help="number of parallel workers for data loader (recommended the same size as batch size)")
    parser.add_argument("--explainer", default='shap', const='shap', nargs='?', choices=['shap','lime','all'], help="choose which explainer to use")
    parser.add_argument("--lang", default='pl', const='pl', nargs='?', choices=['pl', 'en', 'ru'], help="give info about which language should the model handle")
    parser.add_argument("--neptune", default='', type=str)
    parser.add_argument("--quantize", default=False, type=bool, help="whether the model should be quantized")
    args = parser.parse_args()
    return args


def get_model(lang: str, device: str):
    model = None
    if lang=='pl':
        model = AutoModel.from_pretrained("allegro/herbert-base-cased").to(device)
        print("Loaded herbert")
    elif lang=='en':
        model = BertModel.from_pretrained("bert-base-cased").to(device)
        print("Loaded bert")
    elif lang=='ru':
        model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-conversational").to(device)
    else:
        raise ValueError(f"SentimentDataset __init__ : lang:{lang} not supported")

    assert model != None
    return model


def get_tokenizer(lang: str):
    tokenizer = None
    if lang=='pl':
        tokenizer = HerbertTokenizerFast.from_pretrained("allegro/herbert-base-cased")
    elif lang=='en':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif lang=='ru':
        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-conversational")
    else:
        raise ValueError(f"SentimentDataset __init__ : lang:{lang} not supported")

    assert tokenizer != None
    return tokenizer


def initialize_model(
    device, 
    args=None,
    dataloader_train=None, 
    dataloader_val=None, 
    dataloader_test=None, 
    model_path=None, 
    predicting=False,
    explaining=False,
    pretrain_path=None,
    lang="pl",
):
    if not model_path:
        model_path = args.model_path
    
    if not args:
        model = HerbertSentiment(
            train_dataloader = dataloader_train,
            val_dataloader = dataloader_val,
            test_dataloader = dataloader_test,
            predicting = predicting,
            explaining=explaining,
            device = device,
            lang=lang,
        ).to(device)
    else:
        model = HerbertSentiment(
            gru = args.gru,
            lang = args.lang,
            dropout = args.dropout,
            hidden_dim = args.hidden_dim,
            bidirectional = args.bidirectional,
            output_dim = args.output_dim,
            n_layers = args.n_layers,
            lr = args.lr,
            lr_decay = args.lr_decay,
            lr_decay_step_size = args.lr_decay_step_size,
            model_path = model_path,
            train_herbert = args.train_herbert,
            dataset = args.dataset,
            train_dataloader = dataloader_train,
            val_dataloader = dataloader_val,
            test_dataloader = dataloader_test,
            predicting = predicting,
            explaining=explaining,
            device = device,
        ).to(device)
        
    # preparing in case args is none
    lang = args.lang if args else lang
    # initialize bert
    model.herbert = get_model(lang, device)
    # initialize tokenizer
    model.tokenizer = get_tokenizer(lang)
    # load pkl for whole model
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print(f"\nLoaded state dict from {model_path}")
        
    # loading only bert weights pretrained on a different task
    if pretrain_path:
        print(f"Loading bert pretrain weights from {pretrain_path}")
        model.herbert.load_state_dict(torch.load(pretrain_path, map_location=device))
    
    return model


def set_bert_training(con: bool, model):
    for name, param in model.named_parameters():                
        if name.startswith('bert'):
            param.requires_grad = con


def read_txt(filep: str):
    try:
        with open(filep) as f:
            lines = f.readlines()
            return lines
    except Exception as e:
        print(f"Failed to read {filep} file, with error: {e}")
        return None


# colors
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
END = '\033[0m'

def explain_lime(conversation: List[str], test_model, target_names: List[str]):
    explainer = LimeTextExplainer(class_names=[0,1,2])

    # iterate all sentences in conversation
    all_data = []
    all_highest_sentiments = []
    all_index = []
    all_summary = []
    start_time = time.time()
    for input_to_explain in conversation:
        exp = explainer.explain_instance(input_to_explain, test_model.forward, num_features=6)#, labels=[0,1,2])

        # print contribution of tokens to each class
        # print ('\nExplanation for class %s' % target_names[0])
        # print ('\n'.join(map(str, exp.as_list(label=0))))
        # print()
        # print ('Explanation for class %s' % target_names[1])
        # print ('\n'.join(map(str, exp.as_list(label=1))))
        # print()
        # print ('Explanation for class %s' % target_names[2])
        # print ('\n'.join(map(str, exp.as_list(label=2))))
        # print ('\n')

        # collecting all words
        words=[]
        res = exp.as_list(label=0) # whatever label, words & length are the same
        for x in range(len(res)):
            words.append(res[x][0])

        # collecting values so taht we can easily work with them
        values=[]
        for x in range(len(target_names)):
            res = exp.as_list(label=x)
            temp=[]
            for y in range(len(res)):
                temp.append(res[y][1])
            values.append(temp)
        
        # choosing highest sentiment for each word
        highest_sentiment = []
        index = []
        for x in range(len(words)):
            h = -1
            sentiment = -1
            for y in range(len(values)): # iterating through all sentiments and cache highest with its index
                if values[y][x]>h:
                    h = values[y][x]
                    sentiment = y
            highest_sentiment.append(h)
            index.append(sentiment)

        # print words with their dominating sentiment
        for x in range(len(words)):
            color = ENDC # regular by default
            if index[x]==0: # negative
                color = RED
            elif index[x]==1: # neutral
                color = YELLOW
            elif index[x]==2: # positive
                color = GREEN

            print(color, " * ", words[x], " - ", highest_sentiment[x], " - ", index[x], END, "\n")

        print("\n")

        # print the sentence as a whole with different word colors for dominating sentiment value
        for x in range(len(highest_sentiment)):
            color = ENDC # regular by default
            if index[x]==0: # negative
                color = RED
            elif index[x]==1: # neutral
                color = YELLOW
            elif index[x]==2: # positive
                color = GREEN
            print(color, words[x], END, end="")

        sent_sentiment = [sum(values[0]), sum(values[1]), sum(values[2])]
        
        print('\n\nSentence summary sentiment: ', sent_sentiment)
        print()
        
        all_data.append(words)
        all_highest_sentiments.append(highest_sentiment)
        all_summary.append(sent_sentiment)
        all_index.append(index)

    end_time = time.time()

    # print summary for whole conversation
    print("\nThe conversation: \n")
    for y in range(len(all_data)):
        for x in range(len(all_highest_sentiments[y])):
            color = ENDC # regular by default
            if all_index[y][x]==0: # negative
                color = RED
            elif all_index[y][x]==1: # neutral
                color = YELLOW
            elif all_index[y][x]==2: # positive
                color = GREEN
            print(color, all_data[y][x], END, end="")
        print()
    print()

    print("\nThe conversation with whole sentence sentiment: \n")
    for y in range(len(all_summary)):
        sentence_sentiment_index = all_summary[y].index(max(all_summary[y]))
        color = ENDC # regular by default
        if sentence_sentiment_index==0: # negative
            color = RED
        elif sentence_sentiment_index==1: # neutral
            color = YELLOW
        elif sentence_sentiment_index==2: # positive
            color = GREEN
        for x in range(len(all_highest_sentiments[y])):
            print(color, all_data[y][x], END, end="")
        print()
    print()

    conv_sentiment = [0, 0, 0]
    for s in all_summary:
        conv_sentiment[0] += s[0]
        conv_sentiment[1] += s[1]
        conv_sentiment[2] += s[2]
        
    print("Conversation explanation time: ", (end_time-start_time), 's')
    print("Summary conversation sentiment: ", conv_sentiment)
    print("Conversation sentiment: ", target_names[conv_sentiment.index(max(conv_sentiment))])
    print()
    return


def explain_shap(conversation: List[str], test_model, target_names: List[str]):
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    explainer = shap.Explainer(test_model, tokenizer)

    all_data = []
    all_highest_sentiments = []
    all_index = []
    all_summary = []
    start_time = time.time()

    # iterate all sentences in conversation
    for input_to_explain in conversation:
        input_to_explain = [input_to_explain]

        shap_values = explainer(input_to_explain)
        values = shap_values.values
        data = shap_values.data

        # print("\n")
        # for word, shap_value in zip(data[0], values[0]):
        #     print(word,shap_value, '---',target_names[np.argmax(shap_value)])

        highest_sentiment = []
        index = []
        summary_sentiment = [0, 0, 0]
        for x in values[0]:
            elem = max(x)
            highest_sentiment.append(elem)
            index.append(list(x).index(elem))

            summary_sentiment[0] += x[0]
            summary_sentiment[1] += x[1]
            summary_sentiment[2] += x[2]

        print("\n")

        # print words, each in a new line, with their highest sentiment value and appropriate label
        for x in range(len(highest_sentiment)):
            color = ENDC # regular by default
            if index[x]==0: # negative
                color = RED
            elif index[x]==1: # neutral
                color = YELLOW
            elif index[x]==2: # positive
                color = GREEN

            print(color, " * ", data[0][x], " - ", highest_sentiment[x], " - ", index[x], END, "\n")

        print("\n")

        # print the sentence as a whole with different word colors for dominating sentiment value
        for x in range(len(highest_sentiment)):
            color = ENDC # regular by default
            if index[x]==0: # negative
                color = RED
            elif index[x]==1: # neutral
                color = YELLOW
            elif index[x]==2: # positive
                color = GREEN
            print(color, data[0][x], END, end="")

        print("\n\nSentence summary sentiment: ", summary_sentiment)
        print("Sentence sentiment: ", target_names[summary_sentiment.index(max(summary_sentiment))], "\n")

        all_data.append(data)
        all_highest_sentiments.append(highest_sentiment)
        all_index.append(index)
        all_summary.append(summary_sentiment)

    end_time = time.time()

    # print all sentences as a conversation
    print("\nThe conversation: \n")
    for y in range(len(all_data)):
        for x in range(len(all_highest_sentiments[y])):
            color = ENDC # regular by default
            if all_index[y][x]==0: # negative
                color = RED
            elif all_index[y][x]==1: # neutral
                color = YELLOW
            elif all_index[y][x]==2: # positive
                color = GREEN
            print(color, all_data[y][0][x], END, end="")
        print()
    print()

    print("\nThe conversation with whole sentence sentiment: \n")
    for y in range(len(all_summary)):
        sentence_sentiment_index = all_summary[y].index(max(all_summary[y]))
        color = ENDC # regular by default
        if sentence_sentiment_index==0: # negative
            color = RED
        elif sentence_sentiment_index==1: # neutral
            color = YELLOW
        elif sentence_sentiment_index==2: # positive
            color = GREEN
        for x in range(len(all_highest_sentiments[y])):
            print(color, all_data[y][0][x], END, end="")
        print()
    print()

    conv_sentiment = [0, 0, 0]
    for s in all_summary:
        conv_sentiment[0] += s[0]
        conv_sentiment[1] += s[1]
        conv_sentiment[2] += s[2]
        
    print("Conversation explanation time: ", (end_time-start_time), 's')
    print("Summary conversation sentiment: ", conv_sentiment)
    print("Conversation sentiment: ", target_names[conv_sentiment.index(max(conv_sentiment))])
    print()
    return


from model.Herbert import HerbertSentiment