import time
import shap
from lime.lime_text import LimeTextExplainer
from transformers.models.auto.tokenization_auto import AutoTokenizer
from utils.utils import initialize_model, parse_args


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


def explain(args):
    device = 'cpu'
    if args.model_path == '':
        print('Path to the pretrained model not provided! Quitting...')
        return

    test_model = initialize_model(args, None, None, None, device, None, args.model_path, explaining=True)
    
    ## Define what needs to be explained
    conversation = ["Jutro pierwszy w dzień w mojej nowej pracy. Czuję się podekscytowana i lekko zestresowana"]
    ## or read from some file
    # data = pd.read_csv(conv, sep=',', header=None)
    # conversation = []
    # for x in range(len(data)):
    #     conversation.append(data[0][x])
    # print("\nExample: ", conversation)
        
    target_names = ['negative', 'neutral', 'positive']

    if args.explainer=='lime' or args.explainer=='all':
        print('\n###### LIME ######\n')
        explainer = LimeTextExplainer(class_names=[0,1,2])

        # iterate all sentences in conversation
        all_data = []
        all_highest_sentiments = []
        all_index = []
        all_summary = []
        start_time = time.time()
        for input_to_explain in conversation:
            exp = explainer.explain_instance(input_to_explain, test_model.forward, num_features=6, labels=[0,1,2])

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


    if args.explainer=='shap' or args.explainer=='all':
        print('\n####### SHAP #######\n')

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

if __name__=="__main__":
    args = parse_args()
    args.mode = "explain"
    explain(args)
    exit(0)