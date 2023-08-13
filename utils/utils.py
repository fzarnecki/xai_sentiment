import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
    parser.add_argument("--lr_decay", default=0.9, type=float, help="learning rate decay")
    parser.add_argument("--h_dim", default=1024, type=int, help="hidden dimension of the layer after HerBert")
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


def initialize_model(args, dataloader_train, dataloader_val, dataloader_test, device, neptune_run, model_path='', explaining=False):
    if model_path != '':
        best_model_path = model_path
    else:
        best_model_path = args.model_path

    if best_model_path != '':
        model = HerbertSentiment.load_from_checkpoint(best_model_path,
                                                        train_dataloader = dataloader_train,
                                                        val_dataloader = dataloader_val,
                                                        test_dataloader = dataloader_test,
                                                        output_size = args.output_dim, 
                                                        hidden_dim = args.h_dim, 
                                                        n_layers = args.n_layers,
                                                        gru = args.gru, 
                                                        bidirectional = args.bidirectional, 
                                                        dropout = args.dropout,
                                                        herbert_training = args.train_herbert,
                                                        lr = args.lr, 
                                                        training_step_size = args.lr_decay_step_size,
                                                        dataset = args.dataset,
                                                        lang=args.lang,
                                                        gamma = args.lr_decay, device = device, logger = neptune_run, explaining=explaining)
        print("Loaded Herbert from", best_model_path)
    else:
        model = HerbertSentiment(train_dataloader = dataloader_train,
                                    val_dataloader = dataloader_val,
                                    test_dataloader = dataloader_test,
                                    output_size = args.output_dim, 
                                    hidden_dim = args.h_dim, 
                                    n_layers = args.n_layers, 
                                    gru = args.gru,
                                    bidirectional = args.bidirectional, 
                                    dropout = args.dropout,
                                    herbert_training = args.train_herbert,
                                    lr = args.lr, 
                                    training_step_size = args.lr_decay_step_size,
                                    dataset = args.dataset,
                                    lang=args.lang,
                                    gamma = args.lr_decay, device = device, logger = neptune_run, explaining=explaining) 
        print("Initialized new Herbert weights")

    return model


from model.Herbert import HerbertSentiment