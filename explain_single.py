from utils.utils import initialize_model, parse_args, explain_shap, explain_lime


def explain(args):
    device = 'cpu'
    if args.model_path == '':
        print('Path to the pretrained model not provided! Quitting...')
        return

    test_model = initialize_model(
        args=args,
        device=device, 
        model_path=args.model_path, 
        explaining=True,
    )
    
    ## Define what needs to be explained
    conversation = ["Jutro pierwszy w dzień w mojej nowej pracy. Czuję się podekscytowana i lekko zestresowana"]
    target_names = ['negative', 'neutral', 'positive']

    if args.explainer=='lime' or args.explainer=='all':
        print('\n###### LIME ######\n')
        print("\nLime is currently out of service, fix will come soon. For now use SHAP (it achieves better performance anyway..).\n")
        # explain_lime(conversation=conversation, test_model=test_model, target_names=target_names)

    if args.explainer=='shap' or args.explainer=='all':
        print('\n####### SHAP #######\n')
        explain_shap(conversation=conversation, test_model=test_model, target_names=target_names)

        
if __name__=="__main__":
    args = parse_args()
    args.mode = "explain"
    explain(args)
    exit(0)