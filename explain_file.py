from utils.utils import initialize_model, parse_args, read_txt, explain_lime, explain_shap


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
    
    # Load dataset
    data_path = ""
    if not data_path:
        data_path = args.dataset
    conversation = read_txt(data_path)
    if not conversation:
        print(f"No data to explain, quitting.")
        return
    print("\nExample: ", conversation)
        
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