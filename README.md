## About
The repository showcases the use of model agnostic prediction explainers (SHAP and LIME). They allow to get more insights into the model decision-making. Each token is being analysed - how it influences the final output. Custom terminal insight display has been designed, to be able to use outside jupyter notebook.

## Usage
To try out the explainer one needs to provide a path to a model compatible with HerbertSentiment class. Then run ```explain_file.py``` or ```explain_single.py``` file. Provide ```--model_path``` argument and choose explainer with ```--explainer```. When running ```explain_file.py``` one can also specify ```--dataset``` which will be used as a file for prediction (should be txt, one example per line).

Sentiment colors:
- red for negative
- yellow for neutral
- green for positive

## Sample output
First each token is shown, along with its sentiment contribution and class it is contributing to:

<img src="https://github.com/fzarnecki/xai_sentiment/blob/main/output_images/output1.png" width="580" height="630">

Then summary information is displayed, to easily analyse aggregated output. Overall sentiment probability distrbution, sentence colored token-wise and then the sentence in dominating sentiment color:

<img src="https://github.com/fzarnecki/xai_sentiment/blob/main/output_images/output2.png" width="942" height="245">

## XAI Blog post
I have written a blog post describing the methods in detail and providing examples with a deeper dive into the explanation of explanation. 

Link to the article: 
https://voicelab.ai/explainable-artificial-intelligence-xai-in-sentiment-analysis 
