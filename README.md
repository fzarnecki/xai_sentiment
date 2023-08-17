## About
The repository showcases the use of model agnostic explainers. They allow to get more insights into the model decision-making. Each token is being analysed - how it influences the final output. Custom terminal insight display has been designed, to be able to use outside jupyter notebook.

## Usage
To try out the explainer one needs to provide a path to a model compatible with HerbertSentiment class. Then run ```explain_file.py``` or ```explain_single.py``` file. Provide ```--model_path``` argument and choose explainer with ```--explainer```. When running ```explain_file.py``` one can also specify ```--dataset``` which will be used as a file for prediction (should be txt, one example per line).

## Sample output


## XAI Blog post
I have written a blog post describing the methods in detail and providing examples with a deeper dive into the explanation of explanation. 

Link to the article: 

https://voicelab.ai/explainable-artificial-intelligence-xai-in-sentiment-analysis 