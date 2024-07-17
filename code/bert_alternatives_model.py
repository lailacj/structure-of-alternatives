from transformers import AutoTokenizer, BertForMaskedLM
import pdb
import torch

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = BertForMaskedLM.from_pretrained("google-bert/bert-base-uncased")

# inputs = tokenizer("You and your friend Sam go for a long walk together. After the walk, you go back to Sam's house. You say to Sam, 'I'm thirsty.' Sam opens the fridge and responds, 'I only have [MASK].'", return_tensors="pt")
inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# retrieve index of [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
tokenizer.decode(predicted_token_id)

labels = inputs["input_ids"]
# mask labels of non-[MASK] tokens
labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

outputs = model(**inputs, labels=labels)
print(round(outputs.loss.item(), 2))

pdb.set_trace()
