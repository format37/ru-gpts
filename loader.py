from transformers import GPT2LMHeadModel, GPT2Tokenizer
model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"

def load_tokenizer_and_model(model_name_or_path):
  return GPT2Tokenizer.from_pretrained(model_name_or_path), GPT2LMHeadModel.from_pretrained(model_name_or_path).cuda()

def my_generate(
    model, tok, text,
    do_sample=True, max_length=500, repetition_penalty=5.0,
    top_k=5, top_p=0.95, temperature=1,
    num_beams=None,
    no_repeat_ngram_size=3
    ):
  input_ids = tok.encode(text, return_tensors="pt").cuda()
  out = model.generate(
      input_ids.cuda(),
      max_length=max_length,
      repetition_penalty=repetition_penalty,
      do_sample=do_sample,
      top_k=top_k, top_p=top_p, temperature=temperature,
      num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
      )
  return list(map(tok.decode, out))

tok, model = load_tokenizer_and_model("sberbank-ai/rugpt3large_based_on_gpt2")
#text = 'Завтра будет хорошая погода'
#generated = my_generate(model, tok, text, num_beams=10)
#print(generated)
print('ok')
