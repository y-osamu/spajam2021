from flask import Flask, request
from transformers import T5Tokenizer, AutoModelForCausalLM
import MeCab
import json
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")


app = Flask(__name__)

@app.route('/')
def get_request():
    res = [] 
    contents = request.args.get('value', '')
    input = tokenizer.encode(contents, return_tensors="pt")
    output = model.generate(input, do_sample=True, max_length=15, num_return_sequences=4)
    text = tokenizer.batch_decode(output)
    for i in range(4):
        #print(text[i])
        #wakati = MeCab.Tagger('-Ochasen')
        #wakati = MeCab.Tagger("-Owakati")
        num = text[i].find('>')
        text2 = str(text[i][num+1:])
        #print(wakati.parse(text2))
        #result = tagger.parseToNode(text2)
        #result = wakati.parse(text2)
        result = text2.split()
        print(result[0])
        res.append(result[0])
        res2 = {"result":res}
        enc = json.dumps(res2,ensure_ascii=False)
        print(enc)
    
    return enc

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
