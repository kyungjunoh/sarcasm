from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoConfig
from transformers import AutoTokenizer
import torch.nn as nn
import torch


def get_model(args):

    if args.model == 'bert':
        model_name = args.model_name
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    elif args.model == 'gpt':

        class GPTForClassification(nn.Module):
            def __init__(self, model_name, num_labels=2):
                super().__init__()
                self.config = AutoConfig.from_pretrained(model_name)
                self.gpt = AutoModelForCausalLM.from_pretrained(model_name)
                self.classifier = nn.Linear(self.config.hidden_size, num_labels)

            def forward(self, input_ids, attention_mask=None, labels=None):
                outputs = self.gpt(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                last_hidden = outputs.hidden_states[-1][:, -1, :]   # 마지막 토큰
                logits = self.classifier(last_hidden)

                loss = None
                if labels is not None:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(logits, labels)

                return {"loss": loss, "logits": logits}

        model = GPTForClassification(args.model_name, num_labels=2)

        # =========================
        # GPT backbone freeze
        # =========================
        for param in model.gpt.parameters():   # <-- self.gpt 참조
            param.requires_grad = False

        # classifier만 학습
        for param in model.classifier.parameters():
            param.requires_grad = True
        
    elif args.model == 'gpt_prompt':
        
        class GPTPromptModel(nn.Module):
            def __init__(self, model_name):
                super().__init__()
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

                # pad_token 설정
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.model.config.pad_token_id = self.model.config.eos_token_id

                # 전체 모델 Freeze
                for param in self.model.parameters():
                    param.requires_grad = False

                # "0"과 "1" 토큰 ID 확인
                self.token_0 = self.tokenizer.encode("0", add_special_tokens=False)[0]
                self.token_1 = self.tokenizer.encode("1", add_special_tokens=False)[0]
                print(f"Token IDs: '0'={self.token_0}, '1'={self.token_1}")

                # Few-shot 프롬프트 (512 토큰 안에 들어가도록)
                self.prompt_template = """Task: You are really good at detecting the sarcastic response at the last utterance of the given dialog.
Sarcasm is ironic or mocking language where someone says the opposite of what they mean.
If sarcastic, print "1". If not, print "0".

Example 1:
Context: "A: 요리는 잘 되가?
B: 응 지금까지는 순항 중이야. 하나만 빼고.
A: 뭐가 문제야? 잘 안 되는 게 있어?
B: 계란 후라이가 조금 탔어."
Response: "A: 이거 정말 바삭바삭하겠는걸."
Detection Result: 1

Example 2:
"context":"A: 최근에 본 영화 중에 추천할 만한 것이 있나요?
B: 저는 "터미네이터"를 보았는데, 상당히 재미있었습니다.
A: 그렇군요. 저도 그 영화를 볼까 생각 중이었습니다. 어떤 부분이 가장 인상적이었나요?
B: 아마도 그것은 결말 부분일 것입니다. 전형적인 할리우드 영화의 결말처럼 상당히 화려하고 긴장감 넘치는 장면이었습니다."
"response":"A: 아, 그럼 정말로 마지막 장면은 놀라울 정도로 감동적이었겠군요."
Detection Result: 0

Context: "{context}"
Response: "{response}"
Answer:"""

            def forward(self, input_ids, attention_mask=None, labels=None, raw_context=None, raw_response=None):
                device = input_ids.device
                
                # 모델 device 확인
                model_device = next(self.model.parameters()).device
                if model_device != device:
                    self.model = self.model.to(device)

                # 원본 텍스트 사용 (있으면)
                if raw_context is not None and raw_response is not None:
                    contexts = raw_context if isinstance(raw_context, list) else [raw_context]
                    responses = raw_response if isinstance(raw_response, list) else [raw_response]
                else:
                    # Fallback: input_ids 디코딩 (하지만 잘림)
                    input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                    contexts = []
                    responses = []
                    for text in input_texts:
                        if '[SEP]' in text:
                            parts = text.split('[SEP]')
                            contexts.append(parts[0].strip())
                            responses.append(parts[1].strip() if len(parts) > 1 else "")
                        else:
                            contexts.append(text.strip())
                            responses.append("")

                prompts = []
                for context, response in zip(contexts, responses):
                    # 대화 길이 제한 (512 토큰 안에 들어가도록)
                    ctx = context[-150:] if len(context) > 150 else context
                    resp = response[-100:] if len(response) > 100 else response
                    
                    prompt = self.prompt_template.format(context=ctx, response=resp)
                    prompts.append(prompt)

                # 프롬프트 토큰화 (max_length 512로 증가)
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
                if "token_type_ids" in inputs:
                    del inputs["token_type_ids"]

                # Constrained decoding: "0"과 "1"의 확률만 계산
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    next_token_logits = outputs.logits[:, -1, :]  # 마지막 위치의 logits
                
                # "0"과 "1" 토큰의 logits만 추출하여 classification logits로 사용
                logits_0 = next_token_logits[:, self.token_0]
                logits_1 = next_token_logits[:, self.token_1]
                logits = torch.stack([logits_0, logits_1], dim=1)  # [batch_size, 2]

                # Loss 계산
                loss = None
                if labels is not None:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(logits, labels)

                return {"loss": loss, "logits": logits}

        model = GPTPromptModel(args.model_name)

    return model


