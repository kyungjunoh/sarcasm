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
                
                # GPT 생성 모델은 Padding을 왼쪽에 붙여야 생성이 꼬이지 않습니다.
                self.tokenizer.padding_side = 'left'

                # pad_token 설정
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.model.config.pad_token_id = self.model.config.eos_token_id
                
                # 전체 모델 Freeze (메모리 절약)
                for param in self.model.parameters():
                    param.requires_grad = False

                self.system_prompt = """Task Description: You are really good at detecting the sarcastic response at the last utterance of the given dialog.
                Sarcasm is the use of ironic or mocking language in which someone says the opposite of what they really mean, often to criticize, ridicule, or express contempt in a humorous or cutting way.
If the last utterance is sarcastic, print "1". If not sarcastic, print "0" 

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

Example 2: 
Context: "A: 퇴근하고 뭐 하는 거 있어요?
B: 아니 퇴근하면 힘들잖아. 그냥 집에 가서 쉬어야지.
A: 저는 얼마 전에 영어학원 등록했어요.
B: 아 진짜? 영어공부 하려고?? 저번 달에는 중국어 공부할거라며?"
Response: "A: 중국어는 너무 어렵더라고요. 그래서 큰 돈 주고 영어학원 다시 등록했어요."
Detection Result: 0

Example 3:
Context: "A: 어제 하루 종일 잠만 자느라 시험공부 하나도 못 했어."
Response: "B: 정말 성실한 하루를 보냈구나. 잘하는 짓이다."
Detection Result: 1 

Example 4: 
Context: "A: 왜 그렇게 화난 표정이야?
B: 아, 또 그러지 말라니까. 이해가 안 돼?
A: 뭐가 그렇게 힘들고 속상한 건데?
B: 일이 너무 힘들고, 집안 사정도 복잡해. 무엇보다는 내 마음이 참 괴로워.
A: 이제 잠깐 쉬어보면 어때? 좋은 일이 분명 있을거야."
Response: "B: 어차피 내가 아무리 힘들어도 상황이 바뀌는 것은 없을 거야."
Detection Result: 0 
"""

            def forward(self, input_ids, attention_mask=None, labels=None):
                # 현재 device 확인 (input_ids의 device 사용)
                device = input_ids.device
                
                # 디버깅: 모델 device 확인
                model_device = next(self.model.parameters()).device
                if model_device != device:
                    print(f"WARNING: Model is on {model_device}, but input is on {device}. Moving model to {device}")
                    self.model = self.model.to(device)
                
                # 1. 입력 텍스트 디코딩
                input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                
                prompts = []
                for text in input_texts:
                    # [SEP]로 context와 response 분리
                    if '[SEP]' in text:
                        parts = text.split('[SEP]')
                        context = parts[0].strip()
                        response = parts[1].strip() if len(parts) > 1 else ""
                    else:
                        # [SEP]가 없으면 전체를 context로 간주
                        context = text.strip()
                        response = ""
                    
                    user_prompt = f"""Context: "{context}"
Response: "{response}"
Detection Result:"""
                    full_prompt = self.system_prompt + user_prompt
                    prompts.append(full_prompt)

                # 2. 프롬프트 토큰화 (Left Padding 적용됨)
                # max_length를 지정하여 경고 메시지 방지 (필요시 1024 등으로 조절)
                new_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

                # [수정 2] token_type_ids 제거 (GPT에서 사용 안 함)
                if "token_type_ids" in new_inputs:
                    del new_inputs["token_type_ids"]

                # 3. 모델 Generate (추론)
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **new_inputs,
                        max_new_tokens=5, 
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    
                    # 4. 결과 텍스트 추출 및 logits 변환
                    gen_results = []
                    logits_list = []
                    
                    for i, sample_ids in enumerate(generated_ids):
                        input_len = new_inputs['input_ids'][i].shape[0]
                        output_text = self.tokenizer.decode(sample_ids[input_len:], skip_special_tokens=True)
                        gen_results.append(output_text.strip())
                        
                        # 생성된 텍스트를 파싱하여 logits로 변환
                        # "0" -> [1.0, 0.0], "1" -> [0.0, 1.0]
                        if "1" in output_text:
                            logits_list.append([0.0, 1.0])
                        else:
                            logits_list.append([1.0, 0.0])
                
                # logits를 텐서로 변환 (batch_size, num_labels)
                # Few-shot이므로 gradient 필요 없음
                logits = torch.tensor(logits_list, device=device, dtype=torch.float32)

                # Loss 계산 (labels가 있는 경우 - 평가용)
                loss = None
                if labels is not None:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(logits, labels)

                return {"loss": loss, "logits": logits}

        model = GPTPromptModel(args.model_name)

    return model


