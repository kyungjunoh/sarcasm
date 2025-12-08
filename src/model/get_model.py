from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoConfig
import torch.nn as nn


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

    return model


