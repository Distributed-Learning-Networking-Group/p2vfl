from transformers import Wav2Vec2ForSequenceClassification
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn

rank, lora_alpha = 8, 16 # lora 参数 用的lora方法微调
lora_config = LoraConfig(
    r=rank,
    lora_alpha=lora_alpha,
    target_modules=["q_proj", "v_proj"],  # 目标模块
    lora_dropout=0.1,  
    bias="none",
)

class ClientNet(nn.Module):
    def __init__(self,model_path,num_labels):
        super(ClientNet, self).__init__()
        raw_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        raw_model = get_peft_model(raw_model, lora_config)
        # raw_model.print_trainable_parameters()
        base_model = raw_model.get_base_model()

        self.feature_extractor = base_model.wav2vec2.feature_extractor

    def forward(self, input_values):
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = torch.flatten(extract_features,1)
        return extract_features

class ServerNet(nn.Module):
    def __init__(self,model_path,num_labels):
        super(ServerNet, self).__init__()
        raw_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        raw_model = get_peft_model(raw_model, lora_config)
        # raw_model.print_trainable_parameters()
        base_model = raw_model.get_base_model()

        self.feature_projection = base_model.wav2vec2.feature_projection
        self.encoder = base_model.wav2vec2.encoder
        self.projector = base_model.projector
        self.classifier = base_model.classifier

        self._mask_hidden_states = base_model.wav2vec2._mask_hidden_states

    def forward(self, extract_features):
        extract_features = extract_features.view(extract_features.shape[0],-1,512)
        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=None, attention_mask=None
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )

        hidden_states = encoder_outputs[0]
        hidden_states = self.projector(hidden_states)
        pooled_output = hidden_states.mean(dim=1)

        logits = self.classifier(pooled_output)

        return logits

class TestModel(nn.Module):
    def __init__(self,model_path,num_labels):
        super(TestModel, self).__init__()
        self.client_model = ClientNet(model_path,num_labels)
        self.server_model = ServerNet(model_path,num_labels)

    def forward(self, input_values):
        extract_features = self.client_model(input_values)
        output = self.server_model(extract_features)
        return output
