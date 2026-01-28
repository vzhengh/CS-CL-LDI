from timm import create_model
import torch
import torch.nn as nn
from transformers import RobertaModel

EMBEDDING_DIM = 512

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.swin = create_model("swin_base_patch4_window7_224.ms_in22k", pretrained=True, features_only=True)
        for param in self.swin.parameters():
            param.requires_grad = True
        self.swin_output_dim = self.swin.feature_info.channels()[-1] 
        self.fc1 = nn.Linear(self.swin_output_dim * 7 * 7, EMBEDDING_DIM) 
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        for param in self.fc1.parameters():
            param.requires_grad = True


    def forward(self, x):
        swin_features = self.swin(x)[-1]
        swin_features = swin_features.view(swin_features.size(0), -1)  
        output = self.fc1(swin_features)
        return output

class RobertaEncoder(nn.Module):
    def __init__(self, roberta_model_path="roberta-base"):
        super(RobertaEncoder, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model_path)
        self.projection = nn.Linear(self.roberta.config.hidden_size, EMBEDDING_DIM)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        for param in self.roberta.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = roberta_output.last_hidden_state[:, 0, :]
        pooled_output = torch.mean(roberta_output.last_hidden_state, dim=1) 
        return self.projection(cls_token+pooled_output)
        
class LVL(nn.Module):
    def __init__(self):
        super(LVL, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = RobertaEncoder()
        self.t_prime = nn.Parameter(torch.ones([]) * np.log(0.07))
        self.b = nn.Parameter(torch.ones([]) * 0)

    def get_images_features(self,images):
        image_embeddings = self.image_encoder(images) 
        image_embeddings = nn.functional.normalize(image_embeddings, p=2, dim=-1)
        return image_embeddings

    def get_texts_feature(self,input_ids,attention_mask):
        text_embeddings = self.text_encoder(input_ids, attention_mask) 
        text_embeddings = nn.functional.normalize(text_embeddings, p=2, dim=-1)
        return text_embeddings

    def forward(self, images, input_ids, attention_mask):
        image_embeddings = self.get_images_features(images)
        text_embeddings = self.get_texts_feature(input_ids, attention_mask)
        return image_embeddings, text_embeddings
