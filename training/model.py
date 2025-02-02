import torch.nn as nn
import transformers
import torch
from transformers import BertModel
from torchvision import models as visionmodels
from dataset_loader import MELD_Dataset, prepare_dataloader
from sklearn.metrics import precision_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from tqdm import tqdm

class TextEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.projection = nn.Linear(768,128)

    def forward(self,input_ids,attention_mask):
        # extract embeddings
        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)

        #use [cls] token representation

        pooler_output = outputs.pooler_output

        return self.projection(pooler_output)
    

class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone  = visionmodels.video.r3d_18(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features,128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x:torch.FloatTensor):
        # (b,n,c,h,w) -> (b,c,n,h,w)

        x =x.transpose(1,2)

        return self.backbone(x)

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # lower level features
            nn.Conv1d(64,64,kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            #high level features
            nn.Conv1d(64,128,kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)

        )
        for param in self.conv_layers.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(0.2)

        )


    def forward(self, x:torch.Tensor):
        # (b,c=1,freq_bin=64,t=300) -> (b,64,300)
        x = x.squeeze(1)

        features = self.conv_layers(x)
        #(b,128,1)

        return self.projection(features.squeeze(-1))

class MultiModalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.text_encoder = TextEncoder()
        self.video_encoder =VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # fusion layer 
        self.fusion_layer = nn.Sequential(
            nn.Linear(128*3,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        #classifiaction heads 
        self.emotion_classifier =  nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,7)
        )

        self.sentiment_classifier =  nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,3)
        )

    
    def forward(self,text_inputs,video_frames,audio_features):
        
        text_features = self.text_encoder(text_inputs['input_ids'],text_inputs['attention_mask'])
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        combined_features =torch.cat((text_features,video_features,audio_features),dim = 1) # (batch,128*3)

        # fusion_layer 
        fused_features = self.fusion_layer(combined_features)

        # outputs 
        emotion_output = self.emotion_classifier(fused_features)
        sentiment_outputs = self.sentiment_classifier(fused_features)

        return {
            'emotion': emotion_output,
            'sentiment': sentiment_outputs
        }


class MultiModaltrainer():
    def __init__(self,model,train_loader,val_loader):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Log dataset sizes
        train_size = len(train_loader.dataset)
        val_size  = len(val_loader.dataset)

        print(f"Size of the datasets:\nTraining:{train_size:,}\nValidation:{val_size:,}")
        print(f"batches per epoch:{len(self.train_loader):,}")

        timestamp = datetime.now().strftime('%b%d_%H:%M:%S') 
        base_dir = 'opt/ml/output/tensorboard' if 'SM_MODEL_DIR' in os.environ else 'runs'
        log_dir  = f"{base_dir}/run_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

        # define optimizer 
        self.optimizer = torch.optim.Adam([
            {'params':model.text_encoder.parameters(),'lr':8e-6},
            {'params':model.video_encoder.parameters(),'lr':8e-5},
            {'params':model.audio_encoder.parameters(),'lr':8e-5},
            {'params':model.fusion_layer.parameters(),'lr':5e-4},
            {'params':model.emotion_classifier.parameters(),'lr':5e-4},
            {'params':model.sentiment_classifier.parameters(),'lr':5e-4}
        ],weight_decay=1e-5)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience = 2,
        )

        self.emotion_criterion = nn.CrossEntropyLoss(
            label_smoothing= 0.05
        )

        self.sentiment_criterion = nn.CrossEntropyLoss(
            label_smoothing= 0.05
        )

        self.current_train_losses = None

    def log_metrics(self,losses,metrics = None,phase = 'train'):
        if phase == 'train':
            self.current_train_losses = losses
        else:
            self.writer.add_scalar(
                'loss/total/train',self.current_train_losses['total'],self.global_step)
            self.writer.add_scalar(
                'loss/total/val',losses['total'],self.global_step)

            self.writer.add_scalar(
                'loss/emotion/train',self.current_train_losses['emotion'],self.global_step)
            self.writer.add_scalar(
                'loss/emotion/val',losses['emotion'],self.global_step)
            
            self.writer.add_scalar(
                'loss/sentiment/train',self.current_train_losses['sentiment'],self.global_step)
            self.writer.add_scalar(
                'loss/sentiment/val',losses['sentiment'],self.global_step)
            
        if metrics:
            self.writer.add_scalar(
                f'{phase}/emotion_precision',metrics['emotion_precision'],self.global_step)
            
            self.writer.add_scalar(
                f'{phase}/emotion_accuracy',metrics['emotion_accuracy'],self.global_step)
            
            self.writer.add_scalar(
                f'{phase}/sentiment_precision',metrics['sentiment_precision'],self.global_step)
            
            self.writer.add_scalar(
                f'{phase}/sentiment_accuracy',metrics['sentiment_accuracy'],self.global_step)

    
    def train_epoch(self):
        self.model.train()
        running_loss = {'total':0,'emotion':0,'sentiment':0}

        for batch in tqdm(self.train_loader,desc='Batches'):
            device = next(self.model.parameters()).device

            text_inputs = {
                'input_ids':batch['text_features']['input_ids'].to(device),
                'attention_mask': batch['text_features']['attention_mask'].to(device)
            }

            video_inputs = batch['video_features'].to(device)
            audio_inputs = batch['audio_features'].to(device)

            emotion_labels = batch['emotion_labels'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device)

            # Zero gradients
            self.optimizer.zero_grad()

            # forward pass

            outputs = self.model(text_inputs,video_inputs,audio_inputs)

            # calculate losses using raw logits

            emotion_loss = self.emotion_criterion(outputs['emotion'],emotion_labels)
            sentiment_loss = self.sentiment_criterion(outputs['sentiment'],sentiment_labels)

            total_loss = emotion_loss + sentiment_loss

            # backward pass
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),max_norm=1.0
            )

            self.optimizer.step()

            # track losses
            running_loss['total'] += total_loss.item()
            running_loss['emotion'] += emotion_loss.item()
            running_loss['sentiment'] += sentiment_loss.item()


            self.log_metrics({
                'total':total_loss.item(),
                'emotion': emotion_loss.item(),
                'sentiment': sentiment_loss.item()
                })
            self.global_step+=1
        return {k : v/len(self.train_loader) for k,v in running_loss.items()}

    def evaluate(self,data_loader,phase = 'val'):
        self.model.eval()

        losses = {'total':0, 'emotion': 0,'sentiment': 0}
        all_emotion_preds = []
        all_sentiment_preds = []
        all_emotion_labels = []
        all_sentiment_labels = []

        with torch.inference_mode():
            for batch in data_loader:
                device = next(self.model.parameter()).device

                text_inputs = {
                    'input_ids':batch['text_inputs']['input_ids'].to(device),
                    'attention_mask': batch['text_inputs']['attention_mask'].to(device)
                }

                video_inputs = batch['video_features'].to(device)
                audio_inputs = batch['audio_features'].to(device)

                emotion_labels = batch['emotion_labels'].to(device)
                sentiment_labels = batch['sentiment_labels'].to(device)
                
                # forward pass
                outputs = self.model(text_inputs,video_inputs,audio_inputs)

                # Calculate loss

                emotion_loss = self.emotion_criterion(outputs['emotion'],emotion_labels)
                sentiment_loss = self.sentiment_criterion(outputs['sentiment'],sentiment_labels)

                total_loss = emotion_loss + sentiment_loss

                all_emotion_preds.extend(outputs['emotion'].argmax(dim=1).cpu().numpy())
                all_emotion_labels.extend(emotion_labels.cpu().numpy())
                all_sentiment_preds.extend(outputs['sentiment'].argmax(dim=1).cpu().numpy())
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

                # val lossed

                # track losses
                losses['total'] += total_loss.item()
                losses['emotion'] += emotion_loss.item()
                losses['sentiment'] += sentiment_loss.item()
        
            avg_loss = {k : v/len(data_loader) for k,v in losses.items}

            # compute precision and accuracy
            emotion_precision = precision_score(all_emotion_labels,all_emotion_preds,average='weighted')
            sentiment_precision = precision_score(all_sentiment_labels,all_sentiment_preds,average='weighted')
            
            emotion_accuracy = accuracy_score(all_emotion_labels,all_emotion_preds)
            sentiment_accuracy = accuracy_score(all_sentiment_labels,all_sentiment_preds)

            self.log_metrics(losses=avg_loss,
                             metrics={'emotion_precision':emotion_precision,
                                        'sentiment_precision' : sentiment_precision,
                                        'emotion_accuracy' : emotion_accuracy,
                                        'sentiment_accuracy': sentiment_accuracy},
                                        phase=phase)

            if phase == 'val':
                self.scheduler.step(avg_loss['total'])

            return avg_loss,{
                'emotion_precision': emotion_precision,
                'emotion_accuracy': emotion_accuracy,
                'sentiment_precision': sentiment_precision,
                'sentiment_accuracy': sentiment_accuracy
            }





if  __name__ == '__main__':


    train_video_dir = 'data/train/train_splits'
    train_csv = 'data/train/train_sent_emo.csv'

    dev_video_dir = 'data/dev/dev_splits_complete'
    dev_csv = 'data/dev/dev_sent_emo.csv'

    test_video_dir = 'data/test/output_repeated_splits_test'
    test_csv = 'data/test/test_sent_emo.csv'

    train_dataloader,dev_dataloader,test_dataloader = prepare_dataloader(train_csv=train_csv,train_video_dir=train_video_dir,
                                                                         dev_csv=dev_csv,dev_video_dir=dev_video_dir,test_csv=test_csv,test_video_dir=test_video_dir)
    

    model = MultiModalSentimentModel()
    MultiModaltrainer(model= model,train_loader=train_dataloader,val_loader=dev_dataloader)

    # dataset = MELD_Dataset(train_video_dir,train_csv)

    # sample = dataset[0]

    # model = MultiModalSentimentModel()

    # model.eval()

    # text_inputs = {'input_ids': sample['text_features']['input_ids'].unsqueeze(0),
    #                'attention_mask': sample['text_features']['attention_mask'].unsqueeze(0)}
    
    # video_inputs = sample['video_features'].unsqueeze(0)
    # audio_inputs = sample['audio_features'].unsqueeze(0)

    # with torch.inference_mode():
    #     outputs = model(text_inputs,video_inputs,audio_inputs)

    #     emotion_probs = torch.softmax(outputs['emotion'],dim=1)[0]
    #     sentiment_probs = torch.softmax(outputs['sentiment'],dim = 1)[0]

    #     emotion_map = {
    #         0:'anger' ,
    #         1:'disgust',
    #         2:'sadness' ,
    #         3:'joy',
    #         4:'neutral',
    #         5:'surprise',
    #         6:'fear'
    #     }

    #     sentiment_map = {
    #         0:'negative',
    #         1: 'neutral',
    #         2:'positive'
    #     }

    #     print("predictions for utterance:")

    #     for i,prob in enumerate(emotion_probs):
    #         print(f"{emotion_map[i]}:{prob:.2f}")
        
    #     for i,prob in enumerate(sentiment_probs):
    #         print(f"{sentiment_map[i]}:{prob:.2f}")
        
