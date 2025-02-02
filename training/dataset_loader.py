import pandas as pd
import collections
import torch.utils.data.dataloader
from transformers import AutoTokenizer
import os
import torch
from torch.utils.data import Dataset,DataLoader
import cv2
import numpy as np
import subprocess
import torchaudio
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MELD_Dataset(torch.utils.data.Dataset):
    def __init__(self,video_dir,csv_path):
        
        self.label_data = pd.read_csv(csv_path)
                
        self.video_dir = video_dir

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear
        self.emotion_map = {
            'anger' : 0,
            'disgust':1,
            'sadness' :2,
            'joy':3,
            'neutral': 4,
            'surprise': 5,
            'fear': 6
        }

        self.sentiment_map = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }



    def __len__(self):
        return (len(self.label_data))
    

    def process_vframes(self,video_path:str):
        cap= cv2.VideoCapture(video_path)
        frames = []

        try:

            # Check if video path is correct 
            if not cap.isOpened():
                raise ValueError(f'Video not found: {video_path}')
            
            # Check if Video exists in path
            ret,frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f'Video not found: {video_path}')

            # set video capture frame back to 0
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)

            while len(frames) < 30 and cap.isOpened():
                ret,frame = cap.read()
                if not ret :
                    break
                
                frame = cv2.resize(frame,(224,224))
                frame = frame / 255.0
                frames.append(frame)

        except Exception as e:
            raise ValueError(f'Video error:{str(e)}')
        finally:
            cap.release()            
        
        # 
        if len(frames) == 0:
            print('No frames found in video')

        # if len less than 30 pad with zeros
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30-len(frames)) 

        # if len more tan 30 truncate
        else:
            frames = frames[:30]
        
        return torch.FloatTensor(np.array(frames)).permute(0,3,1,2)

    def process_audio(self,video_path:str):
        audio_path = video_path.replace('.mp4','.wav')

        try:
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            waveform,sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate,16000)
                waveform = resampler(waveform)
            
            mel_spectogram = torchaudio.transforms.MelSpectrogram(
                                            sample_rate=sample_rate,
                                            n_mels = 64,
                                            n_fft = 1024,
                                            hop_length=512)
            
            mel_spec = mel_spectogram(waveform)

            # normalize spectogram 
            mel_spec = (mel_spec-mel_spec.mean())/mel_spec.std()

            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec,(0,padding))
            else:
                mel_spec = mel_spec[:,:,:300]
            
            return mel_spec
        
        except subprocess.CalledProcessError as e:
            raise ValueError(f'audio extraction failed: {str(e)}')
        except Exception as e:
            raise ValueError(f'audio error: {str(e)}')
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        


    def __getitem__(self,idx):
        
        if isinstance(idx,torch.Tensor):
            idx = idx.item()


        try:
            row = self.label_data.iloc[idx]

            video_file = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
            video_path = os.path.join(self.video_dir, video_file)

            path_Exists = os.path.exists(video_path)
            if not path_Exists:
                raise FileNotFoundError(f'File not found:{video_path}')
            
            text_inputs = self.tokenizer(text = row['Utterance'],
                                            padding = 'max_length',
                                            truncation= True,
                                            max_length = 128,
                                            return_tensors='pt')
            
            video_frames  = self.process_vframes(video_path)
            # print(video_frames.shape)

            audio_features = self.process_audio(video_path)

            # print(audio_features.shape)        

            # map and sentiment and emotion labels
            emotion_label = self.emotion_map[row['Emotion'].lower()]
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]
            
            data_sample = dict()

            data_sample = {
                'text_features' : {
                    'input_ids' : text_inputs['input_ids'].squeeze(),
                    'attention_mask': text_inputs['attention_mask'].squeeze()},            
                'video_features' : video_frames,
                'audio_features' : audio_features,
                'sentiment_labels': torch.tensor(sentiment_label),
                'emotion_labels': torch.tensor(emotion_label)      
                }

            return data_sample
        except Exception as e:
            print(f"Failed processing {video_path}: {str(e)}")
            return None

def collate_fn(batch):
    # filled out none samples
    batch = list(filter(None,batch))

    return torch.utils.data.dataloader.default_collate(batch)

def prepare_dataloader(train_csv,train_video_dir,
                       dev_csv,dev_video_dir,
                       test_csv,test_video_dir,batch_size = 32):

    train_dataset = MELD_Dataset(train_video_dir,train_csv)
    dev_dataset = MELD_Dataset(dev_video_dir,dev_csv)
    test_dataset = MELD_Dataset(test_video_dir,test_csv)

    train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)

    dev_loader = DataLoader(dataset = dev_dataset,
                                  batch_size=batch_size,
                                  collate_fn=collate_fn)
    test_loader = DataLoader(dataset = test_dataset,
                                  batch_size=batch_size,
                                  collate_fn=collate_fn)
    
    return train_loader,dev_loader,test_loader

if __name__=='__main__':
    video_dir = 'data/dev/dev_splits_complete'
    csv_path = 'data/dev/dev_sent_emo.csv'


    dataset = MELD_Dataset(video_dir, csv_path)

    print(dataset[0])  
     

