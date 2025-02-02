import os 
import argparse
import torchaudio
import torch
from dataset_loader import prepare_dataloader
from model import MultiModalSentimentModel,MultiModaltrainer
from tqdm import tqdm 
import json
from install_ffmpeg import install_ffmpeg
import sys

# AWS sagemaker
SM_MODEL_DIR  = os.environ.get('SM_MODEL_DIR','.')
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING','opt/ml/input/data/training')
SM_CHANNEL_VALIDATION = os.environ.get('SM_CHANNEL_VALIDATION','opt/ml/input/data/validation')
SM_CHANNEL_TEST = os.environ.get('SM_CHANNEL_TEST','opt/ml/input/data/test')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs',type = int,default=20)
    parser.add_argument('--batch-size',type = int,default=16)
    parser.add_argument('--learning-rate',type = int,default=0.001)

    # data directories

    parser.add_argument('--train-dir',type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument('--val-dir',type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument('--test-dir',type=str, default=SM_CHANNEL_TEST)
    parser.add_argument('--model-dir',type = str, default=SM_MODEL_DIR)

    return parser.parse_args()

def main():
    # install ffmpeg on sagemaker pc
    # if not install_ffmpeg():
    #     print('ERROR ffmpeg installation failed. Cannot continue training')
    #     sys.exit(1)

    print("available audio backends:")
    print(str(torchaudio.list_audio_backends()))

    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # track intial gpu memory 
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_consumed = torch.cuda.max_memory_allocated() / 1024**3
    print(f"memory used:{memory_consumed:.2f} GB")

    # prepare data loaders

    train_video_dir = os.path.join(args.train_dir,'train_splits')
    train_csv = os.path.join(args.train_dir,'train_sent_emo.csv')

    dev_video_dir = os.path.join(args.val_dir,'dev_splits_complete')
    dev_csv = os.path.join(args.val_dir,'dev_sent_emo.csv')

    test_video_dir = os.path.join(args.test_dir,'output_repeated_splits_test')
    test_csv = os.path.join(args.test_dir,'test_sent_emo.csv')

    train_dataloader,val_dataloader,test_dataloader = prepare_dataloader(train_csv=train_csv,train_video_dir=train_video_dir,
                                                                         dev_csv=dev_csv,dev_video_dir=dev_video_dir,test_csv=test_csv,test_video_dir=test_video_dir
                                                                         ,batch_size=args.batch_size)
    
    print(f"Training CSV path:{train_csv}")
    print(f"Training Video Dir:{train_video_dir}")

    model = MultiModalSentimentModel().to(device=device)

    trainer = MultiModaltrainer(model=model,train_loader=train_dataloader,val_loader=val_dataloader)

    best_val_loss = float('inf')

    metrics_data = {
        'train_losses' : [],
        'val_losses' :[],
        'epochs': []    }
    
    for epoch in tqdm(range(args.epochs),desc="Epochs"):
        training_loss = trainer.train_epoch()
        val_loss,val_metrics = trainer.evaluate(val_dataloader)
        
        
        
        # track metrics
        metrics_data['train_losses'].append(training_loss['total'])
        metrics_data['val_losses'].append(val_loss['total'])
        metrics_data['epochs'].append(epoch)

        # log metrics in sagemaker format
        print(json.dumps({
            'metrics': [
                {'Name': "train:loss","Value":training_loss['total']},
                {'Name': "val:loss","Value":val_loss['total']},
                {'Name': "val:emotion_precision","Value":val_metrics['emotion_precision']},
                {'Name': "val:emotion_accuracy","Value":val_metrics['emotion_accuracy']},
                {'Name': "val:sentiment_precision","Value":val_metrics['sentiment_precision']},
                {'Name': "val:sentiment_accuracy","Value":val_metrics['sentiment_accuracy']},   
            ]
        }))

        if torch.cuda.is_available():
            memory_consumed = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory used:{memory_consumed:.2f} GB")

        # Save best model 

        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            torch.save(model.state_dict(),os.path.join(args.model_dir,'model.pth'))
    
    # after training is complete ,eval on test set

    print('evaluating on test set...')
    
    test_loss,test_metrics = trainer.evaluate(test_dataloader,phase='test')

    print(json.dumps({
            'metrics': [
                {'Name': "test:loss","Value":test_loss['total']},
                {'Name': "test:emotion_accuracy","Value":test_metrics['emotion_accuracy']},
                {'Name': "test:emotion_precision","Value":test_metrics['emotion_precision']},
                {'Name': "test:sentiment_precision","Value":test_metrics['sentiment_precision']},
                {'Name': "test:sentiment_accuracy","Value":test_metrics['sentiment_accuracy']},   
            ]
        }))








if __name__=='__main__':
    main()