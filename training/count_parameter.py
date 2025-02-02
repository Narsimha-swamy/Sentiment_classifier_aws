from model import MultiModalSentimentModel

def count_parameters(model):
    param_dict = {
        'text_encoder': 0,
        'video_encoder': 0,
        'audio_encoder': 0,
        'fusion_layer': 0,
        'emotion_classifier': 0,
        'sentiment_classifier': 0
    }

    total_params = 0
    for name,param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count


            if 'text_encoder' in name:
                param_dict['text_encoder'] += param_count
            elif 'video_encoder' in name:
                param_dict['video_encoder'] += param_count
            elif 'audio_encoder' in name:
                param_dict['audio_encoder'] += param_count
            elif 'fusion_layer' in name:
                param_dict['fusion_layer'] += param_count
            elif 'emotion_classifier' in name:
                param_dict['emotion_classifier'] += param_count
            elif 'sentiment_classifier' in name:
                param_dict['sentiment_classifier'] += param_count
            

        

    print(f'total parameters:{total_params:,}')
    print(param_dict)

if __name__=='__main__':
    model = MultiModalSentimentModel()
    count_parameters(model)