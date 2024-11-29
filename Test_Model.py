import torch
import torch.nn as nn
from MyModel import Model
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image

def get_args():
    parser = ArgumentParser(description='CNN inference')
    parser.add_argument('--checkpoint', '-c', type=str, default='trained_model/Best_cnn.pth')
    args = parser.parse_args()
    return args

def predictTomato(root):
    categories = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight',
                  'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                  'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus',
                  'Tomato__Tomato_YellowLeaf__Curl_Virus']
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(num_classes=10).to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=True)
        model.load_state_dict(checkpoint['model'])
    else:
        print("No checkpoint found")
        exit(0)
    model.eval()

    image = Image.open(root).convert('RGB')
    transforms = Compose([
        Resize((224, 224)),
        ToTensor()
    ])
    image = transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = nn.Softmax(dim=1)(output)
    max_idx = torch.argmax(probs)

    return categories[max_idx]

if __name__ == '__main__':
    root = '/mnt/d/DataDeepLearning/TestTomato/tomato_late_blight.jpg'
    print(predictTomato(root))