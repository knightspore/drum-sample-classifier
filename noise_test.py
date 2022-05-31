import torch
from util import classes, classes_reverse
from tqdm import tqdm

# Set Device
device = torch.device("cuda:0" if
                      torch.cuda.is_available()
                      else
                      "cpu"
                      )
print(f"Device: {device}")

# Load Model
standardModel = torch.load(
    './models/standard/model_lr0.0002_mlr0.01_e1000_b64.pt',
    map_location=torch.device('cpu')
)
standardModel.eval()
print("Model Loaded")


def run_with_noise(model):
    random_input = torch.randn([1, 2, 64, 344]).to(device)
    output = model(torch.sigmoid(random_input))
    _, prediction = torch.max(output, 1)
    return classes_reverse[prediction.numpy()[0]]

totalGuesses = 1000
countClassGuesses = {}
for k in enumerate(classes.keys()):
    countClassGuesses[k[1]] = 0

print("Running Tests on Random Noise...")
for i in tqdm(range(totalGuesses)):
     guess = run_with_noise(standardModel)
     countClassGuesses[guess] += 1

print("Tests Complete:")
for guess in enumerate(countClassGuesses):
    g = guess[1]
    print(f"{g}: {(countClassGuesses[g] / totalGuesses) * 100:.2f}")
