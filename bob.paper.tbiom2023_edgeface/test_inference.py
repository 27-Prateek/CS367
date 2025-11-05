import torch
from torchvision import transforms
from face_alignment import align
from backbones import get_model

arch = "edgeface_base"  # or "edgeface_s_gamma_05" or "edgeface_xs_gamma_06"
model = get_model(arch)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

checkpoint_path = f'checkpoints/{arch}.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.eval()

path = 'checkpoints/synthface.jpeg'
aligned = align.get_aligned_face(path)
transformed_input = transform(aligned).unsqueeze(0)

print("Testing inference o...")
with torch.no_grad():
    embedding = model(transformed_input)
    print(f"✅ Inference successful!")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding:\n{embedding}")
