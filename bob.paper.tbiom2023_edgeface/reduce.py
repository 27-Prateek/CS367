from backbones import get_model
import torch

# Step 1: Create model instance (pass model name, e.g. 'edgeface_xs_gamma_06')
model = get_model('edgeface_xs_gamma_06')

# Step 2: Load the weights (state_dict)
state_dict = torch.load('./checkpoints/edgeface_xs_gamma_06.pt', map_location='cpu')
model.load_state_dict(state_dict)

# Step 3: Set model to eval mode
model.eval()

# Now you can do inference or quantization properly

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save the quantized model
torch.jit.save(torch.jit.script(quantized_model), 'model_int8.pt')
