import os

def print_model_size(model_path, model_name):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"{model_name} model size: {size_mb:.2f} MB")

if __name__ == "__main__":
    orig_path = "/home/nishit/Desktop/AI/CS367/bob.paper.tbiom2023_edgeface/checkpoints/edgeface_xs_gamma_06.pt"
    quant_path = "/home/nishit/Desktop/AI/CS367/bob.paper.tbiom2023_edgeface/model_int8.pt"
    
    print_model_size(orig_path, "Original 32-bit")
    print_model_size(quant_path, "Quantized 8-bit")
