import os
import sys
import cv2
import torch
import numpy as np
import pickle
from pathlib import Path
from PIL import Image

# Add EdgeFace repository to path
EDGEFACE_PATH = "../"
sys.path.insert(0, EDGEFACE_PATH)

# Import from EdgeFace's face_alignment module
from face_alignment.mtcnn import MTCNN
from face_alignment.align import get_aligned_face

# Import EdgeFace model loader
from backbones import get_model

# Configuration
MODEL_NAME = 'edgeface_xs_gamma_06'
MODEL_PATH = "../checkpoints/edgeface_xs_gamma_06.pt"
EMBEDDINGS_DIR = "./face_embeddings"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create embeddings directory if it doesn't exist
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


def load_edgeface_model():
    """Load the EdgeFace model from checkpoint"""
    try:
        # Check if file exists
        if not os.path.exists(MODEL_PATH):
            print(f"✗ Model file not found at {MODEL_PATH}")
            return None
        
        print(f"Loading model: {MODEL_NAME}")
        
        # Step 1: Create model instance
        model = get_model(MODEL_NAME)
        print(f"✓ Model architecture created")
        
        # Step 2: Load the weights (state_dict)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"✓ Model weights loaded from {MODEL_PATH}")
        
        # Step 3: Set model to eval mode
        model.eval()
        model.to(DEVICE)
        print(f"✓ Model set to eval mode on {DEVICE}")
        
        return model
    
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_mtcnn_model():
    """Load MTCNN for face detection and alignment from EdgeFace repo"""
    try:
        device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        mtcnn = MTCNN(device=device_str, crop_size=(112, 112))
        print(f"✓ MTCNN model loaded on {device_str}")
        return mtcnn
    except Exception as e:
        print(f"✗ Error loading MTCNN: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_image_from_path(image_path):
    """Load image from file path and convert to PIL Image"""
    try:
        if not os.path.exists(image_path):
            print(f"✗ File not found: {image_path}")
            return None
        
        img = Image.open(image_path).convert('RGB')
        print(f"✓ Image loaded from {image_path}")
        return img
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return None


def capture_from_webcam():
    """Capture image from laptop webcam"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Could not open webcam")
            return None
        
        print("📷 Webcam opened. Press SPACE to capture, ESC to cancel")
        
        captured_image = None
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Error reading frame")
                break
            
            cv2.imshow("Capture Face - Press SPACE to capture, ESC to cancel", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE
                # Convert BGR to RGB for PIL
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                captured_image = Image.fromarray(rgb_frame)
                print("✓ Image captured from webcam")
                break
            elif key == 27:  # ESC
                print("✗ Capture cancelled")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return captured_image
    
    except Exception as e:
        print(f"✗ Error capturing from webcam: {e}")
        return None


def align_face_with_mtcnn(pil_image):
    """Align face using EdgeFace's MTCNN alignment"""
    try:
        # Use EdgeFace's get_aligned_face function
        aligned_face = get_aligned_face(image_path=None, rgb_pil_image=pil_image)
        
        if aligned_face is None:
            print("✗ No face detected or alignment failed")
            return None
        
        print("✓ Face detected and aligned (112x112)")
        return aligned_face
    
    except Exception as e:
        print(f"✗ Error in face alignment: {e}")
        return None


def extract_embedding(aligned_face_pil, model):
    """Extract embedding using EdgeFace model"""
    try:
        # Convert PIL Image to numpy array
        face_array = np.array(aligned_face_pil)
        
        # Normalize to [0, 1] if needed
        if face_array.max() > 1:
            face_array = face_array.astype(np.float32) / 255.0
        else:
            face_array = face_array.astype(np.float32)
        
        # Convert to tensor: HWC -> BCHW
        face_tensor = torch.from_numpy(face_array).to(DEVICE)
        face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Extract embedding
        with torch.no_grad():
            embedding = model(face_tensor)
        
        embedding_np = embedding.cpu().numpy().flatten()
        print(f"✓ Embedding extracted (shape: {embedding_np.shape})")
        return embedding_np
    
    except Exception as e:
        print(f"✗ Error extracting embedding: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_embedding(name, embedding):
    """Save embedding to file"""
    try:
        # Sanitize name to create valid filename
        safe_name = "".join(c for c in name if c.isalnum() or c in ('-', '_')).rstrip()
        if not safe_name:
            print("✗ Invalid name/ID")
            return False
        
        embedding_file = os.path.join(EMBEDDINGS_DIR, f"{safe_name}.pkl")
        
        with open(embedding_file, 'wb') as f:
            pickle.dump(embedding, f)
        
        print(f"✓ Embedding saved for '{name}' at {embedding_file}")
        return True
    except Exception as e:
        print(f"✗ Error saving embedding: {e}")
        return False


def register_face(model):
    """Register a new face"""
    print("\n" + "="*50)
    print("REGISTER FACE")
    print("="*50)
    print("1. Upload/Path to image")
    print("2. Use laptop webcam")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    pil_image = None
    if choice == '1':
        image_path = input("Enter image path: ").strip()
        pil_image = load_image_from_path(image_path)
    elif choice == '2':
        pil_image = capture_from_webcam()
    else:
        print("✗ Invalid choice")
        return
    
    if pil_image is None:
        print("✗ Failed to load image")
        return
    
    # Align face using EdgeFace MTCNN
    aligned_face = align_face_with_mtcnn(pil_image)
    if aligned_face is None:
        return
    
    # Extract embedding using EdgeFace model
    embedding = extract_embedding(aligned_face, model)
    if embedding is None:
        return
    
    # Get user name/ID
    name = input("\nEnter student name/ID: ").strip()
    if not name:
        print("✗ Name/ID cannot be empty")
        return
    
    # Save embedding
    if save_embedding(name, embedding):
        print(f"\n✓✓ Face registered successfully for '{name}'")
    else:
        print("\n✗ Failed to save embedding")


def check_face(model):
    """Check/recognize a face"""
    print("\n" + "="*50)
    print("CHECK FACE")
    print("="*50)
    
    # Load all registered embeddings
    registered_embeddings = {}
    embedding_files = list(Path(EMBEDDINGS_DIR).glob("*.pkl"))
    
    if not embedding_files:
        print("✗ No registered faces found. Please register faces first.")
        return
    
    for pkl_file in embedding_files:
        try:
            with open(pkl_file, 'rb') as f:
                registered_embeddings[pkl_file.stem] = pickle.load(f)
        except Exception as e:
            print(f"✗ Error loading {pkl_file}: {e}")
    
    print(f"✓ Loaded {len(registered_embeddings)} registered face(s)")
    print(f"  Registered students: {', '.join(registered_embeddings.keys())}")
    
    # Capture from webcam
    pil_image = capture_from_webcam()
    if pil_image is None:
        return
    
    # Align face using EdgeFace MTCNN
    aligned_face = align_face_with_mtcnn(pil_image)
    if aligned_face is None:
        return
    
    # Extract embedding using EdgeFace model
    embedding = extract_embedding(aligned_face, model)
    if embedding is None:
        return
    
    # Compare with registered embeddings (cosine similarity)
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("\nMatching results:")
    best_match = None
    best_score = -1
    threshold = 0.5  # Adjust threshold as needed
    
    for name, registered_emb in registered_embeddings.items():
        similarity = cosine_similarity([embedding], [registered_emb])[0][0]
       # print(f"  {name}: {similarity:.4f}")
        
        if similarity > best_score:
            best_score = similarity
            best_match = name
    
    if best_score > threshold:
        print(f"\n✓✓ MATCH FOUND: {best_match} (confidence: {best_score:.4f})")
    else:
        print(f"\n✗ NO MATCH - REGISTER FIRST")


def main():
    """Main program"""
    print("\n" + "="*50)
    print("EdgeFace Registration & Recognition System")
    print("="*50 + "\n")
    
    # Load model
    model = load_edgeface_model()
    if model is None:
        print("\n✗ Failed to load model. Exiting...")
        return
    
    # Load MTCNN
    mtcnn = load_mtcnn_model()
    if mtcnn is None:
        print("\n✗ Failed to load MTCNN. Exiting...")
        return
    
    print("\n" + "="*50 + "\n")
    
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. Register face")
        print("2. Check face")
        print("3. Exit")
        
        choice = input("\nEnter choice (1, 2, or 3): ").strip()
        
        if choice == '1':
            register_face(model)
        elif choice == '2':
            check_face(model)
        elif choice == '3':
            print("\n✓ Exiting... Goodbye!")
            break
        else:
            print("\n✗ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
