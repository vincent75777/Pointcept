import os, sys, socket, struct, torch
import numpy as np
from types import ModuleType

# --- CONFIGURATION (LidarView UDP) ---
UDP_IP = "127.0.0.1" 
UDP_PORT = 8888      
OFFSET_ID = 11       
OFFSET_COUNT = 15    
OFFSET_DATA = 17     
BYTES_PER_POINT = 12  
MIN_POINTS_IA = 400   

# --- PARAMÈTRES DE CORRECTION ---
# On estime une perte de ~1.5% de hauteur par mètre de distance
# car les faisceaux s'écartent.
COEFF_CORRECTION = 0.015 

# --- MOCK POINTOPS ---
if "pointops" not in sys.modules:
    m = ModuleType("pointops")
    m.query_and_group_count = lambda *args, **kwargs: None
    sys.modules["pointops"] = m

try:
    from pointcept.engines.defaults import default_config_parser
    from pointcept.datasets.transform import Compose
    from pointcept.models.builder import build_model
except ImportError as e:
    print(f"❌ Erreur Pointcept : {e}")
    sys.exit(1)

def load_model():
    base = os.path.dirname(__file__)
    config_path = os.path.join(base, "configs/3dp/cls-3dp-ptv3-v1m1-0-base.py")
    cfg = default_config_parser(config_path, options={})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model).eval().to(device)
    ckpt_path = os.path.join(base, "exp/3dp/cls-ptv3/model/model_last.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt.get("state_dict", ckpt.get("model", ckpt)), strict=False)
    infer_tf = Compose(cfg.infer.transform)
    return model, infer_tf, device

def main():
    print("="*65)
    print("🚀 ANALYSEUR AVEC CORRECTION DE DISTANCE DYNAMIQUE")
    print("👉 CTRL+C pour quitter")
    print("="*65)
    
    try:
        model, infer_tf, device = load_model()
        print(f"✅ Modèle chargé sur : {device}")
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(1.0)
    
    storage = {} 
    CLASS_NAMES = ["MARCHE", "ACCROUPI", "ESCALADE"]

    try:
        while True:
            try:
                data, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            
            if not data.startswith(b'\x28\x2a'): continue
            
            cid = struct.unpack_from('<H', data, OFFSET_ID)[0]
            num_pts_paquet = struct.unpack_from('<H', data, OFFSET_COUNT)[0]
            
            current_pts = []
            for i in range(num_pts_paquet):
                idx = OFFSET_DATA + (i * 12)
                if idx + 12 <= len(data):
                    current_pts.append(struct.unpack_from('<3f', data, idx))
            
            if cid not in storage: storage[cid] = []
            storage[cid].extend(current_pts)

            if len(storage[cid]) >= MIN_POINTS_IA:
                total_pts = len(storage[cid])
                pts = np.array(storage[cid][:512], dtype=np.float32)
                
                # --- CALCULS DE BASE ---
                z_min, z_max = np.min(pts[:, 2]), np.max(pts[:, 2])
                hauteur_brute = z_max - z_min
                dist_x = np.abs(np.mean(pts[:, 0]))
                
                # --- LOGIQUE DE CORRECTION ---
                # Plus on est loin, plus on multiplie la hauteur pour compenser la perte de points
                # Formule : Hauteur Brute * (1 + (Distance * Coeff))
                hauteur_corrigee = hauteur_brute * (1 + (dist_x * COEFF_CORRECTION))
                
                # --- NORMALISATION POUR L'IA ---
                pts_norm = pts.copy()
                pts_norm[:, 0] -= np.mean(pts[:, 0])
                pts_norm[:, 1] -= np.mean(pts[:, 1])
                pts_norm[:, 2] -= z_min 
                
                input_coords = pts_norm[:, [0, 2, 1]] 
                d = infer_tf({"coord": input_coords})
                for k, v in d.items():
                    if isinstance(v, np.ndarray): d[k] = torch.from_numpy(v).to(device)
                    elif torch.is_tensor(v): d[k] = v.to(device)
                
                d["batch"] = torch.zeros(d["coord"].shape[0], dtype=torch.long, device=device)
                d["offset"] = torch.tensor([d["coord"].shape[0]], dtype=torch.long, device=device)

                with torch.no_grad():
                    output = model(d) 
                    logits = output.get("logits", list(output.values())[0]) if isinstance(output, dict) else output
                    probs = torch.softmax(logits, dim=-1)[0]
                    pred = torch.argmax(probs).item()
                    
                    # --- AFFICHAGE ---
                    print(f"\n📍 [ID {cid}] à {dist_x:.2f}m")
                    print(f"   📏 Hauteur Brute    : {hauteur_brute:.2f} m")
                    print(f"   ✨ Hauteur Corrigée : {hauteur_corrigee:.2f} m")
                    print(f"   ❄️  Points           : {total_pts}")
                    print(f"   🤖 IA Classe        : {CLASS_NAMES[pred]} ({probs[pred]:.1%})")
                
                storage[cid] = []

    except KeyboardInterrupt:
        print("\nArrêt.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()