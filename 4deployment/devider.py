import torch

large_pt_path = "chunk_002_train.pt"  # 900 MB-lıq fayl
small_pt_path = "single_shot.pt"       # Yaranacaq ~5 MB fayl

print("Böyük .pt faylı yüklənir...")
data_dict = torch.load(large_pt_path, map_location="cpu")

single_dict = {}

if isinstance(data_dict, dict):
    for key, val in data_dict.items():
        if isinstance(val, torch.Tensor):
            # .clone() istifadə edərək yaddaş istinadını kəsirik
            single_dict[key] = val[:1].clone()
        else:
            single_dict[key] = val
else:
    # Əgər dict deyilsə, birbaşa tensor özüdürsə:
    single_dict = data_dict[:1].clone()

# Yaddaş kopyasını tam ayıraraq saxlayırıq
torch.save(single_dict, small_pt_path)

print(f"Hazırdır! '{small_pt_path}' faylının ölçüsünü yoxlayın (artıq 900 MB olmayacaq).")