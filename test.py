import torch
print(torch.cuda.is_available())  # ควรได้ True
print(torch.cuda.device_count())  # จำนวน GPU ที่ใช้งานได้
print(torch.cuda.get_device_name(0))  # ชื่อ GPU ตัวแรก