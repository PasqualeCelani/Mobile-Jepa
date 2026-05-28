import pandas as pd
import matplotlib.pyplot as plt
import re


log_data = """
Epoch 1 | Loss: 0.2622 
Epoch 2 | Loss: 0.2284
Epoch 3 | Loss: 0.2507 
Epoch 4 | Loss: 0.2584 
Epoch 5 | Loss: 0.3331
Epoch 6 | Loss: 0.2529 
Epoch 7 | Loss: 0.3430 
Epoch 8 | Loss: 0.2400
Epoch 9 | Loss: 0.3447 
Epoch 10 | Loss: 0.3387 
Epoch 11 | Loss: 0.2383
Epoch 12 | Loss: 0.3322 
Epoch 13 | Loss: 0.3177
Epoch 14 | Loss: 0.2094
Epoch 15 | Loss: 0.2324
Epoch 16 | Loss: 0.2047 
Epoch 17 | Loss: 0.2709 
Epoch 18 | Loss: 0.2560 
Epoch 19 | Loss: 0.1627
Epoch 20 | Loss: 0.1532
Epoch 21 | Loss: 0.2176
Epoch 22 | Loss: 0.1550
Epoch 23 | Loss: 0.1381
Epoch 24 | Loss: 0.1128
Epoch 25 | Loss: 0.1123
Epoch 26 | Loss: 0.1564
Epoch 27 | Loss: 0.1450
Epoch 28 | Loss: 0.0940
Epoch 29 | Loss: 0.1329
Epoch 30 | Loss: 0.1269
Epoch 31 | Loss: 0.1230
Epoch 32 | Loss: 0.0748
Epoch 33 | Loss: 0.0748
Epoch 34 | Loss: 0.0675
Epoch 35 | Loss: 0.0742
Epoch 36 | Loss: 0.0806
Epoch 37 | Loss: 0.0743
Epoch 38 | Loss: 0.0655
Epoch 39 | Loss: 0.0903
Epoch 40 | Loss: 0.0706
Epoch 41 | Loss: 0.0538
Epoch 42 | Loss: 0.0569
Epoch 43 | Loss: 0.0786
Epoch 44 | Loss: 0.0509
Epoch 45 | Loss: 0.0517
Epoch 46 | Loss: 0.0633
Epoch 47 | Loss: 0.0512
Epoch 48 | Loss: 0.0547
Epoch 49 | Loss: 0.0471
Epoch 50 | Loss: 0.0670
Epoch 51 | Loss: 0.0466
Epoch 52 | Loss: 0.0397
Epoch 53 | Loss: 0.0443
Epoch 54 | Loss: 0.0422
Epoch 55 | Loss: 0.0532
Epoch 56 | Loss: 0.0541
Epoch 57 | Loss: 0.0328
Epoch 58 | Loss: 0.0485
Epoch 59 | Loss: 0.0414
Epoch 60 | Loss: 0.0417
Epoch 61 | Loss: 0.0312
Epoch 62 | Loss: 0.0254
Epoch 63 | Loss: 0.0285
Epoch 64 | Loss: 0.0238
Epoch 65 | Loss: 0.0252
Epoch 66 | Loss: 0.0237
Epoch 67 | Loss: 0.0279
Epoch 68 | Loss: 0.0282
Epoch 69 | Loss: 0.0243
Epoch 70 | Loss: 0.0196
Epoch 71 | Loss: 0.0165
Epoch 72 | Loss: 0.0147
Epoch 73 | Loss: 0.0177
Epoch 74 | Loss: 0.0196
Epoch 75 | Loss: 0.0220
Epoch 76 | Loss: 0.0193
Epoch 77 | Loss: 0.0176
Epoch 78 | Loss: 0.0226
Epoch 79 | Loss: 0.0149
Epoch 80 | Loss: 0.0182
Epoch 81 | Loss: 0.0142
Epoch 82 | Loss: 0.0186
Epoch 83 | Loss: 0.0216
Epoch 84 | Loss: 0.0151
Epoch 85 | Loss: 0.0223
Epoch 86 | Loss: 0.0184
Epoch 87 | Loss: 0.0196
Epoch 88 | Loss: 0.0163
Epoch 89 | Loss: 0.0174
Epoch 90 | Loss: 0.0140
Epoch 91 | Loss: 0.0141
Epoch 92 | Loss: 0.0187
Epoch 93 | Loss: 0.0190
Epoch 94 | Loss: 0.0228
Epoch 95 | Loss: 0.0224
Epoch 96 | Loss: 0.0207
Epoch 97 | Loss: 0.0134
Epoch 98 | Loss: 0.0147
Epoch 99 | Loss: 0.0144
Epoch 100 | Loss: 0.0176
"""


pattern = r"Epoch (\d+) \| Loss: ([\d.]+)"
data = re.findall(pattern, log_data)
df = pd.DataFrame(data, columns=['Epoch', 'Loss'])
df['Epoch'] = df['Epoch'].astype(int)
df['Loss'] = df['Loss'].astype(float)


plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Loss'], marker='o', linestyle='-', markersize=3, label='Loss')
plt.title('Training Loss Progress (100 Epochs)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()


plt.savefig('loss_ssl.png')
print("Plot saved successfully.")