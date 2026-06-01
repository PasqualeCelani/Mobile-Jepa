import pandas as pd
import matplotlib.pyplot as plt
import re


log_data = """
Epoch 1 | Loss: 0.2665 
Epoch 2 | Loss: 0.1955
Epoch 3 | Loss: 0.1909 
Epoch 4 | Loss: 0.2251 
Epoch 5 | Loss: 0.3182 
Epoch 6 | Loss: 0.2081
Epoch 7 | Loss: 0.2235 
Epoch 8 | Loss: 0.3276
Epoch 9 | Loss: 0.3285
Epoch 10 | Loss: 0.3245 
Epoch 11 | Loss: 0.3121
Epoch 12 | Loss: 0.2966 
Epoch 14 | Loss: 0.2360
Epoch 15 | Loss: 0.1177 
Epoch 16 | Loss: 0.1985 
Epoch 17 | Loss: 0.1541 
Epoch 18 | Loss: 0.0839 
Epoch 19 | Loss: 0.1153 
Epoch 20 | Loss: 0.0945  
Epoch 21 | Loss: 0.0733 
Epoch 22 | Loss: 0.0700 
Epoch 23 | Loss: 0.0621 
Epoch 24 | Loss: 0.0531 
Epoch 25 | Loss: 0.0547 
Epoch 26 | Loss: 0.0262 
Epoch 27 | Loss: 0.0496                                   
Epoch 28 | Loss: 0.0474
Epoch 29 | Loss: 0.0472
Epoch 30 | Loss: 0.0464 
Epoch 31 | Loss: 0.0452 
Epoch 32 | Loss: 0.0422
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