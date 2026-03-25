# Data Management in PyTorch

## Dataset Class
- Custom class to **store and access data**
- Inherits from `torch.utils.data.Dataset`
- Defines **how data is loaded**

---

### 3 Core Methods

#### 1. `__init__`
- Loads data (files, arrays, preprocessing)
- Runs **once when dataset is created**

```python
def __init__(self, data, labels):
    self.data = data
    self.labels = labels
````

---

#### 2. `__len__`

* Returns total number of samples
* Used for batching and iteration

```python
def __len__(self):
    return len(self.data)
```

---

#### 3. `__getitem__`

* Returns one sample (data + label) at index
* Called during training

```python
def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]
```

---

### DataLoader

* Wraps Dataset → handles:

  * **Batching**
  * **Shuffling**
  * **Parallel loading**

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

### Transforming Images

- Raw images often have **different sizes & formats (PIL)** → not usable directly in PyTorch  
- Use **`transforms` pipeline** to standardize data before training  

---

### Key Steps

#### Resize & Crop
- Ensure all images have the **same size**
- Use:
  - `Resize()` → scale images (keep aspect ratio)
  - `CenterCrop()` → get consistent square input

```python
transforms.Resize(256)
transforms.CenterCrop(224)
````

---

####  Convert to Tensor

* Convert PIL image → PyTorch tensor
* Changes shape → `[C, H, W]`
* Scales pixel values → `[0, 1]`

```python
transforms.ToTensor()
```

---

#### Normalize

* Standardizes pixel distribution using mean & std
* Helps model learn better and stabilizes training

```python
transforms.Normalize(mean, std)
```

---

#### Important Notes

* Order matters:

  * Image transforms → `ToTensor()` → `Normalize()`
* Some transforms only work on:

  * Images (before tensor)
  * Tensors (after conversion)

---

## Splitting, Loading, Augmentation & Debugging

![](<../Images/data_splitting.png>)


### 1. Dataset Splitting
- Split dataset into **train / validation / test**
- Ensures:
  - Train → learning  
  - Val → tuning  
  - Test → final evaluation  

```python
from torch.utils.data import random_split

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)
````

---

### 2. Batching with DataLoader

* Groups data into batches for efficient training
* Handles iteration automatically

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32)
val_loader   = DataLoader(val_dataset, batch_size=32)
test_loader  = DataLoader(test_dataset, batch_size=32)
```

---

### 3. Common Dataset Mistake

* Do NOT load full dataset inside `__getitem__`
* Causes massive slowdown (reloading every sample)

```python
# Wrong
def __getitem__(self, idx):
    data = pd.read_csv("huge_file.csv")
    return data.iloc[idx]
```

* Load once in `__init__`, only index in `__getitem__`

---

### 4. Data Augmentation (Why + How)

**Why:**

* Increase data diversity
* Prevent overfitting
* Make model robust

**How:**

* Apply random transforms only on training data

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),

    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

* Validation/Test → no randomness

```python
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

---

### 5. Handling Corrupt / Bad Images

**Why:**

* Real datasets may contain:

  * corrupted files
  * very small images
  * wrong formats

**How:**

```python
def __getitem__(self, idx):
    try:
        image = Image.open(path)

        image.verify()        # check corruption
        image = Image.open(path)

        if image.size[0] < 32 or image.size[1] < 32:
            raise ValueError("Image too small")

        if image.mode != "RGB":
            image = image.convert("RGB")

    except Exception as e:
        # log error
        return None
```

---

### 6. Tracking Errors

* Keep log of problematic samples for debugging

```python
self.error_log.append({
    "index": idx,
    "error": str(e)
})
```

* Summarize after training

```python
def get_error_summary(self):
    for err in self.error_log[:5]:
        print(err)
```

---

### 7. Visualizing Augmentation

**Why:**

* Ensure augmentations are reasonable (not destructive)

```python
for i in range(8):
    img, _ = dataset[idx]
    plt.imshow(img.permute(1, 2, 0))
```

---

### 8. Data Tracking (Performance Debugging)

* Track:

  * access frequency
  * load time

```python
start = time.time()
result = super().__getitem__(idx)
load_time = time.time() - start

if load_time > 1.0:
    print("Slow load")
```

---

