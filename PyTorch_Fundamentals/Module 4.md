# Convolutional Neural Networks

## 1. What is Convolution?
![](<../Images/c_func.png>)
- Operation where a **filter (kernel)** slides over the image
- Extracts local patterns (edges, textures, shapes)

![](<../Images/visual_convolution.gif>)

```python
nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
````

* Each filter learns a **different feature**
* Convolution Formula:
$$
y(i, j) = \sum_{m} \sum_{n} x(i + m, j + n) \cdot w(m, n) + b
$$


###  Terms
- $ x $ → input image  
- $ w $ → kernel (filter)  
- $ y $ → output feature map  
- $ b $ → bias  
- $ (i, j) $ → position in output  
- $ (m, n) $ → kernel indices  


---

## 2. What is Pooling?

* Downsamples feature maps → reduces spatial size
* Keeps important information while reducing computation

![](<../Images/pooling.png>)
```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

### Effect on size:

* 32 × 32 → 16 × 16 → 8 × 8
* Each pooling halves height & width

---

## 3. Feature Maps (Activation Maps)

* Output of convolution layers
* Represent **detected patterns** (edges, shapes, textures)

![](<../Images/feature_maps.png>)
> More channels = more learned features

---

## 4. Basic CNN Block

* Standard pattern: Conv → ReLU → Pool

```python
self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
self.relu1 = nn.ReLU()
self.pool1 = nn.MaxPool2d(2)
```

---

## 5. Forward Pass Structure

```python
def forward(self, x):
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.pool1(x)

    x = self.conv2(x)
    x = self.relu2(x)
    x = self.pool2(x)

    x = self.flatten(x)
    x = self.fc(x)
    return x
```

---

## 6. Dropout (Why Needed)

* Prevents overfitting
* Randomly disables neurons during training

![](<../Images/dropout.png>)
```python
nn.Dropout(0.5)
```

### Co-adaptation Problem

* Neurons rely too much on each other
* Example:

  * Model learns "snow + animal = wolf"
  * Husky misclassified as wolf

> Dropout forces neurons to learn **independent features**

---

## 7. Weight Decay (Regularization)

* Penalizes large weights → simpler model

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)
```

* Helps prevent overfitting by discouraging extreme values

---

## 8. Dynamic Computation Graphs

* Graph built **during execution (forward pass)**

### Computation Graph:

* Tracks operations → used for backprop

![](<../Images/computation_graph.png>)

### Advantages:

* Flexible (variable inputs, control flow)
* Easier debugging

### Disadvantages:

* Slightly slower than static graphs

---

### Sequential Model

* Fixed pipeline (linear stack of layers)

```python
nn.Sequential(
    nn.Conv2d(...),
    nn.ReLU(),
    nn.MaxPool2d(...)
)
```

* Less flexible than custom `forward()`

---

## 9. Inspecting Model

```python
print(model)
```

* Shows architecture and layers

---

## 10. Counting Parameters

```python
total_params = sum(p.numel() for p in model.parameters())
print(total_params)
```

---

### Inspect Each Layer

```python
for name, param in model.named_parameters():
    print(name, param.shape)
```

---

## 11. Debugging Shape Errors

### Common Issue:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

### Fix:

* Check layer dimensions

```python
print(model.fc1.weight.shape)
```

* Ensure flatten size matches linear layer input

---



* Conv → extract features
* Pool → reduce size
* ReLU → add non-linearity
* Dropout & weight decay → prevent overfitting

