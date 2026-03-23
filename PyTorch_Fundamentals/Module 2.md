# Core Data Tools
## Transforms
``` python
transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((mean,), (std,))
])
```
- `ToTensor()` : converts to pytorch tensors and centres around 0 & 1
- `Normalize()` : mean and standard deviation
## Dataset
![](<../Images/pre-built-datasets.png>)

## DataLoader

`` train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)``

# Model Building
![Model Building](<../Images/model-building.png>)

## Evaluation

```python  
model.eval() # Set evaluation mode (disable dropout, etc.)  
  
with torch.no_grad(): # Disable gradient tracking  
correct = 0  
total = 0  
  
for images, labels in test_loader:  
outputs = model(images)  
  
_, predicted = torch.max(outputs, 1) # predicted class  
  
total += labels.size(0)  
correct += (predicted == labels).sum().item()  
  
accuracy = 100 * correct / total  
print(f'Accuracy: {accuracy}%')
```

----

# Loss Functions

## 📉 MSE Loss (Mean Squared Error)

```python
loss_function = nn.MSELoss()
loss = loss_function(predictions, targets)
````

---

### 🧮 Formula

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2  
$$

---

- Average squared difference
- Just the difference might just give zero error
- So squared error
- Commonly used for **regression tasks**
--- 
 
## 📊Cross Entropy Loss

```python
loss_function = nn.CrossEntropyLoss()
loss = loss_function(predictions, targets)
````

---

### 🧮 Formula

$$
\text{CE} = - \frac{1}{n} \sum_{i=1}^{n} \log(p_{y_i})  
$$

---

- Measures how well predicted **probabilities match true classes**
- Focuses on the **probability of the correct class**
- **Penalizes confident wrong predictions heavily**
--- 
## Other Loss Functions
![Other Loss Functions](<../Images/other-loss-functions.png>)
# 🔁 Backpropagation & Optimizers 
## 📉 1. Backward (Gradient Computation)
- `loss.backward()` computes **gradients for each parameter**
- Gradients = how much each weight contributed to the error :contentReference[oaicite:0]{index=0}  

> ❗ Backward does NOT update weights → only calculates gradients

This is applied layer by layer from output to input. Computing gradients using chain rule:
This is applied layer by layer from output to input. Computing gradients using chain rule:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial w}$$

## 📌 For a Single Neuron

Given:
$z = w \cdot x + b$
$a = \sigma(z)$

Then:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$
---

## 🧠 2. Gradient Descent
- Think: **standing on a hill**
- Gradient = direction of steepest increase  
- To minimize loss → move **opposite direction** (gradient descent)

Weights and biases are updated using gradient descent:

$$
w = w - \eta \frac{\partial L}{\partial w}
$$

$$
b = b - \eta \frac{\partial L}{\partial b}
$$

Where:
- $\eta$: Learning rate
- $\frac{\partial L}{\partial w}$: Gradient computed via backpropagation

---

## ⚙️ 3. Optimizer (Update Step)
- Updates weights using gradients

```python
optimizer.step()
````

- Example: **SGD**
    
    - Big gradient → big update
        
    - Small gradient → small update
        
    - Controlled by **learning rate**
        

---

## 🎚️ Learning Rate

- Too small → slow learning
    
- Too large → unstable / overshooting
    
- Needs tuning
    

---

## 🚀 4. Adam Optimizer

- Adaptive (adjusts updates per parameter)
    
- Often faster and more reliable than SGD
    

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## 🧹 5. Zeroing Gradients

- Gradients accumulate by default → must reset every step

```python
optimizer.zero_grad()
```

---

### 🔄 Training Loop Summary

```python
loss = loss_fn(pred, target)
loss.backward()     # compute gradients
optimizer.step()    # update weights
optimizer.zero_grad()
```

---

### 🏁 Key Idea

- Loss → measures error
    
- Backward → finds cause (gradients)
    
- Optimizer → fixes it (updates weights)

# Device Management

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
- Moving model to device (actually moves the model's parameters)
``` python
model = MyModel().to(device)
```
- Training Loop
``` python
for inputs, targets in dataloader: 
	inputs = inputs.to(device)
	targets = targets.to(device)
```
- always assign the ``x.to(device)`` to the variable
- CUDA OOM Error : Try lowering batch size first
- Data and model should be on the same device
