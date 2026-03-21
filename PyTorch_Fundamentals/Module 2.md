# Core Data Tools
## Transforms
``` python
transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((mean,), (std,))
])
```
- ToTensor() : converts to pytorch tensors and centres around 0 & 1
- Normalize() : mean and standard deviation
## Dataset
![](<Pasted image 20260321115211.png>)

## DataLoader

`` train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model Building
![](<Pasted image 20260321120025.png>)

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

# 🔁 Backpropagation & Optimizers 
## 📉 1. Backward (Gradient Computation)
- `loss.backward()` computes **gradients for each parameter**
- Gradients = how much each weight contributed to the error :contentReference[oaicite:0]{index=0}  

> ❗ Backward does NOT update weights → only calculates gradients

---

## 🧠 2. Gradient Intuition
- Think: **standing on a hill**
- Gradient = direction of steepest increase  
- To minimize loss → move **opposite direction** (gradient descent)

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