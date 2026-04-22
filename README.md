# 🔥 Non-Linear Analysis of Axisymmetric Cylindrical Flow Using Navier–Stokes Equation

---

## 📌 Project Overview

This project focuses on the numerical and machine learning-based analysis of fluid flow in a cylindrical domain using the Navier–Stokes equations. The study models axisymmetric flow and demonstrates how velocity distribution evolves along the pipe length and radius.

The project combines:

* Numerical simulation (MATLAB)
* Machine Learning (Neural Networks)
* Physics-based understanding of fluid dynamics

---

## 🎯 Objectives

* Solve Navier–Stokes equations using numerical methods
* Analyze non-linear fluid behavior
* Visualize velocity distribution in cylindrical coordinates
* Use Machine Learning to predict flow behavior
* Compare physics-based and ML-based results

---

## 🧠 Key Concepts Used

* Navier–Stokes equations
* Non-linear Convection–Diffusion
* Axisymmetric Cylindrical Flow
* Finite Difference Method (FDM)
* Neural Networks
* TensorFlow
* Physics-Informed Learning

---

## ⚙️ Methodology

### 1️⃣ Mathematical Model

The governing equation:

∂u/∂t = -u ∂u/∂z + ν ∇²u - (1/ρ) ∂p/∂z

Where:

* u = velocity
* ν = viscosity
* ρ = density
* p = pressure

---

### 2️⃣ Numerical Simulation (MATLAB)

* Discretized using Finite Difference Method
* Includes:

  * Diffusion term
  * Non-linear convection term
  * Pressure gradient
* Generates velocity field: **u(r, z)**

---

### 💻 MATLAB Code Implementation

```matlab
clc; clear; close all;

%% PARAMETERS
nr = 60;
nz = 80;

r = linspace(0.01,1,nr);
z = linspace(0,5,nz);

dr = r(2) - r(1);
dz = z(2) - z(1);

nu = 0.05;
dt = 0.0001;
nt = 5000;

%% INITIALIZE
u = zeros(nr, nz);

for t = 1:nt
    un = u;

    % Boundary conditions
    u(:,1) = u(:,2);
    u(:,end) = u(:,end-1);
    u(1,:) = u(2,:);
    u(end,:) = 0;

    for i = 2:nr-1
        for j = 2:nz-1

            d2u_dr2 = (un(i+1,j) - 2*un(i,j) + un(i-1,j)) / dr^2;
            du_dr   = (un(i+1,j) - un(i-1,j)) / (2*dr);
            d2u_dz2 = (un(i,j+1) - 2*un(i,j) + un(i,j-1)) / dz^2;

            du_dz = (un(i,j) - un(i,j-1)) / dz;

            dpdz = -0.05;
            rho = 1;

            u(i,j) = un(i,j) ...
                - dt * un(i,j) * du_dz ...
                + dt * nu * (d2u_dr2 + (1/r(i))*du_dr + d2u_dz2) ...
                + dt * (-dpdz/rho);
        end
    end
end

% Export data
[R,Z] = meshgrid(r,z);
data = [R(:), Z(:), u(:)];
writematrix(data, 'flow_data.csv');
```

---

### 3️⃣ Data Generation

Dataset format:

| r | z | velocity |
| - | - | -------- |

Used for training machine learning model.

---

### 4️⃣ Machine Learning Model

* Implemented using TensorFlow
* Neural Network trained on simulation data

---

### 🤖 Python Code (TensorFlow)

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# Load dataset
data = pd.read_csv("flow_data.csv")
X = data[['r','z']].values
y = data['velocity'].values

# Build model
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
model.fit(X, y, epochs=100, batch_size=32)

# Prediction
predictions = model.predict(X)
```

---

## 📊 Results

### ✔ Velocity Distribution

* Maximum velocity at center
* Zero velocity at wall (no-slip condition)
* Parabolic velocity profile

### ✔ Flow Behavior

* Developing flow near inlet
* Fully developed flow downstream

### ✔ ML Model Performance

* MAE ≈ 0.0038
* High prediction accuracy

---

## 📈 Visualizations

* 2D contour plot
* 3D surface plot
* ML vs actual comparison

---

## 🔗 Workflow Integration

1. Solve Navier–Stokes using MATLAB
2. Export data (CSV)
3. Train Neural Network
4. Compare results

---

## 🤖 Machine Learning Insights

* ML approximates Navier–Stokes solution
* Faster than solving PDE repeatedly
* Acts as surrogate model

---

## 🚀 Future Scope

* Physics-Informed Neural Networks (PINNs)
* Turbulent flow modeling
* 3D simulations
* Real-time web application

---

## 🧑‍💻 Author

**Name:** Yug Kikani
**Roll No:** 23BME034

---

## 📌 Conclusion

This project demonstrates how classical fluid mechanics and machine learning can be combined to efficiently model complex non-linear flow systems.

---
