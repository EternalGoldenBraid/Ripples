# Discrete Update for 2D Damped Wave Equation

## Continuous PDE

$$
\frac{\partial^2 Z}{\partial t^2}
= c^2 \nabla^2 Z - \gamma \frac{\partial Z}{\partial t}.
$$

---

## Approximate time derivatives

### Second time derivative

$$
\frac{\partial^2 Z}{\partial t^2} 
\approx 
\frac{Z^{n+1}_{i,j} - 2Z^n_{i,j} + Z^{n-1}_{i,j}}{\Delta t^2}.
$$

### First time derivative (damping)

$$
\frac{\partial Z}{\partial t}
\approx
\frac{Z^n_{i,j} - Z^{n-1}_{i,j}}{\Delta t}.
$$

---

## Approximate spatial derivatives (Laplacian)

### In x

$$
\frac{\partial^2 Z}{\partial x^2}
\approx
\frac{Z_{i+1,j} - 2Z_{i,j} + Z_{i-1,j}}{(\Delta x)^2}.
$$

### In y

$$
\frac{\partial^2 Z}{\partial y^2}
\approx
\frac{Z_{i,j+1} - 2Z_{i,j} + Z_{i,j-1}}{(\Delta y)^2}.
$$

### Combined 2D Laplacian (five-point stencil)

$$
\nabla^2 Z_{i,j}
\approx
\frac{Z_{i+1,j} + Z_{i-1,j} + Z_{i,j+1} + Z_{i,j-1} - 4Z_{i,j}}{(\Delta x)^2}.
$$

---

## Solve for next time value

$$
\frac{Z^{n+1}_{i,j} - 2Z^n_{i,j} + Z^{n-1}_{i,j}}{\Delta t^2}
= c^2 \nabla^2 Z^n_{i,j} - \gamma \frac{Z^n_{i,j} - Z^{n-1}_{i,j}}{\Delta t}.
$$

$$
\Longrightarrow
$$

$$
Z^{n+1}_{i,j}
= 2Z^n_{i,j}
- Z^{n-1}_{i,j}
+ (c \Delta t / \Delta x)^2 \bigl(Z_{i+1,j} + Z_{i-1,j} + Z_{i,j+1} + Z_{i,j-1} - 4Z_{i,j}\bigr)
- \gamma \Delta t \,(Z^n_{i,j} - Z^{n-1}_{i,j}).
$$

---

## Code expression (final form)

$$
Z_{\text{new}}
= \underbrace{2 Z - Z_{\text{old}}}_{\text{leap-frog}}
+ \underbrace{c2\_dt2 \cdot \text{laplacian}(Z)}_{\text{curvature}}
- \underbrace{(1 - \text{damping}) \cdot \Delta t \cdot (Z - Z_{\text{old}})}_{\text{damping correction}}.
$$

Where

$$
c2\_dt2 = \left(\frac{c \Delta t}{\Delta x}\right)^2.
$$

---

## Stability condition

$$
\frac{c \Delta t}{\Delta x} \le \frac{1}{\sqrt{2}}.
$$

---

## Graph Laplacian viewpoint

The discrete Laplacian matrix \(L\) on a regular 2D grid with 4-connected neighbours corresponds to

$$
L Z = -4Z_{i,j} + Z_{i+1,j} + Z_{i-1,j} + Z_{i,j+1} + Z_{i,j-1}.
$$

This is exactly the five-point stencil.

---
