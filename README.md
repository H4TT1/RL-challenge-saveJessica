#  Strategies to Save the Morties  

## INFOS

### **Noms & Logins**
- **Ismail Hatti** — `H4TT1_` —`7800892765d2da5fd9c73ae51e358fede5428ddd`
- **Sekkat Idriss Mohammed** — `imsekkat`—`ce8951b41146f1b03e4385caed60221b824407c2`



## Introduction


Throughout this project, we progressively implemented and compared several reinforcement-learning–inspired algorithms.  
This **README** provides an overview of each strategy, including the underlying intuition, strengths, weaknesses, and the motivation behind each improvement.

> **Note:** All strategies are implemented in the **`strategies/`** directory.

# 1. UCB (Non-contextual baseline)

###  Intuition
Use classical **UCB**, treating each planet as a stationary arm.

###  Observations
- Simple baseline  
- Completely ignores periodicity  
- Reacts slowly to changes



###  Limitations
Assumes stationarity. The environment is periodic → UCB is too blind.

---

# 2. LinUCB — First contextual approach

###  Intuition
Model reward as a *linear* function of time using a contextual bandit.

###  Observations
- Attempt to add time context  
- Linear model cannot capture sinusoidal oscillations


###  Limitations
Linear relation between $t$ and $p(t)$ is false.

---

# 3. LinUCBv2 — Adding sinusoidal features

###  Intuition
Add handcrafted periodic features:
- $cos(t)$
- $sin(t)$

so the linear model can represent oscillations.

###  Observations
- Better generalization  
- Still mixes all planets together  
- Fails on long period $T=200$


###  Limitations
Each planet has its own frequency → a global model is not appropriate.

---

# 4. PeriodStrategy — Phase indexed averages

###  Intuition
Use the known period T of each planet and estimate:
$\text{phase} = t \bmod T$
Then keep a separate mean for each phase.

###  Observations
- Strong performance on $T=10$ and $T=20$
- $T=200$ extremely sparse → noisy phases


###  Limitations
Sensitive to noise, no smoothing, weak slope detection.

---

# 5. PeriodStrategyV2 — Improved phase-based model (Best so far)


###  Improvements
- 3-point smoothing
- Slope estimation across 5 phases
- Balanced variance estimation (Welford)
- UCB using phase uncertainty
- Improved discovery: heavy sampling for $T=200$
- Robust batch choice: rising → batch=3, stable → batch=2, falling → batch=1

###  Observations
- Tracks $T=10$ and $T=20$ perfectly  
- With enough samples, handles $T=200$ well  
- Very stable behaviour run after run  



###  Why it works
- Directly uses known periodic structure  
- Local estimation per phase (no global assumptions)  
- Variance-aware exploration  
- Slope detection helps exploit rising phases

---

# 6. Kalman Periodic Model — Fourier + Kalman

###  Intuition
Represent each planet as:
$p(t) = a_0 + a_c \cos(\omega t) + a_s \sin(\omega t)$
Use a Kalman filter to estimate $θ = (a0, ac, as)$.

###  Observations
- Elegant and mathematically sound  
- Works well on fast planets  
- Needs a lot of data for $T=200$
- Very sensitive to hyperparameters (Q/R)



###  Limitations
- Q too small → slow, rigid  
- Q too large → noisy  
- R too small → overreact  
- R too large → slow reaction  

Currently less stable than *PeriodStrategyV2*.

---
