$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g^2_t
$$


$$
\hat{m}_t = m_t/(1 - \beta_1^t)\\
\hat{v}_t = v_t/(1- \beta_2^t)
$$

$$
\theta_{t} = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

$$
g_{t} = \nabla \hat{L}(\theta_{t}) \\ 
m_{t} = \beta_{1} m_{t-1} + (1-\beta_{1}) g_{t} \\ 
v_{t} = \beta_{2} v_{t-1} + (1-\beta_{2}) g_{t}^{2} \\ 
$$



$$
\hat{m}_{t} = \frac{m_{t}}{1-\beta_{1}^{t}} \\ 
\hat{v}_{t} = \frac{v_{t}}{1-\beta_{2}^{t}} \\ \theta_{t+1} = \theta_{t} - 
$$

$$
\frac{\eta}{\sqrt{\hat{v}_{t}} + \epsilon} \hat{m}_{t}
$$
