$$
\cal{L}(z) = \log \left( \frac{e^{z-c}}{\sum_{i=0}^n e^{z_i - c}} \right) = z - c - \log \left( \sum_{i=0}^n  e^{z_i - c}\right)
$$

$$
\frac{\partial L}{\partial z_i} = 1 - \frac{\partial \log \sum_{i=0}^n e^{z_i - c}}{\partial z_i}
$$

$$
\frac{\partial L}{\partial z_i} = 1 - \frac{1}{\sum_{i=0}^n e^{z_i - c}} \frac{\partial \sum_{i=0}^n e^{z_i - c}}{\partial z_i}\\
= 1 - softmax(z)
$$