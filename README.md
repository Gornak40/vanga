# Vanga
### Vanga class
Let you have a hidden function f and we have n pairs (x, f(x)). If f(x) is similar to:

a[n - 1] * pow(x, (n - 1) * step) + a[n - 2] * pow(x, (n - 1) * step) + ... + a[1] * pow(x, step) + a[0]

then Vanga can easily emulate it very accurately with O(n ** 3) time for training and O(n) time for prediction.

But with huge n it is not optimal way.

# NoiseVanga class
This implementation of vanga algorithm is much more fast. Let us fix the value n, which is now doesn't connected to the size of training set, step and the number of training epochs E. NoiseVanga will use Vanga class E times with n random pairs from training set and calculate the averange of weights.

Now the training time is O(E * n ** 3), but n is constant so O(E) and prediction time is O(1). Also this algorithm can stay some noise with huge datasets.
