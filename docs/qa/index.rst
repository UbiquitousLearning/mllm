FAQ
=====

Precision Issues Caused by ffast-math Flag
------------------------------------------

Precision issues in computations may be caused by the ``ffast-math`` compiler flag which is enabled by default in some tasks. This flag allows the compiler to violate strict IEEE 754 compliance for floating-point operations, potentially resulting in faster execution but less accurate results.

If you encounter precision-related issues, users can remove this flag from their task configuration files. For kernel developers, you can manually specify compilation flags using UnsafeMacros to control the behavior of specific kernels.

For more detailed information about the implications of the fast-math optimizations, please refer to: https://simonbyrne.github.io/notes/fastmath/
